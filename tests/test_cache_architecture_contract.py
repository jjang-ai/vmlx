from pathlib import Path


def test_cache_architecture_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-cache-architecture-contract-20260602-v1554-attention-matmul-refresh.json"
    )


def test_cache_architecture_contract_hashes_real_tq_disk_store_source():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    assert "vmlx_engine/tq_disk_store.py" in gate.SOURCE_HASH_FILES
    assert "vmlx_engine/tq_disk_cache.py" not in gate.SOURCE_HASH_FILES
    assert Path("vmlx_engine/tq_disk_store.py").is_file()


def test_block_prefix_cache_allowed_heads_include_mimo_swa_kv_heads():
    from types import SimpleNamespace

    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="mimo_v2",
            num_key_value_heads=4,
            swa_num_key_value_heads=8,
        )
    )
    cache._n_kv_heads = None
    cache._allowed_n_kv_heads = None

    assert cache._get_allowed_n_kv_heads() == {4, 8}


def test_cache_architecture_contract_api_child_artifact_is_current():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    expected = Path(
        "build/current-api-cache-contract-cache-architecture-check-20260528-named-family-registry-matrix.json"
    )
    command = gate.COMMANDS["api_cache_status_contracts"].cmd

    assert gate.API_CACHE_CONTRACT_ARTIFACT == expected
    assert "--out" in command
    assert Path(command[command.index("--out") + 1]) == expected
    assert Path("build/current-api-cache-contract-cache-architecture-check-20260521.json") != expected


def test_cache_architecture_contract_pins_named_cache_edges():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    required_pytest = gate.REQUIRED_CACHE_TEST_MARKERS
    required_api = gate.REQUIRED_API_CACHE_CHECKS
    required_api_command = gate.REQUIRED_API_CACHE_COMMAND_MARKERS
    required_panel = gate.REQUIRED_PANEL_CACHE_MARKERS

    assert "test_memory_pressure_reuses_shorter_dsv4_terminal_composite_prefix" in required_pytest
    assert "test_memory_pressure_refuses_dsv4_partial_without_terminal_composite" in required_pytest
    assert "test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint" in required_pytest
    assert "test_prompt_disk_l2_hit_backfills_paged_cache_for_partial_reuse" in required_pytest
    assert "test_hybrid_ssm_checkpoint_alignment_falls_back_to_exact_aligned_state" in required_pytest
    assert "test_hybrid_ssm_auto_mode_disables_live_tq_but_keeps_stored_kv_q4" in required_pytest
    assert "test_accepts_scheduler_owned_ssm_l2_store" in required_pytest
    assert "test_scheduler_creates_matching_ssm_companion_l2_for_block_disk" in required_pytest
    assert "test_qwen3_5_moe_linear_attention_keeps_selective_live_tq_and_stored_kv_q4" in required_pytest
    assert "test_serialize_tq_cache_mixed_hybrid" in required_pytest
    assert "test_tq_tensors_roundtrip_via_safetensors" in required_pytest
    assert "test_mllm_stats_include_cache_fields" in required_pytest
    assert "native_cache_status_reports_dsv4_separately_from_tq_kv" in required_api_command
    assert "native_cache_status_reports_zaya_typed_cca" in required_api_command
    assert "native_cache_status_reports_plain_attention_kv" in required_api_command
    assert "dsv4_native_cache_status" in required_api
    assert "zaya_typed_cca_status" in required_api
    assert "plain_attention_kv_status" in required_api
    assert "hybrid_ssm_partial_reuse" in required_api
    assert "turboquant_disk_roundtrip" in required_api

    assert "deepseek-v4 enables native composite prefix cache by default even with stale cache config" in required_panel
    assert "deepseek-v4 native cache path uses DS4 page-sized blocks" in required_panel
    assert "DSV4 pool quant and native prefix controls stay DSV4-only" in required_panel
    assert (
        "enables DSV4 pool quant by default only under DSV4 native composite cache"
        in required_panel
    )
    assert "detected Qwen3.6 hybrid cache forces paged cache over stale saved false" in required_panel
    assert "detected Mamba cache forces paged cache while regular KV respects saved false" in required_panel
    assert "disabling prefix cache disables all dependent features" in required_panel

    command = gate.COMMANDS["cache_family_pytest"].cmd
    assert "-vv" in command
    assert "tests/test_engine_audit.py" in command


def test_cache_architecture_contract_pins_panel_launch_cache_policy():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    assert "vmlx_engine/paged_cache.py" in gate.SOURCE_HASH_FILES
    assert "vmlx_engine/block_disk_store.py" in gate.SOURCE_HASH_FILES
    assert "panel/src/main/sessions.ts" in gate.SOURCE_HASH_FILES
    assert "panel/src/shared/dsv4Env.ts" in gate.SOURCE_HASH_FILES
    assert "panel/src/shared/cacheControlPolicy.ts" in gate.SOURCE_HASH_FILES
    assert "panel/src/renderer/src/components/sessions/CachePanel.tsx" in gate.SOURCE_HASH_FILES
    assert "panel/src/renderer/src/components/sessions/PerformancePanel.tsx" in gate.SOURCE_HASH_FILES
    assert "panel/tests/dsv4-env.test.ts" in gate.SOURCE_HASH_FILES
    assert "panel/tests/settings-flow.test.ts" in gate.SOURCE_HASH_FILES
    assert "panel/tests/cache-control-policy.test.ts" in gate.SOURCE_HASH_FILES

    panel_command = gate.COMMANDS["panel_cache_launch_policy"]
    assert panel_command.cwd == "panel"
    assert panel_command.cmd[:4] == ["npm", "exec", "vitest", "--"]
    assert "tests/settings-flow.test.ts" in panel_command.cmd
    assert "-t" in panel_command.cmd
    assert "deepseek-v4|DSV4|Qwen3\\.6|Mamba|cache" in panel_command.cmd


def test_cache_architecture_contract_publishes_structured_family_matrix():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    expected_rows = {
        "dsv4_native_composite",
        "dsv4_swa_hca_csa_components",
        "zaya_typed_cca",
        "zaya_vl_hybrid_registry",
        "plain_attention_kv",
        "hybrid_ssm",
        "ling_bailing_hybrid_registry",
        "nemotron_h_hybrid_registry",
        "qwen36_hybrid_tq",
        "qwen36_registry_hybrid_parser",
        "gemma4_mixed_swa_kv",
        "step37_full_sliding_kv_registry",
        "lfm25_moe_hybrid_registry",
        "hy_v3_kv_registry",
        "minimax_parser_boundary_registry",
        "prompt_disk_l2",
        "turboquant_disk",
        "mllm_media_cache",
    }

    assert set(gate.REQUIRED_CACHE_FAMILY_MATRIX) == expected_rows
    for row_id, requirement in gate.REQUIRED_CACHE_FAMILY_MATRIX.items():
        assert requirement["checks"], row_id
        assert (
            requirement["markers"]
            or requirement["api_checks"]
            or requirement["api_command_markers"]
            or requirement["panel_markers"]
        ), row_id

    dsv4 = gate.REQUIRED_CACHE_FAMILY_MATRIX["dsv4_native_composite"]
    assert "dsv4_native_composite_cache_status" in dsv4["checks"]
    assert "dsv4_terminal_composite_contracts" in dsv4["checks"]
    assert (
        "test_memory_pressure_refuses_dsv4_partial_without_terminal_composite"
        in dsv4["markers"]
    )

    dsv4_components = gate.REQUIRED_CACHE_FAMILY_MATRIX[
        "dsv4_swa_hca_csa_components"
    ]
    assert "dsv4_swa_hca_csa_component_contracts" in dsv4_components["checks"]
    assert (
        "test_memory_pressure_reuses_shorter_dsv4_terminal_composite_prefix"
        in dsv4_components["markers"]
    )
    assert (
        "test_memory_pressure_refuses_dsv4_partial_without_terminal_composite"
        in dsv4_components["markers"]
    )

    qwen = gate.REQUIRED_CACHE_FAMILY_MATRIX["qwen36_hybrid_tq"]
    assert "turboquant_kv_runtime_contract" in qwen["checks"]
    assert (
        "test_qwen3_5_moe_linear_attention_keeps_selective_live_tq_and_stored_kv_q4"
        in qwen["markers"]
    )

    hybrid = gate.REQUIRED_CACHE_FAMILY_MATRIX["hybrid_ssm"]
    assert "hybrid_ssm_companion_l2_contracts" in hybrid["checks"]
    assert "test_accepts_scheduler_owned_ssm_l2_store" in hybrid["markers"]
    assert (
        "test_scheduler_creates_matching_ssm_companion_l2_for_block_disk"
        in hybrid["markers"]
    )

    gemma4 = gate.REQUIRED_CACHE_FAMILY_MATRIX["gemma4_mixed_swa_kv"]
    assert "gemma4_mixed_swa_kv_status" in gemma4["checks"]
    assert "test_native_cache_status_reports_mixed_swa_kv" in gemma4["markers"]
    assert (
        "test_mllm_rotating_kv_cache_is_kv_like_for_mixed_swa"
        in gemma4["markers"]
    )
    assert (
        "test_paged_cache_mixed_swa_reconstruct_preserves_full_kv_length"
        in gemma4["markers"]
    )
    assert (
        "test_step37_registry_subtype_marks_scheduler_mixed_attention"
        in gate.REQUIRED_CACHE_TEST_MARKERS
    )

    step37 = gate.REQUIRED_CACHE_FAMILY_MATRIX["step37_full_sliding_kv_registry"]
    assert "named_family_registry_cache_parser_contracts" in step37["checks"]
    assert "test_step37_flash_jang_config" in step37["markers"]
    assert (
        "test_native_cache_status_reports_step37_full_sliding_kv_from_registry_subtype"
        in step37["markers"]
    )
    assert (
        "test_step37_registry_subtype_marks_scheduler_mixed_attention"
        in step37["markers"]
    )

    lfm25 = gate.REQUIRED_CACHE_FAMILY_MATRIX["lfm25_moe_hybrid_registry"]
    assert "named_family_registry_cache_parser_contracts" in lfm25["checks"]
    assert "test_lfm2_moe_config" in lfm25["markers"]
    assert "test_lfm2_base_config" in lfm25["markers"]
    assert "test_lfm2_base_model_type_uses_hybrid_ssm_companion_cache" in lfm25["markers"]

    ling = gate.REQUIRED_CACHE_FAMILY_MATRIX["ling_bailing_hybrid_registry"]
    assert "named_family_registry_cache_parser_contracts" in ling["checks"]
    assert (
        "test_jang_stamp_model_type_alias_preserves_registered_family_fields"
        in ling["markers"]
    )

    nemotron = gate.REQUIRED_CACHE_FAMILY_MATRIX["nemotron_h_hybrid_registry"]
    assert "test_nemotron_h_config" in nemotron["markers"]
    assert "test_nemotron_h_v2_config" in nemotron["markers"]
    assert (
        "test_nemotron_h_stale_omni_stamp_without_media_stays_text_hybrid"
        in nemotron["markers"]
    )

    qwen_registry = gate.REQUIRED_CACHE_FAMILY_MATRIX[
        "qwen36_registry_hybrid_parser"
    ]
    assert "test_qwen36_release_rows_intentionally_use_qwen3_5_family_alias" in qwen_registry["markers"]
    assert "test_qwen3_5_linear_attention_config_uses_hybrid_cache" in qwen_registry["markers"]

    zaya_vl = gate.REQUIRED_CACHE_FAMILY_MATRIX["zaya_vl_hybrid_registry"]
    assert "test_zaya1_vl_registered_with_full_contract" in zaya_vl["markers"]

    hy_v3 = gate.REQUIRED_CACHE_FAMILY_MATRIX["hy_v3_kv_registry"]
    assert "test_hy_v3_provisional_contract" in hy_v3["markers"]

    minimax = gate.REQUIRED_CACHE_FAMILY_MATRIX[
        "minimax_parser_boundary_registry"
    ]
    assert "test_minimax_config" in minimax["markers"]
    assert "test_minimax_eos_includes_role_boundary_marker" in minimax["markers"]

    assert "tests/test_model_config_registry.py" in gate.COMMANDS["cache_family_pytest"].cmd
    assert "tests/test_model_config_registry.py" in gate.SOURCE_HASH_FILES
