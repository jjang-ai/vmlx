from pathlib import Path


def test_native_mtp_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_native_mtp_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-native-mtp-contract-after-mimo-capability-snapshot-fix-20260607.json"
    )


def test_native_mtp_contract_pins_named_policy_and_panel_edges():
    from tests.cross_matrix import run_native_mtp_contract as gate

    required = gate.REQUIRED_NATIVE_MTP_TEST_MARKERS

    assert "test_cli_exposes_native_mtp_runtime_flags" in required
    assert "test_qwen36_nested_config_and_layered_tensors_are_native_ready" in required
    assert "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready" in required
    assert "test_native_mtp_detection_uses_weights_not_path_name" in required
    assert "test_runtime_metadata_can_explicitly_drop_configured_mtp" in required
    assert "test_jang2k_profile_blocks_native_mtp_runtime_but_keeps_vl_artifact_route" in required
    assert "test_jang_quant_mode_supports_mxfp8_metadata" in required
    assert "test_native_mtp_depth_defaults_to_three" in required
    assert "test_native_mtp_depth_uses_validated_model_tuning_sidecar_by_default" in required
    assert "test_mllm_generator_runs_depth3_native_mtp_verify_cycle" in required
    assert "test_mllm_native_mtp_only_enables_for_proven_deterministic_sampling" in required
    assert "test_qwen36_vlm_mtp_gdn_sink_does_not_leak_to_upstream_originals" in required
    assert "test_native_mtp_adaptive_depth_lowers_d3_after_poor_third_position" in required
    assert "test_native_mtp_stats_snapshot_exposes_acceptance_depth_and_timings" in required
    assert "test_partial_indexed_layers_flagged" in required
    assert "defaults native-MTP bundles to deterministic measured-depth launch policy without hidden sampler flags" in required
    assert "lets users disable native MTP without leaving deterministic sampling overrides behind" in required
    assert "keeps non-MTP models on bundle-owned generation defaults" in required
    assert "real session launcher and settings form expose native MTP controls" in required

    engine_command = gate.COMMANDS["engine_native_mtp_contracts"][1]
    panel_command = gate.COMMANDS["panel_native_mtp_controls"][1]
    assert "-vv" in engine_command
    assert "--reporter=verbose" in panel_command
