from pathlib import Path


def test_native_mtp_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_native_mtp_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-native-mtp-contract-after-noheavy-contract-refresh-20260608.json"
    )


def test_native_mtp_contract_pins_named_policy_and_panel_edges():
    from tests.cross_matrix import run_native_mtp_contract as gate

    required = gate.REQUIRED_NATIVE_MTP_TEST_MARKERS
    sources = set(gate.SOURCE_HASH_FILES)

    assert "test_cli_exposes_native_mtp_runtime_flags" in required
    assert "test_qwen36_nested_config_and_layered_tensors_are_native_ready" in required
    assert "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready" in required
    assert "test_qwen_text_sanitize_mixed_shard_shifts_only_raw_mtp_norms" in required
    assert "test_native_mtp_detection_uses_weights_not_path_name" in required
    assert "test_runtime_metadata_can_explicitly_drop_configured_mtp" in required
    assert "test_jang2k_profile_alone_does_not_block_native_mtp_runtime" in required
    assert "test_jang_quant_mode_supports_mxfp8_metadata" in required
    assert "test_native_mtp_depth_defaults_to_three" in required
    assert "test_native_mtp_depth_uses_validated_model_tuning_sidecar_by_default" in required
    assert "test_mllm_generator_runs_depth3_native_mtp_verify_cycle" in required
    assert "test_mllm_native_mtp_enables_sampled_requests_with_stochastic_verify" in required
    assert "test_qwen36_vlm_mtp_gdn_sink_does_not_leak_to_upstream_originals" in required
    assert "test_native_mtp_adaptive_depth_lowers_d3_after_poor_third_position" in required
    assert "test_native_mtp_stats_snapshot_exposes_acceptance_depth_and_timings" in required
    assert "test_partial_indexed_layers_flagged" in required
    assert "defaults native-MTP bundles to Adaptive with the bundle sampler (compatible-only), no depth" in required
    assert "lets users choose AR (MTP off) without leaving sampling overrides behind" in required
    assert "keeps non-MTP models on bundle-owned generation defaults" in required
    assert "real session launcher and settings form expose exactly the two native MTP modes" in required
    assert "vmlx_engine/text_mtp/qwen_family.py" in sources
    assert "vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py" in sources

    engine_command = gate.COMMANDS["engine_native_mtp_contracts"][1]
    panel_command = gate.COMMANDS["panel_native_mtp_controls"][1]
    assert "-vv" in engine_command
    assert "--reporter=verbose" in panel_command


def test_native_mtp_contract_rejects_success_output_missing_a_required_edge(monkeypatch):
    from tests.cross_matrix import run_native_mtp_contract as gate

    omitted = "ignores retired fixed-depth / greedy-only persisted keys: legacy rows launch adaptive"
    emitted = "\n".join(m for m in gate.REQUIRED_NATIVE_MTP_TEST_MARKERS if m != omitted)
    monkeypatch.setattr(gate, "_run", lambda *args: {
        "returncode": 0, "counts": {"passed": 200}, "stdout": emitted,
    })
    result = gate.build_artifact(Path(__file__).resolve().parents[1])
    assert result["failed"] == []
    assert result["missing_markers"] == [omitted]
    assert result["checks"]["all_required_policy_edges_executed"] is False
    assert result["status"] == "fail"


def test_native_mtp_contract_accepts_all_required_edges(monkeypatch):
    from tests.cross_matrix import run_native_mtp_contract as gate

    monkeypatch.setattr(gate, "_run", lambda *args: {
        "returncode": 0, "counts": {"passed": 200},
        "stdout": "\n".join(gate.REQUIRED_NATIVE_MTP_TEST_MARKERS),
    })
    result = gate.build_artifact(Path(__file__).resolve().parents[1])
    assert result["missing_markers"] == []
    assert result["status"] == "pass"
