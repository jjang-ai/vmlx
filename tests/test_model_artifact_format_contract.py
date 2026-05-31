from pathlib import Path


def test_model_artifact_format_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_model_artifact_format_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-model-artifact-format-contract-20260531-post-step-lfm-refresh.json"
    )


def test_model_artifact_format_contract_pins_named_artifact_edges():
    from tests.cross_matrix import run_model_artifact_format_contract as gate

    required = gate.REQUIRED_ARTIFACT_TEST_MARKERS
    sources = set(gate.SOURCE_HASH_FILES)

    assert "test_load_jang_model_accepts_affine_weight_format" in required
    assert "test_dsv4_static_audit_reports_mtp_drop_contract" in required
    assert "test_gather_dn_uses_dp_bits" in required
    assert "test_sanitize_repairs_flat_2d_switch_mlp_to_3d" in required
    assert "test_sanitize_no_op_on_correct_3d_shape" in required
    assert "test_sanitize_restores_dwq_split_mla_kv_b_proj" in required
    assert "test_sanitize_trims_absent_mtp_layer_before_strict_load" in required
    assert "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready" in required
    assert "test_mxfp4_vlm_sanitize_shifts_mtp_norms_only" in required
    assert "test_mxfp_vlm_loader_quantizes_with_declared_mode" in required
    assert "test_jang_quant_mode_supports_mxfp8_metadata" in required
    assert "test_qwen36_plain_mlx_4bit_keeps_hybrid_cache_without_jang_or_mxfp" in required
    assert "test_native_mtp_detection_uses_weights_not_path_name" in required

    assert "vmlx_engine/loaders/load_jangtq.py" in sources
    assert "vmlx_engine/loaders/load_jangtq_vlm.py" in sources
    assert "vmlx_engine/loaders/load_jangtq_kimi_vlm.py" in sources
    assert "vmlx_engine/loaders/load_jangtq_dsv4.py" in sources
    assert "vmlx_engine/loaders/load_zaya.py" in sources
    assert "vmlx_engine/loaders/load_laguna.py" in sources
    assert "vmlx_engine/loaders/load_mistral3.py" in sources

    command = gate.COMMANDS["model_artifact_format_pytest"]
    assert "-vv" in command
    assert "mlx_4bit" in " ".join(command)
    assert "bailing" in " ".join(command)
    assert "switch_mlp" in " ".join(command)
