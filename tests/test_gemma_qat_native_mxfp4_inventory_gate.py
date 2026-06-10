import json
from pathlib import Path

from tests.cross_matrix import run_gemma_qat_native_mxfp4_inventory_gate as gate


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _bundle(
    root: Path,
    name: str,
    *,
    model_type: str,
    text_model_type: str | None = None,
    weight_format: str | None = None,
    vision: bool = True,
    audio: bool = True,
    video: bool = True,
) -> Path:
    path = root / name
    config = {"model_type": model_type}
    if text_model_type:
        config["text_config"] = {"model_type": text_model_type}
    if vision:
        config["vision_config"] = {"model_type": f"{model_type}_vision"}
        config["image_token_id"] = 1
    if audio:
        config["audio_config"] = {"model_type": f"{model_type}_audio"}
        config["audio_token_id"] = 2
    if video:
        config["video_token_id"] = 3
    if weight_format:
        config["weight_format"] = weight_format
        config["quantization"] = {
            "method": weight_format,
            "mode": weight_format,
            "bits": 4 if weight_format == "mxfp4" else 8,
            "group_size": 32,
        }
    _write_json(path / "config.json", config)
    _write_json(path / "generation_config.json", {"do_sample": False})
    _write_json(path / "tokenizer_config.json", {"chat_template": "stub"})
    _write_json(path / "processor_config.json", {"processor_class": "stub"})
    if weight_format:
        _write_json(path / "jang_config.json", {"weight_format": weight_format})
    return path


def _write_index(path: Path, keys: list[str]) -> None:
    _write_json(
        path / "model.safetensors.index.json",
        {"weight_map": {key: "model-00001.safetensors" for key in keys}},
    )


def test_gate_reports_missing_gemma4_e2b_e4b_qat_and_open_present_gemma4_rows(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-12B-it-qat-MXFP4",
        model_type="gemma4_unified",
        text_model_type="gemma4_unified_text",
        weight_format="mxfp4",
    )
    _bundle(
        tmp_path,
        "gemma-4-26B-A4B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )
    _bundle(
        tmp_path,
        "gemma-4-31B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )

    artifact = gate.build_artifact((tmp_path,))

    assert artifact["status"] == "open"
    assert artifact["checks"]["gemma4_e2b_qat_native_mxfp4_present"] is False
    assert artifact["checks"]["gemma4_e4b_qat_native_mxfp4_present"] is False
    assert artifact["checks"]["gemma4_12b_native_mxfp4_present"] is True
    assert artifact["checks"]["gemma4_26b_present"] is True
    assert artifact["checks"]["gemma4_31v_or_31b_present"] is True
    assert artifact["required_rows"]["gemma4_12b_native_mxfp4"]["status"] == "open"
    assert artifact["required_rows"]["gemma4_12b_native_mxfp4"]["matching_paths"] == [
        str(tmp_path / "gemma-4-12B-it-qat-MXFP4")
    ]
    assert (
        "responses_function_call_arguments_delta_done_output_item"
        in artifact["required_rows"]["gemma4_12b_native_mxfp4"]["live_proof_required"]
    )
    assert artifact["required_rows"]["gemma4_e2b_qat_native_mxfp4"]["status"] == "missing"


def test_gate_keeps_jangq_gemma4_e2b_e4b_qat_open_until_live_proof_exists(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-E2B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )
    _bundle(
        tmp_path,
        "gemma-4-E4B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )

    artifact = gate.build_artifact((tmp_path,))

    assert artifact["status"] == "open"
    assert artifact["checks"]["gemma4_e2b_qat_native_mxfp4_present"] is True
    assert artifact["checks"]["gemma4_e4b_qat_native_mxfp4_present"] is True
    assert artifact["required_rows"]["gemma4_e2b_qat_native_mxfp4"]["status"] == "open"
    assert artifact["required_rows"]["gemma4_e4b_qat_native_mxfp4"]["tool_parser"] == "gemma4"
    assert artifact["required_rows"]["gemma4_e4b_qat_native_mxfp4"]["reasoning_parser"] == "gemma4"
    assert artifact["required_rows"]["gemma4_e4b_qat_native_mxfp4"]["live_proof_status"] == "missing"
    assert artifact["checks"]["all_required_live_proofs_present"] is False


def test_gate_records_source_live_smoke_without_release_clearance(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-E2B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )
    proof = tmp_path / gate.SOURCE_LIVE_SMOKE_PROOFS["gemma4_e2b_qat_native_mxfp4"]
    _write_json(
        proof,
        {"status": "pass", "failed": 0, "completed": 1, "row_count": 1},
    )

    artifact = gate.build_artifact((tmp_path,), proof_root=tmp_path)

    row = artifact["required_rows"]["gemma4_e2b_qat_native_mxfp4"]
    assert row["status"] == "open"
    assert row["source_live_smoke"]["status"] == "pass"
    assert row["source_live_smoke"]["artifact"] == str(
        gate.SOURCE_LIVE_SMOKE_PROOFS["gemma4_e2b_qat_native_mxfp4"]
    )
    assert artifact["checks"]["all_required_live_proofs_present"] is False


def test_gate_records_source_smoke_video_runtime_and_recovery_proof(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-26B-A4B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
        audio=False,
        video=True,
    )
    proof = tmp_path / gate.SOURCE_LIVE_SMOKE_PROOFS["gemma4_26b_vl"]
    _write_json(
        proof,
        {
            "status": "pass",
            "failed": 0,
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "requests": [
                        {
                            "label": "vl_blue_video",
                            "code": 200,
                            "content": "Blue",
                            "validation_failures": [],
                        },
                        {
                            "label": "text_no_media_after_video",
                            "code": 200,
                            "content": "NONE",
                            "validation_failures": [],
                        },
                    ]
                }
            ],
        },
    )

    artifact = gate.build_artifact((tmp_path,), proof_root=tmp_path)
    source_smoke = artifact["required_rows"]["gemma4_26b_vl"]["source_live_smoke"]

    assert source_smoke["status"] == "pass"
    assert source_smoke["video_runtime_proven"] is True
    assert source_smoke["post_video_text_recovery_proven"] is True
    assert artifact["checks"]["gemma4_26b_video_runtime_proof_required"] is True
    assert artifact["checks"]["gemma4_26b_video_runtime_source_proven"] is True
    assert artifact["checks"]["all_required_live_proofs_present"] is False


def test_gate_uses_gemma4_e2b_e4b_row_ids_for_gemma4_qat_bundles(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-E2B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )
    _bundle(
        tmp_path,
        "gemma-4-E4B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
    )

    artifact = gate.build_artifact((tmp_path,))

    assert "gemma4_e2b_qat_native_mxfp4" in artifact["required_rows"]
    assert "gemma4_e4b_qat_native_mxfp4" in artifact["required_rows"]
    assert "gemma3n_e2b_qat_native4" not in artifact["required_rows"]
    assert "gemma3n_e4b_qat_native4" not in artifact["required_rows"]
    assert artifact["checks"]["gemma4_e2b_qat_native_mxfp4_present"] is True
    assert artifact["checks"]["gemma4_e4b_qat_native_mxfp4_present"] is True


def test_gate_does_not_treat_gemma4_audio_token_metadata_as_native_audio(tmp_path):
    bundle = _bundle(
        tmp_path,
        "gemma-4-12B-it-qat-MXFP4",
        model_type="gemma4_unified",
        text_model_type="gemma4_unified_text",
        weight_format="mxfp4",
        audio=True,
        video=True,
    )
    _write_index(
        bundle,
        [
            "embed_audio.embedding_projection.weight",
            "embed_vision.embedding_projection.weight",
            "vision_embedder.patch_dense.weight",
        ],
    )

    artifact = gate.build_artifact((tmp_path,))
    match = artifact["required_rows"]["gemma4_12b_native_mxfp4"]["matching_rows"][0]

    assert match["modality_backing"]["audio_advertised_by_config"] is True
    assert match["modality_backing"]["audio_weight_backed"] is False
    assert match["modality_backing"]["audio_embed_only"] is True
    assert match["modality_backing"]["audio_runtime_supported"] is False
    assert match["audio"] is False
    assert match["audio_declared_by_config"] is True
    assert match["modality_backing"]["vision_weight_backed"] is True
    row = artifact["required_rows"]["gemma4_12b_native_mxfp4"]
    assert row["declared_required_modalities"] == ["text", "vision", "audio", "video"]
    assert row["required_modalities"] == ["text", "vision", "video"]
    assert artifact["checks"]["gemma4_12b_audio_weight_backed"] is False
    assert artifact["checks"]["gemma4_12b_audio_honestly_gated"] is True
    assert "gemma-4-12B-it-qat-MXFP4: audio metadata present without audio_tower weights" in (
        artifact["required_rows"]["gemma4_12b_native_mxfp4"]["notes"]
    )


def test_gate_records_gemma4_video_as_runtime_proof_required_not_weight_backed(tmp_path):
    bundle = _bundle(
        tmp_path,
        "gemma-4-26B-A4B-it-qat-MXFP4",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="mxfp4",
        audio=False,
        video=True,
    )
    _write_index(bundle, ["vision_tower.encoder.layers.0.input_layernorm.weight"])

    artifact = gate.build_artifact((tmp_path,))
    match = artifact["required_rows"]["gemma4_26b_vl"]["matching_rows"][0]

    assert match["modality_backing"]["video_advertised_by_config"] is True
    assert match["modality_backing"]["video_weight_backed"] is False
    assert match["modality_backing"]["video_runtime_proof_required"] is True
    assert artifact["checks"]["gemma4_26b_video_runtime_proof_required"] is True
    assert "gemma-4-26B-A4B-it-qat-MXFP4: video metadata requires live frame-through-vision proof" in (
        artifact["required_rows"]["gemma4_26b_vl"]["notes"]
    )


def test_gate_tracks_all_gemma4_qat_jang4m_as_separate_open_proof_map_rows(tmp_path):
    for name, model_type, text_type in [
        ("gemma-4-E2B-it-qat-JANG_4M", "gemma4", "gemma4_text"),
        ("gemma-4-E4B-it-qat-JANG_4M", "gemma4", "gemma4_text"),
        ("gemma-4-12B-it-qat-JANG_4M", "gemma4_unified", "gemma4_unified_text"),
        ("gemma-4-26B-A4B-it-qat-JANG_4M", "gemma4", "gemma4_text"),
        ("gemma-4-31B-it-qat-JANG_4M", "gemma4", "gemma4_text"),
    ]:
        _bundle(
            tmp_path,
            name,
            model_type=model_type,
            text_model_type=text_type,
            weight_format="jang_4m",
        )
    _bundle(
        tmp_path,
        "gemma-4-12B-it-qat-MXFP4",
        model_type="gemma4_unified",
        text_model_type="gemma4_unified_text",
        weight_format="mxfp4",
    )

    artifact = gate.build_artifact((tmp_path,))

    for key, path_name in {
        "gemma4_e2b_qat_jang4m": "gemma-4-E2B-it-qat-JANG_4M",
        "gemma4_e4b_qat_jang4m": "gemma-4-E4B-it-qat-JANG_4M",
        "gemma4_12b_qat_jang4m": "gemma-4-12B-it-qat-JANG_4M",
        "gemma4_26b_qat_jang4m": "gemma-4-26B-A4B-it-qat-JANG_4M",
        "gemma4_31b_qat_jang4m": "gemma-4-31B-it-qat-JANG_4M",
    }.items():
        row = artifact["required_rows"][key]
        assert row["status"] == "open"
        assert row["variant"] == "qat_jang4m"
        assert row["matching_paths"] == [str(tmp_path / path_name)]
        assert artifact["checks"][f"{key}_present"] is True
        assert key in artifact["open_required_rows"]
        assert "autodetect_model_family_and_qat_jang4m_variant" in row["live_proof_required"]
        assert "mixed_swa_prefix_cache_first_miss_second_hit" in row["live_proof_required"]
        assert "responses_streaming_args_and_content_deltas" in row["live_proof_required"]
        assert "installed_app_parity" in row["live_proof_required"]
        assert row["live_proof_status"] == "missing"

    assert "gemma4_12b_native_mxfp4" in artifact["open_required_rows"]


def test_gate_does_not_count_non_qat_jang4m_as_qat_jang4m_row(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-12B-it-JANG_4M",
        model_type="gemma4_unified",
        text_model_type="gemma4_unified_text",
        weight_format="jang_4m",
    )
    _bundle(
        tmp_path,
        "gemma-4-12B-it-qat-MXFP4",
        model_type="gemma4_unified",
        text_model_type="gemma4_unified_text",
        weight_format="mxfp4",
    )

    artifact = gate.build_artifact((tmp_path,))
    row = artifact["required_rows"]["gemma4_12b_qat_jang4m"]

    assert row["status"] == "missing"
    assert row["matching_paths"] == []
    assert artifact["checks"]["gemma4_12b_qat_jang4m_present"] is False
    assert "gemma4_12b_qat_jang4m" in artifact["missing_required_rows"]
    assert "gemma4_12b_native_mxfp4" in artifact["open_required_rows"]
