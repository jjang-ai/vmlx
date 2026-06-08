from pathlib import Path


def test_mimo_v2_local_bundle_metadata_contract_pins_both_local_bundles():
    from tests.cross_matrix import run_mimo_v2_local_bundle_metadata_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-mimo-v2-local-bundle-metadata-contract-20260607.json"
    )
    assert set(gate.MIMO_LOCAL_BUNDLES) == {"jangtq2", "jang2l"}
    assert gate.EXPECTED_PRESERVED_MODALITIES == ["vision", "audio"]
    assert gate.EXPECTED_RUNTIME_STATUS == "weights_preserved_text_runtime"


def test_mimo_v2_local_bundle_metadata_contract_reports_text_runtime_sidecars(tmp_path, monkeypatch):
    from tests.cross_matrix import run_mimo_v2_local_bundle_metadata_contract as gate

    bundles = {}
    for name in ("jangtq2", "jang2l"):
        root = tmp_path / name
        root.mkdir()
        (root / "audio_tokenizer").mkdir()
        (root / "jang_config.json").write_text("{}\n", encoding="utf-8")
        (root / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
        (root / "config.json").write_text(
            '{"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],'
            '"vision_config":{},"audio_config":{},'
            '"capabilities":{"modalities":["text"],'
            '"preserved_modalities":["vision","audio"],'
            '"unwired_modalities":["vision","audio"],'
            '"multimodal_status":"weights_preserved_text_runtime"},'
            '"runtime":{"multimodal_mode":"weights_preserved_text_runtime"}}\n',
            encoding="utf-8",
        )
        bundles[name] = root
    monkeypatch.setattr(gate, "MIMO_LOCAL_BUNDLES", bundles)

    artifact = gate.build_artifact()

    assert artifact["status"] == "pass"
    assert artifact["bundles"]["jangtq2"]["sidecars"] == {
        "vision_config": True,
        "audio_config": True,
        "preprocessor_config": True,
        "audio_tokenizer": True,
    }
    assert artifact["bundles"]["jang2l"]["capabilities"] == {
        "modalities": ["text"],
        "preserved_modalities": ["vision", "audio"],
        "unwired_modalities": ["vision", "audio"],
        "multimodal_status": "weights_preserved_text_runtime",
    }


def test_mimo_v2_local_bundle_metadata_contract_rejects_overadvertised_media(tmp_path, monkeypatch):
    from tests.cross_matrix import run_mimo_v2_local_bundle_metadata_contract as gate

    root = tmp_path / "bad"
    root.mkdir()
    (root / "audio_tokenizer").mkdir()
    (root / "jang_config.json").write_text("{}\n", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    (root / "config.json").write_text(
        '{"model_type":"mimo_v2","vision_config":{},"audio_config":{},'
        '"capabilities":{"modalities":["text","vision","audio"]},'
        '"runtime":{"multimodal_mode":"weights_preserved_text_runtime"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "MIMO_LOCAL_BUNDLES", {"jang2l": root})

    artifact = gate.build_artifact()

    assert artifact["status"] == "fail"
    assert "runtime_modalities_not_text_only" in artifact["bundles"]["jang2l"]["failures"]
    assert "preserved_modalities_not_recorded" in artifact["bundles"]["jang2l"]["failures"]
    assert "unwired_modalities_not_recorded" in artifact["bundles"]["jang2l"]["failures"]
