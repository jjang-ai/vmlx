from pathlib import Path


def test_mimo_v2_local_bundle_metadata_contract_pins_both_local_bundles():
    from tests.cross_matrix import run_mimo_v2_local_bundle_metadata_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-mimo-v2-local-bundle-metadata-contract-20260607.json"
    )
    assert gate.DEFAULT_MANIFEST_OUT == Path(
        "build/current-mimo-jangtq2-local-manifest-20260607.tsv"
    )
    assert gate.DEFAULT_STRUCTURAL_OUT == Path(
        "build/current-mimo-jang2l-local-structural-verify-20260606.json"
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


def test_mimo_v2_local_bundle_metadata_contract_builds_noheavy_structural_proof(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import run_mimo_v2_local_bundle_metadata_contract as gate

    bundles = {}
    for name in ("jangtq2", "jang2l"):
        root = tmp_path / name
        root.mkdir()
        (root / "shard.safetensors").write_text("", encoding="utf-8")
        (root / "audio_tokenizer").mkdir()
        (root / "jang_config.json").write_text(
            (
                '{"profile":"JANGTQ_2","expert_layout":"stacked_switch_mlp",'
                '"runtime_expert_module":"switch_mlp","bundle_has_mtp":false}\n'
                if name == "jangtq2"
                else '{"profile":"JANG_2L","expert_layout":"stacked_switch_mlp",'
                '"runtime_expert_module":"switch_mlp","bundle_has_mtp":false}\n'
            ),
            encoding="utf-8",
        )
        (root / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
        (root / "generation_config.json").write_text(
            '{"temperature":1.0,"top_p":0.95,"max_new_tokens":2048}\n',
            encoding="utf-8",
        )
        (root / "config.json").write_text(
            '{"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],'
            '"vision_config":{},"audio_config":{},'
            '"capabilities":{"modalities":["text"],'
            '"preserved_modalities":["vision","audio"],'
            '"unwired_modalities":["vision","audio"],'
            '"multimodal_status":"weights_preserved_text_runtime",'
            '"tools":{"supported":true,"parser":"xml_function"},'
            '"reasoning":{"supported":true,"default":true,"parser":"think_xml"}},'
            '"runtime":{"multimodal_mode":"weights_preserved_text_runtime",'
            '"attention_impl":"hybrid_full_swa_sink","bundle_has_mtp":false,'
            '"mtp_mode":"absent","tq_layout":"prestacked_switch_mlp",'
            '"cache_topology":{"family":"hybrid_full_swa_kv",'
            '"prefix_cache":true,"l2_disk_cache":true,'
            '"turboquant_kv":"full_attention_layers_only",'
            '"swa_layers":"rotating_kv_native"}}}\n',
            encoding="utf-8",
        )
        weight_map = {
            key: "shard.safetensors"
            for key in sorted(gate.REQUIRED_BOOKEND_KEYS)
        }
        weight_map["model.layers.0.mlp.switch_mlp.gate_proj.weight"] = "shard.safetensors"
        (root / "model.safetensors.index.json").write_text(
            '{"weight_map":' + __import__("json").dumps(weight_map) + "}\n",
            encoding="utf-8",
        )
        if name == "jangtq2":
            (root / "jangtq_runtime.safetensors").write_text("", encoding="utf-8")
        bundles[name] = root
    monkeypatch.setattr(gate, "MIMO_LOCAL_BUNDLES", bundles)

    artifact = gate.build_artifact()
    structural = gate.build_structural_artifact(artifact)
    manifest = tmp_path / "manifest.tsv"

    gate.write_manifest(tmp_path, bundles["jangtq2"], manifest)

    assert structural["status"] == "pass"
    assert structural["bundles"]["jangtq2"]["cache_topology"] == {
        "family": "hybrid_full_swa_kv",
        "prefix_cache": True,
        "l2_disk_cache": True,
        "turboquant_kv": "full_attention_layers_only",
        "swa_layers": "rotating_kv_native",
    }
    assert structural["bundles"]["jang2l"]["reasoning"]["parser"] == "think_xml"
    assert structural["bundles"]["jang2l"]["tools"]["parser"] == "xml_function"
    assert "jangtq2/config.json" in manifest.read_text(encoding="utf-8")
