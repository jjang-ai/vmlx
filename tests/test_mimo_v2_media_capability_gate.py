import json
import sys
import types


def _write_mimo_bundle(path, *, audio_token=False):
    (path / "audio_tokenizer").mkdir(parents=True, exist_ok=True)
    (path / "audio_tokenizer" / "model.safetensors").write_bytes(b"stub")
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mimo_v2",
                "vision_config": {"hidden_size": 1280},
                "audio_config": {"hidden_size": 1024},
                "image_token_id": 151655,
                "video_token_id": 151656,
                **({"audio_token_id": 151657} if audio_token else {}),
            }
        ),
        encoding="utf-8",
    )


def _install_fake_mimo_runtime(monkeypatch):
    fake = types.SimpleNamespace(
        VisionModel=object,
        MiMoVisionPatchEmbed=object,
        MiMoVisionBlock=object,
        AudioModel=object,
        MiMoAudioTokenizer=object,
        load_mimo_audio_tokenizer_from_bundle=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.mimo_v2", fake)


def test_mimo_v2_runtime_modalities_are_component_and_token_gated(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server

    _write_mimo_bundle(tmp_path, audio_token=False)
    _install_fake_mimo_runtime(monkeypatch)

    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
    ]

    _write_mimo_bundle(tmp_path, audio_token=True)
    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
        "audio",
    ]


def test_mimo_v2_media_capabilities_filter_runtime_supported_unwired_modalities(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server

    _write_mimo_bundle(tmp_path, audio_token=False)
    _install_fake_mimo_runtime(monkeypatch)
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", None)

    status = server._loaded_media_capability_status(["text", "vision", "video"])

    assert status["runtime_modalities"] == ["text", "vision", "video"]
    assert "vision" not in status["unwired_modalities"]
    assert "image" not in status["unwired_modalities"]
    assert "video" not in status["unwired_modalities"]
    assert status["unwired_modalities"] == ["audio"]
    assert status["status_by_modality"]["vision"] == "runtime_supported"
    assert status["status_by_modality"]["video"] == "runtime_supported"
    assert status["status_by_modality"]["audio"] == "preserved_unwired"
