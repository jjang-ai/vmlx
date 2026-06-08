import json
import sys
import types


def _write_mimo_bundle(path, *, audio_token=False, media_runtime=False):
    (path / "audio_tokenizer").mkdir(parents=True, exist_ok=True)
    (path / "audio_tokenizer" / "model.safetensors").write_bytes(b"stub")
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mimo_v2",
                "vision_config": {"hidden_size": 1280},
                "audio_config": {"hidden_size": 1024},
                "processor_config": {},
                "image_token_id": 151655,
                "video_token_id": 151656,
                **({"audio_token_id": 151657} if audio_token else {}),
                "capabilities": {
                    "modalities": (
                        ["text", "vision", "video", "audio"]
                        if media_runtime and audio_token
                        else ["text", "vision", "video"]
                        if media_runtime
                        else ["text"]
                    ),
                    "preserved_modalities": ["vision", "audio"],
                    "unwired_modalities": [] if media_runtime else ["vision", "audio"],
                    "multimodal_status": (
                        "mimo_v2_multimodal_runtime"
                        if media_runtime
                        else "weights_preserved_text_runtime"
                    ),
                },
                "runtime": {
                    "multimodal_mode": (
                        "mimo_v2_multimodal_runtime"
                        if media_runtime
                        else "weights_preserved_text_runtime"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def _install_fake_mimo_runtime(monkeypatch):
    fake = types.SimpleNamespace(
        VisionModel=object,
        MiMoVisionPatchEmbed=object,
        MiMoVisionPatchMerger=object,
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

    _write_mimo_bundle(tmp_path, audio_token=False, media_runtime=True)
    _install_fake_mimo_runtime(monkeypatch)

    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
    ]

    _write_mimo_bundle(tmp_path, audio_token=True, media_runtime=True)
    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
        "audio",
    ]


def test_mimo_v2_runtime_modalities_fail_closed_for_preserved_text_runtime(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server

    _write_mimo_bundle(tmp_path, audio_token=True, media_runtime=False)
    _install_fake_mimo_runtime(monkeypatch)

    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == ["text"]


def test_mimo_v2_runtime_modalities_auto_enable_when_runtime_and_sidecars_are_complete(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server

    _write_mimo_bundle(tmp_path, audio_token=True, media_runtime=False)
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "visual.patch_embed.proj.weight": "model-00001.safetensors",
                    "audio_encoder.input_local_transformer.weight": "model-00002.safetensors",
                    "speech_embeddings.weight": "model-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    _install_fake_mimo_runtime(monkeypatch)

    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
        "audio",
    ]


def test_mimo_v2_runtime_modalities_register_local_runtime_when_missing(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server
    from vmlx_engine.models import mllm

    _write_mimo_bundle(tmp_path, audio_token=False, media_runtime=True)
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def register_fake_runtime():
        _install_fake_mimo_runtime(monkeypatch)

    monkeypatch.setattr(
        mllm,
        "_register_mimo_v2_mlx_vlm_runtime",
        register_fake_runtime,
    )

    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
    ]


def test_mimo_v2_runtime_modalities_use_processor_audio_token_id(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server

    _write_mimo_bundle(tmp_path, audio_token=False, media_runtime=True)
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    cfg["processor_config"] = {"audio_token_id": 151669}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    _install_fake_mimo_runtime(monkeypatch)

    assert server._mimo_v2_runtime_modalities(str(tmp_path)) == [
        "text",
        "vision",
        "video",
        "audio",
    ]


def test_mimo_v2_local_runtime_registration_exports_capability_classes():
    from vmlx_engine.models.mllm import _register_mimo_v2_mlx_vlm_runtime

    sys.modules.pop("mlx_vlm.models.mimo_v2", None)
    _register_mimo_v2_mlx_vlm_runtime()
    module = sys.modules["mlx_vlm.models.mimo_v2"]

    for name in (
        "VisionModel",
        "MiMoVisionPatchEmbed",
        "MiMoVisionPatchMerger",
        "MiMoVisionBlock",
        "AudioModel",
        "MiMoAudioTokenizer",
        "load_mimo_audio_tokenizer_from_bundle",
    ):
        assert hasattr(module, name), name


def test_mimo_v2_media_capabilities_filter_runtime_supported_unwired_modalities(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine import server

    _write_mimo_bundle(tmp_path, audio_token=False, media_runtime=True)
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
