"""Bundle-preserved media must not be advertised as a runnable input path."""
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


@pytest.fixture
def bolt_runtime(tmp_path, monkeypatch):
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    (tmp_path / "config_omni.json").write_text(json.dumps({
        "vision_config": {"model_type": "radio"},
        "sound_config": None,
        "img_context_token_id": 18,
    }))
    # Bolt preserves video weights/metadata, but the Python wrapper is image-only.
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "has_vision": True, "has_video": True, "has_audio": False,
        "capabilities": {"modality": "vision"},
    }))
    monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "bundle-alias")
    monkeypatch.setattr(server, "_force_text_only", False)
    monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
    return server, tmp_path


@pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_v2"])
def test_bolt_runtime_only_advertises_wired_image_input(bolt_runtime, model_type):
    server, path = bolt_runtime
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert server._bundle_declares_native_video(str(path)) is True
    assert server._loaded_runtime_modalities() == ["text", "vision"]
    status = server._loaded_media_capability_status(server._loaded_runtime_modalities())
    assert status["status_by_modality"]["video"] == "declared_not_runtime_supported"


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("modality", ["video", "audio"])
def test_bolt_unsupported_input_rejected_before_generation(bolt_runtime, endpoint, modality):
    server, _ = bolt_runtime
    with pytest.raises(HTTPException) as error:
        server._reject_unsupported_multimodal(endpoint, {modality})
    assert error.value.status_code == 400
    assert modality in error.value.detail
    server._reject_unsupported_multimodal(endpoint, {"image"})


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_video_http_request_returns_clear_400(bolt_runtime, monkeypatch, endpoint, stream):
    from fastapi.testclient import TestClient

    server, _ = bolt_runtime
    monkeypatch.setattr(server, "_api_key", None)
    parts = [
        {"type": "text", "text": "Describe this clip."},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
    ]
    message = {"role": "user", "content": parts}
    payload = {"model": "bundle-alias", "stream": stream}
    payload["messages" if endpoint.endswith("completions") else "input"] = [message]
    response = TestClient(server.app).post(endpoint, json=payload)
    assert response.status_code == 400, response.text
    assert "unsupported media modality video" in response.text
    assert "text, vision" in response.text


def test_force_text_only_still_takes_precedence(bolt_runtime, monkeypatch):
    server, _ = bolt_runtime
    monkeypatch.setattr(server, "_force_text_only", True)
    assert server._loaded_runtime_modalities() == ["text"]


def test_other_video_runtime_keeps_declared_video(bolt_runtime):
    server, path = bolt_runtime
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert server._loaded_runtime_modalities() == ["text", "vision", "video"]


def test_no_bundle_identity_does_not_break_capabilities(bolt_runtime, monkeypatch):
    server, _ = bolt_runtime
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_name", None)
    assert server._loaded_runtime_modalities() == ["text", "vision"]


def test_nano_omni_keeps_its_separate_runtime_modalities(bolt_runtime, monkeypatch):
    server, path = bolt_runtime
    (path / "config_omni.json").write_text(json.dumps({
        "vision_config": {"model_type": "radio"},
        "sound_config": {"model_type": "parakeet"},
        "img_context_token_id": 18,
    }))
    monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: ["text", "vision", "video", "audio"])
    assert server._loaded_runtime_modalities() == ["text", "vision", "video", "audio"]
