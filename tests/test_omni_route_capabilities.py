"""Capabilities distinguish native media controls and storage from text."""
import pytest

from tests.test_omni_text_only_override import native_bundle


@pytest.mark.parametrize("backend", ["stage1", "stage2"])
@pytest.mark.parametrize("disk_enabled", [False, True])
def test_native_media_route_contract(native_bundle, monkeypatch, backend, disk_enabled):
    from fastapi.testclient import TestClient
    from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher
    server, _ = native_bundle
    monkeypatch.setattr(server, "_force_text_only", False)
    monkeypatch.setattr(OmniMultimodalDispatcher, "_pick_backend", staticmethod(lambda: backend))
    monkeypatch.setattr(server, "_loaded_block_disk_cache_enabled", lambda: disk_enabled)
    monkeypatch.setattr(server, "_loaded_omni_disk_cache_policy", lambda: {
        "root": "/selected/pool", "max_size_bytes": 12345, "ttl_minutes": 0.0})
    client = TestClient(server.app)
    try:
        response = client.get("/v1/capabilities")
    finally:
        client.close()
    assert response.status_code == 200, response.text
    caps = response.json()
    assert caps["supports_thinking_budget"] is False
    text = caps["request_routes"]["text"]
    media = caps["request_routes"]["native_media"]
    assert text["supports_thinking_budget"] is True
    assert media["backend"] == backend
    assert media["selection"] == "media_in_conversation"
    assert media["modalities"] == ["text", "audio", "image", "video"]
    assert media["supports_thinking_budget"] is False
    assert media["supports_tools"] is (backend == "stage1")
    assert media["video_controls"] == ["video_fps", "video_max_frames"]
    assert media["supports_structured_output"] is False
    assert caps["cache"]["scope"] == "text"
    assert caps["cache"]["native_media"] == media["cache"]
    if backend == "stage1":
        assert media["chat_template_kwargs"]["reasoning_budget"]["enforced_token_cap"] is False
        assert media["chat_template_kwargs"]["truncate_history_thinking"]["type"] == "boolean"
        cache = media["cache"]
        assert cache["enabled"] is disk_enabled
        assert cache["type"] == "native_full_state_ssd"
        assert cache["restore"] == "longest_exact_causal_prefix"
        assert cache["write_fence"] == "before_output_and_tool_delivery"
        assert cache["idle_resident_state"] is False
        assert cache["dtype_policy"] == "preserve_native"
        assert cache["policy"]["max_size_bytes"] == 12345
    else:
        assert media["chat_template_kwargs"] == {}
        assert media["cache"]["enabled"] is False
        assert media["cache"]["type"] == "unqualified"


def test_text_only_keeps_existing_hard_budget_capability(native_bundle):
    from fastapi.testclient import TestClient
    server, _ = native_bundle
    client = TestClient(server.app)
    try:
        caps = client.get("/v1/capabilities").json()
    finally:
        client.close()
    assert caps["supports_thinking_budget"] is True
    assert "request_routes" not in caps
    assert "native_media" not in caps["cache"]


def test_media_route_does_not_infer_video_from_omni_family(native_bundle, monkeypatch):
    from fastapi.testclient import TestClient
    server, _ = native_bundle
    monkeypatch.setattr(server, "_force_text_only", False)
    monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: ["text", "audio", "image"])
    client = TestClient(server.app)
    try:
        caps = client.get("/v1/capabilities").json()
    finally:
        client.close()
    assert caps["request_routes"]["native_media"]["video_controls"] == []
