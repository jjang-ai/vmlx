"""/capabilities advertises vision for a DeepSeek-V4.1 VL build (text-routed vision tower, not an MLLM engine).

Before: _loaded_runtime_modalities() checked Omni, M3-VL and MLLM engines only, so a V4.1 VL bundle reported ["text"]
and the Electron composer refused image attachments ("This model does not take image input") although
/v1/chat/completions serves them (served JANGHT bundle, live panel, 2026-10-26)."""
import pytest

server = pytest.importorskip("vmlx_engine.server")


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(server, "_force_text_only", False, raising=False)
    monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
    monkeypatch.setattr(server, "_m3_vl_media_ok", lambda engine: False)
    monkeypatch.setattr(server, "_loaded_mllm_modalities", lambda: None)


def test_dsv41_vl_build_advertises_vision(plain, monkeypatch):
    monkeypatch.setattr(server, "_dsv41_vl_media_ok", lambda engine: True)
    assert server._loaded_runtime_modalities() == ["text", "vision"]


def test_text_only_build_stays_text(plain, monkeypatch):
    monkeypatch.setattr(server, "_dsv41_vl_media_ok", lambda engine: False)
    assert server._loaded_runtime_modalities() == ["text"]


def test_force_text_only_wins(plain, monkeypatch):
    monkeypatch.setattr(server, "_force_text_only", True, raising=False)
    monkeypatch.setattr(server, "_dsv41_vl_media_ok", lambda engine: True)
    assert server._loaded_runtime_modalities() == ["text"]
