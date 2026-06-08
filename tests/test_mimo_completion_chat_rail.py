import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import vmlx_engine.server as server


class _FakeMiMoEngine:
    def __init__(self, text="templated-ok"):
        self.text = text
        self.chat_calls = []

    async def chat(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return SimpleNamespace(
            text=self.text,
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
        )

    async def generate(self, **kwargs):
        raise AssertionError("MiMo completions must not use raw generate")

    async def stream_generate(self, **kwargs):
        raise AssertionError("MiMo streaming completions must not use raw stream_generate")
        yield ""


def _install_mimo_engine(monkeypatch, engine):
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", "MiMo-V2.5-JANG_2L")
    monkeypatch.setattr(server, "_model_path", "/models/MiMo-V2.5-JANG_2L")
    monkeypatch.setattr(server, "_is_loaded_mimo_v2_model", lambda model="": True, raising=False)
    monkeypatch.setattr(server, "_is_loaded_dsv4_model", lambda model="": False)


def test_mimo_legacy_completions_use_chat_template_rail(monkeypatch):
    fake = _FakeMiMoEngine()
    _install_mimo_engine(monkeypatch, fake)

    client = TestClient(server.app)
    response = client.post(
        "/v1/completions",
        json={
            "model": "MiMo-V2.5-JANG_2L",
            "prompt": "Return exactly blue-cat.",
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_prompt_tokens": 512,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "templated-ok"
    assert fake.chat_calls
    messages, kwargs = fake.chat_calls[0]
    assert messages == [{"role": "user", "content": "Return exactly blue-cat."}]
    assert kwargs["max_tokens"] == 8
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["max_prompt_tokens"] == 512
    assert kwargs["enable_thinking"] is False


def test_mimo_streaming_legacy_completions_use_chat_template_rail(monkeypatch):
    fake = _FakeMiMoEngine(text="templated-stream")
    _install_mimo_engine(monkeypatch, fake)

    client = TestClient(server.app)
    with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": "MiMo-V2.5-JANG_2L",
            "prompt": "Return exactly blue-cat.",
            "max_tokens": 8,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        payload = response.read().decode()

    assert fake.chat_calls
    assert "templated-stream" in payload
    assert "text_completion" in payload
    assert "data: [DONE]" in payload
    chunks = [
        json.loads(line[len("data: ") :])
        for line in payload.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks[0]["choices"][0]["text"] == "templated-stream"
