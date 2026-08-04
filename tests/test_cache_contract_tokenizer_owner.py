from types import SimpleNamespace

import vmlx_engine.engine.batched as batched_module
from vmlx_engine.engine.batched import BatchedEngine


class _RecordingTokenizer:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.calls = []

    def encode(self, prompt, *, add_special_tokens):
        self.calls.append((prompt, add_special_tokens))
        return list(self.token_ids)


def test_cache_contract_identity_uses_live_scheduler_tokenizer(monkeypatch):
    scheduler_tokenizer = _RecordingTokenizer([11, 12, 13])
    fallback_tokenizer = _RecordingTokenizer([99])
    engine = object.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._engine = SimpleNamespace(
        engine=SimpleNamespace(
            scheduler=SimpleNamespace(tokenizer=scheduler_tokenizer),
        )
    )
    engine._tokenizer = fallback_tokenizer
    engine._video_frame_fallback_messages = lambda messages: messages
    engine._extract_audio_content = lambda messages: []
    engine._apply_chat_template = lambda *args, **kwargs: "rendered prompt"
    monkeypatch.setattr(
        batched_module,
        "extract_multimodal_content",
        lambda messages: (messages, [], []),
    )

    identity = engine.build_cache_contract_prompt_identity(
        [{"role": "user", "content": "hello"}],
        skip_generation_prompt=True,
    )

    assert identity["token_ids"] == [11, 12, 13]
    assert identity["production_cache_key_token_ids"] == [11, 12, 13]
    assert identity["tokenizer_source"] == "llm_scheduler_tokenizer"
    assert scheduler_tokenizer.calls == [("rendered prompt", False)]
    assert fallback_tokenizer.calls == []


def test_batched_chat_encoding_honors_family_special_token_policy(monkeypatch):
    tokenizer = _RecordingTokenizer([1, 2])
    engine = object.__new__(BatchedEngine)
    engine._model_name = "/models/minicpm"
    registry = SimpleNamespace(
        get_architecture_hints=lambda _path: {
            "chat_encode_add_special_tokens": True
        }
    )
    monkeypatch.setattr(
        batched_module,
        "get_model_config_registry",
        lambda: registry,
    )

    token_ids = engine._encode_rendered_prompt(
        tokenizer.encode,
        "rendered prompt",
        engine._chat_encode_add_special_tokens(),
    )

    assert token_ids == [1, 2]
    assert tokenizer.calls == [("rendered prompt", True)]
