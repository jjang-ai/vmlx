# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Language Model wrapper."""

import platform
import sys

import pytest

# Skip all tests if not on Apple Silicon or MLX not available
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture
def small_model_name():
    """Return a small model for testing."""
    return "mlx-community/Llama-3.2-1B-Instruct-4bit"


def test_model_init():
    """Test model initialization."""
    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel("some-model")
    assert model.model_name == "some-model"
    assert not model._loaded


def test_model_info_not_loaded():
    """Test model info when not loaded."""
    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel("some-model")
    info = model.get_model_info()

    assert info["loaded"] is False
    assert info["model_name"] == "some-model"


def test_model_repr():
    """Test model string representation."""
    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel("test-model")
    repr_str = repr(model)

    assert "MLXLanguageModel" in repr_str
    assert "test-model" in repr_str
    assert "not loaded" in repr_str


def test_generate_installs_guided_json_logits_processor(monkeypatch):
    """response_format should become a logits processor when helper supports it."""
    import mlx.core as mx

    from vmlx_engine.models.llm import MLXLanguageModel

    captured = {}

    def fake_generate(*_args, **kwargs):
        captured.update(kwargs)
        return "{}"

    def fake_processor(_tokens, logits):
        return logits

    fake_processor._vmlx_guided_decoding = "llguidance_json"

    monkeypatch.setattr(
        "vmlx_engine.api.tool_calling.build_guided_json_logits_processor",
        lambda response_format, tokenizer: fake_processor,
    )
    monkeypatch.setattr("mlx_lm.generate", fake_generate)

    model = MLXLanguageModel.__new__(MLXLanguageModel)
    model._loaded = True
    model.model = object()
    model.tokenizer = type("_Tokenizer", (), {"encode": lambda self, text: [1, 2]})()
    model._create_sampler = lambda *args, **kwargs: (lambda logits: mx.argmax(logits, axis=-1))

    output = model.generate(
        "Return JSON",
        max_tokens=4,
        _vmlx_response_format={"type": "json_object"},
    )

    assert output.text == "{}"
    assert captured["logits_processors"] == [fake_processor]


@pytest.mark.slow
def test_model_load(small_model_name):
    """Test loading a model (slow test, downloads model)."""
    pytest.importorskip("mlx_lm")

    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    assert model._loaded
    assert model.model is not None
    assert model.tokenizer is not None


@pytest.mark.slow
def test_model_generate(small_model_name):
    """Test text generation."""
    pytest.importorskip("mlx_lm")

    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    output = model.generate("Hello", max_tokens=10)

    assert output.text is not None
    assert len(output.text) > 0
    assert output.finish_reason is not None


@pytest.mark.slow
def test_model_stream_generate(small_model_name):
    """Test streaming generation."""
    pytest.importorskip("mlx_lm")

    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    chunks = list(model.stream_generate("Hello", max_tokens=10))

    assert len(chunks) > 0
    assert any(chunk.finished for chunk in chunks)


@pytest.mark.slow
def test_model_chat(small_model_name):
    """Test chat interface."""
    pytest.importorskip("mlx_lm")

    from vmlx_engine.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    messages = [{"role": "user", "content": "Hi"}]
    output = model.chat(messages, max_tokens=10)

    assert output.text is not None
