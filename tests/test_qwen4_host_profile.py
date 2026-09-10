"""Host diagnostic semantics, not model runtime or performance acceptance."""

from types import SimpleNamespace

import pytest

from vmlx_engine.models.qwen4_exp.host_profile import profile_decode_forward


def test_disabled_is_original_function(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HOST_PROFILE", raising=False)
    def original(self, inputs):
        return inputs
    assert profile_decode_forward(original) is original


def test_bounded_rows_passthrough_and_single_report(monkeypatch, caplog):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    calls = []
    @profile_decode_forward
    def original(self, inputs, *, value):
        calls.append(inputs.shape)
        return value
    caplog.set_level("INFO")
    result = object()
    for shape in [(1, 8), (2, 1)] + [(1, 1)] * 40:
        assert original(None, SimpleNamespace(shape=shape), value=result) is result
    assert len(calls) == 42
    assert caplog.text.count("QWEN4_HOST_PROFILE calls=32") == 1
    assert "gpu_fences_added=false" in caplog.text


def test_original_error_propagates(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_HOST_PROFILE", "1")
    error = ValueError("original failure")
    @profile_decode_forward
    def original(self, inputs):
        raise error
    with pytest.raises(ValueError) as caught:
        original(None, SimpleNamespace(shape=(1, 1)))
    assert caught.value is error
