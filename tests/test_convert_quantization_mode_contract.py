"""Contract tests for vMLX-to-MLX-LM uniform quantization modes."""

import importlib

from vmlx_engine.commands.convert import _run_conversion


def _capture_conversion(monkeypatch, *, q_mode):
    mlx_convert = importlib.import_module("mlx_lm.convert")
    calls = []
    monkeypatch.setattr(mlx_convert, "convert", lambda **kwargs: calls.append(kwargs))

    _run_conversion(
        hf_path="/models/control",
        mlx_path="/models/output",
        q_bits=4,
        q_group_size=64,
        q_mode=q_mode,
        dtype=None,
        trust_remote_code=False,
    )

    assert len(calls) == 1
    return calls[0]


def test_default_mode_delegates_to_mlx_lm_without_q_mode_keyword(monkeypatch):
    kwargs = _capture_conversion(monkeypatch, q_mode=None)

    assert kwargs == {
        "hf_path": "/models/control",
        "mlx_path": "/models/output",
        "quantize": True,
        "q_group_size": 64,
        "q_bits": 4,
        "dtype": None,
        "trust_remote_code": False,
    }


def test_explicit_mode_is_forwarded_verbatim(monkeypatch):
    kwargs = _capture_conversion(monkeypatch, q_mode="NF4")

    assert kwargs["q_mode"] == "NF4"
