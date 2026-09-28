"""The optional ANE adapter rejects unsupported layouts and keeps Q2 values."""

import sys

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal.bonsai_ane_prefill import (
    _q2_to_q4,
    _validate_projection,
    install,
)
from vmlx_engine.utils.jang_hadamard import HadamardSpec, install_hadamard_modules


def test_bonsai_q2_to_q4_repack_is_lossless_for_ternary_codes():
    words = mx.array([
        [0x00000000, 0x55555555, 0xAAAAAAAA, 0x050A505A],
        [0xAAAAAAAA, 0x050A505A, 0x55555555, 0x00000000],
    ], dtype=mx.uint32)
    scales = mx.array([[0.125], [0.25]], dtype=mx.float16)
    biases = -scales
    repacked = _q2_to_q4(words)
    original = mx.dequantize(words, scales, biases, group_size=64, bits=2, dtype=mx.float16)
    restored = mx.dequantize(repacked, scales, biases, group_size=64, bits=4, dtype=mx.float16)
    mx.eval(original, restored)
    assert bool(mx.array_equal(original, restored).item())


def test_bonsai_q2_to_q4_rejects_nonternary_codes():
    with pytest.raises(ValueError, match="ternary"):
        _q2_to_q4(mx.array([[0xFFFFFFFF]], dtype=mx.uint32))


def test_bonsai_ane_rejects_separate_projection_bias():
    model = nn.Module()
    model.projection = nn.Linear(512, 64, bias=True).to_quantized(group_size=128, bits=2)
    spec = HadamardSpec(512, ["projection"], [], mx.float32)
    install_hadamard_modules(model, spec)
    with pytest.raises(ValueError, match="separate projection bias"):
        _validate_projection(model.projection, "test")


def test_bonsai_ane_fails_loud_without_native_package(monkeypatch):
    monkeypatch.setenv("VMLX_BONSAI_FP16_PREFILL_ONLY", "1")
    monkeypatch.setenv("VMLX_BONSAI_FP16_QMM", "1")
    monkeypatch.setitem(sys.modules, "omlx.custom_kernels.qwen35_prefill", None)
    with pytest.raises(RuntimeError, match="built Qwen custom kernels"):
        install(nn.Module())
