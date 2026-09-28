"""Exact ternary verifier agrees with the packed MLX projection on supported rows."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal.bonsai_ternary_verify import maybe_ternary_verify


@pytest.mark.parametrize("rows", [2, 3])
def test_ternary_verify_matches_packed_projection(rows):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("requires an MLX Metal GPU")
    linear = nn.Linear(512, 32, bias=False).to_quantized(group_size=128, bits=2)
    words = mx.arange(linear.weight.size).reshape(linear.weight.shape)
    linear.weight = mx.where(words % 2 == 0, 0xAAAAAAAA, 0).astype(mx.uint32)
    linear.scales = mx.full(linear.scales.shape, 0.125, mx.float16)
    linear.biases = -linear.scales
    x = ((mx.arange(rows * 512, dtype=mx.float32) % 47) - 23)
    x = (x * 0.03125).reshape(rows, 512).astype(mx.float16)
    expected = linear(x)
    actual = maybe_ternary_verify(x, linear)
    assert actual is not None
    mx.eval(expected, actual)
    assert bool(mx.allclose(expected, actual, atol=0.03, rtol=0.01).item())


def test_ternary_verify_rejects_unsupported_rows():
    linear = nn.Linear(512, 32, bias=False).to_quantized(group_size=128, bits=2)
    assert maybe_ternary_verify(mx.ones((1, 512), mx.float16), linear) is None
