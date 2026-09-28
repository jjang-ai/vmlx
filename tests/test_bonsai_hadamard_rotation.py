"""GPU parity checks for the optional Bonsai signed-Hadamard dispatch."""

import mlx.core as mx
import pytest

from vmlx_engine.metal.bonsai_hadamard_rotation import rotate
from vmlx_engine.utils.jang_hadamard import hadamard_activation


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("rows,width", [(1, 1024), (3, 5120), (512, 5120), (2, 17408)])
def test_fused_rotation_matches_stock_exactly(monkeypatch, dtype, inverse, rows, width):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("requires an MLX Metal GPU")
    x = ((mx.arange(rows * width, dtype=mx.float32) % 103) - 51)
    x = (x * 0.03125).reshape(rows, width).astype(dtype)
    signs = mx.where(mx.arange(width) % 3 == 0, 1, -1).astype(mx.float32)

    monkeypatch.delenv("VMLX_BONSAI_FUSED_HADAMARD", raising=False)
    stock = hadamard_activation(x, 1024, signs, inverse=inverse)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_HADAMARD", "1")
    fused = hadamard_activation(x, 1024, signs, inverse=inverse)
    mx.eval(stock, fused)
    assert bool(mx.array_equal(stock, fused).item())


def test_unsupported_block_falls_back_to_stock(monkeypatch):
    x = mx.arange(512, dtype=mx.float32).reshape(1, 512)
    signs = mx.ones((512,), dtype=mx.float32)
    assert rotate(x, signs, 512) is None
    monkeypatch.delenv("VMLX_BONSAI_FUSED_HADAMARD", raising=False)
    stock = hadamard_activation(x, 512, signs)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_HADAMARD", "1")
    got = hadamard_activation(x, 512, signs)
    assert bool(mx.array_equal(stock, got).item())


def test_long_prefill_rotation_matches_stock_exactly(monkeypatch):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("requires an MLX Metal GPU")
    width = 5120
    x = ((mx.arange(4096 * width, dtype=mx.float32) % 103) - 51)
    x = (x * 0.03125).reshape(4096, width).astype(mx.float16)
    signs = mx.where(mx.arange(width) % 3 == 0, 1, -1).astype(mx.float32)
    monkeypatch.delenv("VMLX_BONSAI_FUSED_HADAMARD", raising=False)
    stock = hadamard_activation(x, 1024, signs)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_HADAMARD", "1")
    fused = hadamard_activation(x, 1024, signs)
    mx.eval(stock, fused)
    assert bool(mx.array_equal(stock, fused).item())
