"""Exact component contract, not full-model speed or runtime acceptance."""

import mlx.core as mx
import pytest

from vmlx_engine.metal.qwen4_hc_combine import exact_hc_combine


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("streams,hidden", [(2, 7), (4, 64), (4, 2560), (8, 33)])
def test_preserves_stock_product_rounding(dtype, streams, hidden):
    mx.random.seed(731)
    residual = mx.random.normal((1, 1, streams * hidden)).astype(dtype)
    block = (mx.random.normal((1, 1, hidden)) * 3).astype(dtype)
    inject = mx.random.uniform(-2, 2, (1, 1, streams)).astype(dtype)
    stock = residual + (block[..., None, :] * inject[..., :, None]).reshape(residual.shape)
    candidate = exact_hc_combine(residual, block, inject, enabled=True)
    assert candidate is not None
    mx.eval(stock, candidate)
    bits = mx.uint32 if dtype == mx.float32 else mx.uint16
    assert bool(mx.array_equal(stock.view(bits), candidate.view(bits)))


def test_unsupported_shapes_and_dtypes_keep_stock_path():
    r, b, i = mx.ones((1, 1, 12)), mx.ones((1, 1, 3)), mx.ones((1, 1, 4))
    assert exact_hc_combine(r, b, i, enabled=False) is None
    assert exact_hc_combine(mx.ones((1, 2, 12)), b, i, enabled=True) is None
    assert exact_hc_combine(r, b.astype(mx.float16), i, enabled=True) is None
    assert exact_hc_combine(r, b, mx.ones((1, 1, 3)), enabled=True) is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_family_combine_wiring_and_multirow_fallback(monkeypatch, enabled, rows):
    from vmlx_engine.models.qwen4_exp.language import GatedResidual, Qwen4ExpTextArgs
    monkeypatch.setenv("VMLX_QWEN4_EXACT_HC_COMBINE", "1" if enabled else "0")
    args = Qwen4ExpTextArgs(hidden_size=32, hc_count=4, hc_lowrank=8)
    module = GatedResidual(args)
    mx.random.seed(291)
    r = mx.random.normal((1, rows, 128)).astype(mx.float16)
    b = mx.random.normal((1, rows, 32)).astype(mx.float16)
    i = mx.random.normal((1, rows, 4)).astype(mx.float16)
    expected = r + (b[..., None, :] * i[..., :, None]).reshape(r.shape)
    got = module.combine(r, b, i)
    mx.eval(expected, got)
    assert bool(mx.array_equal(expected.view(mx.uint16), got.view(mx.uint16)))
