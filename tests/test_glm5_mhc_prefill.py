import pytest
import mlx.core as mx

from vmlx_engine.metal.glm5_mhc_prefill import glm5_mhc_prefill_sinkhorn


def stock(x, eps, iterations):
    x = x / (mx.sum(x, axis=-2, keepdims=True) + eps)
    for _ in range(iterations - 1):
        x = x / (mx.sum(x, axis=-1, keepdims=True) + eps)
        x = x / (mx.sum(x, axis=-2, keepdims=True) + eps)
    return x


@pytest.mark.parametrize("rows", [5, 63, 64, 65, 511, 512, 513, 4594])
@pytest.mark.parametrize("iterations", [1, 20, 64])
def test_exact_normalization(rows, iterations):
    mx.random.seed(73)
    x = mx.softmax(mx.random.normal((1, rows, 4, 4)), axis=-1) + 1e-6
    expected = stock(x, 1e-6, iterations)
    actual = glm5_mhc_prefill_sinkhorn(x, sink_eps=1e-6, iterations=iterations, enabled=True)
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


@pytest.mark.parametrize("shape,dtype,iters,enabled", [
    ((1, 4, 4, 4), mx.float32, 20, True),
    ((2, 5, 4, 4), mx.float32, 20, True),
    ((1, 5, 3, 3), mx.float32, 20, True),
    ((1, 5, 4, 4), mx.bfloat16, 20, True),
    ((1, 5, 4, 4), mx.float32, 0, True),
    ((1, 5, 4, 4), mx.float32, 65, True),
    ((1, 5, 4, 4), mx.float32, 20, False),
])
def test_stock_fallback(shape, dtype, iters, enabled):
    assert glm5_mhc_prefill_sinkhorn(mx.ones(shape, dtype=dtype),
        sink_eps=1e-6, iterations=iters, enabled=enabled) is None


@pytest.mark.parametrize("rows", [5, 512, 513])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_hyperconnection_outputs(rows, dtype):
    from vmlx_engine.models.glm5_next.glm5_next import HyperConnection, ModelArgs

    mx.random.seed(119)
    args = ModelArgs(hidden_size=64)
    layer = HyperConnection(args)
    layer.hc_fn = mx.random.normal(layer.hc_fn.shape).astype(dtype) * 0.02
    layer.hc_base = mx.random.normal(layer.hc_base.shape).astype(dtype) * 0.01
    x = mx.random.normal((1, rows, 4, 64)).astype(dtype)
    layer._fused_decode = False
    layer._fused_prefill = False
    expected = layer(x)
    mx.eval(*expected)
    layer._fused_prefill = True
    actual = layer(x)
    mx.eval(*actual)
    for left, right in zip(expected, actual):
        assert left.dtype == right.dtype
        assert mx.array_equal(left, right).item()
