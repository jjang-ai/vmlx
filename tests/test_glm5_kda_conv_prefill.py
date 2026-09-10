"""Exact stock arithmetic/state checks for the experimental KDA prefill sum."""
import pytest
import mlx.core as mx
from vmlx_engine.models.glm5_next.kda import short_conv
from vmlx_engine.metal.glm5_kda_conv_prefill import kda_conv_prefill

@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("tokens", [2, 3, 7, 64, 512])
@pytest.mark.parametrize("with_state", [False, True])
def test_prefill_sum_and_history(dtype, tokens, with_state):
    mx.random.seed(tokens)
    x = mx.random.normal((1, tokens, 128)).astype(dtype)
    w = mx.random.normal((128, 4)).astype(dtype)
    state = mx.random.normal((1, 3, 128)).astype(dtype) if with_state else None
    expected, expected_state = short_conv(x, w, state)
    actual, actual_state = kda_conv_prefill(x, w, state)
    mx.eval(expected, expected_state, actual, actual_state)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_state, expected_state).item()

@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_strides_mixed_weights_and_continuation(dtype, width):
    mx.random.seed(width)
    x = mx.random.normal((1, 19, 256)).astype(dtype)[:, :, ::2]
    w = mx.random.normal((128, width * 2))[:, ::2]
    state = mx.random.normal((1, width - 1, 256)).astype(dtype)[:, :, ::2]
    original = mx.array(state)
    expected, tail = short_conv(x, w, state)
    left, first_state = kda_conv_prefill(x[:, :2], w, state)
    right, final_state = kda_conv_prefill(x[:, 2:], w, first_state)
    actual = mx.concatenate([left, right], axis=1)
    mx.eval(expected, actual, tail, final_state, original)
    assert mx.array_equal(expected, actual).item()
    assert mx.array_equal(tail, final_state).item()
    assert mx.array_equal(state, original).item()

def test_unsupported_decode_and_batch():
    w = mx.zeros((128, 4))
    assert kda_conv_prefill(mx.zeros((1, 1, 128)), w) is None
    assert kda_conv_prefill(mx.zeros((2, 8, 128)), w) is None

@pytest.mark.parametrize("tokens", [7, 65])
def test_complete_kda_block_state(tokens):
    from vmlx_engine.models.glm5_next.glm5_next import KDAAttention, ModelArgs
    from mlx_lm.models.cache import ArraysCache
    mx.random.seed(tokens)
    layer = KDAAttention(ModelArgs(hidden_size=64, linear_num_heads=2))
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(layer, name, mx.random.normal((256, 4)))
    x = mx.random.normal((1, tokens, 64))
    a, b = ArraysCache(size=4), ArraysCache(size=4)
    layer._fused_kda_prefill = False
    expected = layer(x, cache=a)
    mx.eval(expected, *a.cache)
    layer._fused_kda_prefill = True
    actual = layer(x, cache=b)
    mx.eval(actual, *b.cache)
    assert mx.array_equal(expected, actual).item()
    for stock, fused in zip(a.cache, b.cache):
        assert mx.array_equal(stock, fused).item()
