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
