"""Exact arithmetic, input history and dispatcher gates for QKV preparation."""

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models.cache import ArraysCache

from vmlx_engine.metal import glm5_kda_qkv_prefill as fused
from vmlx_engine.models.glm5_next.kda import short_conv, l2norm


@pytest.fixture(autouse=True)
def reset_dispatch(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)


def assert_exact(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    bits = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(np.asarray(a.view(bits)), np.asarray(b.view(bits)))


def fixture(dtype, tokens=7, width=4, channels=256, with_state=True):
    mx.random.seed(tokens+width+channels)
    # Same strided slices returned by the production packed projection group.
    xs = tuple(mx.split(mx.random.normal((1, tokens, channels*3)).astype(dtype), 3, axis=-1))
    weights = tuple(mx.random.normal((channels, width*2)).astype(dt)[:, ::2]
                    for dt in (mx.float16, mx.bfloat16, mx.float32))
    states = tuple(mx.random.normal((1, width-1, channels*2)).astype(dtype)[:, :, ::2]
                   if with_state else None for _ in range(3))
    return xs, weights, states


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("tokens,width,with_state", [(2, 8, True), (7, 2, False), (65, 4, True), (512, 4, False)])
def test_exact_outputs_norm_and_original_history(dtype, tokens, width, with_state):
    xs, weights, states = fixture(dtype, tokens, width, with_state=with_state)
    snapshots = tuple(mx.array(x) for x in (*xs, *states) if x is not None)
    expected = tuple(short_conv(x, w, s) for x, w, s in zip(xs, weights, states))
    actual = fused.kda_qkv_prefill(xs, weights, states, enabled=True)
    assert actual is not None
    for i, (ref, got) in enumerate(zip(expected, actual)):
        assert_exact(ref[0], got[0])
        assert_exact(ref[1], got[1])
        if i < 2:
            assert_exact(l2norm(ref[0].reshape(1, tokens, -1, 128)),
                         l2norm(got[0].reshape(1, tokens, -1, 128)))
    for before, after in zip(snapshots, (x for x in (*xs, *states) if x is not None)):
        assert_exact(before, after)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_prefill_segments_continue_raw_histories(dtype):
    xs, weights, states = fixture(dtype, tokens=19, width=8)
    whole = tuple(short_conv(x, w, s) for x, w, s in zip(xs, weights, states))
    left = fused.kda_qkv_prefill(tuple(x[:, :2] for x in xs), weights, states, enabled=True)
    right = fused.kda_qkv_prefill(tuple(x[:, 2:] for x in xs), weights,
                                tuple(p[1] for p in left), enabled=True)
    for ref, a, b in zip(whole, left, right):
        assert_exact(ref[0], mx.concatenate([a[0], b[0]], axis=1))
        assert_exact(ref[1], b[1])


def test_upper_prefill_bound_real_channel_count():
    xs, weights, states = fixture(mx.bfloat16, tokens=2048, channels=8192)
    expected = tuple(short_conv(x, w, s) for x, w, s in zip(xs, weights, states))
    actual = fused.kda_qkv_prefill(xs, weights, states, enabled=True)
    for ref, got in zip(expected, actual):
        assert_exact(ref[0], got[0])
        assert_exact(ref[1], got[1])


def test_three_dimensional_weights_and_cast_history():
    xs, weights, states = fixture(mx.float16)
    weights = tuple(w[:, None, :] for w in weights)
    states = tuple(s.astype(mx.float32) for s in states)
    actual = fused.kda_qkv_prefill(xs, weights, states, enabled=True)
    for x, w, s, got in zip(xs, weights, states, actual):
        ref = short_conv(x, w, s)
        assert_exact(ref[0], got[0])
        assert_exact(ref[1], got[1])


def test_default_off_and_unsupported_geometry(monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_KDA_QKV_PREFILL", raising=False)
    assert not fused.qkv_prefill_requested()
    monkeypatch.setenv("VMLX_GLM5_KDA_QKV_PREFILL", "1")
    assert fused.qkv_prefill_requested()
    xs, weights, states = fixture(mx.float16)
    def unexpected(*args, **kwargs):
        pytest.fail("ineligible row attempted a Metal dispatch")
    monkeypatch.setattr(fused, "_kernel", unexpected)
    assert fused.kda_qkv_prefill(xs, weights, states, enabled=False) is None
    for inputs in (tuple(x[:, :1] for x in xs),
                   tuple(mx.broadcast_to(x, (2, *x.shape[1:])) for x in xs),
                   (xs[0], xs[1].astype(mx.float32), xs[2]),
                   tuple(mx.zeros((1, 2049, 256), dtype=mx.float16) for _ in xs)):
        assert fused.kda_qkv_prefill(inputs, weights, states, enabled=True) is None
    assert fused.kda_qkv_prefill(xs, tuple(w[:, :1] for w in weights), states, enabled=True) is None
    assert fused.kda_qkv_prefill(xs, weights, (states[0][:, :1], *states[1:]), enabled=True) is None
    with mx.stream(mx.cpu):
        monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)
        assert fused.kda_qkv_prefill(xs, weights, states, enabled=True) is None


def test_first_launch_failure_is_logged_and_keeps_state(monkeypatch, caplog):
    xs, weights, states = fixture(mx.float16)
    before = tuple(mx.array(s) for s in states)
    def fail(width):
        raise RuntimeError("controlled compile failure")
    monkeypatch.setattr(fused, "_kernel", fail)
    assert fused.kda_qkv_prefill(xs, weights, states, enabled=True) is None
    assert fused._FAILED
    assert "disabled after launch failure" in caplog.text
    for a, b in zip(before, states):
        assert_exact(a, b)
    # A disabled helper returns before another compiler attempt.
    monkeypatch.setattr(fused, "_kernel", lambda width: pytest.fail("retry"))
    assert fused.kda_qkv_prefill(xs, weights, states, enabled=True) is None


@pytest.mark.parametrize("bits,group_size", [(2, 64), (4, 32), (6, 64), (8, 128)])
@pytest.mark.parametrize("tokens", [7, 65])
def test_quantized_native_kda_block_and_next_segments(bits, group_size, tokens):
    from vmlx_engine.models.glm5_next.glm5_next import KDAAttention, ModelArgs
    mx.random.seed(bits*100+tokens)
    dtype = mx.bfloat16 if bits in (2, 6) else mx.float16
    layer = KDAAttention(ModelArgs(hidden_size=128, linear_num_heads=2))
    layer.update(tree_map(lambda a: a.astype(dtype), layer.parameters()))
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(layer, name, nn.QuantizedLinear.from_linear(
            getattr(layer, name), bits=bits, group_size=group_size))
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(layer, name, (mx.random.normal((256, 4))*0.1).astype(dtype))
    assert layer.prepare_runtime()
    assert (layer.qkv_group.bits, layer.qkv_group.group_size) == (bits, group_size)
    layer._fused_kda_prefill = False
    a, b = ArraysCache(size=4), ArraysCache(size=4)
    for count in (tokens, 3, 1):
        x = (mx.random.normal((1, count, 128))*0.1).astype(dtype)
        layer._fused_kda_qkv_prefill = False
        expected = layer(x, cache=a)
        mx.eval(expected, *a.cache)
        layer._fused_kda_qkv_prefill = True
        actual = layer(x, cache=b)
        mx.eval(actual, *b.cache)
        assert_exact(expected, actual)
        for left, right in zip(a.cache, b.cache):
            assert_exact(left, right)


def test_native_caller_uses_stock_after_preparation_failure(monkeypatch):
    from vmlx_engine.models.glm5_next import glm5_next as model
    mx.random.seed(2223)
    layer = model.KDAAttention(model.ModelArgs(hidden_size=64, linear_num_heads=2))
    x = mx.random.normal((1, 7, 64))
    a, b = ArraysCache(size=4), ArraysCache(size=4)
    layer._fused_kda_prefill = False
    layer._fused_kda_qkv_prefill = False
    expected = layer(x, cache=a)
    mx.eval(expected, *a.cache)
    monkeypatch.setattr(model, "kda_qkv_prefill", lambda *args, **kwargs: None)
    layer._fused_kda_qkv_prefill = True
    actual = layer(x, cache=b)
    assert_exact(expected, actual)
    for left, right in zip(a.cache, b.cache):
        assert_exact(left, right)
