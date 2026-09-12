"""Opt-in KDA output rounding leaves the recurrent state schema untouched."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models.cache import ArraysCache
import pytest

from vmlx_engine.models.glm5_next import glm5_next as model


def make_layer(dtype):
    mx.random.seed(927)
    layer = model.KDAAttention(model.ModelArgs(hidden_size=128, linear_num_heads=2))
    layer.update(tree_map(lambda value: value.astype(dtype), layer.parameters()))
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(layer, name, (mx.random.normal((256, 4)) * 0.1).astype(dtype))
    return layer


def test_output_boundary_is_opt_in_and_default_is_identity(monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_KDA_NATIVE_OUTPUT_DTYPE", raising=False)
    layer = make_layer(mx.bfloat16)
    value = mx.array([[[[1.00001, 1.004, -0.017]]]], dtype=mx.float32)
    assert layer._output_dtype_boundary(value, mx.bfloat16) is value
    monkeypatch.setenv("VMLX_GLM5_KDA_NATIVE_OUTPUT_DTYPE", "1")
    enabled = make_layer(mx.bfloat16)
    actual = enabled._output_dtype_boundary(value, mx.bfloat16)
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, value.astype(mx.bfloat16)).item()


def check_segments(layer, dtype, monkeypatch):
    observed = []
    def capture(output, *args, **kwargs):
        observed.append(output)
        return None  # Exercise the real stock gated RMS consumer.
    monkeypatch.setattr(model, "sigmoid_gated_rmsnorm_small_rows", capture)
    original, native = ArraysCache(size=4), ArraysCache(size=4)
    for count in (65, 3, 1):
        x = (mx.random.normal((1, count, 128)) * 0.15).astype(dtype)
        layer._native_kda_output_dtype = False
        off = layer(x, cache=original)
        mx.eval(off, *original.cache)
        prior = observed[-1]
        layer._native_kda_output_dtype = True
        on = layer(x, cache=native)
        mx.eval(on, *native.cache)
        bounded = observed[-1]
        assert prior.dtype == mx.float32 and bounded.dtype == dtype
        assert mx.array_equal(bounded, prior.astype(dtype)).item()
        assert on.dtype == off.dtype == dtype
        for left, right in zip(original.cache, native.cache):
            assert left.dtype == right.dtype
            assert mx.array_equal(left, right).item()
        assert native.cache[model.KDA_STATE].dtype == mx.float32
        if dtype == mx.float32:
            assert mx.array_equal(on, off).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_output_boundary_on_prefill_continuation_and_decode(dtype, monkeypatch):
    check_segments(make_layer(dtype), dtype, monkeypatch)


@pytest.mark.parametrize("bits,group_size", [(2, 64), (4, 32), (6, 64), (8, 128)])
def test_packed_projection_format_does_not_change_state_boundary(bits, group_size, monkeypatch):
    layer = make_layer(mx.bfloat16)
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(layer, name, nn.QuantizedLinear.from_linear(
            getattr(layer, name), bits=bits, group_size=group_size))
    assert layer.prepare_runtime()
    check_segments(layer, mx.bfloat16, monkeypatch)
