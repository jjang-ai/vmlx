import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.models.qwen4_exp.projection_cache import validated_projection_group


def make_linears(bits=4, group_size=32):
    result = []
    for size in (64, 32, 96):
        m = nn.Linear(64, size, bias=False)
        m.weight = m.weight.astype(mx.float16)
        result.append(m.to_quantized(group_size=group_size, bits=bits))
    return tuple(result)


def test_cache_hit_and_exact_group_outputs(caplog):
    caplog.set_level("INFO", logger="vmlx_engine.models.qwen4_exp.projection_cache")
    linears = make_linears()
    group = validated_projection_group(linears, mx.float16)
    assert validated_projection_group(linears, mx.float16) is group
    assert caplog.text.count("Qwen guarded projection group prepared:") == 1
    assert "projections=3 bits=4 group_size=32" in caplog.text
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    expected = tuple(m(x) for m in linears)
    actual = group(x)
    mx.eval(*expected, *actual)
    assert all(bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
               for a, b in zip(expected, actual))


@pytest.mark.parametrize("field", ["weight", "scales", "biases"])
def test_replaced_tensor_rebuilds(field):
    linears = make_linears()
    old = validated_projection_group(linears, mx.float16)
    setattr(linears[1], field, mx.array(getattr(linears[1], field)))
    assert validated_projection_group(linears, mx.float16) is not old


@pytest.mark.parametrize("field,value", [("bits", 8), ("group_size", 64), ("mode", "mxfp4")])
def test_metadata_change_revalidates(field, value):
    linears = make_linears()
    assert validated_projection_group(linears, mx.float16) is not None
    setattr(linears[1], field, value)
    assert validated_projection_group(linears, mx.float16) is None


def test_dtype_post_bias_and_module_replacement():
    linears = make_linears()
    old = validated_projection_group(linears, mx.float16)
    assert validated_projection_group(linears, mx.float32) is None
    linears[1].bias = mx.zeros((32,), dtype=mx.float16)
    assert validated_projection_group(linears, mx.float16) is None
    del linears[1].bias
    replaced = (linears[0], make_linears()[1], linears[2])
    assert validated_projection_group(replaced, mx.float16) is not old


def test_nonquantized_keeps_existing_path():
    assert validated_projection_group((nn.Linear(64, 32),), mx.float16) is None


def test_live_gdn_q8_group64_exact_outputs_and_replacement():
    # Observed projection metadata in Flash-Next 4S, not its routed-expert bits.
    linears = make_linears(bits=8, group_size=64)
    group = validated_projection_group(linears, mx.float16)
    assert validated_projection_group(linears, mx.float16) is group
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    expected = tuple(m(x) for m in linears)
    actual = group(x)
    mx.eval(*expected, *actual)
    assert all(bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
               for a, b in zip(expected, actual))
    linears[1].weight = mx.array(linears[1].weight)
    assert validated_projection_group(linears, mx.float16) is not group
