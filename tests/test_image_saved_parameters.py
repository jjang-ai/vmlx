from types import SimpleNamespace

import pytest

mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')
from vmlx_engine.image_saved_parameters import restore_output_modulation_bias


@pytest.mark.parametrize('bits', [None, 2, 4, 6, 8])
def test_saved_bias_preserves_output_and_original_projection(bits):
    linear = (nn.Linear(128, 256, bias=False) if bits is None else
              nn.QuantizedLinear(128, 256, bias=False, bits=bits, group_size=64))
    target = SimpleNamespace(norm_out=SimpleNamespace(linear=linear))
    original = dict(linear.parameters())
    bias = mx.arange(256, dtype=mx.float32) / 1024
    x = mx.ones((1, 128))
    expected = linear(x) + bias
    mx.eval(expected)
    assert restore_output_modulation_bias(target, {'norm_out': {'linear': {'bias': bias}}})
    assert target.norm_out.linear is linear
    assert mx.array_equal(linear(x), expected).item()
    for key, value in original.items():
        assert linear.parameters()[key] is value
    assert linear.bias is bias
    if bits is not None:
        assert linear.bits == bits and linear.group_size == 64


def test_bias_free_save_is_noop():
    linear = nn.Linear(128, 256, bias=False)
    target = SimpleNamespace(norm_out=SimpleNamespace(linear=linear))
    assert not restore_output_modulation_bias(target, {})
    assert 'bias' not in linear


@pytest.mark.parametrize('bias', [mx.zeros((255,)), mx.zeros((256,), dtype=mx.uint32)])
def test_invalid_saved_bias_rejected_without_mutation(bias):
    linear = nn.Linear(128, 256, bias=False)
    target = SimpleNamespace(norm_out=SimpleNamespace(linear=linear))
    with pytest.raises(ValueError, match='bias shape or dtype'):
        restore_output_modulation_bias(target, {'norm_out': {'linear': {'bias': bias}}})
    assert 'bias' not in linear
