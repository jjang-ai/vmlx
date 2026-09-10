"""Keep exact gate/up fusion on SwitchGLU's sorted-route arithmetic path."""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU
from vmlx_engine.metal.qwen4_affine_moe_decode import (
    _ExactGateUpProjection, _exact_gate_up_switchglu, _EXACT_PROJ_ATTR,
)


@pytest.fixture(scope="module", params=[(b, g) for b in (2, 4, 6, 8) for g in (64, 128)])
def switch(request):
    bits, group = request.param
    mx.random.seed(78)
    module = SwitchGLU(2560, 640, 16)
    nn.quantize(module, group_size=group, bits=bits)
    setattr(module, _EXACT_PROJ_ATTR, _ExactGateUpProjection(module.up_proj, module.gate_proj))
    return module


@pytest.mark.parametrize("rows", [1, 6, 7, 64])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_sorted_route_boundary_preserves_output(switch, rows, dtype):
    mx.random.seed(79)
    x = mx.random.normal((1, rows, 2560)).astype(dtype)
    indices = mx.random.randint(0, 16, (1, rows, 10)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, rows, 10)), axis=-1).astype(dtype)
    expected = (switch(x, indices) * scores[..., None]).sum(axis=-2)
    actual = _exact_gate_up_switchglu(switch, x, indices, scores)
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


@pytest.mark.parametrize("bits,group", [(3, 32), (3, 64), (3, 128), (5, 32), (5, 64), (5, 128)])
def test_additional_affine_layouts(bits, group):
    mx.random.seed(80)
    module = SwitchGLU(2560, 640, 16)
    nn.quantize(module, group_size=group, bits=bits)
    setattr(module, _EXACT_PROJ_ATTR, _ExactGateUpProjection(module.up_proj, module.gate_proj))
    for rows in (1, 7, 64):
        x = mx.random.normal((1, rows, 2560)).astype(mx.bfloat16)
        idx = mx.random.randint(0, 16, (1, rows, 10)).astype(mx.uint32)
        scores = mx.full((1, rows, 10), .1, dtype=mx.bfloat16)
        expected = (module(x, idx) * scores[..., None]).sum(-2)
        actual = _exact_gate_up_switchglu(module, x, idx, scores)
        assert mx.array_equal(expected, actual).item()


def test_exact_installer_preserves_mismatched_pairs(monkeypatch):
    from vmlx_engine.metal import qwen4_affine_moe_decode as fusion
    good = SwitchGLU(2560, 640, 2)
    bad = SwitchGLU(2560, 640, 2)
    nn.quantize(good, group_size=64, bits=2)
    nn.quantize(bad, group_size=64, bits=3)
    bad.gate_proj.bits = 2  # Metadata mismatch must be rejected before projection use.
    original = bad.up_proj
    monkeypatch.setattr(fusion, "_exact_self_test", lambda *args: None)
    assert fusion._install_exact_gate_up([good, bad]) == 1
    assert good.up_proj is None
    assert bad.up_proj is original
    assert not getattr(bad, fusion._EXACT_OK_ATTR, False)
