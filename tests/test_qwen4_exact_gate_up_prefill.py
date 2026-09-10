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
