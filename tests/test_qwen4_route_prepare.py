import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU
from vmlx_engine.metal import qwen4_route_prepare as route


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(17, 64), (2, 17, 64)])
def test_sorted_affine_output_exact(monkeypatch, bits, dtype, shape):
    monkeypatch.setattr(route, "_ENABLED", True)
    mx.random.seed(19)
    module = SwitchGLU(64, 64, 8)
    module.set_dtype(dtype)
    nn.quantize(module, group_size=32, bits=bits)
    module.eval()
    x = mx.random.normal(shape).astype(dtype)
    indices = mx.random.randint(0, 8, (*shape[:-1], 4), dtype=mx.uint32)
    expected = module(x, indices)
    actual = route.scatter_route_switchglu(module, x, indices)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual))


def test_disabled_small_rows_and_training_fall_back(monkeypatch):
    module = SwitchGLU(64, 64, 8)
    x = mx.zeros((1, 1, 64))
    indices = mx.zeros((1, 1, 4), mx.uint32)
    monkeypatch.setattr(route, "_ENABLED", False)
    assert route.scatter_route_switchglu(module, x, indices) is None
    monkeypatch.setattr(route, "_ENABLED", True)
    module.eval()
    assert route.scatter_route_switchglu(module, x, indices) is None
    module.train()
    assert route.scatter_route_switchglu(
        module, mx.zeros((17, 64)), mx.zeros((17, 4), mx.uint32)
    ) is None
