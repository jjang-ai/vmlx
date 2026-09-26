"""Qualified routing and native numerical edges for the opt-in expert tiles."""
from types import SimpleNamespace
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from vmlx_engine.jangh import kernels, switch


def geometry():
    def projection(i, o):
        return SimpleNamespace(input_dims=i, output_dims=o, num_experts=288, bits=2, rotated=True)
    return SimpleNamespace(gate_proj=projection(4096, 2048), up_proj=projection(4096, 2048),
                           down_proj=projection(2048, 4096))


def test_admission_requires_opt_in_native_geometry(monkeypatch):
    module = geometry()
    x = SimpleNamespace(dtype=mx.bfloat16, shape=(1024,4096))
    monkeypatch.setattr(kernels, 'nax_available', lambda: True)
    monkeypatch.setattr(switch, 'EXPERT_TILES', '0')
    assert not switch.TQSwitchGLU._use_expert_tiles(module, x, 8)
    monkeypatch.setattr(switch, 'EXPERT_TILES', '1')
    assert switch.TQSwitchGLU._use_expert_tiles(module, x, 8)
    assert not switch.TQSwitchGLU._use_expert_tiles(module, x, 1)
    assert not switch.TQSwitchGLU._use_expert_tiles(module, SimpleNamespace(dtype=mx.float16), 8)
    for field, value in [('bits', 4), ('rotated', False), ('num_experts', 287), ('input_dims', 2048)]:
        old = getattr(module.gate_proj, field)
        setattr(module.gate_proj, field, value)
        assert not switch.TQSwitchGLU._use_expert_tiles(module, x, 8)
        setattr(module.gate_proj, field, old)
    module.up_proj.input_dims=2048
    assert not switch.TQSwitchGLU._use_expert_tiles(module, x, 8)
    module.up_proj.input_dims=4096
    assert not switch.TQSwitchGLU._use_expert_tiles(module, SimpleNamespace(dtype=mx.bfloat16,shape=(1024,2048)), 8)
    monkeypatch.setattr(kernels, 'nax_available', lambda: False)
    assert not switch.TQSwitchGLU._use_expert_tiles(module, x, 8)


@pytest.mark.parametrize('bits,fused', [(2, False), (2, True), (3, False), (3, True)])
def test_expert_tiles_empty_and_multitile_segments(bits, fused):
    if not mx.metal.is_available() or not kernels.nax_available():
        pytest.skip('Native NAX required')
    ids = np.repeat([2,47,99,144,200,230,250,270,280,286], [1,31,32,33,63,64,65,127,128,129]).astype(np.uint32)
    rng = np.random.default_rng(7301)
    x = mx.array(rng.normal(0, .25, (len(ids),64)).astype(np.float32)).astype(mx.bfloat16)
    packed = mx.array(rng.integers(0, 2**32, (288,64,64*bits//32),dtype=np.uint32))
    scales = mx.array(rng.uniform(.01,.1,(288,64)).astype(np.float16))
    idx = mx.array(ids)
    options = dict(packed_u=packed,scales_u=scales*2,limit=10.0) if fused else {}
    reference = kernels.gather_qmm_sorted(x,packed,scales,None,idx,bits,**options)
    actual = kernels.gather_qmm_expert_sorted(x,packed,scales,idx,bits,kernels.expert_tile_plan(idx,288),**options)
    mx.eval(reference,actual)
    assert np.isfinite(np.asarray(actual.astype(mx.float32))).all()
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)),np.asarray(reference.astype(mx.float32)))
    mx.clear_cache()


@pytest.mark.parametrize('bits,fused', [(2, True), (3, False)])
def test_expert_tiles_large_extent_uses_int_before_narrowing(bits, fused):
    if not mx.metal.is_available() or not kernels.nax_available():
        pytest.skip('Native NAX required')
    rows, width = 32776, 64
    packed = mx.full((288,width,width*bits//32),0xFFFFFFFF,dtype=mx.uint32)
    scales = mx.full((288,width),0.03125,dtype=mx.float16)
    idx = mx.full((rows,),287,dtype=mx.uint32)
    x = mx.full((rows,width),0.125,dtype=mx.bfloat16)
    options = dict(packed_u=packed,scales_u=scales*2,limit=10.0) if fused else {}
    reference = kernels.gather_qmm_sorted(x[:8],packed,scales,None,idx[:8],bits,**options)
    actual = kernels.gather_qmm_expert_sorted(x,packed,scales,idx,bits,kernels.expert_tile_plan(idx,288),**options)
    mx.eval(reference,actual)
    expected = np.asarray(reference.astype(mx.float32))
    result = np.asarray(actual.astype(mx.float32))
    assert np.isfinite(expected).all() and (expected > 0).all()
    np.testing.assert_array_equal(expected,np.full(expected.shape,expected[0,0]))
    np.testing.assert_array_equal(result,np.full(result.shape,expected[0,0]))
    mx.clear_cache()
