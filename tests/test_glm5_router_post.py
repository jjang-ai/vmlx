"""Stock-exact router scheduling and unchanged packed MoE ownership."""

import logging
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np
import pytest

from vmlx_engine.metal import glm5_router_post as compiled
from vmlx_engine.models.glm5_next import glm5_next as native


@pytest.fixture(autouse=True)
def reset_dispatch(monkeypatch):
    monkeypatch.setattr(compiled, "_FAILED", False)
    monkeypatch.setattr(compiled, "_OBSERVED", False)


def stock(logits, bias, k=8, norm=True, scale=2.5):
    scores = mx.sigmoid(logits)
    choice = scores + bias
    idx = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, idx, axis=-1)
    if norm:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return idx, weights * scale


def assert_exact(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    bits = mx.uint32 if a.itemsize == 4 else mx.uint16
    np.testing.assert_array_equal(np.asarray(a.view(bits)), np.asarray(b.view(bits)))


@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("experts,k", [(8, 1), (288, 8)])
@pytest.mark.parametrize("norm", [False, True])
def test_exact_random_selection_weights_and_inputs(tokens, experts, k, norm):
    mx.random.seed(tokens+experts)
    logits = mx.random.normal((1,tokens,experts*2))[..., ::2]
    bias = mx.random.normal((experts*2,))[::2]
    before = tuple(mx.array(v) for v in (logits,bias))
    expected = stock(logits,bias,k,norm,1.23456789012345)
    actual = compiled.glm5_router_post(logits,bias,top_k=k,norm_topk=norm,
                                      scaling=1.23456789012345,enabled=True)
    assert actual is not None
    for a,b in zip(expected,actual):
        assert_exact(a,b)
    for a,b in zip(before,(logits,bias)):
        assert_exact(a,b)


@pytest.mark.parametrize("scale", [0.0, -0.0, 2.5, 2.50001])
@pytest.mark.parametrize("case", ["ties", "extreme", "nonfinite"])
def test_ties_extremes_nonfinite_and_exact_scaling(case, scale):
    values = {"ties":[0.0]*8,
              "extreme":[-1000.,-100.,-1e-38,-0.,0.,1e-38,100.,1000.],
              "nonfinite":[-float('inf'),float('inf'),float('nan'),0.,1.,-1.,2.,3.]}[case]
    logits = mx.array(values).reshape(1,1,8)
    bias = mx.zeros((8,))
    expected = stock(logits,bias,4,True,scale)
    actual = compiled.glm5_router_post(logits,bias,top_k=4,norm_topk=True,
                                      scaling=scale,enabled=True)
    assert actual is not None
    for a,b in zip(expected,actual):
        assert_exact(a,b)


@pytest.mark.parametrize("logit", [-100., -46., -44.])
def test_underflow_normalization_boundary(logit):
    logits = mx.full((1,1,8),logit)
    bias = mx.zeros((8,))
    expected = stock(logits,bias,4,True,2.5)
    actual = compiled.glm5_router_post(logits,bias,top_k=4,norm_topk=True,
                                      scaling=2.5,enabled=True)
    assert actual is not None
    for a,b in zip(expected,actual):
        assert_exact(a,b)


def test_default_off_and_unsupported_rows_keep_stock(monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_ROUTER_COMPILE",raising=False)
    assert not compiled.router_compile_requested()
    monkeypatch.setenv("VMLX_GLM5_ROUTER_COMPILE","1")
    assert compiled.router_compile_requested()
    def unexpected(*args,**kwargs):
        pytest.fail("unsupported row reached compiler")
    monkeypatch.setattr(compiled,"_compiled_post",unexpected)
    logits,bias = mx.zeros((1,1,8)),mx.zeros((8,))
    options = dict(top_k=4,norm_topk=True,scaling=2.5,enabled=True)
    assert compiled.glm5_router_post(logits,bias,**{**options,"enabled":False}) is None
    for shape in ((1,9,8),(2,1,8),(1,8),(1,0,8)):
        assert compiled.glm5_router_post(mx.zeros(shape),bias,**options) is None
    assert compiled.glm5_router_post(logits.astype(mx.bfloat16),bias,**options) is None
    assert compiled.glm5_router_post(logits,bias.astype(mx.float16),**options) is None
    assert compiled.glm5_router_post(logits,bias[None],**options) is None
    for bad in ({"top_k":0},{"top_k":9},{"top_k":True},{"norm_topk":1},
                {"scaling":float('inf')},{"scaling":float('nan')}):
        assert compiled.glm5_router_post(logits,bias,**{**options,**bad}) is None
    monkeypatch.setattr(mx,"default_device",lambda:mx.cpu)
    assert compiled.glm5_router_post(logits,bias,**options) is None


def test_launch_failure_disables_without_mutation(monkeypatch,caplog):
    def fail(*args):
        raise RuntimeError("controlled router compilation failure")
    monkeypatch.setattr(compiled,"_compiled_post",fail)
    logits,bias = mx.ones((1,1,8)),mx.zeros((8,))
    before = tuple(mx.array(v) for v in (logits,bias))
    options = dict(top_k=4,norm_topk=True,scaling=2.5,enabled=True)
    assert compiled.glm5_router_post(logits,bias,**options) is None
    assert compiled._FAILED
    assert 'disabled after launch failure' in caplog.text
    for a,b in zip(before,(logits,bias)):
        assert_exact(a,b)
    monkeypatch.setattr(compiled,"_compiled_post",lambda *a:pytest.fail('retry'))
    assert compiled.glm5_router_post(logits,bias,**options) is None


def test_observed_log_requires_materialized_result(caplog):
    with caplog.at_level(logging.INFO):
        result = compiled.glm5_router_post(mx.zeros((1,1,8)),mx.zeros((8,)),
            top_k=4,norm_topk=True,scaling=2.5,enabled=True)
    assert result is not None and compiled._OBSERVED
    assert 'GLM compiled router active' in caplog.text
    assert 'weight_storage=unchanged' in caplog.text


@pytest.mark.parametrize("bits,group_size,dtype", [
    (2,64,mx.bfloat16),(4,32,mx.float16),(6,64,mx.bfloat16),(8,128,mx.float32)])
def test_whole_moe_exact_and_packed_weights_unchanged(bits,group_size,dtype):
    mx.random.seed(bits)
    args = native.ModelArgs(hidden_size=128,n_routed_experts=8,num_experts_per_tok=2,
                            moe_intermediate_size=128,n_shared_experts=1)
    block = native.MoEBlock(args)
    block.update(tree_map(lambda a:a.astype(dtype),block.parameters()))
    # Actual class hooks create packed affine switch/shared projections; the
    # router deliberately retains its original floating storage.
    nn.quantize(block.switch_mlp,bits=bits,group_size=group_size)
    nn.quantize(block.shared_experts,bits=bits,group_size=group_size)
    assert block.shared_experts.prepare_runtime()
    mx.eval(block.parameters())
    before = [(name,id(v),mx.array(v)) for name,v in tree_flatten(block.parameters())]
    for tokens in (3,1,8,9):
        x = mx.random.normal((1,tokens,128)).astype(dtype)
        block._compiled_router = False
        expected = block(x)
        mx.eval(expected)
        block._compiled_router = True
        actual = block(x)
        assert_exact(expected,actual)
    for (name,identity,a),(name_after,b) in zip(before,tree_flatten(block.parameters())):
        assert name==name_after and identity==id(b)
        assert_exact(a,b)
    assert block.gate.weight.dtype == dtype


def test_preparation_enables_only_base_not_mtp(monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_ROUTER_COMPILE","1")
    args = native.ModelArgs(hidden_size=64,n_routed_experts=8,num_experts_per_tok=2,
                            moe_intermediate_size=32,n_shared_experts=1)
    base,mtp = native.MoEBlock(args),native.MoEBlock(args)
    monkeypatch.setattr(native,"install_affine_moe_pair_decode",lambda *a,**kw:0)
    owner = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False,mlp=base)]),
                            mtp=SimpleNamespace(mlp=mtp))
    summary = native.Model.prepare_acceleration(owner)
    assert base._compiled_router and not mtp._compiled_router
    assert summary['base_compiled_router_modules']==1
