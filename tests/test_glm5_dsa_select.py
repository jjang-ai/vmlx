"""GLM selector integration, packaged-helper ownership and safe fallback."""

import fnmatch
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
import tomllib

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np
import pytest

from vmlx_engine.metal import glm5_dsa_select as selection
from vmlx_engine.models.glm5_next import glm5_next as native


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(selection, "_FAILED", False)
    monkeypatch.setattr(selection, "_OBSERVED", False)
    monkeypatch.setattr(selection, "_VERIFIED_VARIANTS", set())


def assert_exact(left, right):
    mx.eval(left, right)
    assert left.shape == right.shape and left.dtype == right.dtype
    if left.dtype == mx.bfloat16:
        # NumPy cannot consume MLX's BF16 buffer format; compare storage bits
        # directly rather than converting or weakening metadata preservation.
        left, right = left.view(mx.uint16), right.view(mx.uint16)
    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_packaged_helper_matches_pinned_mlx_and_is_in_package_data():
    original = Path(mx.__file__).parent / "include/mlx/backend/metal/kernels/sort.h"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        "3171f47dabf2cf501dffc2e388e3d66446d6f71712e8d070028258a084b63574")
    packaged = Path(selection.__file__).with_name("glm5_dsa_sort.metal")
    assert packaged.read_text().rstrip() == original.read_text().split("// Kernel sort")[0].rstrip()
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root/"pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["vmlx_engine"]
    for relative in ["metal/glm5_dsa_sort.metal", "metal/third_party/mlx_dsa_sort/LICENSE",
                     "metal/third_party/mlx_dsa_sort/NOTICE"]:
        assert (root/"vmlx_engine"/relative).is_file()
        assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def test_default_off_and_guarded_inputs_never_launch(monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_DSA_BLOCK_SELECT", raising=False)
    assert not selection.glm5_dsa_select_requested()
    monkeypatch.setenv("VMLX_GLM5_DSA_BLOCK_SELECT", "1")
    assert selection.glm5_dsa_select_requested()
    monkeypatch.setattr(selection, "_kernel", lambda: pytest.fail("unexpected launch"))
    scores = mx.zeros((1,4,4097))
    assert selection.glm5_dsa_select(scores,k=512,enabled=False) is None
    for shape in [(1,1,4097),(2,4,4097),(1,4,2048),(1,0,4097),(4,4097)]:
        assert selection.glm5_dsa_select(mx.zeros(shape),k=512,enabled=True) is None
    for dtype in [mx.float16,mx.bfloat16,mx.int32]:
        assert selection.glm5_dsa_select(scores.astype(dtype),k=512,enabled=True) is None
    for k in [0,511,513,True,512.0]:
        assert selection.glm5_dsa_select(scores,k=k,enabled=True) is None
    monkeypatch.setattr(selection, "_compatible_sort_version", lambda:False)
    assert selection.glm5_dsa_select(scores,k=512,enabled=True) is None
    monkeypatch.setattr(selection, "_compatible_sort_version", lambda:True)
    monkeypatch.setattr(mx, "default_device", lambda:mx.cpu)
    assert selection.glm5_dsa_select(scores,k=512,enabled=True) is None


@pytest.mark.parametrize("error", [RuntimeError("compile rejected"), OSError("helper missing")])
def test_first_launch_failure_keeps_stock_and_input(monkeypatch,caplog,error):
    scores = mx.ones((1,4,4097))
    mx.eval(scores)
    before = mx.array(scores)
    def fail(): raise error
    monkeypatch.setattr(selection, "_kernel", fail)
    assert selection.glm5_dsa_select(scores,k=512,enabled=True) is None
    assert selection._FAILED and not selection._OBSERVED
    assert "disabled after launch failure" in caplog.text
    assert_exact(scores,before)
    monkeypatch.setattr(selection, "_kernel", lambda:pytest.fail("retry"))
    assert selection.glm5_dsa_select(scores,k=512,enabled=True) is None


def test_new_variant_materializes_before_observed(monkeypatch):
    scores = mx.zeros((1,4,4097))
    mx.eval(scores)
    def fail(*arrays): raise RuntimeError("controlled lazy launch failure")
    monkeypatch.setattr(mx,"eval",fail)
    assert selection.glm5_dsa_select(scores,k=512,enabled=True) is None
    assert selection._FAILED and not selection._OBSERVED
    assert not selection._VERIFIED_VARIANTS


def test_packaged_dispatch_exact_strided_and_bounded_variants(caplog):
    rng = np.random.default_rng(9523)
    for n in (2049,4097,16387):
        scores = mx.array(rng.normal(size=(1,4,n*2)).astype(np.float32))[...,::2]
        expected = mx.argpartition(scores,kth=511,axis=-1)[...,:512]
        with caplog.at_level(logging.INFO):
            actual = selection.glm5_dsa_select(scores,k=512,enabled=True)
        assert actual is not None
        assert_exact(expected,actual)
    assert selection._OBSERVED
    assert len(selection._VERIFIED_VARIANTS) <= 3
    assert "GLM DSA block selection active" in caplog.text
    assert "scope=base_prefill" in caplog.text


@pytest.mark.parametrize("bits,group_size,dtype", [
    (2,32,mx.float16),(4,64,mx.bfloat16),(6,32,mx.bfloat16),(8,64,mx.float32)])
def test_actual_indexer_mixed_quant_projection_and_causal_tail(bits,group_size,dtype):
    mx.random.seed(9524+bits)
    args = native.ModelArgs(hidden_size=128,q_lora_rank=128,index_n_heads=4,
                            index_head_dim=32,index_topk=2048,index_kpool=4)
    indexer = native.Glm5NextIndexer(args)
    indexer.update(tree_map(lambda a:a.astype(dtype),indexer.parameters()))
    nn.quantize(indexer, bits=bits, group_size=group_size)
    mx.eval(indexer.parameters())
    before = [(name,id(v),mx.array(v)) for name,v in tree_flatten(indexer.parameters())]
    for rows in (4,1):
        x = mx.random.normal((1,rows,128)).astype(dtype)
        q_resid = mx.random.normal((1,rows,128)).astype(dtype)
        n,tail = 2051,3
        packed = mx.random.normal((1,n*4+tail,64))
        pools = indexer.compress_pool_keys(packed[:,:n*4,:])
        positions = (mx.array([0,3,2048,n*4+tail-1]) if rows==4
                     else mx.array([n*4+tail-1]))
        indexer._block_select_prefill = False
        expected = indexer.topk_indices(x,q_resid,packed,positions,pools)
        mx.eval(expected)
        indexer._block_select_prefill = True
        actual = indexer.topk_indices(x,q_resid,packed,positions,pools)
        for a,b in zip(expected,actual): assert_exact(a,b)
    for (name,identity,a),(after_name,b) in zip(before,tree_flatten(indexer.parameters())):
        assert name==after_name and identity==id(b)
        assert_exact(a,b)


def test_prepare_enables_base_indexer_not_mtp(monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_DSA_BLOCK_SELECT","1")
    monkeypatch.setattr(native,"install_affine_moe_pair_decode",lambda *a,**kw:0)
    args = native.ModelArgs(hidden_size=128,q_lora_rank=128,index_n_heads=4,index_head_dim=32,
                            n_routed_experts=8,num_experts_per_tok=2,moe_intermediate_size=128)
    base,mtp = native.Glm5NextIndexer(args),native.Glm5NextIndexer(args)
    base_mlp,mtp_mlp = native.MoEBlock(args),native.MoEBlock(args)
    owner = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(
        is_linear=False,self_attn=SimpleNamespace(indexer=base),mlp=base_mlp)]),
        mtp=SimpleNamespace(self_attn=SimpleNamespace(indexer=mtp),mlp=mtp_mlp))
    summary = native.Model.prepare_acceleration(owner)
    assert base._block_select_prefill and not mtp._block_select_prefill
    assert summary['base_dsa_block_select_modules']==1
    monkeypatch.delenv("VMLX_GLM5_DSA_BLOCK_SELECT")
    summary = native.Model.prepare_acceleration(owner)
    assert not base._block_select_prefill and not mtp._block_select_prefill
    assert summary['base_dsa_block_select_modules']==0
