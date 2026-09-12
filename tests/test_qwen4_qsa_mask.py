# SPDX-License-Identifier: Apache-2.0
"""Exact mask/consumer and gate tests; no weight quantization assumptions."""

import mlx.core as mx
import numpy as np
import pytest

import vmlx_engine.metal.qwen4_qsa_mask as mask_module
from vmlx_engine.models.qwen4_exp.language import QSAIndexer, Qwen4ExpTextArgs
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_QSA_MASK", raising=False)
    monkeypatch.setattr(mask_module, "_FAILED", False)
    monkeypatch.setattr(mask_module, "_OBSERVED", set())


def reference(hits, counts, length):
    b, s, blocks = hits.shape
    complete = mx.arange(blocks)[None, :] < counts[:, None]
    keep = hits & complete[None]
    keep = mx.repeat(keep[..., None], 4, axis=-1).reshape(b, s, blocks * 4)
    if length > blocks * 4:
        keep = mx.concatenate([keep, mx.ones((b, s, length-blocks*4), mx.bool_)], axis=-1)
    keep = keep | (mx.arange(length)[None, :] >= (counts*4)[:, None])[None]
    return mx.where(keep[:, None], mx.array(0.0, mx.float32), mx.array(-float("inf"), mx.float32))


def inputs(b, s, length, early=False, strided=False):
    rng = np.random.default_rng(2200 + length + s)
    hits = mx.array(rng.random((b, s, (length//4)*(2 if strided else 1))) < .3)
    if strided:
        hits = hits[..., ::2]
    counts = mx.array((np.arange(s)+1 if early else np.arange(length-s+1,length+1))//4, mx.int32)
    return hits, counts


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    mx.eval(a,b)
    bits = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(np.asarray(a.view(bits)), np.asarray(b.view(bits)))


@pytest.mark.parametrize("batch,rows,length,early,strided", [
    (1,1,2052,False,False), (1,2,2053,False,False),
    (1,3,8191,False,False), (1,4,32768,False,True),
    (2,4,131072,False,False), (1,4,2053,True,False),
])
def test_exact_partial_tail_future_blocks_and_strides(batch, rows, length, early, strided):
    h,c = inputs(batch,rows,length,early,strided)
    out = mask_module.qsa_block_mask(h,c,ratio=4,key_length=length,enabled=True)
    assert out is not None
    exact(reference(h,c,length),out)


@pytest.mark.parametrize("dtype", [mx.float16,mx.bfloat16,mx.float32])
def test_consumer_preserves_causal_mask_and_attention(dtype):
    length, rows = 2053, 4
    h,c = inputs(1,rows,length)
    a = reference(h,c,length)
    z = mask_module.qsa_block_mask(h,c,ratio=4,key_length=length,enabled=True)
    # The index mask permits the future tail; only the consumer removes it.
    assert bool(mx.all(mx.isfinite(z[0,0,0,length-rows:])))
    causal = mx.where(mx.arange(length)[None,:] <= mx.arange(length-rows,length)[:,None],0.0,-float("inf")).astype(dtype)[None,None]
    q=mx.random.normal((1,4,rows,32)).astype(dtype)
    k=mx.random.normal((1,2,length,32)).astype(dtype)
    v=mx.random.normal((1,2,length,32)).astype(dtype)
    left=mx.fast.scaled_dot_product_attention(q,k,v,scale=32**-.5,mask=a.astype(dtype)+causal)
    right=mx.fast.scaled_dot_product_attention(q,k,v,scale=32**-.5,mask=z.astype(dtype)+causal)
    exact(left,right)


def test_default_off_and_unsupported_shapes_never_launch(monkeypatch):
    assert not mask_module.qsa_mask_requested()
    monkeypatch.setenv("VMLX_QWEN4_QSA_MASK", "1")
    assert mask_module.qsa_mask_requested()
    monkeypatch.setattr(mask_module,"_kernel",lambda: pytest.fail("must retain stock path"))
    h,c=inputs(1,1,2053)
    assert mask_module.qsa_block_mask(h,c,ratio=4,key_length=2053,enabled=False) is None
    for b,s,length,ratio in ((3,1,2053,4),(1,5,2053,4),(1,1,2048,4),
                             (1,1,131073,4),(1,1,2053,2)):
        h,c=inputs(b,s,length)
        assert mask_module.qsa_block_mask(h,c,ratio=ratio,key_length=length,enabled=True) is None
    h,c=inputs(1,1,2053)
    assert mask_module.qsa_block_mask(h.astype(mx.int32),c,ratio=4,key_length=2053,enabled=True) is None
    assert mask_module.qsa_block_mask(h,c.astype(mx.int64),ratio=4,key_length=2053,enabled=True) is None
    assert mask_module.qsa_block_mask(h,c,ratio=4,key_length=2056,enabled=True) is None
    with mx.stream(mx.cpu):
        # Stream selection is independent of default device; explicitly guard it.
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            assert mask_module.qsa_block_mask(h,c,ratio=4,key_length=2053,enabled=True) is None
        finally:
            mx.set_default_device(previous)


def test_launch_failure_retains_inputs_and_stops_retrying(monkeypatch,caplog):
    h,c=inputs(1,1,2053)
    before_h,before_c=np.asarray(h).copy(),np.asarray(c).copy()
    attempts=[]
    def broken():
        attempts.append(True)
        raise RuntimeError("controlled compile failure")
    monkeypatch.setattr(mask_module,"_kernel",broken)
    for _ in range(2):
        assert mask_module.qsa_block_mask(h,c,ratio=4,key_length=2053,enabled=True) is None
    assert len(attempts)==1 and "disabled after launch failure" in caplog.text
    np.testing.assert_array_equal(np.asarray(h),before_h)
    np.testing.assert_array_equal(np.asarray(c),before_c)


def test_real_indexer_masks_and_native_state_unchanged(monkeypatch):
    args=Qwen4ExpTextArgs(hidden_size=32,indexer_n_heads=2,indexer_kv_heads=1,
        indexer_head_dim=16,indexer_budget=2048,indexer_compress_ratio=4,
        head_dim=32,rope_theta=10000.0,partial_rotary_factor=.25,mrope_section=[2,1,1])
    ix=QSAIndexer(args)
    assert not ix._fused_block_mask
    reference_cache,candidate_cache=MiniMaxM3SparseCache(),MiniMaxM3SparseCache()
    for rows in (2049,4,1,3):
        hidden=mx.random.normal((1,rows,32))
        outputs=[]
        for enabled,cache in ((False,reference_cache),(True,candidate_cache)):
            offset=cache.offset
            kv=mx.zeros((1,1,rows,4))
            cache.update_and_fetch(kv,kv)
            ix._fused_block_mask=enabled
            out=ix(hidden,cache,offset=offset)
            if out is not None:
                mx.eval(out)
            outputs.append(out)
        if outputs[0] is None:
            assert outputs[1] is None
        else:
            exact(*outputs)
        assert reference_cache.offset==candidate_cache.offset
        assert reference_cache._idx_offset==candidate_cache._idx_offset
        for left,right in zip(reference_cache.state,candidate_cache.state):
            exact(left,right)
    assert (1,1) in mask_module._OBSERVED and (1,4) in mask_module._OBSERVED


def test_indexer_direct_block_consumer_does_not_construct_mask(monkeypatch):
    import vmlx_engine.models.qwen4_exp.language as language
    args=Qwen4ExpTextArgs(hidden_size=32,indexer_n_heads=2,indexer_kv_heads=1,
        indexer_head_dim=16,indexer_budget=8,indexer_compress_ratio=4,
        head_dim=32,rope_theta=10000.0,partial_rotary_factor=.25,mrope_section=[2,1,1])
    ix=QSAIndexer(args)
    monkeypatch.setattr(language,"qsa_block_mask",lambda *a,**kw: pytest.fail("direct path must not construct token mask"))
    selected,valid=ix(mx.random.normal((1,16,32)),None,return_blocks=True)
    mx.eval(selected,valid)
    assert selected.shape==(16,2) and valid.dtype==mx.bool_
