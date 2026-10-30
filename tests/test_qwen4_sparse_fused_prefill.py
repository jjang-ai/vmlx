"""Current-SDPA numerical contract, dispatch bounds and kernel-math cache isolation."""

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.metal import qwen4_sparse_fused_prefill as fused


@pytest.fixture
def active(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL", "1")
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "0")
    monkeypatch.setattr(fused, "_state", "unproven")
    monkeypatch.setattr(fused, "_first_dispatch", False)
    if not fused.ready():
        pytest.skip("measured M5 / MLX 0.32.2 runtime unavailable")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL", raising=False)
    assert not fused.enabled()
    assert not fused.ready()


def test_preflight_preserves_sampling_rng(active):
    before = list(mx.random.state)
    fused._preflight()
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(before, mx.random.state))


def test_failed_preflight_disables_without_retry(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL", "1")
    monkeypatch.setattr(fused, "_state", "unproven")
    monkeypatch.setattr(fused, "_hardware_allowed", lambda: True)
    monkeypatch.setattr(fused.direct, "lane_unavailable_reason", lambda **kw: None)
    # Isolate the preflight-failure logic from the runtime gates ready()
    # checks first (MLX version / GPU pin, matrix units).
    monkeypatch.setattr(fused, "enabled", lambda: True)
    calls = []
    def fail():
        calls.append(1)
        raise RuntimeError("injected wrong math")
    monkeypatch.setattr(fused, "_preflight", fail)
    assert not fused.ready()
    assert not fused.ready()
    assert calls == [1]


@pytest.mark.parametrize("rows,total,dtype", [
    (1,32768,mx.float16), (4,32768,mx.float16), (255,32768,mx.float16),
    (4097,32768,mx.float16), (256,32767,mx.float16),
    (256,131073,mx.float16), (256,32768,mx.bfloat16),
])
def test_outside_measured_shapes_retains_stock(rows, total, dtype, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL", "1")
    q = mx.zeros((1,24,rows,256), dtype=dtype)
    ids = mx.zeros((rows,512), dtype=mx.int32)
    assert not fused.supported(q, None, None, ids, ids == 0,
                               pos_start=total-rows,total_tokens=total,scale=0.0625)


@pytest.mark.parametrize("rows,total", [
    (256,32768), (256,32771), (256,65539),
    (1025,32771), (2048,32768), (4096,65539), (4096,131072),
])
def test_current_attention_partial_tail_and_logical_views(active, rows, total):
    offset = total - rows
    q = mx.random.normal((1,24,rows,256), key=mx.random.key(11)).astype(mx.float16)
    k = mx.random.normal((1,2,total+19,256), key=mx.random.key(12)).astype(mx.float16)[:,:,:total]
    v = mx.random.normal((1,2,total+19,256), key=mx.random.key(13)).astype(mx.float16)[:,:,:total]
    pos = mx.arange(offset,total)
    complete = (pos+1)//4
    ids = (mx.arange(512)[None]*complete[:,None]//512).astype(mx.int32)
    valid = mx.ones(ids.shape,dtype=mx.bool_)
    selected = mx.put_along_axis(mx.zeros((rows,total//4),dtype=mx.bool_),ids,mx.array(True),axis=-1)
    keep = mx.concatenate([mx.repeat(selected,4,axis=-1),mx.zeros((rows,total%4),dtype=mx.bool_)],axis=-1)
    tokens = mx.arange(total)[None]
    mask = mx.where((keep | (tokens >= complete[:,None]*4)) & (tokens <= pos[:,None]),0,-mx.inf).astype(q.dtype)[None,None]
    ref = mx.fast.scaled_dot_product_attention(q,k,v,scale=0.0625,mask=mask,force_fused=True)
    got = fused.attention(q,k,v,ids,valid,pos_start=offset,total_tokens=total,scale=0.0625)
    mx.eval(ref,got)
    assert bool(mx.all(mx.isfinite(got)))
    assert bool(mx.array_equal(got, ref))


@pytest.mark.parametrize("rows", [256,4096])
def test_actual_indexer_cache_and_restored_suffix(active, monkeypatch, rows):
    from vmlx_engine.models.qwen4_exp import language
    attention = language.QSAAttention(language.Qwen4ExpTextArgs(hidden_size=64))
    attention.set_dtype(mx.float16)
    attention.eval()
    cache = language.QSACache()
    offset = 32768
    k = mx.random.normal((1,2,offset,256),key=mx.random.key(21)).astype(mx.float16)
    v = mx.random.normal(k.shape,key=mx.random.key(22)).astype(mx.float16)
    raw = mx.random.normal((1,1,offset,128),key=mx.random.key(23))
    pos = mx.broadcast_to(mx.arange(offset)[None,None,:,None],(1,1,offset,3)).astype(mx.float32)
    mx.eval(cache.update_and_fetch(k,v),cache.update_index(mx.concatenate([raw,pos],axis=-1)))
    stock, candidate = deepcopy(cache), deepcopy(cache)
    x = mx.random.normal((1,rows,64),key=mx.random.key(24)).astype(mx.float16)
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL","0")
    ref = attention(x,cache=stock)
    mx.eval(ref)
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL","1")
    before = fused.DISPATCH_COUNT
    got = attention(x,cache=candidate)
    mx.eval(got)
    assert fused.DISPATCH_COUNT == before+1
    assert stock.offset == candidate.offset == offset+rows
    assert bool(mx.array_equal(ref, got))
    assert all(bool(mx.array_equal(a,b)) for a,b in zip(stock.state,candidate.state))
    # This layer's persisted state is unchanged; a disk-format reconstruction
    # plus short AR continuation must be bit-identical on these same inputs.
    restored = language.QSACache.from_state(candidate.state,candidate.meta_state)
    tail = mx.random.normal((1,1,64),key=mx.random.key(25)).astype(mx.float16)
    a,b = attention(tail,cache=candidate),attention(tail,cache=restored)
    mx.eval(a,b)
    assert bool(mx.array_equal(a,b))
    assert fused.DISPATCH_COUNT == before+1


def test_namespace_separates_math_and_artifact(monkeypatch):
    from vmlx_engine import prefix_cache
    model = SimpleNamespace(args=SimpleNamespace(model_type="qwen4_exp"))
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT","0")
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL","0")
    stock = prefix_cache.compute_model_cache_key(model)
    monkeypatch.setenv("VMLX_QWEN4_SPARSE_FUSED_PREFILL","1")
    monkeypatch.setattr(prefix_cache,"_qwen4_qsa_kernel_identity",lambda: "math-a")
    first = prefix_cache.compute_model_cache_key(model)
    monkeypatch.setattr(prefix_cache,"_qwen4_qsa_kernel_identity",lambda: "math-b")
    second = prefix_cache.compute_model_cache_key(model)
    assert len({stock,first,second}) == 3
