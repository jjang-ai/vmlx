"""Exact pairwise KDA prefill arithmetic and continued-state contract."""
import pytest
import mlx.core as mx
from vmlx_engine.models.glm5_next import kda

@pytest.mark.parametrize("tokens", [63,64,65,127,511,512,513])
def test_pairwise_full_recurrence(tokens, monkeypatch):
    mx.random.seed(tokens)
    shape=(1,tokens,2,128)
    q=kda.l2norm(mx.random.normal(shape))
    k=kda.l2norm(mx.random.normal(shape))
    v=mx.random.normal(shape)
    g=-mx.sigmoid(mx.random.normal(shape))
    beta=mx.sigmoid(mx.random.normal(shape[:-1]))
    initial=mx.random.normal((1,2,128,128))*.01
    monkeypatch.setattr(kda,"_EXACT_PAIRWISE_REQUESTED",False)
    a,sa=kda.kda_chunked(q,k,v,g,beta,initial)
    mx.eval(a,sa)
    monkeypatch.setattr(kda,"_EXACT_PAIRWISE_REQUESTED",True)
    b,sb=kda.kda_chunked(q,k,v,g,beta,initial)
    mx.eval(b,sb)
    assert mx.array_equal(a,b).item()
    assert mx.array_equal(sa,sb).item()
    # State from the optimized prefill feeds unchanged next-token recurrence.
    x=mx.random.normal((1,2,128))
    args=(kda.l2norm(x),kda.l2norm(x),x,-mx.ones_like(x),mx.full((1,2),.5))
    oa,na=kda.kda_step(*args,sa)
    ob,nb=kda.kda_step(*args,sb)
    mx.eval(oa,ob,na,nb)
    assert mx.array_equal(oa,ob).item()
    assert mx.array_equal(na,nb).item()

def test_pairwise_unsupported(monkeypatch):
    monkeypatch.setattr(kda,"_EXACT_PAIRWISE_REQUESTED",True)
    x=mx.zeros((1,2,32,128))
    assert kda._exact_pairwise_product(x,x,x) is None
    x=mx.zeros((1,2,64,128),dtype=mx.bfloat16)
    assert kda._exact_pairwise_product(x,x,x) is None
