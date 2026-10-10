"""Speculative-verify slabs (2..16 rows) take absorbed per-row MLA attention == the stock materialized path.

bf16 MLA layer with the real absorbed geometry (rank 512, 64 heads not required: 8 heads), 50-token cache, then a
4-row and a 7-row slab through both paths from identical cache copies (VMLX_GLM5_MLA_DECODE_ATTN=0 = stock).
"""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.glm5_next.glm5_next import Glm5MLACache, MLAAttention, ModelArgs


def _layer():
    mx.random.seed(0)
    args = ModelArgs(hidden_size=512, num_attention_heads=8, q_lora_rank=128, kv_lora_rank=512, qk_nope_head_dim=64,
                     v_head_dim=64, index_n_heads=2, index_head_dim=64, num_hidden_layers=1, vocab_size=64,
                     layer_types=["deepseek_sparse_attention"])
    a = MLAAttention(args)
    from mlx.utils import tree_map
    a.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, a.parameters()))
    for n, lin in (("q_a_proj", a.q_a_proj), ("q_b_proj", a.q_b_proj), ("kv_a_proj_with_mqa", a.kv_a_proj_with_mqa),
                   ("kv_b_proj", a.kv_b_proj), ("o_proj", a.o_proj)):
        lin.weight = (mx.random.normal(lin.weight.shape) * lin.weight.shape[1] ** -0.5).astype(mx.bfloat16)
    mx.eval(a.parameters())
    return a, args


def _copy(c):
    from vmlx_engine.models.glm5_next.glm5_next import clone_glm5_next_layer_cache
    return clone_glm5_next_layer_cache(c, copy_fn=lambda x: mx.array(x), copy_is_detached=True)


@pytest.mark.parametrize("T", [4, 7])
def test_verify_slab_matches_stock(monkeypatch, T):
    a, args = _layer()
    x = (mx.random.normal((1, 50 + T, 512))).astype(mx.bfloat16)
    c = Glm5MLACache(args.index_kpool)
    mx.eval(a(x[:, :50], cache=c), c.state)
    ca, cb = _copy(c), _copy(c)
    monkeypatch.setenv("VMLX_GLM5_MLA_DECODE_ATTN", "0")
    ref = a(x[:, 50:], cache=ca)
    monkeypatch.setenv("VMLX_GLM5_MLA_DECODE_ATTN", "1")
    from vmlx_engine.metal import glm5_mla_decode_attn as K
    n0 = K._OBSERVED
    got = a(x[:, 50:], cache=cb)
    assert K._OBSERVED == n0 + T            # fail closed: one kernel call per slab row, not the stock path
    mx.eval(ref, got)
    r, g = np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32))
    print("max|diff|", np.max(np.abs(r - g)), "max|ref|", np.max(np.abs(r)))
    assert np.max(np.abs(r - g)) <= 2e-2 * np.max(np.abs(r)), np.max(np.abs(r - g))
