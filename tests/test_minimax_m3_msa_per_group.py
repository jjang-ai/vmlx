# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA: every KV group keeps its own top-k key blocks.

The upstream indexer (transformers ``MiniMaxM3VLIndexer``) selects key blocks per
indexer head, one per KV group, and each group's query heads attend only their
group's blocks. A selection pooled over the indexer heads matches that only while
every block is visible (context <= top-k * block size), so these tests run past
that point, at a length that is not a multiple of the block size, and assert the
groups really disagree somewhere (otherwise the comparison proves nothing).

The NumPy reference below is written from the upstream semantics, independently
of the module under test.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

H, KV, D, ROT, HID = 4, 2, 8, 4, 32
BLOCK, TOPK, SEQ = 4, 2, 23
# Relative to max |reference|. MLX runs float32 matmuls as TF32 on M5 by default (measured 2.2e-3 here; 2.2e-7 with
# MLX_ENABLE_TF32=0); one selection shared by all groups measured 0.71.
REL_TOL = 1e-2


def _args(**kw):
    from vmlx_engine.models.minimax_m3.minimax_m3 import ModelArgs

    base = dict(
        hidden_size=HID, num_hidden_layers=4, num_attention_heads=H, num_key_value_heads=KV, head_dim=D,
        rotary_dim=ROT, rope_theta=10_000.0, index_n_heads=KV, index_head_dim=D, index_block_size=BLOCK,
        index_topk_blocks=TOPK, index_local_blocks=1, sparse_attention_freq=[1, 1, 1, 1],
        moe_layer_freq=[0, 0, 0, 1],
    )
    base.update(kw)
    return ModelArgs(**base)


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda *s: (rng.standard_normal(s) * 0.3).astype(np.float32)  # noqa: E731
    return {
        "q": w(H * D, HID), "k": w(KV * D, HID), "v": w(KV * D, HID), "o": w(HID, H * D),
        "qn": w(D), "kn": w(D), "iq": w(KV * D, HID), "ik": w(D, HID), "iqn": w(D), "ikn": w(D),
    }


def _attention(W):
    from vmlx_engine.models.minimax_m3.minimax_m3 import Attention

    a = Attention(_args(), 0)
    a.q_proj.weight, a.k_proj.weight = mx.array(W["q"]), mx.array(W["k"])
    a.v_proj.weight, a.o_proj.weight = mx.array(W["v"]), mx.array(W["o"])
    a.q_norm.weight, a.k_norm.weight = mx.array(W["qn"]), mx.array(W["kn"])
    ix = a.indexer
    ix.index_q_proj.weight, ix.index_k_proj.weight = mx.array(W["iq"]), mx.array(W["ik"])
    ix.index_q_norm.weight, ix.index_k_norm.weight = mx.array(W["iqn"]), mx.array(W["ikn"])
    return a


def _norm(x, w):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1.0 + w)


def _rope(x, pos):  # x [S, heads, D]; first ROT dims rotated, halves layout
    half = ROT // 2
    inv = 10_000.0 ** (-np.arange(0, ROT, 2) / ROT)
    ang = pos[:, None, None] * inv[None, None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:ROT]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, x[..., ROT:]], axis=-1)


def _reference(x, W):
    """Upstream MSA for one sequence: returns (output [S, HID], selection [G, S, n_blocks] bool)."""
    S = x.shape[0]
    pos = np.arange(S, dtype=np.float64)
    q = _rope(_norm((x @ W["q"].T).reshape(S, H, D), W["qn"]), pos)
    k = _rope(_norm((x @ W["k"].T).reshape(S, KV, D), W["kn"]), pos)
    v = (x @ W["v"].T).reshape(S, KV, D)
    iq = _rope(_norm((x @ W["iq"].T).reshape(S, KV, D), W["iqn"]), pos)
    ik = _rope(_norm((x @ W["ik"].T).reshape(S, 1, D), W["ikn"]), pos)[:, 0]
    nb = -(-S // BLOCK)
    causal = np.arange(S)[None, :] <= np.arange(S)[:, None]
    sel = np.zeros((KV, S, nb), bool)
    for g in range(KV):
        sc = np.where(causal, iq[:, g] @ ik.T, -np.inf)
        sc = np.pad(sc, ((0, 0), (0, nb * BLOCK - S)), constant_values=-np.inf)
        bs = sc.reshape(S, nb, BLOCK).max(-1)
        bs[np.arange(S), np.arange(S) // BLOCK] = np.inf  # local block always kept
        for i in range(S):
            valid = np.flatnonzero(bs[i] > -np.inf)
            sel[g, i, valid[np.argsort(-bs[i, valid], kind="stable")[:TOPK]]] = True
    allowed_blocks = sel if nb > TOPK else np.ones_like(sel)  # all blocks visible
    out = np.zeros((S, H, D))
    for h in range(H):
        g, kv = h // (H // KV), h // (H // KV)
        keep = causal & np.repeat(allowed_blocks[g], BLOCK, axis=1)[:, :S]
        logits = np.where(keep, (q[:, h] @ k[:, kv].T) * D ** -0.5, -np.inf)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        out[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, kv]
    return out.reshape(S, H * D) @ W["o"].T, sel


def test_reference_groups_disagree_past_topk_blocks():
    x = np.random.default_rng(1).standard_normal((SEQ, HID)).astype(np.float32)
    _, sel = _reference(x, _weights())
    assert -(-SEQ // BLOCK) > TOPK  # some blocks are not visible to some queries
    assert (sel[0] != sel[1]).any(-1).sum() > 0, "groups never disagree: the test would prove nothing"


def test_prefill_matches_per_group_reference():
    W = _weights()
    x = np.random.default_rng(1).standard_normal((SEQ, HID)).astype(np.float32)
    ref, _ = _reference(x, W)
    out = np.array(_attention(W)(mx.array(x)[None], mask="causal", cache=None))[0]
    np.testing.assert_allclose(out, ref, rtol=0, atol=REL_TOL * np.abs(ref).max())


def test_decode_and_chunked_prefill_through_the_cache_match_reference():
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    W = _weights()
    x = np.random.default_rng(1).standard_normal((SEQ, HID)).astype(np.float32)
    ref, _ = _reference(x, W)
    a, xm = _attention(W), mx.array(x)[None]
    tol = REL_TOL * np.abs(ref).max()

    c = MiniMaxM3SparseCache()
    a(xm[:, : SEQ - 1], mask="causal", cache=c)
    last = np.array(a(xm[:, SEQ - 1:], mask=None, cache=c))[0, 0]
    np.testing.assert_allclose(last, ref[SEQ - 1], rtol=0, atol=tol)

    c = MiniMaxM3SparseCache()
    rows = [np.array(a(xm[:, i:i + 7], mask="causal", cache=c))[0] for i in range(0, SEQ, 7)]
    np.testing.assert_allclose(np.concatenate(rows), ref, rtol=0, atol=tol)


def test_model_cache_follows_sparse_attention_freq_not_the_moe_layout():
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache, make_minimax_m3_cache

    # M3.1: sparse attention on every layer, dense MLP on the first layers.
    caches = make_minimax_m3_cache(_args(sparse_attention_freq=[1, 1, 1, 1], moe_layer_freq=[0, 0, 0, 1]))
    assert all(isinstance(c, MiniMaxM3SparseCache) for c in caches)
    # M3: dense attention on the first three layers.
    caches = make_minimax_m3_cache(_args(sparse_attention_freq=[0, 0, 0, 1], moe_layer_freq=[0, 0, 0, 1]))
    assert [isinstance(c, MiniMaxM3SparseCache) for c in caches] == [False, False, False, True]
