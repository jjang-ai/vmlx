"""Absorbed-MLA decode attention kernel (metal/glm5_mla_decode_attn) == fp32 SDPA (dense) / fp32 gather+mask (DSA).

Row counts that do not divide the 128-row split (the partial-split tail is where flash-decoding bugs hide), a full
2048-row selection, invalid slots and indices past the query position (masked like the stock path).
"""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal.glm5_mla_decode_attn import glm5_mla_decode_attn

H, R = 64, 512
SCALE = 0.0625


def _inputs(n, seed):
    mx.random.seed(seed)
    q = (mx.random.normal((1, H, 1, R)) * 2).astype(mx.bfloat16)
    lat = mx.random.normal((1, 1, n, R)).astype(mx.bfloat16)
    return q, lat


def _ok(got, ref):
    g, r = np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32))
    return np.max(np.abs(g - r)) <= 8e-3 * np.max(np.abs(r)) + 1e-6, np.max(np.abs(g - r))   # <= 1 bf16 ULP


@pytest.mark.parametrize("n", [1, 129, 300, 1634, 2048])
def test_dense_matches_fp32_sdpa(n):
    q, lat = _inputs(n, n)
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), lat.astype(mx.float32), lat.astype(mx.float32),
                                               scale=SCALE).astype(q.dtype)
    ok, err = _ok(glm5_mla_decode_attn(q, lat, SCALE), ref)
    assert ok, err


@pytest.mark.parametrize("total,w", [(5000, 2048), (3001, 777), (16214, 2048)])
def test_indexed_matches_gather_mask(total, w):
    q, lat = _inputs(total, total)
    rng = np.random.default_rng(w)
    past = total - 1
    idx = rng.choice(total + 40, w, replace=False).astype(np.int32)          # some indices > past (masked)
    valid = rng.random(w) > 0.05                                              # some invalid slots
    allowed = valid & (idx <= past)
    safe = np.where(valid, np.minimum(idx, total - 1), 0)
    keys = mx.take(lat[0, 0], mx.array(safe), axis=0)[None, None].astype(mx.float32)
    bias = mx.where(mx.array(allowed), 0.0, -mx.inf).astype(mx.float32).reshape(1, 1, 1, w)
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), keys, keys, scale=SCALE, mask=bias).astype(q.dtype)
    got = glm5_mla_decode_attn(q, lat, SCALE, mx.array(np.where(valid, np.minimum(idx, total - 1 + 40), 0))[None, None],
                               mx.array(valid)[None, None], past)
    # kernel reads latent[idx] only for allowed rows; idx > past never dereferenced
    ok, err = _ok(got, ref)
    assert ok, err


def test_rejects_non_decode_shapes():
    q, lat = _inputs(10, 0)
    assert glm5_mla_decode_attn(mx.concatenate([q, q], axis=2), lat, SCALE) is None
    assert glm5_mla_decode_attn(q.astype(mx.float32), lat, SCALE) is None
