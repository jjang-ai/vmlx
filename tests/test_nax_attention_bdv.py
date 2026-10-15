"""NAX 192/128 attention (metal/nax_attention_bdv.py) against fp32 and MLX's existing routes, on the shapes MiMo-V2.6
JANGHT prefill produces: full layers (64 q heads, 4 kv heads, causal or bool masks, long key ranges with key-range
passes) and sliding-window layers (8 kv heads, window 128, sinks, the model's own RotatingKVCache window mask).
Sizes deliberately do not divide by the 64-row / 32-key tiles."""
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.metal import nax_attention_bdv as nab  # noqa: E402

pytestmark = pytest.mark.skipif(not nab.available(), reason="needs a NAX (M5) GPU")

SCALE = 192 ** -0.5


def _qkv(H, Hk, L, T, seed):
    kq, kk, kv = mx.random.split(mx.random.key(seed), 3)
    q = (0.5 * mx.random.normal((1, H, L, 192), key=kq)).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((1, Hk, T, 192), key=kk)).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((1, Hk, T, 128), key=kv)).astype(mx.bfloat16)
    return q, k, v


def _err(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


@pytest.mark.parametrize("L,T", [(9, 9), (100, 100), (777, 3001), (2048, 2048 + 4093)])
def test_causal_full_layer_matches_fp32(L, T):
    q, k, v = _qkv(64, 4, L, T, L + T)
    out = nab.nax_attention_bdv(q, k, v, scale=SCALE, mask="causal")
    assert out is not None and out.shape == (1, 64, L, 128)
    assert _err(out, nab.reference_attention(q, k, v, SCALE, "causal")) < 2e-2


def test_window_with_sinks_matches_model_rotating_cache_route():
    """A 2048-row chunk after 5,000 cached tokens through mlx_lm's RotatingKVCache(128) and the model's own mask."""
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import RotatingKVCache

    cache = RotatingKVCache(max_size=128, keep=0)
    for n, seed in ((3000, 1), (2000, 2)):
        _, k0, v0 = _qkv(64, 8, 1, n, seed)
        cache.update_and_fetch(k0, v0)
    L = 2048
    q, k1, v1 = _qkv(64, 8, L, L, 3)
    mask = create_attention_mask(mx.zeros((1, L, 8)), cache, window_size=128)
    k, v = cache.update_and_fetch(k1, v1)
    sinks = mx.random.normal((64,), key=mx.random.key(4)).astype(mx.bfloat16)
    ref = scaled_dot_product_attention(q, k, v, cache, SCALE, mask=mask, sinks=sinks)
    out = nab.nax_attention_bdv(q, k, v, scale=SCALE, mask=mask, sinks=sinks, window=128)
    assert out is not None
    assert _err(out, ref) < 2e-2
    assert _err(out, nab.reference_attention(q, k, v, SCALE, mask, sinks, 128)) < 2e-2


@pytest.mark.parametrize("offset,L", [(0, 300), (50, 70), (0, 2048)])
def test_window_short_prompt_causal_string_mask(offset, L):
    """offset + L <= window gives mask "causal"; longer first chunks give the band; both through the model's mask."""
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import RotatingKVCache

    cache = RotatingKVCache(max_size=128, keep=0)
    if offset:
        _, k0, v0 = _qkv(64, 8, 1, offset, 5)
        cache.update_and_fetch(k0, v0)
    q, k1, v1 = _qkv(64, 8, L, L, 6)
    mask = create_attention_mask(mx.zeros((1, L, 8)), cache, window_size=128)
    k, v = cache.update_and_fetch(k1, v1)
    sinks = mx.random.normal((64,), key=mx.random.key(7)).astype(mx.bfloat16)
    out = nab.nax_attention_bdv(q, k, v, scale=SCALE, mask=mask, sinks=sinks, window=128)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=mask, sinks=sinks)
    assert out is not None and _err(out, ref) < 2e-2


def test_bool_padding_mask_is_honoured():
    """Batched-engine style left padding: the first 37 keys masked for every row on top of causal."""
    L, T = 333, 1000
    q, k, v = _qkv(64, 4, L, T, 8)
    rows = mx.arange(L)[:, None] + (T - L)
    cols = mx.arange(T)[None]
    mask = (rows >= cols) & (cols >= 37)
    out = nab.nax_attention_bdv(q, k, v, scale=SCALE, mask=mask)
    assert out is not None
    assert _err(out, nab.reference_attention(q, k, v, SCALE, mask)) < 2e-2


def test_key_range_passes_match_padded_fused_route():
    """512 rows x 64 heads over 20,001 keys: 3 key-range passes with the head-dim split vs MLX's padded fused call."""
    L, T = 512, 20001
    q, k, v = _qkv(64, 4, L, T, 9)
    assert len(nab._pass_edges(64 * ((L + 63) // 64), (T + 31) // 32, 0)) > 2
    out = nab.nax_attention_bdv(q, k, v, scale=SCALE, mask="causal")
    pad = lambda a: mx.pad(a, [(0, 0), (0, 0), (0, 0), (0, 256 - a.shape[-1])])
    ref = mx.fast.scaled_dot_product_attention(pad(q), pad(k), pad(v), scale=SCALE, mask="causal")[..., :128]
    assert out is not None and _err(out, ref) < 2e-2


def test_declines_what_it_does_not_handle():
    q, k, v = _qkv(64, 4, 8, 100, 10)
    assert nab.nax_attention_bdv(q, k, v, scale=SCALE, mask="causal") is None  # L <= 8 (decode / small verify)
    q, k, v = _qkv(64, 4, 100, 100, 11)
    fmask = mx.zeros((100, 100), dtype=mx.bfloat16)
    assert nab.nax_attention_bdv(q, k, v, scale=SCALE, mask=fmask) is None  # additive masks
    assert nab.nax_attention_bdv(q[..., :128], k[..., :128], v, scale=SCALE, mask="causal") is None  # other dims
