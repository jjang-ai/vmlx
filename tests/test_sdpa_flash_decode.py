"""metal/sdpa_flash_decode.py (oMLX port) on MiMo-V2.6 full-attention decode shapes: 64 q heads over 4 kv heads,
q/k 192, v 128, 1-4 rows, key counts that do not divide the 8-key tiles or the split chunk; and the model routing."""
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.metal import nax_attention_bdv as nab  # noqa: E402
from vmlx_engine.metal.sdpa_flash_decode import sdpa_flash  # noqa: E402

pytestmark = pytest.mark.skipif(not nab.available(), reason="needs an M5-class GPU (float32 simdgroup MMA)")
SCALE = 192 ** -0.5


def _qkv(L, T, seed, Hk=4):
    kq, kk, kv = mx.random.split(mx.random.key(seed), 3)
    q = (0.5 * mx.random.normal((1, 64, L, 192), key=kq)).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((1, Hk, T, 192), key=kk)).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((1, Hk, T, 128), key=kv)).astype(mx.bfloat16)
    return q, k, v


@pytest.mark.parametrize("L,T", [(1, 40001), (2, 5003), (3, 2049), (4, 33333), (5, 9001), (8, 4099)])
def test_matches_mlx_and_fp32(L, T):
    q, k, v = _qkv(L, T, L * 7 + T)
    mask = None if L == 1 else "causal"
    out = sdpa_flash(q, k, v, SCALE, mask, None)
    assert out is not None and out.shape == (1, L, 64 * 128)
    out = out.reshape(1, L, 64, 128).transpose(0, 2, 1, 3).astype(mx.float32)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=mask).astype(mx.float32)
    assert mx.abs(out - ref).max().item() < 2e-3
    assert mx.abs(out - nab.reference_attention(q, k, v, SCALE, mask)).max().item() < 2e-3


def test_model_routes_decode_rows_to_flash_and_matches(monkeypatch):
    import vmlx_engine.models.mimo_v26_janght as M

    for L, T in ((1, 40000), (2, 9000), (4, 3000), (7, 5000)):
        q, k, v = _qkv(L, T, 100 + L)
        mask = None if L == 1 else "causal"
        monkeypatch.setattr(M, "_FLASH_DECODE", True)
        assert M._flash_decode(q, k, v, SCALE, mask, None) is not None
        got = M._full_attention(q, k, v, SCALE, mask, None)
        monkeypatch.setattr(M, "_FLASH_DECODE", False)
        want = M._full_attention(q, k, v, SCALE, mask, None)
        assert got.shape == want.shape == (1, 64, L, 128)
        assert mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() < 2e-3


def test_short_caches_and_wide_calls_keep_the_existing_route():
    import vmlx_engine.models.mimo_v26_janght as M

    q, k, v = _qkv(1, 8192, 3)
    assert M._flash_decode(q, k, v, SCALE, None, None) is None        # one row under 32k keys
    q, k, v = _qkv(9, 8192, 4)
    assert M._flash_decode(q, k, v, SCALE, "causal", None) is None    # 9+ rows: grouped / NAX paths


def test_fused_sorted_combine_matches_the_batched_matmul():
    from vmlx_engine.metal.moe_combine import sorted_combine

    for T in (2047, 64):
        k, D = 8, 4096
        idx = mx.random.randint(0, 256, (T, k), key=mx.random.key(T)).astype(mx.int32).reshape(-1)
        order = mx.argsort(idx)
        w2 = mx.random.uniform(shape=(T, k), key=mx.random.key(T + 1)).astype(mx.float32)
        y_s = (0.1 * mx.random.normal((T * k, D), key=mx.random.key(T + 2))).astype(mx.bfloat16)
        ref = (w2[:, None, :] @ y_s[mx.argsort(order)].reshape(T, k, -1).astype(mx.float32))[:, 0, :]
        out = sorted_combine(y_s, order, w2, mx.float32)
        assert out is not None and out.shape == (T, D) and out.dtype == mx.float32
        assert (mx.abs(out - ref).max() / mx.abs(ref).max()).item() < 1e-6


def test_attention_kernel_warmup_builds_the_long_context_variants(caplog):
    """warm_attention_kernels runs every routed variant on a stand-in with MiMo's attention geometry without error."""
    import types

    import vmlx_engine.models.mimo_v26_janght as M

    def attn(is_swa):
        return types.SimpleNamespace(self_attn=types.SimpleNamespace(
            is_swa=is_swa, n_heads=64, n_kv=8 if is_swa else 4, head_dim=192, v_head_dim=128, scale=192 ** -0.5,
            has_sink=is_swa, _window=128 if is_swa else 0))

    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[attn(True), attn(False)]))
    with caplog.at_level("WARNING"):
        assert M.warm_attention_kernels(model) >= 0.0
    assert "warm-up skipped" not in caplog.text
