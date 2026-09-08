"""Correctness and dispatch guards for the opt-in MTP verification path."""

import mlx.core as mx
import pytest

from vmlx_engine.metal.qwen4_verify_sdpa import qwen4_verify_sdpa


def tensors(rows, context=8193, dtype=mx.bfloat16, strided=False):
    mx.random.seed(7 + rows + context)
    q = mx.random.normal((1, 24, rows * (2 if strided else 1), 256)).astype(dtype)
    if strided:
        q = q[:, :, ::2, :]
    k = mx.random.normal((1, 2, context + 19, 256)).astype(dtype)[:, :, :context]
    v = mx.random.normal((1, 2, context + 19, 256)).astype(dtype)[:, :, :context]
    scores = mx.random.uniform(shape=(1, rows, context // 4))
    positions = mx.arange(context - rows, context)
    complete = mx.arange(context // 4)[None, :] < ((positions + 1) // 4)[:, None]
    scores = mx.where(complete[None], scores, -1e9)
    top = mx.argpartition(-scores, kth=511, axis=-1)[..., :512]
    keep = mx.put_along_axis(
        mx.zeros(scores.shape, dtype=mx.bool_), top, mx.array(True), axis=-1
    )
    keep = mx.repeat(keep & complete[None], 4, axis=-1)
    if context % 4:
        keep = mx.concatenate(
            [keep, mx.zeros((1, rows, context % 4), dtype=mx.bool_)], axis=-1
        )
    tokens = mx.arange(context)
    tail = tokens[None, None, :] >= (((positions + 1) // 4) * 4)[None, :, None]
    keep = (keep | tail) & (tokens[None, None, :] <= positions[None, :, None])
    mask = mx.where(keep[:, None], 0.0, -float("inf")).astype(dtype)
    return q, k, v, mask


@pytest.mark.parametrize("rows", [3, 4])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("context", [8192, 8193, 8194, 8195, 16384])
def test_sparse_causal_verification_agrees(rows, dtype, context, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    q, k, v, mask = tensors(rows, context, dtype, strided=True)
    got = qwen4_verify_sdpa(q, k, v, mask, scale=256**-0.5)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=256**-0.5)
    assert got is not None
    assert bool(mx.allclose(got, ref, atol=0.002, rtol=0.005))


@pytest.mark.parametrize("rows", [1, 2, 5])
def test_decode_and_prefill_not_admitted(rows, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    assert qwen4_verify_sdpa(*tensors(rows), scale=256**-0.5) is None


def test_default_disabled(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_VERIFY_SDPA", raising=False)
    assert qwen4_verify_sdpa(*tensors(4), scale=256**-0.5) is None


@pytest.mark.parametrize("context,dtype", [(4096, mx.bfloat16), (8192, mx.float32)])
def test_unqualified_shapes_fall_back(context, dtype, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    assert qwen4_verify_sdpa(*tensors(4, context, dtype), scale=256**-0.5) is None


def test_unmasked_rows_preserve_attention(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    q, k, v, _ = tensors(4)
    got = qwen4_verify_sdpa(q, k, v, None, scale=256**-0.5)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=256**-0.5)
    assert bool(mx.allclose(got, ref, atol=0.002, rtol=0.005))


def test_qsa_attention_routes_and_preserves_cache(monkeypatch):
    """Exercise the model call site, including RoPE, gates and output projection."""
    import mlx.nn as nn

    from vmlx_engine.models.qwen4_exp import language

    rows = 4
    _, k, v, mask = tensors(rows, 8195, mx.bfloat16)

    class Cache:
        def __init__(self):
            self.keys = k[:, :, :-rows]
            self.values = v[:, :, :-rows]
            self.offset = self.keys.shape[2]

        def update_and_fetch(self, new_k, new_v):
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
            self.offset = self.keys.shape[2]
            return self.keys, self.values

    class Indexer(nn.Module):
        def __call__(self, *args, **kwargs):
            return mask

    attention = language.QSAAttention(language.Qwen4ExpTextArgs(hidden_size=64))
    attention.set_dtype(mx.bfloat16)
    attention.indexer = Indexer()
    x = mx.random.normal((1, rows, 64)).astype(mx.bfloat16)
    old_cache, new_cache = Cache(), Cache()
    monkeypatch.delenv("VMLX_QWEN4_VERIFY_SDPA", raising=False)
    ref = attention(x, cache=old_cache)
    real = language.qwen4_verify_sdpa
    routed = []

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        routed.append(result is not None)
        return result

    monkeypatch.setattr(language, "qwen4_verify_sdpa", spy)
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    got = attention(x, cache=new_cache)
    mx.eval(ref, got)
    assert routed == [True]
    assert bool(mx.allclose(got, ref, atol=0.002, rtol=0.005))
    assert old_cache.offset == new_cache.offset == 8195
    assert bool(mx.array_equal(old_cache.keys, new_cache.keys))
    assert bool(mx.array_equal(old_cache.values, new_cache.values))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", [3, 4])
@pytest.mark.parametrize("context", [8192, 8195, 32768])
def test_certified_sparse_rows_preserve_materialized_precision(
    dtype, rows, context, monkeypatch
):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    q, k, v, mask = tensors(rows, context, dtype, strided=True)
    got = qwen4_verify_sdpa(q, k, v, mask, scale=0.0625, selected_token_bound=2051)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=0.0625)
    mx.eval(ref, got)
    assert bool(mx.array_equal(got, ref))


@pytest.mark.parametrize("rows", [3, 4])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("context", [*range(8192, 8208), 32771, 32783, 131073])
def test_certified_block_pv_preserves_every_mma_tail(rows, dtype, context, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    q, k, v, mask = tensors(rows, context, dtype, strided=True)
    got = qwen4_verify_sdpa(
        q, k, v, mask, scale=256**-0.5,
        selected_token_bound=2051, selected_four_token_block_bound=513,
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=256**-0.5)
    mx.eval(got, ref)
    assert bool(mx.array_equal(got, ref))


@pytest.mark.parametrize("block_bound", [None, 0, -1, True, 0.5, 4096])
def test_uncertified_or_dense_block_pv_retains_exact_fallback(block_bound, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_VERIFY_SDPA", "1")
    q, k, v, mask = tensors(4, 8195)
    got = qwen4_verify_sdpa(
        q, k, v, mask, scale=256**-0.5,
        selected_token_bound=2051, selected_four_token_block_bound=block_bound,
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=256**-0.5)
    mx.eval(got, ref)
    assert bool(mx.array_equal(got, ref))
