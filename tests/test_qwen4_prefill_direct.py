"""Native sparse prefill numerical parity at causal and partial-block boundaries."""

import mlx.core as mx
import pytest

from vmlx_engine.metal.qwen4_prefill_direct import (
    qsa_prefill_direct,
    qsa_prefill_direct_ready,
)


def test_missing_extension_falls_back(monkeypatch):
    from vmlx_engine.metal import qwen4_prefill_direct as direct

    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    monkeypatch.setattr(direct, "_EXT", None)
    assert not direct.qsa_prefill_direct_ready()


def test_preflight_rejects_executable_but_wrong_math(monkeypatch):
    from vmlx_engine.metal import qwen4_prefill_direct as direct

    monkeypatch.setattr(direct, "_PIPELINE_STATE", direct._PIPELINE_UNPROVEN)
    monkeypatch.setattr(direct, "_PIPELINE_PROVEN_DTYPES", frozenset())
    monkeypatch.setattr(direct, "qsa_prefill_direct", lambda q, *a, **kw: mx.ones(q.shape, dtype=q.dtype))
    assert not direct._prove_pipeline_locked(mx.bfloat16)
    assert direct._PIPELINE_STATE == direct._PIPELINE_FAILED
    assert not direct._PIPELINE_PROVEN_DTYPES


def test_dispatch_execution_does_not_publish_numeric_readiness(monkeypatch):
    from vmlx_engine.metal import qwen4_prefill_direct as direct

    monkeypatch.setattr(direct, "_PIPELINE_STATE", direct._PIPELINE_UNPROVEN)
    monkeypatch.setattr(direct, "_PIPELINE_PROVEN_DTYPES", frozenset())
    direct._prove_first_dispatch(mx.zeros((1,), mx.bfloat16), dtype=mx.bfloat16)
    assert direct._PIPELINE_STATE == direct._PIPELINE_UNPROVEN
    assert not direct._PIPELINE_PROVEN_DTYPES


def test_preflight_accepts_stock_math_without_changing_sampling_rng(monkeypatch):
    from vmlx_engine.metal import qwen4_prefill_direct as direct

    monkeypatch.setattr(direct, "_PIPELINE_STATE", direct._PIPELINE_UNPROVEN)
    monkeypatch.setattr(direct, "_PIPELINE_PROVEN_DTYPES", frozenset())

    def stock(q, k, v, ids, valid, *, pos_start, total_tokens, scale, **kw):
        positions = mx.arange(pos_start, total_tokens)
        keys = mx.arange(total_tokens)
        keep = ((keys[None] < 2048) | (keys[None] >= ((positions + 1) // 4)[:, None] * 4))
        keep &= keys[None] <= positions[:, None]
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
            mask=mx.where(keep, 0, -mx.inf).astype(q.dtype)[None, None])

    monkeypatch.setattr(direct, "qsa_prefill_direct", stock)
    previous_rng = list(mx.random.state)
    assert direct._prove_pipeline_locked(mx.bfloat16)
    actual_rng = list(mx.random.state)
    assert len(actual_rng) == len(previous_rng)
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(actual_rng, previous_rng))
    assert direct._dtype_proven(mx.bfloat16)


def test_incompatible_nanobind_disables_extension():
    from types import SimpleNamespace

    from vmlx_engine.metal.qwen4_prefill_direct import _verify_abi

    def reject(value):
        raise TypeError("incompatible function arguments")

    module, error = _verify_abi(SimpleNamespace(abi_probe=reject), None)
    assert module is None
    assert isinstance(error, TypeError)


def test_incompatible_mlx_receipt_disables_extension(monkeypatch):
    from types import SimpleNamespace

    from vmlx_engine.metal import qwen4_prefill_direct as direct

    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    monkeypatch.setattr(direct, "_EXT", SimpleNamespace(BUILT_AGAINST_MLX="0.0.0"))
    assert "built against mlx 0.0.0" in direct._build_receipt_mismatch()
    assert not direct.qsa_prefill_direct_ready()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_model_indexer_and_attention_integration(dtype, monkeypatch):
    from copy import deepcopy

    from vmlx_engine.models.qwen4_exp import language

    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    if not qsa_prefill_direct_ready():
        pytest.skip("optional native extension unavailable")
    mx.random.seed(37)
    attention = language.QSAAttention(language.Qwen4ExpTextArgs(hidden_size=64))
    attention.set_dtype(dtype)
    attention.eval()
    cache = language.QSACache()
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "0")
    for _ in range(8):
        mx.eval(attention(mx.random.normal((1, 1024, 64)).astype(dtype), cache=cache))
    old, new = deepcopy(cache), deepcopy(cache)
    x = mx.random.normal((1, 256, 64)).astype(dtype)
    ref = attention(x, cache=old)
    mx.eval(ref)
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    got = attention(x, cache=new)
    mx.eval(got)
    assert getattr(attention, "_direct_prefill_logged", False)
    assert old.offset == new.offset == 8448
    assert bool(
        mx.allclose(
            got.astype(mx.float32), ref.astype(mx.float32), atol=0.005, rtol=0.01
        )
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "rows,total",
    [
        (256, 2051),
        (256, 2052),
        (257, 4096),
        (4096, 4096),
        (256, 8195),
        (256, 32768),
        (256, 131072),
    ],
)
def test_native_matches_dense_sparse_mask(dtype, rows, total, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    if not qsa_prefill_direct_ready():
        pytest.skip("optional native extension unavailable")
    mx.random.seed(total + rows)
    offset = total - rows
    q = mx.random.normal((1, 24, rows, 256)).astype(dtype)
    # Logical views into padded allocations exercise nontrivial head strides.
    k = mx.random.normal((1, 2, total + 19, 256)).astype(dtype)[:, :, :total]
    v = mx.random.normal((1, 2, total + 19, 256)).astype(dtype)[:, :, :total]
    complete = mx.arange(offset + 1, total + 1) // 4
    scores = mx.random.uniform(shape=(rows, total // 4))
    scores = mx.where(mx.arange(total // 4)[None] < complete[:, None], scores, -mx.inf)
    ids = mx.sort(mx.argpartition(-scores, kth=511, axis=-1)[:, :512], axis=-1).astype(
        mx.int32
    )
    valid = ids < complete[:, None]
    selected = mx.put_along_axis(
        mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
    )
    keep = mx.repeat(selected, 4, axis=-1)
    if total % 4:
        keep = mx.concatenate(
            [keep, mx.zeros((rows, total % 4), dtype=mx.bool_)], axis=-1
        )
    positions = mx.arange(total)[None]
    mask = mx.where(
        (keep | (positions >= complete[:, None] * 4))
        & (positions <= mx.arange(offset, total)[:, None]),
        0,
        -mx.inf,
    ).astype(dtype)[None, None]
    if total == 2051:
        with pytest.raises(ValueError, match="dense/sparse boundary"):
            qsa_prefill_direct(
                q, k, v, ids, valid, pos_start=offset, total_tokens=total, scale=0.0625
            )
        return
    got = qsa_prefill_direct(
        q, k, v, ids, valid, pos_start=offset, total_tokens=total, scale=0.0625
    )
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=0.0625, mask=mask,
    )
    mx.eval(ref, got)
    assert bool(mx.all(mx.isfinite(got)))
    assert bool(mx.array_equal(got, ref))


def test_old_extension_without_score_api_is_disabled():
    from types import SimpleNamespace

    from vmlx_engine.metal.qwen4_prefill_direct import _verify_abi

    module, error = _verify_abi(SimpleNamespace(abi_probe=lambda a: a.size), None)
    assert module is None
    assert "materialized-score API" in str(error)
