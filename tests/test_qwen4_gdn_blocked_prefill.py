"""Dispatch invariants for experimental Qwen4 blocked prefill."""

import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_update as stock
from vmlx_engine.metal.qwen4_gdn_blocked_prefill import (
    _normalize_block_t,
    qwen4_blocked_gated_delta_update as candidate,
)


def data(rows, key_dim=128):
    mx.random.seed(4)
    q = (mx.random.normal((1, rows, 2, key_dim)) * 0.03).astype(mx.float32)
    k = q * 0.8
    v = mx.random.normal((1, rows, 4, 32))
    a = mx.zeros((1, rows, 4))
    return q, k, v, a, a, mx.zeros((4,)), mx.zeros((4,))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 15])
def test_decode_and_verify_remain_exact(rows, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_GDN_BLOCKED_PREFILL", "1")
    args = data(rows)
    assert all(
        bool(mx.array_equal(a, b)) for a, b in zip(candidate(*args), stock(*args))
    )


def test_disabled_default_is_exact(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_GDN_BLOCKED_PREFILL", raising=False)
    args = data(32)
    assert all(
        bool(mx.array_equal(a, b)) for a, b in zip(candidate(*args), stock(*args))
    )


@pytest.mark.parametrize("key_dim,masked", [(64, False), (128, True)])
def test_unsupported_inputs_fall_back(key_dim, masked, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_GDN_BLOCKED_PREFILL", "1")
    args = data(32, key_dim)
    mask = mx.array([[True] * 16 + [False] * 16]) if masked else None
    assert all(
        bool(mx.array_equal(a, b))
        for a, b in zip(candidate(*args, mask=mask), stock(*args, mask=mask))
    )


def test_float32_threadgroup_memory_guard():
    assert _normalize_block_t(None, mx.float32) == 16
    with pytest.raises(ValueError, match="threadgroup memory"):
        _normalize_block_t(32, mx.float32)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_partial_time_blocks_batch_and_resumed_state(dtype, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_GDN_BLOCKED_PREFILL", "1")
    args = list(data(65))
    args[:5] = [mx.concatenate([x, x * 0.7], axis=0).astype(dtype) for x in args[:5]]
    ref = stock(*args)
    got = candidate(*args)
    for a, b in zip(got, ref):
        assert bool(mx.allclose(a, b, atol=0.002, rtol=0.002))
    first = candidate(*(x[:, :33] for x in args[:5]), *args[5:])
    second = candidate(*(x[:, 33:] for x in args[:5]), *args[5:], state=first[1])
    resumed = (mx.concatenate([first[0], second[0]], axis=1), second[1])
    for a, b in zip(got, resumed):
        assert bool(mx.array_equal(a, b))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [17, 257])
@pytest.mark.parametrize("value_dim", [32, 128])
def test_stock_reduction_order_is_exact_with_nonzero_state(
    dtype, rows, value_dim, monkeypatch
):
    monkeypatch.setenv("VMLX_QWEN4_GDN_BLOCKED_PREFILL", "1")
    mx.random.seed(rows + value_dim)
    q, k = [mx.random.normal((2, rows, 2, 128)) for _ in range(2)]
    q = (q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))).astype(dtype)
    k = (k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True))).astype(dtype)
    v = mx.random.normal((2, rows, 4, value_dim)).astype(dtype)
    a, b = [mx.random.normal((2, rows, 4)).astype(dtype) for _ in range(2)]
    A_log, dt_bias = [mx.random.normal((4,)) for _ in range(2)]
    state = mx.random.normal((2, 4, value_dim, 128)) * 0.2
    args = (q, k, v, a, b, A_log, dt_bias)
    reference = stock(*args, state=state)
    actual = candidate(*args, state=state)
    mx.eval(reference, actual)
    for expected, got in zip(reference, actual):
        assert bool(mx.array_equal(expected, got))
