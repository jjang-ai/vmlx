"""Component contract only; these tests do not establish model throughput."""
import mlx.core as mx
import pytest

from vmlx_engine.metal.qwen4_qsa_select import qwen4_qsa_select


@pytest.mark.parametrize("n", [513, 4097, 16384, 65536])
@pytest.mark.parametrize("kind", ["random", "ties", "zero", "negative", "nonfinite"])
def test_chronological_selection_matches_stock(n, kind):
    mx.random.seed(76)
    x = mx.random.normal((1, 1, n))
    if kind == "ties":
        x = mx.random.randint(-8, 8, (1, 1, n)).astype(mx.float32)
    elif kind == "zero":
        x = mx.zeros((1, 1, n))
    elif kind == "negative":
        x = mx.full((1, 1, n), -1e30)
    elif kind == "nonfinite":
        i = mx.arange(n)
        x = mx.where(i % 3 == 0, float("nan"),
                     mx.where(i % 3 == 1, float("-inf"), float("inf"))).reshape(1, 1, n)
    expected = mx.sort(mx.argpartition(-x, kth=511)[..., :512])
    actual = qwen4_qsa_select(x, enabled=True)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("k", [1, 7, 511, 512])
def test_strided_causal_scores_and_signed_zero(k):
    x = mx.arange(2048).astype(mx.float32).reshape(1, 1, -1)[..., ::2]
    x = mx.where(mx.arange(1024) >= 700, -1e30, x)
    x = mx.where(mx.arange(1024) < 20, -0.0, x)
    expected = mx.sort(mx.argpartition(-x, kth=k - 1)[..., :k])
    assert mx.array_equal(qwen4_qsa_select(x, k=k, enabled=True), expected).item()


def test_disabled_and_unsupported_paths_remain_stock():
    x = mx.zeros((1, 1, 513))
    assert qwen4_qsa_select(x) is None
    for invalid in [x.astype(mx.float16), mx.zeros((2, 1, 513)), mx.zeros((1, 2, 513)), mx.zeros((513,))]:
        assert qwen4_qsa_select(invalid, enabled=True) is None
    for k in [0, 513, True, 1.5]:
        assert qwen4_qsa_select(x, k=k, enabled=True) is None
