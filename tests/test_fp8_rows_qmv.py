"""Multi-row mxfp8 QMV (vmlx_engine/metal/fp8_rows_qmv.py): every row bit-identical to stock M=1 mxfp8 calls, shared and
batched weights, 2..8 rows, bf16/fp16/fp32; inadmissible shapes return None (caller keeps mx.quantized_matmul)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.metal import fp8_rows_qmv as F  # noqa: E402


def _w(N, K, B=None, seed=0):
    shape = (N, K) if B is None else (B, N, K)
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * 0.05).astype(mx.bfloat16)
    wq, s = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    return wq, s


CONFIGS = [(4, 2, False), (2, 2, False), (2, 2, True), (1, 2, True), (2, 4, True), (1, 8, True)]


@pytest.mark.parametrize("K,N", [(1024, 512), (2304, 640), (512, 8)])
@pytest.mark.parametrize("M", [2, 3, 5, 6, 8])
@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_rows_bit_identical_to_m1(K, N, M, dt, cfg):
    """Every launch config (rows per simdgroup, simdgroups, x register dtype) leaves each row's arithmetic unchanged."""
    wq, s = _w(N, K, seed=K + N)
    x = (mx.random.normal((1, M, K), key=mx.random.key(M)) * 3.0).astype(dt)
    y = F._launch(x, wq, s, False, cfg)
    assert y.shape == (1, M, N)
    for r in range(M):
        ref = mx.quantized_matmul(x[:, r:r + 1], wq, s, None, transpose=True, group_size=32, bits=8, mode="mxfp8")
        assert np.abs(np.array(ref.astype(mx.float32))).max() > 0
        assert mx.array_equal(y[:, r:r + 1], ref).item(), r


@pytest.mark.parametrize("M", [3, 6])
def test_batched_weights(M):
    G, R, K = 4, 64, 512
    wq, s = _w(R, K, B=G, seed=3)
    x = (mx.random.normal((G, M, K), key=mx.random.key(9))).astype(mx.bfloat16)
    y = F.fp8_rows_qmv(x, wq, s)
    ref = mx.quantized_matmul(x, wq, s, None, transpose=True, group_size=32, bits=8, mode="mxfp8")
    assert y is not None and y.shape == (G, M, R)
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32)), rtol=2e-2, atol=2e-2)
    for r in range(M):
        r1 = mx.quantized_matmul(x[:, r:r + 1], wq, s, None, transpose=True, group_size=32, bits=8, mode="mxfp8")
        assert mx.array_equal(y[:, r:r + 1], r1).item()


def test_dispatch_uses_config_table():
    wq, s = _w(512, 1024)
    for M in range(1, 10):
        y = F.fp8_rows_qmv(mx.ones((1, M, 1024), mx.bfloat16), wq, s)
        assert (y is not None) == (F.CONFIG.get(M) is not None), M


def test_inadmissible_returns_none(monkeypatch):
    wq, s = _w(512, 1024)
    assert F.fp8_rows_qmv(mx.zeros((1, 1, 1024), mx.bfloat16), wq, s) is None      # M = 1: stock
    assert F.fp8_rows_qmv(mx.zeros((1, 9, 1024), mx.bfloat16), wq, s) is None      # M > 8
    wq2, s2 = _w(512, 1056)                                                           # K % 256 != 0
    assert F.fp8_rows_qmv(mx.zeros((1, 4, 1056), mx.bfloat16), wq2, s2) is None
    monkeypatch.setenv("VMLX_FP8_ROWS_QMV", "0")
    assert F.fp8_rows_qmv(mx.zeros((1, 4, 1024), mx.bfloat16), wq, s) is None
