# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify kernels added for the MiniMax-M3.1 DSpark verify:

* metal/row_exact_gemv.row_exact_gemv: the fp32 MoE router (x bf16 @ W.T, W float32) for 2..16 rows in one launch,
  every row bit-equal to its one-row call (MLX's one-row gemv arithmetic); declined outside MLX's K >= 16 N
  configuration. Control: MLX's own multi-row matmul is not row-exact here.
* metal/row_exact_qmv.row_exact_qmv_wide: affine qmv for 2..16 rows in one weight walk (off by default, measured
  slower), every row bit-equal to its one-row call, at M3.1 trunk shapes and 4 / 6 / 8 bits.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.metal import row_exact_gemv as G  # noqa: E402
from vmlx_engine.metal import row_exact_qmv as R  # noqa: E402


def _rows(rows, K, seed, dtype=mx.bfloat16):
    k1, k2 = mx.random.split(mx.random.key(seed))
    return (mx.random.normal((1, rows, K), key=k1) * 0.7 + mx.random.uniform(-2, 2, (1, rows, 1), key=k2)).astype(dtype)


@pytest.mark.parametrize("K,N", [(6144, 128), (2048, 64), (4096, 128)])
def test_router_gemv_rows_equal_one_row_calls(K, N):
    lin = nn.Linear(K, N, bias=False)
    lin.weight = mx.random.normal((N, K), key=mx.random.key(K)) * 0.02
    for rows in (2, 3, 8, 13, 16):
        for trial in range(3):
            x = _rows(rows, K, 10 * rows + trial)
            y = G.row_exact_gemv(x, lin.weight)
            assert y is not None and y.dtype == mx.float32
            ref = mx.concatenate([lin(x[:, r:r + 1]) for r in range(rows)], axis=1)
            assert bool(mx.array_equal(y, ref).item()), (rows, trial)


def test_router_gemv_declines_and_control():
    lin = nn.Linear(1024, 128, bias=False)                       # K < 16 N: another MLX gemv configuration
    assert G.row_exact_gemv(_rows(4, 1024, 0), lin.weight) is None
    lin = nn.Linear(6144, 128, bias=False)
    lin.weight = mx.random.normal((128, 6144), key=mx.random.key(1)) * 0.02
    assert G.row_exact_gemv(_rows(17, 6144, 0), lin.weight) is None
    differs = 0
    for trial in range(6):
        x = _rows(8, 6144, trial)
        ref = mx.concatenate([lin(x[:, r:r + 1]) for r in range(8)], axis=1)
        differs += not bool(mx.array_equal(lin(x), ref).item())
    assert differs >= 3                                          # MLX's multi-row matmul is not row-exact


@pytest.mark.parametrize("K,N,bits", [(6144, 8192, 4), (8192, 6144, 4), (3072, 6144, 4), (6144, 4096, 6),
                                      (6144, 1024, 8)])
def test_wide_qmv_rows_equal_one_row_calls(K, N, bits):
    mx.random.seed(K + N + bits)
    lin = nn.Linear(K, N, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    for rows in (2, 7, 16):
        x = _rows(rows, K, rows + bits)
        y = R.row_exact_qmv_wide(x, q["weight"], q["scales"], q["biases"], group_size=64, bits=bits)
        assert y is not None
        ref = mx.concatenate([q(x[:, r:r + 1]) for r in range(rows)], axis=1)
        assert bool(mx.array_equal(y, ref).item()), rows
