# SPDX-License-Identifier: Apache-2.0
"""Row-exact multi-row MXFP8 / MXFP4 qmv (vmlx_engine/metal/row_exact_fp_qmv.py): every row of one multi-row launch
is bit-equal to MLX's own one-row quantized_matmul, at M3.1 trunk widths (6,144 inputs) and small shapes; control:
MLX's own multi-row call is not. Shapes outside MLX's fast single-vector traversal are declined (None)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.metal import row_exact_fp_qmv as F  # noqa: E402

SHAPES = [(512, 1024), (2048, 4096), (6144, 512), (6144, 128), (768, 256)]


def _layer(in_dim, out_dim, mode, seed):
    mx.random.seed(seed)
    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.set_dtype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(lin, group_size=32, bits=F.MODES[mode][0], mode=mode)


def _rows(rows, in_dim, seed):
    k1, k2 = mx.random.split(mx.random.key(seed))
    return (mx.random.normal((1, rows, in_dim), key=k1) * 0.5
            + mx.random.uniform(-3, 3, (1, rows, 1), key=k2)).astype(mx.bfloat16)


@pytest.mark.parametrize("mode", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("in_dim, out_dim", SHAPES)
def test_every_row_equals_the_one_row_call(mode, in_dim, out_dim):
    q = _layer(in_dim, out_dim, mode, seed=in_dim + out_dim)
    if in_dim % ((32 // F.MODES[mode][0]) * 64):
        assert F.row_exact_fp_qmv(_rows(2, in_dim, 0), q["weight"], q["scales"], group_size=32,
                                  bits=F.MODES[mode][0], mode=mode) is None
        return
    for rows in (2, 3, 4):
        for trial in range(4):
            x = _rows(rows, in_dim, 100 * rows + trial)
            y = F.row_exact_fp_qmv(x, q["weight"], q["scales"], group_size=32, bits=F.MODES[mode][0], mode=mode)
            assert y is not None and y.shape == (1, rows, out_dim)
            ref = mx.concatenate([q(x[:, r:r + 1]) for r in range(rows)], axis=1)
            assert float(mx.abs(ref).max()) > 0
            assert bool(mx.array_equal(y, ref).item()), (rows, trial)


def test_control_stock_multi_row_mxfp8_is_not_row_exact():
    q = _layer(2048, 4096, "mxfp8", seed=1)
    differs = 0
    for trial in range(6):
        x = _rows(4, 2048, trial)
        ref = mx.concatenate([q(x[:, r:r + 1]) for r in range(4)], axis=1)
        differs += not bool(mx.array_equal(q(x), ref).item())
    assert differs >= 3


def test_declines_unsupported_inputs():
    q = _layer(512, 1024, "mxfp8", seed=2)
    x = _rows(5, 512, 0)
    assert F.row_exact_fp_qmv(x, q["weight"], q["scales"], group_size=32, bits=8, mode="mxfp8") is None   # > 4 rows
    assert F.row_exact_fp_qmv(x[:, :1], q["weight"], q["scales"], group_size=32, bits=8, mode="mxfp8") is None
    assert F.row_exact_fp_qmv(x[:, :2], q["weight"], q["scales"], group_size=16, bits=4, mode="nvfp4") is None
