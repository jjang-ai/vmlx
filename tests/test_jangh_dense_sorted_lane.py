# SPDX-License-Identifier: Apache-2.0
"""JANGH dense sorted lane: every row count runs the sorted tile QMM, and row r of an M-row call is bit-equal to the
1-row call (what lets a DFlash2 target verify wide windows while staying equal to its own AR steps)."""
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangh import kernels as K  # noqa: E402
from vmlx_engine.jangh.dense import TQLinear  # noqa: E402


@pytest.mark.skipif(not K.nax_available(), reason="sorted tile QMM needs the matrix-unit kernels")
@pytest.mark.parametrize("k,n", [(512, 1024), (1024, 256)])
def test_sorted_lane_rows_equal_their_one_row_call(k, n):
    mx.random.seed(3)
    m = TQLinear(k, n, 2)
    m.tq2_packed = mx.random.randint(0, 2**31, (1, n, k * 2 // 32)).astype(mx.uint32)
    m.tq2_scales = (mx.random.uniform(shape=(1, n)) * 0.02).astype(mx.float16)
    m.sorted_lane = True
    x = mx.random.normal((1, 16, k)).astype(mx.bfloat16)
    one = [m(x[:, i:i + 1]) for i in range(16)]
    for rows in (2, 3, 5, 8, 16):
        y = m(x[:, :rows])
        for i in range(rows):
            assert mx.array_equal(y[:, i:i + 1], one[i]).item(), (rows, i)
    # and it is the same projection as the default kernels, to rounding
    m.sorted_lane = False
    ref = m(x)
    m.sorted_lane = True
    assert mx.allclose(m(x).astype(mx.float32), ref.astype(mx.float32), atol=0.05, rtol=0.05).item()
