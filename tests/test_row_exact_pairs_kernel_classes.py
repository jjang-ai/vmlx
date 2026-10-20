# SPDX-License-Identifier: Apache-2.0
"""attend_rows_in_pairs must equal each row's one-row decode step at every key count.

MLX 0.32.3's vector SDPA changes its key-block count with the key count (M5 Max: steps at 1,024 / 8,192 / 32,768 /
65,536) and gives an unmasked one-query call a GQA kernel past 8,192 keys at D 128 with GQA 8 / 12 / 16. A pair that
straddles a step, or any pair past 8,192 keys at those geometries, used to differ from decode (greedy MTP != AR there).
"""

import mlx.core as mx
import pytest

from vmlx_engine.metal.row_exact_qmv import attend_rows_in_pairs


def _sdpa(scale):
    return lambda q, k, v, m: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)


@pytest.mark.parametrize("heads, kv, dim", [(32, 2, 256), (32, 2, 128), (24, 4, 256), (16, 2, 128)])
@pytest.mark.parametrize("n", [1023, 1024, 1025, 4097, 8192, 8193, 9001, 32769, 32770])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_pair_rows_equal_decode(heads, kv, dim, n, rows):
    mx.random.seed(n * 7 + rows)
    q = mx.random.normal((1, heads, rows, dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, kv, n, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, kv, n, dim)).astype(mx.bfloat16)
    attend = _sdpa(dim ** -0.5)
    out = attend_rows_in_pairs(attend, q, k, v)
    mx.eval(out)
    for r in range(rows):
        end = n - rows + r + 1
        one = attend(q[:, :, r:r + 1], k[:, :, :end], v[:, :, :end], None)
        assert mx.array_equal(out[:, :, r:r + 1], one).item(), f"row {r} at {end} keys"
