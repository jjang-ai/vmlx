"""The dense split-K verify tile against the sorted tile it replaces: same values up to bf16/f16 rounding, on row counts and
K lengths that do not divide evenly by the tile or the slice."""
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangh import kernels as K
from vmlx_engine.jangh.dense import TQLinear

pytestmark = pytest.mark.skipif(not K.nax_available(), reason="needs NAX tensor ops")


def _bank(k, n, bits, seed=0):
    mx.random.seed(seed)
    packed = mx.random.randint(0, 2**31, (1, n, k * bits // 32)).astype(mx.uint32)
    scales = (mx.random.uniform(shape=(1, n)) * 0.02 + 0.001).astype(mx.float16)
    x = mx.random.normal((64, k))
    mx.eval(packed, scales, x)
    return packed, scales, x


# (k, n): 41 K steps split 4 gives slices 11,11,11,8; 40 divides evenly; a wide N takes fewer slices.
SHAPES = [(64 * 41, 64), (64 * 40, 64 * 3), (64 * 8, 64 * 20)]


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [4, 5, 7, 15, 16, 17, 31, 32])
def test_splitk_matches_sorted_within_rounding(k, n, bits, dtype, rows):
    packed, scales, x = _bank(k, n, bits)
    xr = x[:rows].astype(dtype)
    idx = mx.zeros((rows,), dtype=mx.uint32)
    assert K.splitk_eligible(rows, k, n, dtype)
    y = K.gather_qmm_dense(xr, packed, scales, None, idx, bits)
    ref = K.gather_qmm_sorted(xr, packed, scales, None, idx, bits)
    mx.eval(y, ref)
    assert y.shape == ref.shape == (rows, n) and y.dtype == dtype
    a, b = y.astype(mx.float32), ref.astype(mx.float32)
    assert float(mx.abs(b).max().item()) > 0
    # Each store rounds once to the output dtype: one unit in the last place of the larger value.
    ulp = 2.0 ** (-7 if dtype == mx.bfloat16 else -10)
    limit = mx.maximum(mx.abs(a), mx.abs(b)) * ulp + 1e-4 * mx.abs(b).max()
    assert bool(mx.all(mx.abs(a - b) <= limit).item())


def test_every_row_and_column_is_written_on_ragged_rows():
    k, n = 64 * 41, 64 * 3
    packed, scales, x = _bank(k, n, 4, seed=3)
    for rows in (5, 17):
        y = K.gather_qmm_dense(x[:rows].astype(mx.bfloat16), packed, scales, None, mx.zeros((rows,), dtype=mx.uint32), 4)
        one = K.gather_qmm_dense(x[rows - 5:rows].astype(mx.bfloat16), packed, scales, None, mx.zeros((5,), dtype=mx.uint32), 4)
        mx.eval(y, one)
        assert bool(mx.all(mx.any(y != 0, axis=1)).item()) and bool(mx.all(mx.any(y != 0, axis=0)).item())
        # A row's value does not depend on which other rows share the window, up to rounding.
        assert float(mx.abs(y[-1].astype(mx.float32) - one[-1].astype(mx.float32)).max().item()) < 0.02 * float(mx.abs(one[-1].astype(mx.float32)).max().item()) + 1e-3


def test_slice_count_grows_with_k_and_stops_at_about_1024_threadgroups():
    assert K.splitk_slices(40960, 5120) == 16           # Ripple down: 80 tiles x 640 steps
    assert K.splitk_slices(5120, 40960) == 2            # Ripple gate/up: 640 tiles x 80 steps
    assert K.splitk_slices(64 * 8, 64) == 1             # too few steps to slice


@pytest.mark.parametrize("rows,routed", [(3, False), (4, True), (32, True), (33, False)])
def test_tqlinear_routing_and_env_off(monkeypatch, rows, routed):
    k, n = 5120, 640
    m = TQLinear(k, n, 2)
    m.tq2_packed, m.tq2_scales, x = _bank(k, n, 2)
    calls = []
    real = K._splitk_kernel
    monkeypatch.setattr(K, "_splitk_kernel", lambda *a: calls.append(a) or real(*a))
    xx = (x[:rows] if rows <= 64 else None).astype(mx.bfloat16)[None]
    mx.eval(m(xx))
    assert bool(calls) is routed
    calls.clear()
    monkeypatch.setenv("VMLX_JANGH_SPLITK", "0")
    mx.eval(m(xx))
    assert not calls


def test_row_exact_verify_stays_on_the_per_row_kernel(monkeypatch):
    from vmlx_engine.metal.row_exact_qmv import row_exact_verify_scope

    k, n = 5120, 640
    m = TQLinear(k, n, 2)
    m.tq2_packed, m.tq2_scales, x = _bank(k, n, 2)
    m.row_exact_verify = True
    calls = []
    real = K._splitk_kernel
    monkeypatch.setattr(K, "_splitk_kernel", lambda *a: calls.append(a) or real(*a))
    with row_exact_verify_scope(rows=8, required=True):
        mx.eval(m(x[:8].astype(mx.bfloat16)[None]))
    assert not calls
