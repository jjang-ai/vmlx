"""Native numerical regressions for large sorted routed-token/output extents."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from vmlx_engine.jangtq2 import kernels


@pytest.fixture(autouse=True)
def require_native_nax():
    if not mx.metal.is_available() or not kernels.nax_available():
        pytest.skip("Requires native NAX support")
    yield
    mx.clear_cache()


def _constant_projection(rows, columns, bits, *, fused=False):
    # Every packed bit set selects the highest positive code, including codes
    # crossing word boundaries at3 bits. Exact dyadic inputs/scales avoid an
    # arbitrary numerical tolerance; the safe-size kernel is the oracle.
    width = 64
    x = mx.full((rows, width), 0.125, dtype=mx.float16)
    packed = mx.full((1, columns, width * bits // 32), 0xFFFFFFFF, dtype=mx.uint32)
    scales = mx.full((1, columns), 0.03125, dtype=mx.float16)
    indices = mx.zeros((rows,), dtype=mx.uint32)
    options = {}
    if fused:
        options = dict(packed_u=packed, scales_u=scales * 2, limit=10.0)
    return kernels.gather_qmm_sorted(
        x, packed, scales, None, indices, bits, **options
    )


def _assert_matches_safe_extent(rows, columns, bits, *, fused=False):
    reference = _constant_projection(8, 64, bits, fused=fused)
    actual = _constant_projection(rows, columns, bits, fused=fused)
    mx.eval(reference, actual)
    expected = np.asarray(reference)
    result = np.asarray(actual)
    assert np.isfinite(expected).all() and (expected > 0).all()
    # Constant expert rows and inputs produce the same positive value for
    # every position. Check all elements, including early tiles and tails.
    np.testing.assert_array_equal(expected, np.full(expected.shape, expected[0, 0]))
    np.testing.assert_array_equal(result, np.full(result.shape, expected[0, 0]))


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("rows,columns", [(32768, 64), (38016, 64), (8, 32768)])
def test_nax_large_extent_matches_safe_projection(rows, columns, bits):
    _assert_matches_safe_extent(rows, columns, bits)


@pytest.mark.parametrize("bits", [2, 3])
def test_nax_large_routed_extent_fused_gate_up(bits):
    _assert_matches_safe_extent(38016, 64, bits, fused=True)
