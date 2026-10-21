"""Dense JANGTQ2 multi-row decode (jangh.kernels.gather_qmv_rows): every row equals the per-row gather QMV bit for bit
at every width the kernel takes, with and without rotation, at non-aligned shapes and row counts; TQLinear routes
1..ROWS_THRESHOLD-1 rows (and row-exact scopes) through it and keeps the sorted path above."""
import mlx.core as mx
import pytest

from vmlx_engine.jangh import dense as D
from vmlx_engine.jangh import kernels as K


def _weight(N, Kd, bits, seed):
    mx.random.seed(seed)
    packed = mx.random.randint(0, 2 ** 31 - 1, (1, N, Kd * bits // 32), dtype=mx.int32).astype(mx.uint32)
    scales = (mx.random.uniform(0.5, 1.5, (1, N)) * 0.01).astype(mx.float16)
    mx.eval(packed, scales)
    return packed, scales


def _per_row(x, packed, scales, bits, rotate):
    idx = mx.zeros((x.shape[0],), dtype=mx.uint32)
    return K.gather_qmv(x, packed, scales, None, idx, bits, x_per_dispatch=True, rotate=rotate)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("shape", [(96, 64), (104, 96), (160, 544), (24, 1024)])   # 104: guarded rows; 544: K tail
@pytest.mark.parametrize("rotate", [False, True])
def test_rows_kernel_equals_per_row_kernel_bitwise(bits, shape, rotate):
    N, Kd = shape
    packed, scales = _weight(N, Kd, bits, 7 + bits)
    for M in (1, 2, 3, 5, 7, 8, 11, 16):
        for dt in (mx.bfloat16, mx.float16):
            mx.random.seed(M)
            x = mx.random.normal((M, Kd)).astype(dt)
            ref = _per_row(x, packed, scales, bits, rotate)
            idx = mx.zeros((1,), dtype=mx.uint32)
            if M == 1:
                got = K.gather_qmv_rows(x, packed, scales, idx, bits, rotate=rotate)
            else:
                xin = K.h32_rows(x, mx.float32) if rotate else x
                got = K.gather_qmv_rows(xin, packed, scales, idx, bits)
            mx.eval(ref, got)
            assert got.shape == (M, N) and got.dtype == mx.float32
            assert mx.array_equal(ref, got).item(), (bits, shape, rotate, M, dt)


def test_rows_kernel_every_register_geometry_is_exact():
    """The geometry knobs (rows per simdgroup, activation chunk, launch split) never change a row's bits."""
    packed, scales = _weight(104, 544, 2, 3)
    x = mx.random.normal((11, 544)).astype(mx.bfloat16)
    ref = _per_row(x, packed, scales, 2, True)
    xr = K.h32_rows(x, mx.float32)
    idx = mx.zeros((1,), dtype=mx.uint32)
    for rpsm, chunk, per_launch in ((1, 4, 8), (2, 8, 3), (4, 16, 11), (8, 8, 5), (4, 4, 1)):
        got = K.gather_qmv_rows(xr, packed, scales, idx, 2, rows_per_simdgroup=rpsm, chunk=chunk,
                                rows_per_launch=per_launch)
        assert mx.array_equal(ref, got).item(), (rpsm, chunk, per_launch)


def test_h32_rows_fp32_matches_in_kernel_rotation_and_is_reused_per_activation():
    packed, scales = _weight(96, 64, 2, 5)
    x = mx.random.normal((3, 64)).astype(mx.bfloat16)
    ref = _per_row(x, packed, scales, 2, True)
    a = K.h32_rows_cached(x)
    assert K.h32_rows_cached(x) is a                       # same array object -> reused
    got = K.gather_qmv_rows(a, packed, scales, mx.zeros((1,), dtype=mx.uint32), 2)
    assert mx.array_equal(ref, got).item()
    y = mx.random.normal((3, 64)).astype(mx.bfloat16)
    assert K.h32_rows_cached(y) is not a                   # a different activation is rotated afresh
    x3 = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)  # a 3-D module input is keyed as is and returned 2-D
    r3 = K.h32_rows_cached(x3)
    assert r3.shape == (3, 64) and K.h32_rows_cached(x3) is r3


def test_gate_and_up_share_one_rotation_per_layer_input(monkeypatch):
    """Two TQLinear modules called with the SAME input object rotate it once (the old key was a per-call reshape)."""
    calls = []
    real = K.h32_rows

    def spy(x, out_dtype=None):
        calls.append(x.shape); return real(x, out_dtype)

    monkeypatch.setattr(K, "h32_rows", spy)
    monkeypatch.setattr(D, "_ROWS_ENV", True)
    gate, up = D.TQLinear(64, 96, 2), D.TQLinear(64, 96, 2)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    mx.eval(gate(x), up(x))
    assert len(calls) == 1
    D._ROWS_ENV = None


def test_rows_kernel_refuses_unsupported_formats():
    assert not K.rows_kernel_supports(3) and not K.rows_kernel_supports(6)
    with pytest.raises(ValueError):
        K._rows_kernel(2, 2, 4, 8, "bfloat16_t", False, True)   # rotation needs the whole 16-value chunk


class _Spy:
    def __init__(self):
        self.calls = []

    def rows(self, x, packed, scales, idx, bits, **kw):
        self.calls.append(("rows", int(x.shape[0]), bool(kw.get("rotate", False))))
        return mx.zeros((int(x.shape[0]), int(packed.shape[1])), dtype=mx.float32)

    def per_row(self, x, packed, scales, cb, idx, bits, **kw):
        self.calls.append(("per_row", int(x.shape[0]), False))
        return mx.zeros((int(x.shape[0]), int(packed.shape[1])), dtype=mx.float32)

    def sorted(self, x, packed, scales, cb, idx, bits, **kw):
        self.calls.append(("sorted", int(x.shape[0]), False))
        return mx.zeros((int(x.shape[0]), int(packed.shape[1])), dtype=mx.float32)


def test_tqlinear_routes_narrow_calls_to_rows_kernel_and_wide_calls_to_sorted(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(K, "gather_qmv_rows", spy.rows)
    monkeypatch.setattr(K, "gather_qmv", spy.per_row)
    monkeypatch.setattr(K, "gather_qmm_sorted", spy.sorted)
    monkeypatch.setattr(D, "_ROWS_ENV", True)
    lin = D.TQLinear(64, 96, 2)
    for T in (1, 2, 5, 6, 16):
        lin(mx.zeros((1, T, 64), dtype=mx.bfloat16))
    assert spy.calls == [("rows", 1, True), ("rows", 2, False), ("rows", 5, False), ("sorted", 6, False),
                         ("sorted", 16, False)]
    # a 3-bit dense module keeps the old per-row / sorted routing (no rows kernel for its format)
    spy.calls.clear()
    lin3 = D.TQLinear(64, 96, 3)
    for T in (1, 2, 4):
        lin3(mx.zeros((1, T, 64), dtype=mx.bfloat16))
    assert spy.calls == [("per_row", 1, False), ("per_row", 2, False), ("sorted", 4, False)]
    # the A/B control restores the old routing
    monkeypatch.setattr(D, "_ROWS_ENV", False)
    spy.calls.clear()
    for T in (1, 3, 4):
        lin(mx.zeros((1, T, 64), dtype=mx.bfloat16))
    assert spy.calls == [("per_row", 1, False), ("per_row", 3, False), ("sorted", 4, False)]


def test_tqlinear_output_equals_the_per_row_kernel_for_every_narrow_width():
    """Live module: for 1..5 rows the new routing's bf16 output equals the per-row (1-row-exact) path; the old routing
    matched it only below SORT_THRESHOLD (4), where 4-5 rows took the sorted tile path with its own reduction order."""
    lin = D.TQLinear(64, 96, 2)
    packed, scales = _weight(96, 64, 2, 9)
    lin.tq2_packed, lin.tq2_scales = packed, scales
    for T in (1, 2, 3, 4, 5):
        x = mx.random.normal((1, T, 64)).astype(mx.bfloat16)
        D._ROWS_ENV = True
        try:
            new = lin(x)
        finally:
            D._ROWS_ENV = None
        ref = _per_row(x.reshape(T, 64), packed, scales, 2, True).astype(mx.bfloat16).reshape(1, T, 96)
        mx.eval(new, ref)
        assert mx.array_equal(new, ref).item(), T


def test_rows_kernel_partial_output_simdgroup_is_bitwise_exact():
    """N=13 exercises a partially filled output group, not only whole empty groups."""
    for bits in (2, 4, 8):
        packed, scales = _weight(13, 96, bits, 40 + bits)
        for rows in (1, 2, 5, 8, 11):
            for dtype in (mx.bfloat16, mx.float16):
                x = mx.random.normal((rows, 96)).astype(dtype)
                for rotate in (False, True):
                    ref = _per_row(x, packed, scales, bits, rotate)
                    xin = K.h32_rows(x, mx.float32) if rotate and rows > 1 else x
                    got = K.gather_qmv_rows(
                        xin, packed, scales, mx.zeros((1,), dtype=mx.uint32), bits,
                        rotate=rotate and rows == 1,
                    )
                    mx.eval(ref, got)
                    assert mx.array_equal(ref, got).item(), (bits, rows, dtype, rotate)
