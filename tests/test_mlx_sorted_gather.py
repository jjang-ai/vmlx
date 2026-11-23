# SPDX-License-Identifier: Apache-2.0
"""MLX shared sorted-expert boundary, real quantized Metal calls and routing."""
import mlx.core as mx
import pytest

from vmlx_engine.patches import mlx_sorted_gather as guard


@pytest.mark.parametrize("rows", [32768, 32769, 48824, 65537, 98305])
def test_balanced_boundaries_cover_once(rows):
    bounds = guard._bounds(rows)
    assert bounds[0][0] == 0 and bounds[-1][1] == rows
    assert all(0 < b-a <= 32768 for a, b in bounds)
    assert all(b == c for (_, b), (c, _) in zip(bounds, bounds[1:]))
    assert max(b-a for a,b in bounds) - min(b-a for a,b in bounds) <= len(bounds)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [2, 4, 6, 8])
@pytest.mark.parametrize("rows", [32769, 48824])
def test_actual_quantized_rows_match_aligned_control(monkeypatch, dtype, bits, rows):
    monkeypatch.setattr(guard, "_ROW_DEFECT", True)
    key = mx.random.key(41)
    weights = (mx.random.normal((8, 64, 128), key=key) * 0.1).astype(dtype)
    packed, scales, biases = mx.quantize(weights, group_size=64, bits=bits)
    x = (mx.random.normal((rows, 1, 128), key=key) * 0.1).astype(dtype)
    idx = mx.minimum(mx.arange(rows) // (rows // 8), 7).astype(mx.uint32)
    pad = (-rows) % 64
    control = guard._ORIGINAL(
        mx.concatenate([x, mx.zeros((pad, 1, 128), dtype=dtype)]),
        packed, scales, biases,
        rhs_indices=mx.concatenate([idx, mx.full((pad,), 7, dtype=idx.dtype)]),
        bits=bits, group_size=64, sorted_indices=True,
    )[:rows]
    actual = mx.gather_qmm(x, packed, scales, biases, None, idx, True, 64, bits,
                           "affine", sorted_indices=True)
    mx.eval(control, actual)
    assert bool(mx.all(mx.isfinite(actual)))
    view = mx.uint32 if dtype == mx.float32 else mx.uint16
    assert bool(mx.array_equal(control.view(view), actual.view(view)))


def test_native_row_canary_observes_installed_build(monkeypatch):
    monkeypatch.setattr(guard, "_ROW_DEFECT", None)
    result = guard._row_overflow_present()
    assert isinstance(result, bool)
    # Current M5/0.32.2 is the reproduced owner; other builds may have the fix.
    if mx.__version__ == "0.32.2" and "g17" in mx.device_info().get("architecture", ""):
        assert result is True


@pytest.mark.parametrize("change", ["short", "unsorted", "lhs", "transpose", "healthy", "cpu"])
def test_unaffected_calls_are_identical_passthrough(monkeypatch, change):
    calls = []
    sentinel = object()
    monkeypatch.setattr(guard, "_ORIGINAL", lambda *a, **kw: calls.append((a, kw)) or sentinel)
    monkeypatch.setattr(guard, "_ROW_DEFECT", change != "healthy")
    x, idx = mx.zeros((32769, 1, 64)), mx.zeros((32769,), dtype=mx.uint32)
    kwargs = dict(rhs_indices=idx, sorted_indices=True)
    if change == "short": x, kwargs["rhs_indices"] = x[:64], idx[:64]
    if change == "unsorted": kwargs["sorted_indices"] = False
    if change == "lhs": kwargs["lhs_indices"] = idx
    if change == "transpose": kwargs["transpose"] = False
    if change == "cpu": kwargs["stream"] = mx.cpu
    w, scales = mx.zeros((1,)), mx.ones((1,))
    assert guard._gather(x, w, scales, **kwargs) is sentinel
    assert len(calls) == 1 and calls[0][0][0] is x
    assert all(calls[0][1][k] is v for k,v in kwargs.items())


def test_unknown_broadcast_layout_drops_only_hint(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(guard, "_ORIGINAL", lambda *a, **kw: calls.append((a, kw)) or sentinel)
    monkeypatch.setattr(guard, "_ROW_DEFECT", True)
    x, idx = mx.zeros((2, 16385, 1, 64)), mx.zeros((2, 16385), mx.uint32)
    assert guard._gather(x, mx.zeros((1,)), mx.ones((1,)), rhs_indices=idx,
                         sorted_indices=True) is sentinel
    assert len(calls) == 1 and calls[0][0][0] is x
    assert calls[0][1]["rhs_indices"] is idx
    assert calls[0][1]["sorted_indices"] is False


def test_install_is_idempotent_and_does_not_probe(monkeypatch):
    monkeypatch.setattr(guard, "_row_overflow_present", lambda: pytest.fail("import-time GPU probe"))
    before = mx.gather_qmm
    assert guard.install() is False
    assert mx.gather_qmm is before


@pytest.mark.parametrize("rows,expected_calls", [(8, 1), (32769, 2)])
def test_nvfp4_expert_scales_survive_row_guard(monkeypatch, rows, expected_calls):
    calls = []

    def native(x, w, scales, biases, *, global_scale, rhs_indices, **kwargs):
        calls.append((global_scale, rhs_indices))
        return x

    monkeypatch.setattr(guard, "_ORIGINAL", native)
    monkeypatch.setattr(guard, "_on_gpu", lambda _: True)
    monkeypatch.setattr(guard, "_ROW_DEFECT", True)
    x = mx.zeros((rows, 1, 64))
    idx = mx.arange(rows, dtype=mx.uint32) % 2
    expert_scales = mx.array([1.25, 0.75])
    w, scales = mx.zeros((1,)), mx.ones((1,))

    actual = guard._gather(
        x, w, scales, rhs_indices=idx, mode="nvfp4",
        global_scale=expert_scales, sorted_indices=True,
    )

    assert actual.shape == x.shape
    assert len(calls) == expected_calls
    assert all(scale is expert_scales for scale, _ in calls)
    assert bool(mx.array_equal(mx.concatenate([part for _, part in calls]), idx))
