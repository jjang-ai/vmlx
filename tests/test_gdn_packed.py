# SPDX-License-Identifier: Apache-2.0
"""The packed GDN recurrence must equal the stock mlx-lm 0.31.3 kernel BIT FOR BIT on this device.

Production shapes: Ornith-1.5-122B (Hk 16, Hv 64), a 2:1 head mix and a 1:1 mix, Dk = Dv = 128, decode (T = 1),
verify widths (2..8) and prefill lengths including non-aligned ones, with a carried-in state (not only zeros: a
zero state hides the decay path). Dispatcher coverage: ineligible shapes (mask, Dk != 128) must take the stock path.
"""

import mlx.core as mx
import pytest

from mlx_lm.models import gated_delta as gd
from vmlx_engine.metal import gdn_packed as gp

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal only")


def _inputs(T, Hk, Hv, seed, D=128):
    mx.random.seed(seed)
    q = (mx.random.normal((1, T, Hk, D)) * 0.08).astype(mx.bfloat16)
    k = (mx.random.normal((1, T, Hk, D)) * 0.08).astype(mx.bfloat16)
    v = (mx.random.normal((1, T, Hv, D)) * 0.5).astype(mx.bfloat16)
    a = (mx.random.normal((1, T, Hv)) * 0.5).astype(mx.bfloat16)
    b = (mx.random.normal((1, T, Hv)) * 0.5).astype(mx.bfloat16)
    A_log = (mx.random.uniform(low=-1.0, high=1.5, shape=(Hv,))).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((Hv,)) * 0.3).astype(mx.bfloat16)
    state = (mx.random.normal((1, Hv, D, D)) * 0.05).astype(mx.float32)
    return q, k, v, a, b, A_log, dt_bias, state


@pytest.mark.parametrize("heads", [(16, 64), (16, 32), (8, 8)])
@pytest.mark.parametrize("T", [1, 2, 3, 5, 8, 37, 129, 2048, 2051])
def test_packed_equals_stock_bitwise(heads, T):
    Hk, Hv = heads
    q, k, v, a, b, A_log, dt_bias, state = _inputs(T, Hk, Hv, seed=T * 31 + Hv)
    y0, s0 = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True)
    before = gp.STATS["packed_calls"]
    y1, s1 = gp.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True)
    assert gp.STATS["packed_calls"] == before + 1, "packed kernel did not run (vacuous comparison)"
    mx.eval(y0, s0, y1, s1)
    assert y0.dtype == y1.dtype and s0.dtype == s1.dtype
    assert mx.array_equal(y0, y1), f"y differs: max {mx.max(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32))).item()}"
    assert mx.array_equal(s0, s1), f"state differs: max {mx.max(mx.abs(s0 - s1)).item()}"


def test_chained_calls_stay_bitwise():
    # prefill chunk, then single-token decode steps, carrying the state through both paths
    q, k, v, a, b, A_log, dt_bias, state = _inputs(300, 16, 64, seed=5)
    s0 = s1 = state
    for lo, hi in ((0, 257), (257, 258), (258, 259), (259, 300)):
        y0, s0 = gd.gated_delta_update(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], a[:, lo:hi], b[:, lo:hi], A_log, dt_bias, s0)
        y1, s1 = gp.gated_delta_update(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], a[:, lo:hi], b[:, lo:hi], A_log, dt_bias, s1)
        mx.eval(y0, y1, s0, s1)
        assert mx.array_equal(y0, y1) and mx.array_equal(s0, s1)


def test_none_state_matches_stock():
    q, k, v, a, b, A_log, dt_bias, _ = _inputs(9, 16, 64, seed=11)
    y0, s0 = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, None)
    y1, s1 = gp.gated_delta_update(q, k, v, a, b, A_log, dt_bias, None)
    mx.eval(y0, s0, y1, s1)
    assert mx.array_equal(y0, y1) and mx.array_equal(s0, s1)


def test_ineligible_shapes_take_stock_path(monkeypatch):
    q, k, v, a, b, A_log, dt_bias, state = _inputs(6, 16, 64, seed=3)
    mask = mx.array([[True, True, False, True, True, True]])
    before = dict(gp.STATS)
    y1, s1 = gp.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
    y0, s0 = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
    mx.eval(y0, s0, y1, s1)
    assert gp.STATS["stock_calls"] == before["stock_calls"] + 1
    assert mx.array_equal(y0, y1) and mx.array_equal(s0, s1)
    q64, k64, v64, a64, b64, Al, dt, st64 = _inputs(4, 4, 8, seed=4, D=64)
    before = dict(gp.STATS)
    gp.gated_delta_update(q64, k64, v64, a64, b64, Al, dt, st64)
    assert gp.STATS["stock_calls"] == before["stock_calls"] + 1
    monkeypatch.setenv("VMLX_GDN_PACKED", "0")
    before = dict(gp.STATS)
    gp.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state)
    assert gp.STATS["stock_calls"] == before["stock_calls"] + 1
