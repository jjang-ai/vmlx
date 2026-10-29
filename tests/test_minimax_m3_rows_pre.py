# SPDX-License-Identifier: Apache-2.0
"""VMLX_M3_ROWS_PRE: the multi-row fused-attention projections (K1 / K5) on precomputed input chunks
(kernels/attn_xprep.metal + attn_qkv_pre / attn_oproj_pre). The one-row decode build is untouched, so a row of the
multi-row build must equal the one-row build bit for bit: checked at 2..16 rows (row counts that are not multiples of
the rows per threadgroup), rows of very different magnitude side by side, 2 / 3 / 4 rows per threadgroup.
Control: a perturbed prep (the chunk sums scaled) is caught."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

from tests import test_minimax_m3_fused_attn as TA  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import kernels as K  # noqa: E402


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _rows(n, hidden, seed):
    x = np.random.default_rng(seed).standard_normal((n, hidden)).astype(np.float32) * 2.0
    if n >= 5:
        x[1] *= 300.0
        x[3] *= 1e-3
    return mx.array(x).astype(mx.bfloat16)


@pytest.fixture(scope="module")
def layer():
    return TA.Layer(hidden=1024, seed=5)


@pytest.mark.parametrize("rt", [2, 3, 4])
def test_pre_rows_equal_one_row_builds(layer, monkeypatch, rt):
    p = F.plan(layer)
    assert F._pre_ok(p)
    x = _rows(16, 1024, rt)
    o = mx.array(np.random.default_rng(rt).standard_normal((16, p.H * p.D)).astype(np.float32))
    ones = [F.qkv(layer, p, x[r:r + 1]) for r in range(16)]
    ones5 = [F.oproj(layer, p, o[r:r + 1], x[r:r + 1]) for r in range(16)]
    monkeypatch.setenv(F.PRE_ENV, "1")
    monkeypatch.setenv("VMLX_M3_ROWS_PRE_RT", str(rt))
    for R in (2, 3, 5, 7, 9, 16):
        y = F.qkv(layer, p, x[:R])
        h = F.oproj(layer, p, o[:R], x[:R])
        assert float(mx.abs(y).max()) > 0
        assert all(_eq(y[r:r + 1], ones[r]) for r in range(R)), (rt, R)
        assert all(_eq(h[r:r + 1], ones5[r]) for r in range(R)), (rt, R)


def test_control_perturbed_prep_is_caught(layer, monkeypatch):
    p = F.plan(layer)
    x = _rows(6, 1024, 9)
    ones = [F.qkv(layer, p, x[r:r + 1]) for r in range(6)]
    real = K.body.__wrapped__

    def bad(name):
        s = real(name)
        return s.replace("xs[(size_t)r * (KIN / VPT) + k / VPT + int(lane)] = s;",
                         "xs[(size_t)r * (KIN / VPT) + k / VPT + int(lane)] = s * 1.0001f;") if name == "xprep" else s

    monkeypatch.setattr(K, "body", bad)
    K._KERNELS.clear()
    monkeypatch.setenv(F.PRE_ENV, "1")
    try:
        y = F.qkv(layer, p, x)
        assert not all(_eq(y[r:r + 1], ones[r]) for r in range(6))
    finally:
        K._KERNELS.clear()
