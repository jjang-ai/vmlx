"""Expert-aligned NAX prefill tiles at the Ornith-1.5-122B geometry (D=3072, I=1024, E=256, k=8, H32 experts).

The expert-tile route runs the same tq_gather_qmm_nax arithmetic as the generic sorted route; only the tile -> row
plan differs. Pinned here: admission of exactly this geometry (with the per-module opt-in the installer sets), and
BIT equality of gate/up (fused SwiGLU) and down outputs between the two routes, on non-aligned expert segments
(empty experts, 1-row and 63/64/65-row segments) at the real widths.
"""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from vmlx_engine.jangh import kernels, switch


def _geometry(gu=2, dn=3, experts=256):
    def projection(i, o, bits):
        return SimpleNamespace(input_dims=i, output_dims=o, num_experts=experts, bits=bits, rotated=True)
    return SimpleNamespace(gate_proj=projection(3072, 1024, gu), up_proj=projection(3072, 1024, gu),
                           down_proj=projection(1024, 3072, dn), use_expert_tiles=True)


def test_admission_d3072(monkeypatch):
    monkeypatch.setattr(kernels, "nax_available", lambda: True)
    monkeypatch.setattr(switch, "EXPERT_TILES", "0")
    x = SimpleNamespace(dtype=mx.bfloat16, shape=(4096, 3072))
    for gu, dn in ((2, 2), (2, 3), (3, 2), (3, 3)):
        assert switch.TQSwitchGLU._use_expert_tiles(_geometry(gu, dn), x, 8)
    m = _geometry()
    m.use_expert_tiles = False
    assert not switch.TQSwitchGLU._use_expert_tiles(m, x, 8)            # opt-in required
    assert not switch.TQSwitchGLU._use_expert_tiles(_geometry(dn=4), x, 8)
    assert not switch.TQSwitchGLU._use_expert_tiles(_geometry(experts=288), x, 8)
    assert not switch.TQSwitchGLU._use_expert_tiles(_geometry(), x, 6)
    assert not switch.TQSwitchGLU._use_expert_tiles(_geometry(), SimpleNamespace(dtype=mx.float16, shape=(8, 3072)), 8)
    m = _geometry()
    m.down_proj.rotated = False
    assert not switch.TQSwitchGLU._use_expert_tiles(m, x, 8)
    monkeypatch.setattr(kernels, "nax_available", lambda: False)
    assert not switch.TQSwitchGLU._use_expert_tiles(_geometry(), x, 8)


@pytest.mark.parametrize("gu,dn", [(2, 2), (2, 3), (3, 3)])
def test_d3072_tiles_bitwise_equal_generic(gu, dn):
    if not mx.metal.is_available() or not kernels.nax_available():
        pytest.skip("Native NAX required")
    E, D, I = 256, 3072, 1024
    rng = np.random.default_rng(3072 + gu * 10 + dn)
    counts = np.zeros(E, dtype=np.int64)
    for e, c in zip([0, 3, 40, 41, 77, 128, 200, 254, 255], [1, 63, 64, 65, 7, 129, 2, 31, 96]):
        counts[e] = c                                    # empty experts in between, ragged segments
    ids = np.repeat(np.arange(E), counts).astype(np.uint32)
    idx = mx.array(ids)
    x = mx.array(rng.normal(0, 0.25, (len(ids), D)).astype(np.float32)).astype(mx.bfloat16)
    gp = mx.array(rng.integers(0, 2**32, (E, I, D * gu // 32), dtype=np.uint32))
    up = mx.array(rng.integers(0, 2**32, (E, I, D * gu // 32), dtype=np.uint32))
    dp = mx.array(rng.integers(0, 2**32, (E, D, I * dn // 32), dtype=np.uint32))
    gs = mx.array(rng.uniform(0.002, 0.02, (E, I)).astype(np.float16))
    us = mx.array(rng.uniform(0.002, 0.02, (E, I)).astype(np.float16))
    ds = mx.array(rng.uniform(0.002, 0.02, (E, D)).astype(np.float16))
    plan = kernels.expert_tile_plan(idx, E)
    h0 = kernels.gather_qmm_sorted(x, gp, gs, None, idx, gu, packed_u=up, scales_u=us, limit=0.0)
    h1 = kernels.gather_qmm_expert_sorted(x, gp, gs, idx, gu, plan, packed_u=up, scales_u=us, limit=0.0)
    y0 = kernels.gather_qmm_sorted(h0, dp, ds, None, idx, dn)
    y1 = kernels.gather_qmm_expert_sorted(h0, dp, ds, idx, dn, plan)
    mx.eval(h0, h1, y0, y1)
    assert np.isfinite(np.asarray(h1.astype(mx.float32))).all()
    np.testing.assert_array_equal(np.asarray(h1.astype(mx.float32)), np.asarray(h0.astype(mx.float32)))
    np.testing.assert_array_equal(np.asarray(y1.astype(mx.float32)), np.asarray(y0.astype(mx.float32)))
    mx.clear_cache()
