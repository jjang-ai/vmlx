"""Multi-row decode MoE kernels with the expert slot on the fastest grid axis are bit-identical to the original order.

A speculative verify block (W rows x top-k) runs the decode expert kernels with W*k slots. With the slot on the
slowest grid axis, every row tile of slot 0 ran before slot 1, so an expert picked by several rows was fetched once per
row; putting the slot on grid axis y only changes the threadgroup schedule (VMLX_MOE_SLOT_FASTEST). This pins that the
outputs are bit-identical for JANGT (trellis) gate/up + down and JANGH (codebook) gate/up + weighted down, that the
switch turns the swap off, and that 1-row decode never takes the swapped kernels.
"""

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")

from vmlx_engine.jangh import kernels as K  # noqa: E402
from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import v2l as VL  # noqa: E402

E, D, I, T, TOPK = 32, 1024, 512, 8, 8


def _routing(rng):
    pool = rng.choice(E, size=12, replace=False)                 # overlapping experts across the 8 rows
    return mx.array(np.stack([rng.choice(pool, size=TOPK, replace=False) for _ in range(T)]).astype(np.int32))


def _jangt_unit(rng, kin, n, kbits=2.5, col_scales=False):
    w = VL.words_per_row(kin, 12, kbits)
    u = {"K": kbits, "kin": kin, "N": n, "n_out": 0,
         "packed": mx.array(rng.integers(0, 2 ** 32, size=(E, n, w), dtype=np.uint64).astype(np.uint32)),
         "scale": mx.array(rng.uniform(0.5, 1.5, size=(E, n)).astype(np.float32)),
         "su": mx.array(rng.choice([-1.0, 1.0], size=kin).astype(np.float32)),
         "ocol": mx.full((E, 1), -1, dtype=mx.int32), "wout": mx.zeros((E, n, 1), dtype=mx.float16)}
    if col_scales:
        u["cs"] = mx.array(rng.uniform(0.5, 1.5, size=(E, kin)).astype(np.float32))
    return u


def test_jangt_gate_up_and_down_bit_identical(monkeypatch):
    rng = np.random.default_rng(1)
    g, up, dn = _jangt_unit(rng, D, I), _jangt_unit(rng, D, I), _jangt_unit(rng, I, D, col_scales=True)
    U = {"L": 12, "gate": g, "up": up, "down": dn, "K": 2.5, "K_dn": 2.5, "su_gu": g["su"],
         "sv_dn": dn["cs"] * dn["su"][None, :]}
    idx = _routing(rng)
    x = (mx.random.normal((T, D)) * 2).astype(mx.bfloat16)
    w = mx.softmax(mx.random.normal((T, TOPK)), axis=-1)
    xr = MK.rotate_rows(x, U["su_gu"])
    outs = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("VMLX_MOE_SLOT_FASTEST", flag)
        for rot_out in (False, True):
            h = MK.gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.float32, rot_out=rot_out, slot_fastest=True)
            outs[(flag, "gu", rot_out)] = h
        h = outs[(flag, "gu", False)]
        y = MK.gather_down_weighted_v3(MK.rotate_rows(h, U["sv_dn"], idx.reshape(-1)), h, idx, w, U, slot_fastest=True)
        outs[(flag, "dn")] = y
    mx.eval(list(outs.values()))
    for key in (("gu", False), ("gu", True), ("dn",)):
        a, b = outs[("1",) + key], outs[("0",) + key]
        assert mx.array_equal(a, b).item(), f"JANGT {key} differs between grid orders"


def test_jangh_gate_up_and_weighted_down_bit_identical(monkeypatch):
    rng = np.random.default_rng(2)
    bits = 3
    cb = mx.zeros((1 << bits,), dtype=mx.float32)

    def packed(kin, n):
        return mx.array(rng.integers(0, 2 ** 32, size=(E, n, kin * bits // 32), dtype=np.uint64).astype(np.uint32))

    def scales(n):
        return mx.array(rng.uniform(0.01, 0.02, size=(E, n)).astype(np.float16))

    gp, gs, upk, ups = packed(D, I), scales(I), packed(D, I), scales(I)
    dp, ds = packed(I, D), scales(D)
    idx = _routing(rng)
    x = mx.random.normal((T, D)).astype(mx.float32)
    w = mx.softmax(mx.random.normal((T, TOPK)), axis=-1)
    outs = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("VMLX_MOE_SLOT_FASTEST", flag)
        h = K.gather_qmv(x, gp, gs, cb, idx.reshape(-1).astype(mx.uint32), bits, x_per_dispatch=False,
                         packed_u=upk, scales_u=ups, slot_fastest=True)
        y = K.gather_qmv_weighted_down(h, dp, ds, cb, idx.astype(mx.uint32), w, bits, mx.float32, slot_fastest=True)
        y1 = K.gather_qmv(h, dp, ds, cb, idx.reshape(-1).astype(mx.uint32), bits, x_per_dispatch=True, slot_fastest=True)
        outs[flag] = (h, y, y1)
    mx.eval(list(outs.values()))
    for i, name in enumerate(("gate/up", "weighted down", "per-slot down")):
        assert mx.array_equal(outs["1"][i], outs["0"][i]).item(), f"JANGH {name} differs between grid orders"


def test_one_row_decode_keeps_the_original_kernels(monkeypatch):
    """The callers request the swap only for multi-row calls; the kernel names carry the order."""
    monkeypatch.setenv("VMLX_MOE_SLOT_FASTEST", "1")
    import inspect
    import vmlx_engine.jangt.switch as JTS
    import vmlx_engine.jangh.switch as JHS
    src = inspect.getsource(JTS.JTMixedSwitchGLU._decode_h) + inspect.getsource(JTS.JTMixedSwitchGLU._decode_down)
    # _decode_h: JANGT gate/up, JANGH fused gate/up, JANGH single-weight gate + up (SwiGLU-OAI); _decode_down: 2
    assert src.count("slot_fastest=x2.shape[0] > 1") == 3 and src.count("slot_fastest=idx2.shape[0] > 1") == 2
    jh = inspect.getsource(JHS)
    assert "slot_fastest=xf.shape[0] > 1" in jh and "slot_fastest=idx2.shape[0] > 1" in jh
    assert "threadgroup_position_in_grid.z" in K.swap_grid_yz("threadgroup_position_in_grid.y")
    assert K.swap_grid_yz("threadgroup_position_in_grid.z") == "threadgroup_position_in_grid.y"
