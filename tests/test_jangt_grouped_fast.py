# SPDX-License-Identifier: Apache-2.0
"""Expert-grouped fast JANGT decode (vmlx_engine/jangt/grouped_fast.py, VMLX_JANGT_GROUPED_FAST=1).

Checked:
  * bit identity: gate / up + SwiGLU of every (row, slot), the per-slot down outputs and the combined routed block
    equal fast_decode's per-slot kernels exactly, for 2..16 rows (sizes that do not divide the group size), routing
    that shares experts in every pattern (one expert in every row, drift, a small pool, all distinct), trellis K
    1 / 1.5 / 2 / 2.5 / 3, outlier columns on / off, groups of 2 / 3 / 4 members;
  * the group lists: every slot exactly once, a group holds slots of one expert in slot order;
  * the lever: routed_fast with the lever == without; the DSpark target's verify rows == decode steps with it;
  * control: a member's fold perturbed (not the first member's) is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

import tests.test_jangt_grouped_decode as TG  # noqa: E402
from vmlx_engine.jangt import fast_decode as FD  # noqa: E402
from vmlx_engine.jangt import grouped_fast as GF  # noqa: E402

E = TG.E


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _route(T, k, seed, mode):
    rng = np.random.default_rng(seed)
    if mode == "hot":                                 # one expert in every row, the other slots spread
        idx = np.stack([[0] + list(rng.choice(np.arange(1, E), k - 1, replace=False)) for _ in range(T)])
        idx = np.array([rng.permutation(r) for r in idx])
    elif mode == "drift":                             # consecutive rows keep most experts
        cur = list(rng.choice(E, k, replace=False))
        rows = []
        for _ in range(T):
            cur = [int(rng.choice([c for c in range(E) if c not in cur])) if rng.random() < 0.4 else c for c in cur]
            rows.append(list(cur))
        idx = np.array(rows)
    else:
        return TG._routing(T, k, seed, mode)[:2]
    w = rng.uniform(0.05, 0.4, size=(T, k)).astype(np.float32)
    return mx.array(idx.astype(np.int32)), mx.array(w)


def _all(m, x, idx, w, rp):
    U = m._unit()
    i2 = idx.astype(mx.uint32)
    h_ref = FD.routed_gate_up(x, i2, U, m.limit, m.act, m.alpha)
    h_grp = GF.routed_gate_up(x, i2, U, m.limit, m.act, m.alpha, rp=rp)
    packed = FD.pack_rows1(h_ref, U["sv_dn"], i2.reshape(-1))
    ysl_ref = FD.down4(packed, h_ref, i2, w, U, KSPLIT=1, combine=False, **FD.DN_CFG)
    gl = GF.group_slots(i2.reshape(-1), rp)
    ysl_grp = GF._launch("dn", packed, h_ref, gl, U, int(i2.size), int(i2.shape[1]), mx.float32, 0.0, "silu", 0.0, rp)
    y_ref = FD.routed_fast(x, i2, w, U, m.limit, m.act, m.alpha)
    y_grp = GF.routed_grouped(x, i2, w, U, m.limit, m.act, m.alpha, rp=rp)
    mx.eval(h_ref, h_grp, ysl_ref, ysl_grp, y_ref, y_grp)
    assert float(mx.abs(h_ref).max()) > 0 and float(mx.abs(y_ref).max()) > 0     # non-empty baseline
    return _eq(h_ref, h_grp), _eq(ysl_ref, ysl_grp), _eq(y_ref, y_grp)


@pytest.mark.parametrize("Kgu,Kdn", [(1, 1.5), (1.5, 2.5), (2, 2), (2.5, 1), (3, 2.5)])
@pytest.mark.parametrize("n_out", [0, 3])
def test_grouped_fast_equals_per_slot_kernels(Kgu, Kdn, n_out):
    m = TG._module(1024, 512, Kgu, Kdn, n_out, 7.0, "oai", seed=int(Kgu * 10 + Kdn))
    for T, mode in ((2, "same"), (5, "hot"), (9, "drift"), (13, "pool"), (16, "drift")):
        idx, w = _route(T, 4, seed=T, mode=mode)
        assert _all(m, TG._x(T, 1024, T), idx, w, 4) == (True, True, True), (T, mode)


@pytest.mark.parametrize("rp", [2, 3, 4])
@pytest.mark.parametrize("mode", ["same", "hot", "drift", "pool", "distinct"])
def test_grouped_fast_group_sizes_and_routing(rp, mode):
    m = TG._module(1024, 512, 1.5, 2.5, 2, 7.0, "oai", seed=5)
    for T in (3, 7, 11):
        idx, w = _route(T, 4, seed=100 + T, mode=mode)
        assert _all(m, TG._x(T, 1024, 3 * T), idx, w, rp) == (True, True, True), (T, mode, rp)


@pytest.mark.parametrize("rp", [2, 4])
def test_group_lists_cover_every_slot_once(rp):
    for T, mode in ((4, "hot"), (9, "drift"), (16, "same"), (6, "distinct")):
        idx, _ = _route(T, 4, seed=T, mode=mode)
        flat = np.array(idx).reshape(-1)
        ns = flat.size
        gl = np.array(GF.group_slots(idx.reshape(-1), rp))
        w = rp + 2
        c0, c1 = int(gl[2 * ns * w]), int(gl[2 * ns * w + 1])
        seen = []
        for li, cnt in ((0, c0), (1, c1)):
            for g in range(cnt):
                e = gl[(li * ns + g) * w]
                n = gl[(li * ns + g) * w + 1]
                slots = list(gl[(li * ns + g) * w + 2:(li * ns + g) * w + 2 + n])
                assert 1 <= n <= rp and all(flat[s] == e for s in slots) and slots == sorted(slots)
                assert (li == 1) == (np.sum(flat == e) == 1)
                seen += slots
        assert sorted(seen) == list(range(ns)), (T, mode)


def test_lever_through_routed_fast(monkeypatch):
    m = TG._module(1024, 512, 1.5, 2.5, 2, 7.0, "oai", seed=9)
    U = m._unit()
    x = TG._x(8, 1024, 1)
    idx, w = _route(8, 4, 3, "drift")
    i2 = idx.astype(mx.uint32)
    monkeypatch.setenv(GF.ENV, "0")
    ref = FD.routed_fast(x, i2, w, U, m.limit, m.act, m.alpha)
    monkeypatch.setenv(GF.ENV, "1")
    calls = GF.STATS["calls"]
    got = FD.routed_fast(x, i2, w, U, m.limit, m.act, m.alpha)
    assert GF.STATS["calls"] == calls + 1 and _eq(ref, got)
    one = FD.routed_fast(x[:1], i2[:1], w[:1], U, m.limit, m.act, m.alpha)       # decode rows keep the one-row path
    assert GF.STATS["calls"] == calls + 1 and _eq(one, got[:1])


def test_dspark_verify_with_grouped_fast_equals_decode_steps(monkeypatch):
    import tests.test_minimax_m3_fused_moe as TF
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    monkeypatch.setenv("VMLX_JANGT_FAST", "1")
    monkeypatch.setenv(GF.ENV, "1")
    for fused in ("0", "1"):
        monkeypatch.setenv("VMLX_M3_FUSED_MOE", fused)
        tgt = MiniMaxM3DSparkTarget(TF._model())
        prompt = mx.array([np.random.default_rng(1).integers(0, 511, 61).tolist()])
        for width in (5, 16):
            rows = mx.array([np.random.default_rng(width).integers(0, 511, width).tolist()])
            calls = GF.STATS["calls"]
            cache = tgt.make_cache()
            tgt.prefill(prompt, cache, (), 32)
            logits, _, _ = tgt.verify(rows, cache, ())
            mx.eval(logits)
            assert GF.STATS["calls"] > calls                      # the grouped kernels ran
            ref = tgt.make_cache()
            tgt.prefill(prompt, ref, (), 32)
            steps = [tgt.decode(rows[0, r:r + 1], ref) for r in range(width)]
            assert all(_eq(logits[r], steps[r]) for r in range(width)), (fused, width)


def test_control_perturbed_member_fold_is_caught(monkeypatch):
    """Control: a perturbation of the second member's fold makes grouped != per-slot, so the identity bites."""
    m = TG._module(1024, 512, 1.5, 2.5, 0, 7.0, "oai", seed=4)
    x = TG._x(6, 1024, 6)
    idx, w = _route(6, 4, 2, "same")
    real = GF._member_code

    def perturbed(*a, **k):
        return real(*a, **k).replace("fma(float(a_.x), x0_1,", "fma(float(a_.x), x0_1 * 1.0001f,")

    monkeypatch.setattr(GF, "_member_code", perturbed)
    GF.gen_grouped.cache_clear()
    monkeypatch.setattr(GF, "GU_NOG", 2)                        # a kernel name the cache has not seen
    try:
        got = _all(m, x, idx, w, 4)
        assert not got[0]
    finally:
        GF.gen_grouped.cache_clear()
