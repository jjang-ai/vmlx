"""JANGT prefill with expert-aligned tiles (JANGT_EXPERT_TILES) and the H32 store epilogue (JANGT_H32_EPILOGUE)
equals the row-tiled path: all-JANGT units, JANGT gate/up feeding a JANGH down, gate/up with outlier columns; empty
experts and row counts that are not multiples of the 64-row tile."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

if not JK.nax_available():
    pytest.skip("NAX prefill kernels need macOS 26.2+ on M5", allow_module_level=True)

D, I, E = 1024, 512, 16


def _module(dn_mode: str, n_out: int, seed: int):
    rng = np.random.default_rng(seed)
    dn = {"mode": "jangt", "kbits": 2.5} if dn_mode == "jangt" else {"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"}
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": 2}, dn, n_out, 0, 0.0)
    for p in (m.gate_proj, m.up_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
        if n_out:
            p.jt_ocol = mx.array(np.tile(np.arange(n_out, dtype=np.int32) * 7, (E, 1)))
            p.jt_wout = mx.array(rng.standard_normal(p.jt_wout.shape).astype(np.float16) * 0.05)
    m.up_proj.jt_su = m.gate_proj.jt_su
    d = m.down_proj
    if dn_mode == "jangt":
        d.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=d.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        d.jt_scale = mx.array(rng.uniform(0.01, 0.03, size=d.jt_scale.shape).astype(np.float32))
        d.jt_su = mx.array(rng.choice([-1.0, 1.0], size=d.jt_su.shape).astype(np.float32))
    else:
        d.tq2_packed = mx.array(rng.integers(0, 2 ** 32, size=d.tq2_packed.shape, dtype=np.uint64).astype(np.uint32))
        d.tq2_scales = mx.array(rng.uniform(0.01, 0.03, size=d.tq2_scales.shape).astype(np.float16))
    return m


def _inputs(T, seed, k=4):
    rng = np.random.default_rng(seed)
    x = mx.array((rng.standard_normal((T, D)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    # experts 0..9 only: 6 of 16 are empty, and per-expert row counts are ragged
    idx = mx.array(np.stack([rng.choice(10, size=k, replace=False) for _ in range(T)]).astype(np.int32))
    w = mx.array(rng.uniform(0.1, 0.4, size=(T, k)).astype(np.float32))
    return x, idx, w


def _prefill(m, x, idx, w, monkeypatch, tiles: bool, h32: bool):
    monkeypatch.setattr(SW, "EXPERT_TILES", tiles)
    monkeypatch.setattr(SW, "H32_EPILOGUE", h32)
    mx.clear_cache()                       # a fresh output buffer: an unwritten row must not inherit an earlier result
    y = m._prefill(x, idx, w)
    mx.eval(y)
    return np.array(y.astype(mx.float32))


@pytest.mark.parametrize("dn_mode,n_out", [("jangt", 0), ("jangh", 0), ("jangh", 3), ("jangt", 3)])
@pytest.mark.parametrize("T", [37, 200])
def test_expert_tiles_match_row_tiles(dn_mode, n_out, T, monkeypatch):
    m = _module(dn_mode, n_out, seed=T + n_out)
    x, idx, w = _inputs(T, seed=T)
    et = _prefill(m, x, idx, w, monkeypatch, tiles=True, h32=False)       # first: no earlier result to reuse
    ref = _prefill(m, x, idx, w, monkeypatch, tiles=False, h32=False)
    assert np.isfinite(ref).all() and np.isfinite(et).all()
    scale = np.max(np.abs(ref))
    # the same tiles accumulate in the same order: expert alignment only changes which rows a tile holds
    assert np.max(np.abs(et - ref)) <= 1e-5 * scale
    if dn_mode == "jangh":
        h32 = _prefill(m, x, idx, w, monkeypatch, tiles=True, h32=True)
        # H32 epilogue: bf16 store, fp32 transform, bf16 store, as h32_rows on the stored activations
        assert np.max(np.abs(h32 - ref)) <= 1e-5 * scale


@pytest.mark.parametrize("dn_mode", ["jangt", "jangh"])
@pytest.mark.parametrize("T", [37, 200])
def test_fused_combine_opt_in_equals_the_batched_matmul(dn_mode, T, monkeypatch):
    """MiMo-V2.6 JANGHT opts its switches into metal/moe_combine.sorted_combine; the result must match the batched
    matmul combine (fp32 class: only the k-term summation order may differ). Modules without the flag are untouched."""
    m = _module(dn_mode, 3, 11)
    x, idx, w = _inputs(T, 12)
    base = _prefill(m, x, idx, w, monkeypatch, True, True)
    object.__setattr__(m, "_fused_combine", True)
    fused = _prefill(m, x, idx, w, monkeypatch, True, True)
    scale = float(np.abs(base).max())
    assert scale > 0 and float(np.abs(fused - base).max()) <= 1e-6 * scale
