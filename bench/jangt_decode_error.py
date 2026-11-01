# SPDX-License-Identifier: Apache-2.0
"""Error of the JANGT decode kernels (today's v3 vs fast_decode v4) against a float64 dequantize-and-matmul reference,
at MiniMax-M3.1 widths (hidden 6144, intermediate 3072, L 12) for every trellis width the final plan uses.

Reference: the JANGT units dequantized in float64 into the raw input basis (trellis codes -> code statistics -> row
scales -> Hadamard-128 -> input signs / column scales, outlier columns added), then g = Wg x, u = Wu x,
h = SwiGLU-OAI(g, u) (alpha 1.702, limit 7) and y = sum_slot w_slot Wd[e_slot] h_slot, all float64. The down error is
measured on the same h for both kernels (v3's), so it is the down kernel's own error.

  python bench/jangt_decode_error.py [--rows 1,16]
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

from vmlx_engine.jangt import fast_decode as FD
from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL

D, I, E, TOPK, L = 6144, 3072, 4, 4, 12


def _h128():
    H = np.ones((1, 1))
    while H.shape[0] < 128:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(128)


H128 = _h128()


def unit(n_in, n_out, K, n_cols, col_scales, rng, su=None, cols=None):
    W = VL.words_per_row(n_in, L, K)
    packed = rng.integers(0, 2 ** 32, size=(E, n_out, W), dtype=np.uint64).astype(np.uint32)
    sc = rng.uniform(0.02, 0.05, size=(E, n_out)).astype(np.float32)
    su = rng.choice([-1.0, 1.0], size=n_in).astype(np.float32) if su is None else su
    if cols is None:
        if not n_cols:
            cols = np.full((E, 1), -1, np.int32)
        elif col_scales:                                  # down: per-expert outlier columns
            cols = np.stack([rng.choice(n_in, n_cols, replace=False) for _ in range(E)]).astype(np.int32)
        else:                                             # gate / up: layer-wide columns (the bundle contract)
            cols = np.tile(rng.choice(n_in, n_cols, replace=False), (E, 1)).astype(np.int32)
    nc = cols.shape[1]
    wo = (rng.standard_normal((E, n_out, nc)) * 0.05 * (cols >= 0)[:, None, :]).astype(np.float16)
    cs = rng.uniform(0.5, 1.5, size=(E, n_in)).astype(np.float32) if col_scales else None
    raw = np.zeros((E, n_out, n_in))
    for e in range(E):
        st = VL.unpack_states(packed[e], n_in, L, K)
        w_rot = np.array(VL.values_of(mx.array(st), 1.0, L, K)).astype(np.float64) * sc[e][:, None].astype(np.float64)
        w_h = (w_rot.reshape(n_out, n_in // 128, 128) @ H128).reshape(n_out, n_in)
        keep = np.ones(n_in)
        keep[cols[e][cols[e] >= 0]] = 0.0
        s = su.astype(np.float64) * keep * (cs[e].astype(np.float64) if cs is not None else 1.0)
        raw[e] = w_h * s[None, :]
        for j, c in enumerate(cols[e]):
            if c >= 0:
                raw[e][:, c] += wo[e, :, j].astype(np.float64)
    P = {"K": K, "kin": n_in, "packed": mx.array(packed), "scale": mx.array(sc), "su": mx.array(su),
         "ocol": mx.array(cols), "wout": mx.array(wo), "n_out": n_cols, "cs": None if cs is None else mx.array(cs),
         "N": n_out}
    return P, raw, su, cols


def swiglu64(g, u, alpha=1.702, limit=7.0):
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    return g / (1.0 + np.exp(-alpha * g)) * (u + 1.0)


def err(got, ref):
    got = np.asarray(got, np.float64)
    m = np.abs(ref).max()
    return float(np.abs(got - ref).max()), float(np.abs(got - ref).max() / m), float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,16")
    ap.add_argument("--gu", default="1,1.5,2")
    ap.add_argument("--dn", default="1,1.5,2,2.5")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for T in (int(r) for r in a.rows.split(",")):
        idx = np.stack([rng.permutation(E)[:TOPK] for _ in range(T)]).astype(np.int32)
        w = rng.uniform(0.1, 0.6, (T, TOPK)).astype(np.float32)
        x = (rng.standard_normal((T, D)) * 1.0).astype(np.float32)
        xb = mx.array(x).astype(mx.bfloat16)
        x64 = np.array(xb.astype(mx.float32)).astype(np.float64)
        for Kg in (float(k) for k in a.gu.split(",") if k):
            Kg = int(Kg) if Kg.is_integer() else Kg
            g, gr, su, cols = unit(D, I, Kg, 1, False, rng)
            u, ur, _, _ = unit(D, I, Kg, 1, False, rng, su=su, cols=cols)
            m = np.ones(D, np.float32)
            m[cols[0][cols[0] >= 0]] = 0.0
            U = {"L": L, "gate": g, "up": u, "K": Kg, "su_gu": mx.array(su * m)}
            ref = np.stack([swiglu64(gr[e] @ x64[t], ur[e] @ x64[t]) for t in range(T) for e in idx[t]])
            xr = MK.rotate_rows(xb, U["su_gu"])
            v3 = MK.gather_gate_up_swiglu_v3(xr, xb, mx.array(idx), U, out_dtype=mx.float32, limit=7.0, act="oai",
                                             alpha=1.702)
            v4 = FD.routed_gate_up(xb, mx.array(idx), U, 7.0, "oai", 1.702)
            e3, e4 = err(v3, ref), err(v4, ref)
            print(f"gate/up K={Kg} rows={T}: v3 max abs {e3[0]:.3e} max rel {e3[1]:.3e} rms rel {e3[2]:.3e} | "
                  f"v4 max abs {e4[0]:.3e} max rel {e4[1]:.3e} rms rel {e4[2]:.3e}", flush=True)
        h = (rng.standard_normal((T * TOPK, I)) * 1.5).astype(np.float32)
        h64 = h.astype(np.float64)
        for Kd in (float(k) for k in a.dn.split(",") if k):
            Kd = int(Kd) if Kd.is_integer() else Kd
            d, dr, sud, colsd = unit(I, D, Kd, 8, True, rng)
            md = np.ones((E, I), np.float32)
            for e in range(E):
                md[e, colsd[e][colsd[e] >= 0]] = 0.0
            U = {"L": L, "down": d, "K_dn": Kd, "sv_dn": d["cs"] * mx.array(sud)[None, :] * mx.array(md)}
            ref = np.stack([sum(float(w[t, j]) * (dr[idx[t, j]] @ h64[t * TOPK + j]) for j in range(TOPK))
                            for t in range(T)])
            hm = mx.array(h)
            v3 = MK.gather_down_weighted_v3(MK.rotate_rows(hm, U["sv_dn"], mx.array(idx).reshape(-1)), hm,
                                            mx.array(idx), mx.array(w), U)
            v4 = FD.routed_down(hm, mx.array(idx), mx.array(w), U)
            e3, e4 = err(v3, ref), err(v4, ref)
            print(f"down    K={Kd} rows={T}: v3 max abs {e3[0]:.3e} max rel {e3[1]:.3e} rms rel {e3[2]:.3e} | "
                  f"v4 max abs {e4[0]:.3e} max rel {e4[1]:.3e} rms rel {e4[2]:.3e}", flush=True)


if __name__ == "__main__":
    main()
