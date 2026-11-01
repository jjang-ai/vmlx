# SPDX-License-Identifier: Apache-2.0
"""JANGT trellis decode kernels at MiniMax-M3.1 widths: today's v3 kernels vs vmlx_engine/jangt/fast_decode.py
variants, per K, 1 / 4 / 8 / 16 rows (synthetic banks: 128 experts, hidden 6144, intermediate 3072, L 12).

Each timing = one eval of NL independent layer calls (distinct banks so weights stream from DRAM), interleaved over
configurations for --rounds rounds; min and median reported, plus effective GB/s at the min (bytes of the DISTINCT
experts the rows touch: packed words + row scales + outlier weights).

  python bench/jangt_decode_profile.py --part gu --K 1,1.5,2 --rows 1,4,8,16
  python bench/jangt_decode_profile.py --part dn --K 1,1.5,2,2.5 --rows 1,16 --variants v3,fast,fast_O8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from vmlx_engine.jangt import fast_decode as FD
from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL

D, I, E, TOPK, L = 6144, 3072, 128, 4, 12
DISTINCT = {1: 4, 2: 7, 4: 10, 8: 16, 12: 21, 16: 28}

def _v4(spec: str):
    """v4 variant from a name like v4_O4G2S1_scalar, or v4c_O4G2_scalar (down, slot combine in-kernel)."""
    import re
    if spec.startswith("v4g_"):
        body, wload = spec[4:].split("_")
        m = re.match(r"O(\d+)G(\d+)R(\d+)", body)
        return ("v4g", {"O": int(m.group(1)), "NOG": int(m.group(2)), "RP": int(m.group(3)), "wload": wload})
    if spec.startswith("v4c_"):
        body, wload = spec[4:].split("_")
        m = re.match(r"O(\d+)G(\d+)", body)
        return ("v4c", {"O": int(m.group(1)), "NOG": int(m.group(2)), "wload": wload})
    body, wload = spec[3:].split("_")
    m = re.match(r"O(\d+)G(\d+)S(\d+)", body)
    return ("v4", {"O": int(m.group(1)), "NOG": int(m.group(2)), "KSPLIT": int(m.group(3)), "wload": wload})


VARIANTS = {
    # name: (kind, options)
    "v3": ("v3", {}),
    "prod": ("prod", {}),       # fast_decode.routed_gate_up / routed_down (input packing included)
    "fast2": ("fast", {"fold": "two"}),
    "fast_ld": ("fast", {"fold": "one", "wload": "scalar"}),
    "fast_pf": ("fast", {"fold": "one", "wload": "prefetch"}),
    "fast_pk": ("fast", {"fold": "one", "wload": "packed"}),
    "lut": ("fast", {"fold": "one", "wload": "scalar", "hashop": "lut"}),
    "lut_S8": ("fast", {"fold": "one", "wload": "scalar", "hashop": "lut", "NSG": 8}),
    "lut_S8O2": ("fast", {"fold": "one", "wload": "scalar", "hashop": "lut", "NSG": 8, "O": 2}),
    "lut_S16O2": ("fast", {"fold": "one", "wload": "scalar", "hashop": "lut", "NSG": 16, "O": 2}),
    "fast_pk2": ("fast", {"fold": "one", "wload": "packed", "O": 2}),
    "fast_pk8": ("fast", {"fold": "one", "wload": "packed", "O": 8}),
    "fast_pf2": ("fast", {"fold": "one", "wload": "prefetch", "O": 2}),
    "fast_pf2S4": ("fast", {"fold": "one", "wload": "prefetch", "O": 2, "NSG": 4}),
    "fast": ("fast", {"fold": "one"}),
    "fast_O8": ("fast", {"fold": "one", "O": 8}),
    "fast_O2": ("fast", {"fold": "one", "O": 2}),
    "fast_S4": ("fast", {"fold": "one", "NSG": 4}),
    "fast_O8S1": ("fast", {"fold": "one", "O": 8, "NSG": 1}),
    "fast_O8S4": ("fast", {"fold": "one", "O": 8, "NSG": 4}),
    "fast_nobfe": ("fast", {"fold": "one", "bfe": False}),
    "abl_nohash": ("fast", {"fold": "one", "hashop": "none"}),
    "abl_noload": ("fast", {"fold": "one", "wconst": True}),
    "abl_small": ("fast", {"fold": "one", "wload": "scalar", "wsmall": True}),
    "abl_xor": ("fast", {"fold": "one", "wload": "scalar", "abl_fma": "xor"}),
    "abl_xor_noload": ("fast", {"fold": "one", "wconst": True, "abl_fma": "xor"}),
    "abl_all": ("fast", {"fold": "one", "wconst": True, "abl_fma": "xor", "hashop": "none"}),
    "abl_skel": ("fast", {"fold": "one", "wconst": True, "abl_fma": "skeleton"}),
    "abl_skel_ld": ("fast", {"fold": "one", "wload": "scalar", "abl_fma": "skeleton"}),
    "abl_extract": ("fast", {"fold": "one", "wconst": True, "abl_fma": "extract_only"}),
    "abl_extract_nobfe": ("fast", {"fold": "one", "wconst": True, "abl_fma": "extract_only", "bfe": False}),
    "abl_nohash_noload": ("fast", {"fold": "one", "hashop": "none", "wconst": True}),
}


def _bits(shape, key):
    k1, k2 = mx.random.split(key)
    hi = mx.random.randint(0, 1 << 16, shape, key=k1).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, shape, key=k2).astype(mx.uint32)
    return (hi << 16) | lo


def _proj(n_in, n_out, K, n_cols, seed, per_expert_cols):
    W = VL.words_per_row(n_in, L, K)
    ks = mx.random.split(mx.random.key(seed), 4)
    rng = np.random.default_rng(seed)
    if n_cols:
        cols = (np.stack([rng.choice(n_in, n_cols, replace=False) for _ in range(E)]) if per_expert_cols
                else np.tile(rng.choice(n_in, n_cols, replace=False), (E, 1))).astype(np.int32)
        wout = (mx.random.normal((E, n_out, n_cols), key=ks[2]) * 0.05).astype(mx.float16)
    else:
        cols, wout = np.full((E, 1), -1, np.int32), mx.zeros((E, n_out, 1), mx.float16)
    return {"K": K, "kin": n_in, "packed": _bits((E, n_out, W), ks[0]),
            "scale": mx.random.uniform(0.02, 0.05, (E, n_out), key=ks[1]), "ocol": mx.array(cols), "wout": wout,
            "n_out": n_cols, "N": n_out}


def build(part, K, nl, n_cols):
    units = []
    for li in range(nl):
        s = 1000 * li + int(K * 10)
        su = mx.where(mx.random.uniform(shape=(D,), key=mx.random.key(s)) < 0.5, -1.0, 1.0)
        if part == "gu":
            g, u = _proj(D, I, K, n_cols, s + 1, False), _proj(D, I, K, n_cols, s + 2, False)
            u["ocol"] = g["ocol"]
            U = {"L": L, "gate": g, "up": u, "K": K, "su_gu": su}
        else:
            d = _proj(I, D, K, n_cols, s + 3, True)
            sv = mx.random.uniform(0.5, 1.5, (E, I), key=mx.random.key(s + 5)) * \
                mx.where(mx.random.uniform(shape=(I,), key=mx.random.key(s + 6)) < 0.5, -1.0, 1.0)
            U = {"L": L, "down": d, "K_dn": K, "sv_dn": sv}
        mx.eval([v for p in U.values() if isinstance(p, dict) for v in p.values() if isinstance(v, mx.array)]
                + [v for v in U.values() if isinstance(v, mx.array)])
        units.append(U)
    return units


def routing(rows, seed):
    rng = np.random.default_rng(seed)
    want = max(TOPK, min(DISTINCT.get(rows, 4 + int(1.6 * (rows - 1))), rows * TOPK))
    pool = list(rng.choice(E, want, replace=False))
    idx, k = np.zeros((rows, TOPK), np.int32), 0
    for r in range(rows):
        chosen = []
        while len(chosen) < TOPK:
            e = pool[k % len(pool)] if k < len(pool) else pool[rng.integers(len(pool))]
            k += 1
            if e not in chosen:
                chosen.append(e)
        idx[r] = chosen
    return mx.array(idx), mx.array(rng.uniform(0.1, 0.6, (rows, TOPK)).astype(np.float32))


def active_bytes(part, U, idx):
    experts = len(set(np.array(idx).reshape(-1).tolist()))
    ps = [U["gate"], U["up"]] if part == "gu" else [U["down"]]
    per = 0
    for p in ps:
        per += p["packed"][0].nbytes + p["scale"][0].nbytes + (p["wout"][0].nbytes if p["n_out"] else 0)
    return experts * per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gu")
    ap.add_argument("--K", default="1,1.5,2")
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--variants", default="v3,fast2,fast,fast_O8")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--outliers", type=int, default=-1, help="outlier columns (default: gu 0, dn 8)")
    ap.add_argument("--prep", action="store_true", help="include the input preparation (rotate / pack) in the timing")
    ap.add_argument("--same", action="store_true", help="every row picks the same 4 experts")
    a = ap.parse_args()
    part = a.part
    n_cols = a.outliers if a.outliers >= 0 else (0 if part == "gu" else 8)
    variants = a.variants.split(",")
    for K in (float(k) for k in a.K.split(",")):
        K = int(K) if K.is_integer() else K
        units = build(part, K, a.layers, n_cols)
        for rows in (int(r) for r in a.rows.split(",")):
            rts = [routing(rows, 7 * rows + li) for li in range(a.layers)]
            if a.same:
                rts = [(mx.broadcast_to(i[:1], i.shape), w) for i, w in rts]
            if part == "gu":
                x = (mx.random.normal((rows, D), key=mx.random.key(rows)) * 0.5).astype(mx.bfloat16)
                xr = [MK.rotate_rows(x, U["su_gu"]) for U in units]
                pk = [FD.pack_rows(x, U["su_gu"]) for U in units]
            else:
                hh = mx.random.normal((rows * TOPK, I), key=mx.random.key(rows + 50)) * 2.0
                hr = [MK.rotate_rows(hh, U["sv_dn"], i.reshape(-1)) for U, (i, _) in zip(units, rts)]
                pk = [FD.pack_rows(hh, U["sv_dn"], i.reshape(-1)) for U, (i, _) in zip(units, rts)]
            mx.eval(xr if part == "gu" else hr, pk)

            def make(name):
                kind, opts = VARIANTS[name]
                opts = dict(opts)
                if "abl_fma" in opts:
                    opts["abl"] = opts.pop("abl_fma")

                def run():
                    outs = []
                    for li, (U, (idx, w)) in enumerate(zip(units, rts)):
                        if part == "gu":
                            if kind == "v3":
                                xin = MK.rotate_rows(x, U["su_gu"]) if a.prep else xr[li]
                                outs.append(MK.gather_gate_up_swiglu_v3(xin, x, idx, U, out_dtype=mx.float32, limit=7.0,
                                                                        slot_fastest=rows > 1, act="oai", alpha=1.702))
                            elif kind == "prod":
                                outs.append(FD.routed_gate_up(x, idx, U, 7.0, "oai", 1.702))
                            elif kind == "v4g":
                                p = FD.pack_rows1(x, U["su_gu"]) if a.prep else pk[li]
                                outs.append(FD.gate_up4g(p, x, idx, U, limit=7.0, act="oai", alpha=1.702, **opts))
                            elif kind == "v4":
                                p = FD.pack_rows1(x, U["su_gu"]) if a.prep else pk[li]
                                outs.append(FD.gate_up4(p, x, idx, U, limit=7.0, act="oai", alpha=1.702, **opts))
                            else:
                                p = FD.pack_rows(x, U["su_gu"]) if a.prep else pk[li]
                                outs.append(FD.gate_up(p, x, idx, U, limit=7.0, act="oai", alpha=1.702, **opts))
                        else:
                            if kind == "v3":
                                hin = MK.rotate_rows(hh, U["sv_dn"], idx.reshape(-1)) if a.prep else hr[li]
                                outs.append(MK.gather_down_weighted_v3(hin, hh, idx, w, U, slot_fastest=rows > 1))
                            elif kind == "v4":
                                p = FD.pack_rows1(hh, U["sv_dn"], idx.reshape(-1)) if a.prep else pk[li]
                                outs.append(FD.down4(p, hh, idx, w, U, **opts))
                            elif kind == "prod":
                                outs.append(FD.routed_down(hh, idx, w, U))
                            elif kind == "v4g":
                                p = FD.pack_rows1(hh, U["sv_dn"], idx.reshape(-1)) if a.prep else pk[li]
                                outs.append(FD.down4g(p, hh, idx, w, U, **opts))
                            elif kind == "v4c":
                                p = FD.pack_rows1(hh, U["sv_dn"], idx.reshape(-1)) if a.prep else pk[li]
                                outs.append(FD.down4c(p, hh, idx, w, U, **opts))
                            else:
                                p = FD.pack_rows(hh, U["sv_dn"], idx.reshape(-1)) if a.prep else pk[li]
                                outs.append(FD.down_weighted(p, hh, idx, w, U, **opts))
                    return outs
                return run

            for v in variants:
                if v.startswith(("v4_", "v4c_", "v4g_")) and v not in VARIANTS:
                    VARIANTS[v] = _v4(v)
            fns = {v: make(v) for v in variants}
            for f in fns.values():
                for _ in range(2):
                    mx.eval(f())
            ts = {v: [] for v in fns}
            for _ in range(a.rounds):
                for v, f in fns.items():
                    out = f()
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval(out)
                    ts[v].append((time.perf_counter() - t0) * 1e3)
            nbytes = sum(active_bytes(part, U, idx) for U, (idx, _) in zip(units, rts))
            line = []
            for v, t in ts.items():
                mn = min(t)
                line.append(f"{v} {mn * 57 / a.layers:.2f}/{statistics.median(t) * 57 / a.layers:.2f}ms "
                            f"{nbytes / (mn * 1e-3) / 1e9:.0f}GB/s")
            print(f"{part} K={K} rows={rows:2d} (57 layers, min/median): " + " | ".join(line), flush=True)
        del units


if __name__ == "__main__":
    sys.exit(main())
