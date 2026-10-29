# SPDX-License-Identifier: Apache-2.0
"""Per-component cost of a MiniMax-M3.1 DSpark verify at W = 1..16 rows vs its weights-read-once floor (synthetic).

Real M3.1 shapes and formats, random weights, no bundle: hidden 6144, 64 q heads / 4 KV heads of 128, indexer 4 x 128
(MXFP8), q / k / v / o + shared expert affine 4-bit g64, fp32 router [128, 6144], JANGT routed experts (128 experts,
top-4, intermediate 3072, gate / up K 1 / 1.5 / 2, down K 1.5 / 2 / 2.5, outlier columns as in the bundle), three dense
MLP layers (12288, affine 4-bit), lm_head affine 6-bit g64 [200064, 6144]. Every part runs through the code the DSpark
target runs for a verify of W rows with the levers in the environment (``fused_attn.attention_block`` and its five
kernels, ``fused_moe`` K6 / K7 / shared / routed / K8, ``MiniMaxM3DSparkTarget._linear`` for the head and the dense
MLPs), over several distinct layers so the weights stream from DRAM, scaled to 60 attention / 57 MoE / 3 dense layers.

Routing: drift routing calibrated to the real bundle's distinct-experts-per-layer (coding row, 1,500 tokens: W 1 / 2 /
4 / 8 / 12 / 16 -> 4 / 6.75 / 9.89 / 16.12 / 21.25 / 27.47): consecutive rows keep most of the previous row's experts
(real consecutive tokens share experts, skewed: a few experts serve many rows), each slot replaced with a probability
fitted per W to hit the target mean. ``--routing uniform`` (every expert used by ~the same number of rows) and
``--routing same`` (all rows the same 4 experts) are the controls.

Floors per component: DRAM = bytes the W rows must read once (dense weights once; routed: the distinct experts' bytes;
attention: the K / V / index rows of the visible blocks) / --bw GB/s (539 = measured streaming peak of the M5 Max);
routed also an ALU floor = distinct / 4 x the measured one-row routed time (the trellis decode is ALU-bound, one
decode per distinct expert). Timings: configurations interleaved in rounds, median of --rounds after one discarded
warm-up round, each probe a fresh evaluation of every layer.

  python bench/minimax_m3_flat_verify.py [--rows 1,2,4,8,12,16] [--ctx 1500] [--parts all] [--out f.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimax_m3_fused_attn_bench as AB  # noqa: E402
import minimax_m3_verify_parts as VP  # noqa: E402

D, I, E, TOPE, VOCAB = 6144, 3072, 128, 4, 200064
REAL_DISTINCT = {1: 4.0, 2: 6.75, 4: 9.89, 8: 16.12, 12: 21.25, 16: 27.47}


def target_distinct(w: int) -> float:
    ks = sorted(REAL_DISTINCT)
    if w in REAL_DISTINCT:
        return REAL_DISTINCT[w]
    lo = max(k for k in ks if k <= w)
    hi = min(k for k in ks if k >= w)
    return REAL_DISTINCT[lo] + (REAL_DISTINCT[hi] - REAL_DISTINCT[lo]) * (w - lo) / (hi - lo)


def drift_routing(rows: int, p: float, rng) -> np.ndarray:
    cur = list(rng.choice(E, TOPE, replace=False))
    out = [list(cur)]
    for _ in range(1, rows):
        nxt = []
        for e in cur:
            if rng.random() < p:
                cand = int(rng.integers(E))
                while cand in cur or cand in nxt:
                    cand = int(rng.integers(E))
                nxt.append(cand)
            else:
                nxt.append(int(e))
        cur = nxt
        out.append(list(cur))
    return np.array(out, np.int32)


def fit_drift(rows: int, seed: int = 0) -> float:
    """Slot-replacement probability whose mean distinct-expert count over 400 draws hits the real target."""
    if rows == 1:
        return 0.0
    want = target_distinct(rows)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        rng = np.random.default_rng(seed)
        got = np.mean([len(set(drift_routing(rows, mid, rng).reshape(-1).tolist())) for _ in range(400)])
        lo, hi = (mid, hi) if got < want else (lo, mid)
    return (lo + hi) / 2


def make_routing(rows: int, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "same":
        idx = np.tile(rng.choice(E, TOPE, replace=False), (rows, 1)).astype(np.int32)
    elif kind == "uniform":
        idx = np.array(VP.routing(rows, seed, distinct=target_distinct(rows))[0])[0]
    else:
        idx = drift_routing(rows, fit_drift(rows), rng)
    scores = rng.uniform(0.1, 0.6, (rows, TOPE)).astype(np.float32)
    scores = scores / scores.sum(-1, keepdims=True) * 2.0
    return mx.array(idx), mx.array(scores)


def rows_per_expert(idx: np.ndarray) -> list:
    u, c = np.unique(idx.reshape(-1), return_counts=True)
    return sorted(c.tolist(), reverse=True)


# ------------------------------------------------------------------------------------------------ timing
def _split(r):
    return r if isinstance(r, tuple) else (r, None)


ARMS: dict = {"env": {}}


def _arm_env(arm: str):
    for k, v in ARMS[arm].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def interleaved_arms(make_fns, rounds: int) -> dict:
    """{arm: {name: median ms}}: every arm's fns (``make_fns()`` under that arm's levers) interleaved per round."""
    fns = {}
    for arm in ARMS:
        _arm_env(arm)
        fns[arm] = make_fns()
        for f in fns[arm].values():
            out, post = _split(f())
            mx.eval(out)
            if post:
                post()
    ts = {arm: {k: [] for k in fns[arm]} for arm in ARMS}
    for _ in range(rounds):
        for arm in ARMS:
            _arm_env(arm)
            for k, f in fns[arm].items():
                out, post = _split(f())
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(out)
                ts[arm][k].append((time.perf_counter() - t0) * 1e3)
                if post:
                    post()
    return {arm: {k: statistics.median(v) for k, v in d.items()} for arm, d in ts.items()}


def interleaved(fns: dict, rounds: int) -> dict:
    """{name: median ms} over ``rounds`` rounds (one discarded warm-up), each fn returns arrays to evaluate (or
    (arrays, post): post() runs after the timed evaluation, e.g. a cache rollback)."""
    for f in fns.values():
        out, post = _split(f())
        mx.eval(out)
        if post:
            post()
    ts = {k: [] for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            out, post = _split(f())
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(out)
            ts[k].append((time.perf_counter() - t0) * 1e3)
            if post:
                post()
    return {k: statistics.median(v) for k, v in ts.items()}


def qbytes(m) -> int:
    return int(sum(v.nbytes for k, v in m.parameters().items() if k in ("weight", "scales", "biases")))


# ------------------------------------------------------------------------------------------------ attention
def _flat_on() -> bool:
    from vmlx_engine.models.minimax_m3 import flat_dense as FL

    return FL.active()


def bench_attention(rows_list, ctx, nl, rounds, bw):
    from vmlx_engine.models.minimax_m3 import fused_attn as F
    from vmlx_engine.models.minimax_m3 import flat_dense as FL

    layers = AB.make_layers(nl, seed=11)
    caches = [AB.make_cache(ctx, 50 + i) for i in range(nl)]
    a0 = layers[0].self_attn
    wbytes = sum(qbytes(m) for m in (a0.q_proj, a0.k_proj, a0.v_proj, a0.o_proj, a0.indexer.index_q_proj,
                                      a0.indexer.index_k_proj))
    res = {arm: {} for arm in ARMS}
    for rows in rows_list:
        x = mx.array(np.random.default_rng(rows).standard_normal((1, rows, D), dtype=np.float32)).astype(mx.bfloat16)
        x2 = x.reshape(rows, D)
        stage = []
        for L, c in zip(layers, caches):
            p = F.plan(L)
            off = int(c.offset)
            y = F.qkv(L, p, x2)
            dts = F.lane_dtypes(L, c, x.dtype)
            nb = -(-(off + rows) // p.block)
            meta = mx.array([off, rows, nb, 0, 0, 0, 0, 0], dtype=mx.int32)
            q, iq, cnt, new = F.heads(L, p, y, meta, c.fused_lanes(), dts, False)
            mx.eval(y, q, iq, cnt, new)
            stage.append((L, p, c, y, dts, meta, nb, q, iq, new))
        o = mx.zeros((rows, 64 * 128), dtype=mx.float32) + 0.01

        def make():
            def block():
                outs = [F.attention_block(L, x, c) for L, c in zip(layers, caches)]

                def post():
                    for c in caches:
                        c.trim(rows)
                return outs, post

            def k1():
                if _flat_on():
                    return [FL.attn_qkv(L, x2) for (L, *_r) in stage]
                return [F.qkv(L, p, x2) for (L, p, *_r) in stage]

            def k2():
                return [F.heads(L, p, y, meta, c.fused_lanes(), dts, False)[0]
                        for (L, p, c, y, dts, meta, *_r) in stage]

            def k3():
                return [F.scores(p, iq, c.fused_lanes()[2], meta, nb)
                        for (L, p, c, y, dts, meta, nb, q, iq, new) in stage if nb > p.keep] or [mx.zeros((1,))]

            def k4():
                outs = []
                for (L, p, c, y, dts, meta, nb, q, iq, new) in stage:
                    kl, vl, il = c.fused_lanes()
                    sc = F.scores(p, iq, il, meta, nb) if nb > p.keep else F._dummy()
                    cnt = mx.zeros((max(8, rows * p.KVH),), dtype=mx.uint32)
                    outs.append(F.msa(p, q, kl, vl, sc, cnt, meta))
                return outs

            def k5():
                if _flat_on():
                    return [FL.attn_oproj(L, o, x2) for (L, *_r) in stage]
                return [F.oproj(L, p, o, x2) for (L, p, *_r) in stage]

            return {"block": block, "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}

        t = interleaved_arms(make, rounds)
        scale = 60 / nl
        sparse = any(s_[6] > s_[1].keep for s_ in stage)
        kv_bytes = (ctx + rows) * 4 * 128 * 2 * 2 + (ctx + rows) * 128 * 2
        for arm, tt in t.items():
            out = {k: round(v * scale, 2) for k, v in tt.items()}
            if not sparse:
                out["K3"] = 0.0
            out["floor_proj"] = round(wbytes / bw / 1e6 * 60, 2)
            out["floor_kv"] = round(kv_bytes / bw / 1e6 * 60, 2)
            res[arm][rows] = out
            print(f"[{arm}] attn W={rows:2d} ctx={ctx}: block {out['block']:7.2f} | K1 {out['K1']:6.2f} K2 "
                  f"{out['K2']:5.2f} K3 {out['K3']:5.2f} K4 {out['K4']:6.2f} K5 {out['K5']:6.2f} | floors proj "
                  f"{out['floor_proj']:.2f} kv {out['floor_kv']:.2f} ms / 60 layers", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ MoE
def bench_moe(rows_list, nm, rounds, bw, routing_kind):
    from vmlx_engine.jangt import fast_decode as FD
    from vmlx_engine.models.minimax_m3 import flat_dense as FL
    from vmlx_engine.models.minimax_m3 import fused_moe as FM

    blocks = VP.build_moe(nm)
    sh0 = blocks[0].shared_experts
    sh_bytes = sum(qbytes(m) for m in (sh0.gate_proj, sh0.up_proj, sh0.down_proj))
    exp_bytes = []
    for b in blocks:
        U = b.switch_mlp._unit()
        n = 0
        for part in ("gate", "up", "down"):
            g = U[part]
            n += g["packed"].nbytes // E + g["scale"].nbytes // E
            if g.get("n_out"):
                n += g["wout"].nbytes // E
        exp_bytes.append(n)
    exp_b = float(np.mean(exp_bytes))
    res = {arm: {} for arm in ARMS}
    for rows in rows_list:
        x = VP._x(rows, D, rows)
        h = VP._x(rows, D, rows + 100)
        x2, r2 = x[0], h[0]
        rts = [make_routing(rows, routing_kind, 7 * rows + li) for li in range(len(blocks))]
        rts = [(i.astype(mx.uint32), s_) for i, s_ in rts]
        distinct = float(np.mean([len(set(np.array(i).reshape(-1).tolist())) for i, _ in rts]))
        rpe = rows_per_expert(np.array(rts[0][0]))
        sigs = []
        for b in blocks:
            U = b.switch_mlp._unit()
            sig, _ = FM.router(x2, b.gate.weight, U["su_gu"], rows, rot=False)
            mx.eval(sig)
            sigs.append(sig)

        flags = []

        def make():
            flat = _flat_on()
            in_kernel = (not flat) and rows <= FM._shared_rows()
            flags.append(in_kernel)
            pre = []
            for b, sig in zip(blocks, sigs):
                hh, hs, inds, scores = FM.gate_up(b, x2, sig, None, in_kernel, routed=False)
                mx.eval(hh, hs, inds, scores)
                pre.append((b, hh, hs))
            ry = [FD.routed_fast(x2, i_, s_, b.switch_mlp._unit(), b.switch_mlp.limit, b.switch_mlp.act,
                                 b.switch_mlp.alpha) for b, (i_, s_) in zip(blocks, rts)]
            sdo = [None if in_kernel else (FL.mlp(b.shared_experts, x2) if flat else
                                           FM._shared_rows_exact(b.shared_experts, x2)) for b in blocks]
            mx.eval(ry, [s_ for s_ in sdo if s_ is not None])

            def block():
                return [FM.fused_moe_residual(b, x, h) for b in blocks]

            def k6():
                return [FM.router(x2, b.gate.weight, b.switch_mlp._unit()["su_gu"], rows, rot=False)[0]
                        for b in blocks]

            def k7():
                return [FM.gate_up(b, x2, sig, None, in_kernel, routed=False)[2] for b, sig in zip(blocks, sigs)]

            def shared():
                if in_kernel:
                    return [mx.zeros((1,))]
                if flat:
                    return [FL.mlp(b.shared_experts, x2) for b in blocks]
                return [FM._shared_rows_exact(b.shared_experts, x2) for b in blocks]

            def routed():
                return [FD.routed_fast(x2, i_, s_, b.switch_mlp._unit(), b.switch_mlp.limit, b.switch_mlp.act,
                                       b.switch_mlp.alpha) for b, (i_, s_) in zip(blocks, rts)]

            def k8():
                return [FM.down(b, None, hh, i_, s_, hs if in_kernel else None, r2, sd, ry=y_)
                        for (b, hh, hs), (i_, s_), y_, sd in zip(pre, rts, ry, sdo)]

            return {"block": block, "K6": k6, "K7": k7, "shared": shared, "routed": routed, "K8": k8}

        t = interleaved_arms(make, rounds)
        scale = 57 / len(blocks)
        for (arm, tt), in_kernel in zip(t.items(), flags):
            out = {k: round(v * scale, 2) for k, v in tt.items()}
            if in_kernel:
                out["shared"] = 0.0                 # inside K7 / K8
            out["distinct"] = round(distinct, 2)
            out["rows_per_expert"] = rpe
            out["floor_dense"] = round((sh_bytes + E * D * 4) / bw / 1e6 * 57, 2)
            out["floor_routed_dram"] = round(distinct * exp_b / bw / 1e6 * 57, 2)
            res[arm][rows] = out
            print(f"[{arm}] moe  W={rows:2d}: block {out['block']:7.2f} | K6 {out['K6']:5.2f} K7 {out['K7']:5.2f} "
                  f"shared {out['shared']:6.2f} routed {out['routed']:6.2f} K8 {out['K8']:5.2f} | distinct "
                  f"{distinct:5.2f} (rows/expert {rpe[:6]}) | floors dense {out['floor_dense']:.2f} routed dram "
                  f"{out['floor_routed_dram']:.2f} ms / 57 layers", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ head, dense MLPs
class _MLP:
    def __init__(self, g, u, d):
        self.gate_proj, self.up_proj, self.down_proj, self.alpha, self.limit = g, u, d, 1.702, 7.0


def bench_head_dense(rows_list, rounds, bw):
    from vmlx_engine.models.minimax_m3 import flat_dense as FL

    t = VP._target()
    head = VP._affine(D, VOCAB, 6, seed=7)
    dense = [_MLP(VP._affine(D, 12288, 4, seed=40 + li), VP._affine(D, 12288, 4, seed=50 + li),
                  VP._affine(12288, D, 4, seed=60 + li)) for li in range(3)]
    mx.eval(head.parameters(), [[m.gate_proj.parameters(), m.up_proj.parameters(), m.down_proj.parameters()]
                                for m in dense])
    res = {arm: {} for arm in ARMS}
    for rows in rows_list:
        x = VP._x(rows, D, rows)
        xi = VP._x(rows, 12288, rows + 1)

        def make():
            def hd():
                return [FL.linear(head, x[0])] if _flat_on() else [t._linear(head, x)]

            def dn():
                if _flat_on():
                    return [FL.mlp(m, x[0]) for m in dense]
                return [t._linear(mm_, xi if mm_ is m.down_proj else x) for m in dense
                        for mm_ in (m.gate_proj, m.up_proj, m.down_proj)]

            return {"head": hd, "dense3": dn}

        tt = interleaved_arms(make, rounds)
        for arm, d in tt.items():
            out = {k: round(v, 2) for k, v in d.items()}
            out["floor_head"] = round(qbytes(head) / bw / 1e6, 2)
            out["floor_dense3"] = round(sum(qbytes(mm_) for m in dense for mm_ in (m.gate_proj, m.up_proj,
                                                                                    m.down_proj)) / bw / 1e6, 2)
            res[arm][rows] = out
            print(f"[{arm}] head W={rows:2d}: head {out['head']:6.2f} (floor {out['floor_head']:.2f}) dense MLPs x3 "
                  f"{out['dense3']:6.2f} (floor {out['floor_dense3']:.2f}) ms", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,8,12,16")
    ap.add_argument("--ctx", type=int, default=1500)
    ap.add_argument("--parts", default="attn,moe,head")
    ap.add_argument("--attn-layers", type=int, default=8)
    ap.add_argument("--moe-layers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--bw", type=float, default=539.0)
    ap.add_argument("--routing", default="drift", choices=("drift", "uniform", "same"))
    ap.add_argument("--arms", default="env",
                    help="arms interleaved in one process: name=VAR:VAL,VAR:VAL;name2=... ('-' unsets a VAR); "
                         "'env' = the environment as is")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ARMS.clear()
    for spec in a.arms.split(";"):
        name, _, body = spec.partition("=")
        ARMS[name.strip()] = {kv.split(":")[0]: (None if kv.split(":")[1] == "-" else kv.split(":")[1])
                              for kv in body.split(",") if kv}
    rows_list = [int(r) for r in a.rows.split(",")]
    parts = a.parts.split(",")
    levers = {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("VMLX_")}
    print(f"levers {levers}", flush=True)
    res = {"levers": levers, "ctx": a.ctx, "routing": a.routing}
    if "attn" in parts:
        res["attn"] = bench_attention(rows_list, a.ctx, a.attn_layers, a.rounds, a.bw)
    if "moe" in parts:
        res["moe"] = bench_moe(rows_list, a.moe_layers, a.rounds, a.bw, a.routing)
    if "head" in parts:
        res["head"] = bench_head_dense(rows_list, a.rounds, a.bw)
    if all(k in res for k in ("attn", "moe", "head")):
        res["total"] = {}
        base_w1 = None
        for arm in ARMS:
            tot = {}
            for r in rows_list:
                m = res["moe"][arm][r]
                moe_parts = m["K6"] + m["K7"] + m["shared"] + m["routed"] + m["K8"]      # drift routing
                tot[r] = res["attn"][arm][r]["block"] + moe_parts + res["head"][arm][r]["head"] + \
                    res["head"][arm][r]["dense3"]
            base_w1 = base_w1 or tot[rows_list[0]]
            print(f"[{arm}] total per verify (attention block + MoE parts at drift routing + head + dense MLPs), "
                  f"ms and x the first arm's W={rows_list[0]}:")
            for r in rows_list:
                print(f"  W={r:2d}: {tot[r]:7.2f} ms = {tot[r] / base_w1:.2f}x")
            res["total"][arm] = tot
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
