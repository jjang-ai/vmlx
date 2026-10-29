# SPDX-License-Identifier: Apache-2.0
"""Synthetic per-part cost of a MiniMax-M3.1 DSpark verify (MiniMaxM3DSparkTarget.verify) at k = 1..16 rows.

Real M3.1 shapes, random weights (no bundle): hidden 6144, 64 query heads / 4 KV heads of 128, indexer 4 x 128 (MXFP8),
q / k / v / o + shared expert affine 4-bit g64, fp32 router [128, 6144], JANGT routed experts (128 experts, top-4,
intermediate 3072, gate / up K = 1 / 1.5 / 2 / 2, down K = 1.5 / 2 / 2.5 / 2.5, outlier columns as in the bundle),
lm_head affine 6-bit g64 [200064, 6144]. Each part runs through the target's own verify code (``_linear``,
``_routed``, ``_attention``) over several distinct layers so the weights stream from DRAM as in the model, and is
reported per 57 MoE layers (the head once). Every timing is a median of --reps after warm-up, one eval per rep.

  python bench/minimax_m3_verify_parts.py --part all --rows 1,2,4,8,12,16

The kernel-family levers (env) select the before / after path, e.g. VMLX_M3_VERIFY_WIDE_QMV=0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

D, HEADS, KVH, HD, ROT = 6144, 64, 4, 128, 64
IDX_H, IDX_D, BLOCK, TOPK = 4, 128, 128, 16
E, I, TOPE, VOCAB = 128, 3072, 4, 200064
MOE_LAYERS = 57
# distinct experts per MoE layer measured on the real bundle (coding row, 1,500 tokens): k -> distinct
DISTINCT = {1: 4.0, 2: 6.8, 3: 8.1, 4: 9.8, 5: 11.4, 6: 13.0, 8: 16.1, 10: 18.1, 12: 21.4, 16: 27.7}
# (gate/up K, down K, gate/up outlier columns, down outlier columns): representative M3.1 layers
JT_LAYERS = [(1.0, 1.5, 1, 8), (1.5, 2.0, 0, 8), (1.5, 2.5, 0, 8), (2.0, 2.5, 0, 6)]


def _target():
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    t = object.__new__(MiniMaxM3DSparkTarget)
    t.verify_paths = {"linear_batched": 0, "linear_rows": 0, "routed_batched": 0, "routed_rows": 0}
    return t


def _bits(shape, key):
    """Uniform random uint32 words (two 16-bit draws: randint over the full 32-bit range leaves the low bits zero)."""
    k1, k2 = mx.random.split(key)
    hi = mx.random.randint(0, 1 << 16, shape, key=k1).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, shape, key=k2).astype(mx.uint32)
    return (hi << 16) | lo


def _affine(in_dim, out_dim, bits, gs=64, seed=0):
    """A QuantizedLinear with random packed words and plausible scales / biases (no float weight materialized)."""
    q = nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=gs, bits=bits)
    k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
    q.weight = _bits(q.weight.shape, k1)
    q.scales = (mx.random.uniform(0.004, 0.012, q.scales.shape, key=k2)).astype(mx.bfloat16)
    q.biases = (-q.scales.astype(mx.float32) * (2 ** (bits - 1))
                + mx.random.normal(q.biases.shape, key=k3) * 1e-3).astype(mx.bfloat16)
    return q


def _mxfp8(in_dim, out_dim, seed=0):
    mx.random.seed(seed)
    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.set_dtype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(lin, group_size=32, bits=8, mode="mxfp8")


def _x(rows, dim, seed):
    k1, k2 = mx.random.split(mx.random.key(seed))
    return (mx.random.normal((1, rows, dim), key=k1) * 0.5
            + mx.random.uniform(-1, 1, (1, rows, 1), key=k2)).astype(mx.bfloat16)


def _time(fn, reps, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        out = fn()
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def _time_build(fn, reps, warm=2):
    """Graph build + GPU time (the target builds its graph in Python every verify)."""
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


# ------------------------------------------------------------------------------------------------ projections
def build_proj(nl):
    layers = []
    for li in range(nl):
        s = 100 * li
        layers.append({
            "q": _affine(D, HEADS * HD, 4, seed=s + 1), "k": _affine(D, KVH * HD, 4, seed=s + 2),
            "v": _affine(D, KVH * HD, 4, seed=s + 3), "o": _affine(HEADS * HD, D, 4, seed=s + 4),
            "iq": _mxfp8(D, IDX_H * IDX_D, seed=s + 5), "ik": _mxfp8(D, IDX_D, seed=s + 6),
            "sg": _affine(D, I, 4, seed=s + 7), "su": _affine(D, I, 4, seed=s + 8), "sd": _affine(I, D, 4, seed=s + 9),
        })
    mx.eval([[m.parameters() for m in L.values()] for L in layers])
    return layers


def bench_proj(layers, rows_list, reps):
    t = _target()
    res = {}
    for rows in rows_list:
        xs = {D: _x(rows, D, rows), HEADS * HD: _x(rows, HEADS * HD, rows + 1), I: _x(rows, I, rows + 2)}

        def run():
            outs = []
            for L in layers:
                for name, m in L.items():
                    outs.append(t._linear(m, xs[m.weight.shape[1] * 32 // m.bits]))
            return outs

        ms = _time(run, reps)
        res[rows] = round(ms * MOE_LAYERS / len(layers), 2)
        print(f"proj   rows={rows:2d}: {res[rows]:8.2f} ms / 57 layers", flush=True)
    return res


def bench_head(rows_list, reps):
    t = _target()
    head = _affine(D, VOCAB, 6, seed=7)
    mx.eval(head.parameters())
    res = {}
    for rows in rows_list:
        x = _x(rows, D, rows)
        res[rows] = round(_time(lambda: t._linear(head, x), reps), 2)
        print(f"head   rows={rows:2d}: {res[rows]:8.2f} ms", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ router
def bench_router(nl, rows_list, reps):
    t = _target()
    gates = []
    for li in range(nl):
        g = nn.Linear(D, E, bias=False)
        g.weight = mx.random.normal((E, D), key=mx.random.key(li)) * 0.02
        gates.append(g)
    mx.eval([g.parameters() for g in gates])
    res = {}
    for rows in rows_list:
        x = _x(rows, D, rows)
        res[rows] = round(_time(lambda: [t._linear(g, x) for g in gates], reps) * MOE_LAYERS / nl, 2)
        print(f"router rows={rows:2d}: {res[rows]:8.2f} ms / 57 layers", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ routed experts
def build_routed(nl):
    from vmlx_engine.jangt.switch import JTMixedSwitchGLU

    blocks = []
    for li in range(nl):
        kg, kd, og, od = JT_LAYERS[li % len(JT_LAYERS)]
        sw = JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": kg}, {"mode": "jangt", "kbits": kd}, og, od,
                              7.0, "oai", 1.702)
        keys = mx.random.split(mx.random.key(1000 + li), 12)
        for j, (p, n_in, n_out) in enumerate(((sw.gate_proj, D, og), (sw.up_proj, D, og), (sw.down_proj, I, od))):
            p.jt_packed = _bits(p.jt_packed.shape, keys[4 * j])
            p.jt_scale = mx.random.uniform(0.02, 0.05, p.jt_scale.shape, key=keys[4 * j + 1])
            p.jt_su = mx.where(mx.random.uniform(shape=(n_in,), key=keys[4 * j + 2]) < 0.5, -1.0, 1.0)
            if n_out:
                if p is sw.down_proj:
                    cols = np.stack([np.random.default_rng(li * 7 + e).choice(n_in, n_out, replace=False)
                                     for e in range(E)]).astype(np.int32)
                else:
                    cols = np.tile(np.random.default_rng(li).choice(n_in, n_out, replace=False), (E, 1)).astype(np.int32)
                p.jt_ocol = mx.array(cols)
                p.jt_wout = (mx.random.normal(p.jt_wout.shape, key=keys[4 * j + 3]) * 0.05).astype(mx.float16)
            if p is sw.down_proj:
                p.jt_cs = mx.random.uniform(0.5, 1.5, (E, n_in), key=keys[4 * j + 3])
        sw.up_proj.jt_su = sw.gate_proj.jt_su
        sw.up_proj.jt_ocol = sw.gate_proj.jt_ocol
        mx.eval(sw.parameters())
        sw._unit()

        class _Blk:
            def __init__(self, s):
                self.switch_mlp = s

            def routed(self, x, inds, scores):
                return self.switch_mlp.routed(x, inds, scores)

        blocks.append(_Blk(sw))
    return blocks


def routing(rows, seed, distinct=None):
    """(1, rows, 4) expert ids with ~``distinct`` different experts over the rows (real consecutive tokens share)."""
    rng = np.random.default_rng(seed)
    want = int(round(distinct if distinct is not None else DISTINCT.get(rows, min(E, 4 + 1.6 * (rows - 1)))))
    want = max(TOPE, min(want, rows * TOPE))
    pool = rng.choice(E, want, replace=False)
    order = list(pool)
    idx = np.zeros((rows, TOPE), np.int32)
    k = 0
    for r in range(rows):
        chosen = []
        while len(chosen) < TOPE:
            e = order[k % len(order)] if k < len(order) else rng.choice(pool)
            k += 1
            if e not in chosen:
                chosen.append(e)
        idx[r] = chosen
    scores = rng.uniform(0.1, 0.6, (rows, TOPE)).astype(np.float32)
    return mx.array(idx)[None], mx.array(scores)[None]


def bench_routed(blocks, rows_list, reps, same=False):
    t = _target()
    res = {}
    for rows in rows_list:
        x = _x(rows, D, rows)
        rts = [routing(rows, 31 * rows + li, distinct=TOPE if same else None) for li in range(len(blocks))]

        def run():
            return [t._routed(b)(x, i, s) for b, (i, s) in zip(blocks, rts)]

        ms = _time(run, reps)
        res[rows] = round(ms * MOE_LAYERS / len(blocks), 2)
        print(f"routed rows={rows:2d}{' same' if same else ''}: {res[rows]:8.2f} ms / 57 layers "
              f"(distinct {len(set(np.array(rts[0][0]).reshape(-1).tolist()))})", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ attention
def build_attn(nl, ctx):
    from vmlx_engine.models.minimax_m3 import minimax_m3 as M
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    args = M.ModelArgs(hidden_size=D, num_hidden_layers=nl, num_attention_heads=HEADS, num_key_value_heads=KVH,
                       head_dim=HD, rotary_dim=ROT, rope_theta=1e7, index_n_heads=IDX_H, index_head_dim=IDX_D,
                       index_block_size=BLOCK, index_topk_blocks=TOPK, index_local_blocks=1,
                       sparse_attention_freq=[1] * nl)
    out = []
    for li in range(nl):
        a = M.Attention(args, li)
        s = 100 * li
        a.q_proj, a.k_proj, a.v_proj = (_affine(D, HEADS * HD, 4, seed=s + 1), _affine(D, KVH * HD, 4, seed=s + 2),
                                        _affine(D, KVH * HD, 4, seed=s + 3))
        a.o_proj = _affine(HEADS * HD, D, 4, seed=s + 4)
        a.indexer.index_q_proj, a.indexer.index_k_proj = _mxfp8(D, IDX_H * IDX_D, seed=s + 5), _mxfp8(D, IDX_D, seed=s + 6)
        for n in (a.q_norm, a.k_norm, a.indexer.index_q_norm, a.indexer.index_k_norm):
            n.weight = (mx.random.normal(n.weight.shape, key=mx.random.key(s + 9)) * 0.1).astype(mx.bfloat16)
        c = MiniMaxM3SparseCache()
        k1, k2, k3 = mx.random.split(mx.random.key(s + 11), 3)
        c.update_and_fetch((mx.random.normal((1, KVH, ctx, HD), key=k1)).astype(mx.bfloat16),
                           (mx.random.normal((1, KVH, ctx, HD), key=k2)).astype(mx.bfloat16))
        c.update_index((mx.random.normal((1, 1, ctx, IDX_D), key=k3)).astype(mx.bfloat16))
        mx.eval(a.parameters(), c.state)
        out.append((a, c))
    return out


def bench_attn(layers, rows_list, reps, ctx):
    t = _target()
    res = {}
    for rows in rows_list:
        x = _x(rows, D, rows)

        def run():
            outs = [t._attention(a, x, c, ctx) for a, c in layers]
            return outs

        def once():
            outs = run()
            mx.eval(outs)
            for _, c in layers:
                c.trim(rows)
            return outs

        for _ in range(2):
            once()
        ts = []
        for _ in range(reps):
            mx.synchronize()
            t0 = time.perf_counter()
            once()
            ts.append((time.perf_counter() - t0) * 1e3)
        res[rows] = round(statistics.median(ts) * 60 / len(layers), 2)
        print(f"attn   rows={rows:2d} ctx={ctx}: {res[rows]:8.2f} ms / 60 layers (incl. q/k/v/o/index projections, "
              f"graph build)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--rows", default="1,2,4,8,12,16")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--layers", type=int, default=8, help="distinct trunk layers")
    ap.add_argument("--jt-layers", type=int, default=4, help="distinct JANGT routed layers (~1.7 GB each)")
    ap.add_argument("--ctx", default="1500")
    ap.add_argument("--same", action="store_true", help="routed: all rows pick the same 4 experts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows_list = [int(r) for r in a.rows.split(",")]
    parts = a.part.split(",") if a.part != "all" else ["proj", "head", "router", "routed", "attn"]
    res = {}
    if "proj" in parts:
        res["proj"] = bench_proj(build_proj(a.layers), rows_list, a.reps)
    if "head" in parts:
        res["head"] = bench_head(rows_list, a.reps)
    if "router" in parts:
        res["router"] = bench_router(a.layers, rows_list, a.reps)
    if "routed" in parts:
        blocks = build_routed(a.jt_layers)
        res["routed"] = bench_routed(blocks, rows_list, a.reps, same=a.same)
        del blocks
    if "attn" in parts:
        for ctx in (int(c) for c in a.ctx.split(",")):
            res[f"attn@{ctx}"] = bench_attn(build_attn(a.layers, ctx), rows_list, a.reps, ctx)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
