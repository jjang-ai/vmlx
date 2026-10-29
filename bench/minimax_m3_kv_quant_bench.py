# SPDX-License-Identifier: Apache-2.0
"""Synthetic real-shape timing of the MiniMax-M3 KV lane formats (models/minimax_m3/kv_quant.py).

M3.1 attention geometry and formats (bench/minimax_m3_fused_attn_bench.py's layers: hidden 6144, 64 x 128 q heads,
4 KV heads, index 4 x 128 / 1 x 128, blocks of 128, top-16; affine-4 q / k / v / o, MXFP8 index projections), random
weights, NL distinct layers so weights stream from DRAM. Per (K / V format, index format) and context:

* ``block``: the fused attention half (VMLX_M3_FUSED_ATTN's five launches) chained through the NL layers for 1 and 16
  rows, ms per layer (medians of PROBES after a discarded warm-up; caches trimmed back after every probe);
* ``k3`` / ``k4``: the index scoring and MSA launches alone (4 x NL independent launches, GPU time per launch);
* ``read``: the unfused path's per-layer cost of decoding the visible extent (K, V, index keys), i.e. what a prefill
  chunk or an unfused decode step adds per layer;
* bytes per token of the three lanes, per layer.

No real model is loaded. Usage: python bench/minimax_m3_kv_quant_bench.py [--ctx 4000,32000,100000] [--layers 16]
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from vmlx_engine.models.minimax_m3 import fused_attn as F
from vmlx_engine.models.minimax_m3 import kv_quant as Q
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

_spec = importlib.util.spec_from_file_location("_fab", Path(__file__).with_name("minimax_m3_fused_attn_bench.py"))
FAB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FAB)
HID, KVH, HD = FAB.HID, FAB.KVH, FAB.HD


def make_cache(ctx: int, seed: int, kf: str, idf: str) -> MiniMaxM3SparseCache:
    rng = np.random.default_rng(seed)
    cap = ((ctx + 16 + 255) // 256) * 256
    lanes = [mx.array(rng.standard_normal((1, h, cap, HD), dtype=np.float32)).astype(mx.bfloat16) for h in (KVH, KVH, 1)]
    kfm, ifm = Q.get(kf), Q.get(idf)
    enc = [Q.quantize(lanes[0], kfm) if kfm.quantized else lanes[0],
           Q.quantize(lanes[1], kfm) if kfm.quantized else lanes[1],
           Q.quantize(lanes[2], ifm) if ifm.quantized else lanes[2]]
    mx.eval(enc)
    del lanes
    c = MiniMaxM3SparseCache(kf, idf)
    c.state = tuple(enc)
    c.trim(cap - ctx)
    # one throw-away step owns the lanes (in-place appends from then on), then trim it back
    return c


def _own(layer, c):
    x = mx.zeros((1, 1, HID), dtype=mx.bfloat16)
    mx.eval(F.attention_block(layer, x, c))
    c.trim(1)


def block_ms(layers, caches, rows, probes):
    x = mx.array(np.random.default_rng(rows).standard_normal((1, rows, HID), dtype=np.float32)).astype(mx.bfloat16)
    ts = []
    for p in range(probes + 1):
        mx.synchronize()
        h = x
        for layer, c in zip(layers, caches):
            h = F.attention_block(layer, h, c)
        t0 = time.perf_counter()
        mx.eval(h)
        ts.append((time.perf_counter() - t0) * 1e3 / len(layers))
        for c in caches:
            c.trim(rows)
    return statistics.median(ts[1:])


def kernel_us(layers, caches, probes):
    """(K3 us, K4 us) per launch, one row, 4 x NL independent launches evaluated together."""
    p = F.plan(layers[0])
    res = {"k3": [], "k4": []}
    for _ in range(probes + 1):
        k3s, k4s = [], []
        for _rep in range(4):
            for layer, c in zip(layers, caches):
                off = int(c.offset)
                nb = -(-(off + 1) // p.block)
                meta = mx.array([off, 1, nb, 0, 0, 0, 0, 0], dtype=mx.int32)
                x = mx.zeros((1, HID), dtype=mx.bfloat16)
                y = F.qkv(layer, p, x)
                dts = F.lane_dtypes(layer, c, mx.bfloat16)
                fm = F._fmts(c)
                q, iq, cnt, _ = F.heads(layer, p, y, meta, None, dts, False, fm)
                mx.eval(q, iq, cnt)
                k, v, i = c.fused_lanes()
                k3s.append((p, iq, i, meta, nb, fm, dts))
                k4s.append((p, q, k, v, cnt, meta, fm, dts, nb))
        mx.synchronize()
        t0 = time.perf_counter()
        sc = [F.scores(p, iq, i, meta, nb, fm[1], dts[2]) for (p, iq, i, meta, nb, fm, dts) in k3s]
        mx.eval(sc)
        t1 = time.perf_counter()
        outs = [F.msa(p, q, k, v, s, cnt, meta, fm[0], dts[0], dts[1])
                for (p, q, k, v, cnt, meta, fm, dts, nb), s in zip(k4s, sc)]
        mx.eval(outs)
        t2 = time.perf_counter()
        res["k3"].append((t1 - t0) * 1e6 / len(k3s))
        res["k4"].append((t2 - t1) * 1e6 / len(k4s))
    return statistics.median(res["k3"][1:]), statistics.median(res["k4"][1:])


def read_ms(caches, probes):
    ts = []
    for _ in range(probes + 1):
        outs = []
        mx.synchronize()
        t0 = time.perf_counter()
        for c in caches:
            k, v = c.read_kv(mx.bfloat16)
            outs += [k, v, c.read_index(mx.bfloat16)]
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) * 1e3 / len(caches))
        del outs
    return statistics.median(ts[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="4000,32000,100000")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--formats", default="bf16:bf16,mxfp8:mxfp8,nvfp4:nvfp4,mxfp8:nvfp4,bf16:nvfp4,affine8:bf16")
    ap.add_argument("--rows", default="1,16")
    a = ap.parse_args()
    import os

    os.environ[F.ENV] = "1"
    layers = FAB.make_layers(a.layers)
    fmts = [tuple(f.split(":")) for f in a.formats.split(",")]
    print("| keys | K/V | index | B/token/layer | block 1 row ms | block 16 rows ms | K3 us | K4 us | unfused read ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ctx in (int(x) for x in a.ctx.split(",")):
        for kf, idf in fmts:
            caches = [make_cache(ctx, s, kf, idf) for s in range(a.layers)]
            for layer, c in zip(layers, caches):
                _own(layer, c)
            bpt = Q.bytes_per_token(kf, KVH, HD) * 2 + Q.bytes_per_token(idf, 1, HD)
            blk = {r: block_ms(layers, caches, int(r), a.probes) for r in a.rows.split(",")}
            k3, k4 = kernel_us(layers, caches, a.probes)
            rd = read_ms(caches, a.probes)
            print(f"| {ctx} | {kf} | {idf} | {bpt} | {blk.get('1', float('nan')):.3f} | "
                  f"{blk.get('16', float('nan')):.3f} | {k3:.1f} | {k4:.1f} | {rd:.3f} |", flush=True)
            del caches
            mx.clear_cache()


if __name__ == "__main__":
    main()
