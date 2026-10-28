# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 MSA attention microbenchmark on synthetic tensors (no model, no weights).

One sparse layer's attention at the released geometry (64 query heads, 4 KV groups, head_dim 128, 4 index heads of
128, blocks of 128, top-16, local 1), bf16, through the model's own Indexer.select / Attention.attend with each MSA
path forced by VMLX_MINIMAX_MSA:

  decode   one query row at the end of Sk cached keys (the cache's K / V buffers are step-padded views, as served)
  prefill  an L-row chunk at the end of Sk keys

Reported per call, median [min-max] over probes (each probe = --iters calls, each depending on the previous one's
output as consecutive layers do, after --warmup discarded probes): "eval" = evaluating the built graph (GPU work and
command encoding; what decode pays once graph building overlaps the previous step), "wall" = building + evaluating.
Paths alternate inside every probe, so other GPU load hits each path alike. "attn" times the attention given the selection; "layer" adds the selection.

    python bench/minimax_m3_msa_bench.py --mode decode --sk 8192 32768 131072
    python bench/minimax_m3_msa_bench.py --mode prefill --rows 2048 --sk 8192 32768
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx

H, G, D, IDX_D, BLOCK, TOPK = 64, 4, 128, 128, 128, 16


def _attention(groups: int):
    from vmlx_engine.models.minimax_m3.minimax_m3 import Attention, ModelArgs

    args = ModelArgs(hidden_size=256, num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=groups,
                     head_dim=D, rotary_dim=64, index_n_heads=groups, index_head_dim=IDX_D,
                     index_block_size=BLOCK, index_topk_blocks=TOPK, index_local_blocks=1,
                     sparse_attention_freq=[1])
    return Attention(args, 0)


def _inputs(sk: int, rows: int, groups: int, seed: int, step: int = 256):
    mx.random.seed(seed)
    cap = -(-sk // step) * step + step
    kbuf = mx.random.normal((1, groups, cap, D)).astype(mx.bfloat16)
    vbuf = mx.random.normal((1, groups, cap, D)).astype(mx.bfloat16)
    ibuf = mx.random.normal((1, 1, cap, IDX_D)).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, rows, D)) * 0.3).astype(mx.bfloat16)
    iq = mx.random.normal((1, groups, rows, IDX_D)).astype(mx.bfloat16)
    mx.eval(kbuf, vbuf, ibuf, q, iq)
    return q, kbuf[:, :, :sk], vbuf[:, :, :sk], iq, ibuf[:, :, :sk]


def _probe(fn, iters: int):
    """(wall ms, eval ms) per call for ``iters`` DEPENDENT calls (each call's query takes a tiny term of the previous
    output, as a layer's input depends on the layer before). wall = building the lazy graph + evaluating it; eval =
    evaluating the already-built graph only (GPU work + command encoding), i.e. what remains once graph building
    overlaps the previous step (async eval)."""
    mx.synchronize()
    t0 = time.perf_counter()
    prev = None
    for i in range(iters):
        prev = fn(i, prev)
    t1 = time.perf_counter()
    mx.eval(prev)
    mx.synchronize()
    t2 = time.perf_counter()
    return (t2 - t0) / iters * 1e3, (t2 - t1) / iters * 1e3


def _summary(times):
    return statistics.median(times), min(times), max(times)


def run(mode: str, sks, rows: int, groups: int, iters: int, probes: int, warmup: int, paths):
    """Paths are interleaved inside every probe (order alternating), so drift and other GPU load hit each alike."""
    attn = _attention(groups)
    ix = attn.indexer
    results = []
    for sk in sks:
        L = 1 if mode == "decode" else rows
        q, k, v, iq, ik = _inputs(sk, L, groups, seed=sk)
        offset = sk - L
        sel0 = ix.select(iq, ik, offset)
        mx.eval(sel0.top_idx)
        mask = "causal" if L > 1 else None
        eps = mx.array(1e-3, dtype=mx.bfloat16)

        def _q(prev):
            return q if prev is None else q + eps * prev

        def attn_only(i, prev):
            return attn.attend(_q(prev), k, v, sel0, mask, None)

        def layer(i, prev):
            iqi = iq if prev is None else iq + eps * prev[:, :groups, :, :IDX_D]
            sel = ix.select(iqi, ik, offset)
            return attn.attend(_q(prev), k, v, sel, mask, None)

        times = {(p, part, kind): [] for p in paths for part in ("attn", "layer") for kind in ("wall", "eval")}
        peaks = {}
        for n in range(warmup + probes):
            order = paths if n % 2 == 0 else list(reversed(paths))
            for path in order:
                os.environ["VMLX_MINIMAX_MSA"] = path
                for part, fn in (("attn", attn_only), ("layer", layer)):
                    mx.reset_peak_memory()
                    wall, ev = _probe(fn, iters)
                    peaks[(path, part)] = mx.get_peak_memory() / 2**20
                    if n >= warmup:
                        times[(path, part, "wall")].append(wall)
                        times[(path, part, "eval")].append(ev)
        for path in paths:
            row = {"mode": mode, "sk": sk, "rows": L, "groups": groups, "path": path,
                   "peak_mib_attn": round(peaks[(path, "attn")]), "peak_mib_layer": round(peaks[(path, "layer")])}
            line = f"{mode:7s} Sk={sk:6d} L={L:4d} G={groups} {path:6s}"
            for part in ("attn", "layer"):
                for kind in ("eval", "wall"):
                    t = times[(path, part, kind)]
                    med, lo, hi = _summary(t)
                    row[f"{part}_{kind}_ms"] = [med, lo, hi]
                    row[f"{part}_{kind}_probes"] = t
                    line += f"  {part}-{kind} {med:7.3f} [{lo:.3f}-{hi:.3f}]"
            results.append(row)
            print(line + f"  peak {peaks[(path, 'attn')]:.0f}/{peaks[(path, 'layer')]:.0f} MiB", flush=True)
        del q, k, v, iq, ik, sel0
        mx.clear_cache()
    os.environ.pop("VMLX_MINIMAX_MSA", None)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--sk", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--groups", type=int, default=G)
    ap.add_argument("--iters", type=int, default=0, help="calls per probe (default: 32 decode, 1 prefill)")
    ap.add_argument("--probes", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--paths", nargs="+", default=["dense", "sparse"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    iters = a.iters or (32 if a.mode == "decode" else 1)
    res = run(a.mode, a.sk, a.rows, a.groups, iters, a.probes, a.warmup, a.paths)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
