# SPDX-License-Identifier: Apache-2.0
"""Synthetic real-shape timing of the MiniMax-M3 fused attention half (VMLX_M3_FUSED_ATTN) vs today's ops.

M3.1 geometry (hidden 6144, 64 q heads x 128, 4 KV heads, partial RoPE 64, theta 1e7, index 4 x 128 / 1 x 128,
blocks of 128, top-16), q / k / v / o affine 4-bit g64, index q / k MXFP8, random weights; NL distinct layers so the
weights stream from DRAM as on the real model. Each probe chains the attention half through all NL layers (the
residual of layer l is the input of layer l + 1, as in decode), evaluates once, and reports ms per layer; caches sit
at a fixed context (each probe trims what it appended). Medians of PROBES probes after one discarded warm-up.

Paths: ``stock`` = today's ops (rows = 1: the model's decode attention; rows > 1: the DSpark verify's per-row
attention with row-exact projections), ``fused`` = the five fused launches. ``--kernels`` adds per-kernel GPU times
(NL independent launches of one kernel over the NL layers' weights, evaluated together: throughput per launch) for
the fused kernels and their unfused op groups.

No real model is loaded. Usage: python bench/minimax_m3_fused_attn_bench.py [--rows 1,4,8,16] [--ctx 1000,4000,16000]
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget
from vmlx_engine.models.minimax_m3 import fused_attn as F
from vmlx_engine.models.minimax_m3 import minimax_m3 as M
from vmlx_engine.models.minimax_m3 import msa
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

HID, H, KVH, HD, ROT = 6144, 64, 4, 128, 64


class _Layer:
    def __init__(self, attn, norm):
        self.self_attn, self.input_layernorm = attn, norm


def _args():
    tc = {"hidden_size": HID, "num_attention_heads": H, "num_key_value_heads": KVH, "head_dim": HD, "rotary_dim": ROT,
          "rope_theta": 1e7, "rms_norm_eps": 1e-6, "num_hidden_layers": 1,
          "sparse_attention_config": {"sparse_num_index_heads": 4, "sparse_index_dim": 128, "sparse_block_size": 128,
                                      "sparse_topk_blocks": 16, "sparse_local_block": 1,
                                      "sparse_attention_freq": [1]}}
    return M.ModelArgs.from_dict({"text_config": tc})


def _qlinear(rng, out_d, in_d, mode):
    w = mx.array(rng.standard_normal((out_d, in_d), dtype=np.float32) * (1.5 / np.sqrt(in_d))).astype(mx.bfloat16)
    if mode == "mxfp8":
        m = nn.QuantizedLinear(in_d, out_d, bias=False, group_size=32, bits=8, mode="mxfp8")
        m.weight, m.scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    else:
        m = nn.QuantizedLinear(in_d, out_d, bias=False, group_size=64, bits=4)
        m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4)
    return m


def make_layers(n, seed=0):
    args = _args()
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(n):
        attn = M.Attention(args, 0)
        attn.q_proj = _qlinear(rng, H * HD, HID, "affine")
        attn.k_proj = _qlinear(rng, KVH * HD, HID, "affine")
        attn.v_proj = _qlinear(rng, KVH * HD, HID, "affine")
        attn.o_proj = _qlinear(rng, HID, H * HD, "affine")
        ix = attn.indexer
        ix.index_q_proj = _qlinear(rng, 4 * 128, HID, "mxfp8")
        ix.index_k_proj = _qlinear(rng, 128, HID, "mxfp8")
        for nm in (attn.q_norm, attn.k_norm, ix.index_q_norm, ix.index_k_norm):
            nm.weight = mx.array(rng.standard_normal(nm.weight.shape, dtype=np.float32) * 0.1).astype(mx.bfloat16)
        norm = M.GemmaRMSNorm(HID, 1e-6)
        norm.weight = mx.array(rng.standard_normal((HID,), dtype=np.float32) * 0.1).astype(mx.bfloat16)
        layers.append(_Layer(attn, norm))
        mx.eval(attn.parameters(), norm.parameters())
    return layers


def make_cache(ctx, seed):
    rng = np.random.default_rng(seed)
    c = MiniMaxM3SparseCache()
    cap = ((ctx + 16 + 255) // 256) * 256
    k = mx.array(rng.standard_normal((1, KVH, cap, HD), dtype=np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, KVH, cap, HD), dtype=np.float32)).astype(mx.bfloat16)
    i = mx.array(rng.standard_normal((1, 1, cap, 128), dtype=np.float32)).astype(mx.bfloat16)
    c.state = (k, v, i)
    c.trim(cap - ctx)
    mx.eval(c.state)
    return c


def _verify_target():
    t = MiniMaxM3DSparkTarget.__new__(MiniMaxM3DSparkTarget)
    t.verify_paths = {"linear_batched": 0, "linear_rows": 0, "routed_batched": 0, "routed_rows": 0}
    return t


def block(path, layer, x, cache, target):
    rows = int(x.shape[1])
    if path == "fused":
        return F.attention_block(layer, x, cache)
    if rows == 1:
        return x + layer.self_attn(layer.input_layernorm(x), None, cache)
    off = int(cache.offset)
    return x + target._attention(layer.self_attn, layer.input_layernorm(x), cache, off)


def probe(path, layers, caches, x, target):
    """(eval ms per layer, graph-build ms per layer): the chain (every layer of every pass: ``caches`` is one list per
    pass) is built first, then evaluated once; eval = encode + GPU."""
    rows = int(x.shape[1])
    mx.synchronize()
    t0 = time.perf_counter()
    h = x
    for pass_caches in caches:
        for layer, c in zip(layers, pass_caches):
            h = block(path, layer, h, c, target)
    t1 = time.perf_counter()
    mx.eval(h)
    t2 = time.perf_counter()
    n = 0
    for pass_caches in caches:
        for c in pass_caches:
            c.trim(rows)
            n += 1
    return (t2 - t1) * 1e3 / n, (t1 - t0) * 1e3 / n


def run_blocks(layers, ctxs, rows_list, probes, passes=1):
    target = _verify_target()
    out = []
    for ctx in ctxs:
        caches = [[make_cache(ctx, s + 1000 * p) for s in range(len(layers))] for p in range(passes)]
        for rows in rows_list:
            x = mx.array(np.random.default_rng(rows).standard_normal((1, rows, HID), dtype=np.float32)).astype(
                mx.bfloat16)
            res, build = {}, {}
            for path in ("stock", "fused"):
                ts = [probe(path, layers, caches, x, target) for _ in range(probes + 1)][1:]
                res[path] = statistics.median(t[0] for t in ts)
                build[path] = statistics.median(t[1] for t in ts)
            out.append((ctx, rows, res["stock"], res["fused"], build["stock"], build["fused"]))
            print(f"ctx {ctx:6d} rows {rows:2d}: eval stock {res['stock']:.3f} fused {res['fused']:.3f} ms/layer "
                  f"({res['stock'] / res['fused']:.2f}x) | graph build stock {build['stock']:.3f} fused "
                  f"{build['fused']:.3f} ms/layer [inplace {F.STATS['inplace']} copy {F.STATS['copy_appends']}]",
                  flush=True)
    return out


def _time_independent(fn_per_layer, n_layers, probes):
    """Eval ms per layer of n_layers independent launches (graph built before the clock starts)."""
    ts = []
    for _ in range(probes + 1):
        outs = [fn_per_layer(i) for i in range(n_layers)]
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) * 1e3 / n_layers)
    return statistics.median(ts[1:])


def run_kernels(layers, ctx, rows, probes):
    """Per-kernel GPU time per layer (independent launches over the layers' weights)."""
    n = len(layers)
    caches = [make_cache(ctx, s) for s in range(n)]
    x = mx.array(np.random.default_rng(1).standard_normal((1, rows, HID), dtype=np.float32)).astype(mx.bfloat16)
    x2 = x.reshape(rows, HID)
    plans = [F.plan(l) for l in layers]
    target = _verify_target()
    lin = (lambda m, a: m(a)) if rows == 1 else target._linear
    res = {}
    # K1 vs input norm + 5 projections
    res["K1 norm+qkv+index proj"] = (
        _time_independent(lambda i: [lin(m, layers[i].input_layernorm(x)) for m in (
            layers[i].self_attn.q_proj, layers[i].self_attn.k_proj, layers[i].self_attn.v_proj,
            layers[i].self_attn.indexer.index_q_proj, layers[i].self_attn.indexer.index_k_proj)], n, probes),
        _time_independent(lambda i: F.qkv(layers[i], plans[i], x2), n, probes))
    ys = [F.qkv(layers[i], plans[i], x2) for i in range(n)]
    mx.eval(ys)
    meta = mx.array([ctx, rows, -(-(ctx + rows) // 128), 0, 0, 0, 0, 0], dtype=mx.int32)
    dts = [F.lane_dtypes(l, c, mx.bfloat16) for l, c in zip(layers, caches)]

    def stock_heads(i):
        a, ix = layers[i].self_attn, layers[i].self_attn.indexer
        y = ys[i].astype(mx.bfloat16)
        q = a.q_norm(y[:, :H * HD].reshape(1, rows, H, HD)).transpose(0, 2, 1, 3)
        k = a.k_norm(y[:, H * HD:(H + KVH) * HD].reshape(1, rows, KVH, HD)).transpose(0, 2, 1, 3)
        iq = ix.index_q_norm(y[:, (H + 2 * KVH) * HD:(H + 2 * KVH + 4) * HD].reshape(1, rows, 4, HD)).transpose(0, 2, 1, 3)
        ik = ix.index_k_norm(y[:, (H + 2 * KVH + 4) * HD:].reshape(1, rows, 1, HD)).transpose(0, 2, 1, 3)
        return [a.rope(q, offset=ctx), a.rope(k, offset=ctx), ix.rope(iq, offset=ctx), ix.rope(ik, offset=ctx)]

    def fused_heads(i):
        caches[i].fused_reserve(rows, KVH, HD, *dts[i][:2], HD, dts[i][2])
        return F.heads(layers[i], plans[i], ys[i], meta, caches[i].fused_lanes(), dts[i], True)[:3]

    res["K2 head norms+rope(+appends)"] = (_time_independent(stock_heads, n, probes),
                                            _time_independent(fused_heads, n, probes))
    outs = [fused_heads(i) for i in range(n)]
    mx.eval(outs)
    nb = -(-(ctx + rows) // 128)
    if nb > 16:
        def stock_scores(i):
            iq = outs[i][1].astype(mx.bfloat16).transpose(1, 0, 2)[None]
            ik = caches[i].fused_lanes()[2][..., :ctx + rows, :]
            return [layers[i].self_attn.indexer.select(iq[:, :, r:r + 1], ik[:, :, :ctx + r + 1], ctx + r).top_idx
                    for r in range(rows)]
        res["K3 block scores(+select)"] = (_time_independent(stock_scores, n, probes),
                                           _time_independent(lambda i: F.scores(plans[i], outs[i][1],
                                                                                caches[i].fused_lanes()[2], meta, nb),
                                                             n, probes))
        scs = [F.scores(plans[i], outs[i][1], caches[i].fused_lanes()[2], meta, nb) for i in range(n)]
    else:
        scs = [F._dummy() for _ in range(n)]
    mx.eval(scs)

    def stock_attn(i):
        a = layers[i].self_attn
        q = outs[i][0].astype(mx.bfloat16).transpose(1, 0, 2)[None]
        kl, vl, il = caches[i].fused_lanes()
        res_ = []
        for r in range(rows):
            end = ctx + r + 1
            sel = None
            if nb > 16:
                iq = outs[i][1].astype(mx.bfloat16).transpose(1, 0, 2)[None]
                sel = a.indexer.select(iq[:, :, r:r + 1], il[..., :end, :], ctx + r)
            res_.append(a.attend(q[:, :, r:r + 1], kl[..., :end, :], vl[..., :end, :], sel, None, None))
        return res_

    def fused_attn(i):
        kl, vl, _ = caches[i].fused_lanes()
        return F.msa(plans[i], outs[i][0], kl, vl, scs[i], outs[i][2], meta)

    res["K4 select+MSA attention"] = (_time_independent(stock_attn, n, probes),
                                      _time_independent(fused_attn, n, probes))
    os_ = [fused_attn(i) for i in range(n)]
    mx.eval(os_)
    res["K5 o_proj+residual"] = (
        _time_independent(lambda i: x + lin(layers[i].self_attn.o_proj,
                                            os_[i].astype(mx.bfloat16).reshape(1, rows, H * HD)), n, probes),
        _time_independent(lambda i: F.oproj(layers[i], plans[i], os_[i].reshape(rows, H * HD), x2), n, probes))
    for name, (s, f) in res.items():
        print(f"  ctx {ctx:6d} rows {rows:2d} {name:32s} stock {s * 1e3:8.1f} us  fused {f * 1e3:8.1f} us", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--ctx", default="1000,4000,16000")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--passes", type=int, default=1, help="chain the layers this many times per evaluation")
    a = ap.parse_args()
    layers = make_layers(a.layers)
    rows_list = [int(r) for r in a.rows.split(",")]
    ctxs = [int(c) for c in a.ctx.split(",")]
    if not a.no_blocks:
        run_blocks(layers, ctxs, rows_list, a.probes, a.passes)
    if a.kernels:
        for ctx in ctxs:
            for rows in rows_list:
                run_kernels(layers, ctx, rows, a.probes)


if __name__ == "__main__":
    main()
