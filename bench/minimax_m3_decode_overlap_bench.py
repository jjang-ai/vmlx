# SPDX-License-Identifier: Apache-2.0
"""Served MiniMax-M3 decode loop (SingleBatchGenerator) per token: the old synchronous loop vs the new one.

A real-width synthetic M3.1 (hidden 6144, 64 x 128 heads, 4 KV heads, M3.1 attention formats: q / k / v / o affine 4
g64, index MXFP8) with ``--layers`` decoder layers and ``--experts`` routed experts (affine 4) per MoE layer, random
weights. Arms, each from a copy of one primed cache at ``--ctx`` keys, ``--tokens`` greedy tokens:

* ``old``: the loop before perf/minimax-m3-decode-overlap: synchronous, and after every step the per-layer
  ``cache.state`` evaluated through ``_eval_on_stream`` (each array rebound = copied);
* ``sync``: the new loop with VMLX_M3_DECODE_OVERLAP=0 (synchronous, lanes in the step's evaluation, no copy);
* ``overlap``: the new loop (one-token overlap).

Each arm with VMLX_M3_FUSED_ATTN off and on. Reported: ms per token after the first (median of ``--probes``,
arms interleaved per probe) and the fused path's in-place / copy-on-write append counts. No real model is loaded.

Usage: python bench/minimax_m3_decode_overlap_bench.py [--ctx 1000,8000,32000] [--layers 8] [--tokens 48]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from vmlx_engine.models.minimax_m3 import fused_attn as F
from vmlx_engine.models.minimax_m3 import minimax_m3 as M
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache, clone_minimax_m3_sparse
from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator


def build(layers: int, experts: int):
    tc = {"model_type": "minimax_m3", "hidden_size": 6144, "num_hidden_layers": layers, "intermediate_size": 3072,
          "dense_intermediate_size": 12288, "shared_intermediate_size": 3072, "num_attention_heads": 64,
          "num_key_value_heads": 4, "head_dim": 128, "rotary_dim": 64, "rope_theta": 1e7, "rms_norm_eps": 1e-6,
          "vocab_size": 200064, "num_local_experts": experts, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
          "norm_topk_prob": True, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [1] * layers,
          "sparse_attention_config": {"sparse_num_index_heads": 4, "sparse_index_dim": 128, "sparse_block_size": 128,
                                      "sparse_topk_blocks": 16, "sparse_local_block": 1,
                                      "sparse_attention_freq": [1] * layers}}
    model = M.Model(M.ModelArgs.from_dict({"model_type": "minimax_m3_vl", "text_config": tc}))
    def plan(path, module):
        if path.endswith("mlp.gate"):
            return False
        if "switch_mlp" in path:
            return {"group_size": 64, "bits": 4} if hasattr(module, "to_quantized") else False
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if "index_" in path:
            return {"mode": "mxfp8", "group_size": 32, "bits": 8}
        if path.endswith("embed_tokens") or path == "lm_head":
            return {"group_size": 64, "bits": 6}
        return {"group_size": 64, "bits": 4}

    mx.random.seed(0)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("norm.weight"):
            w = (mx.random.normal(v.shape) * 0.1).astype(mx.bfloat16)
        elif k.endswith("e_score_correction_bias"):
            w = mx.random.normal(v.shape) * 0.05
        elif k.endswith(".mlp.gate.weight"):
            w = mx.random.normal(v.shape) * 0.3
        else:
            w = (mx.random.normal(v.shape) * (1.5 / np.sqrt(v.shape[-1]))).astype(mx.bfloat16)
        weights.append((k, w))
    model.load_weights(weights)                     # lazy: materialized module by module under quantization
    nn.quantize(model, class_predicate=plan)
    mx.eval(model.parameters())
    return model


def prime(model, ctx: int):
    cache = model.make_cache()
    ids = np.random.default_rng(1).integers(0, 200000, ctx).tolist()
    for a in range(0, ctx, 2048):
        out = model(mx.array([ids[a:a + 2048]]), cache=cache)
        mx.eval(out, [c.state for c in cache])
    return cache


def clone(cache):
    out = [clone_minimax_m3_sparse(c, copy_fn=lambda a: a * 1) for c in cache]
    mx.eval([c.state for c in out])
    return out


def run(model, primed, tokens: int, arm: str) -> float:
    from vmlx_engine.sampling import make_minimax_m3_sampler

    sampler = make_minimax_m3_sampler(temp=0.0)
    gen = SingleBatchGenerator(model, sampler=sampler, max_tokens=tokens + 1, prefill_step_size=2048,
                               stop_tokens=[])
    if arm in ("old", "sync"):
        gen._can_overlap_decode = lambda request: False
    if arm == "old":
        real = gen._compute_next_from_input_array

        def old_step(req, input_tokens):
            real(req, input_tokens)
            arrs = [req.next_token]
            for c in req.cache:
                arrs.extend(x for x in c.state if x is not None)
            req.next_token = gen._eval_on_stream(*arrs)[0]

        gen._compute_next_from_input_array = old_step
    gen.insert([[7]], caches=[clone(primed)])
    first, = gen.next()[0]
    t0 = time.perf_counter()
    n = 0
    while True:
        resp = gen.next()[1]
        if not resp:
            break
        n += 1
        if resp[0].finish_reason is not None:
            break
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3 / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="1000,8000,32000")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--probes", type=int, default=3)
    a = ap.parse_args()
    model = build(a.layers, a.experts)
    for ctx in (int(c) for c in a.ctx.split(",")):
        primed = prime(model, ctx)
        for fused in (False, True):
            if fused:
                os.environ[F.ENV] = "1"
            else:
                os.environ.pop(F.ENV, None)
            run(model, primed, 8, "overlap")                       # warm-up (kernel builds)
            times = {arm: [] for arm in ("old", "sync", "overlap")}
            stats = {}
            for _ in range(a.probes):
                for arm in times:
                    F.STATS.update(inplace=0, copy_appends=0)
                    times[arm].append(run(model, primed, a.tokens, arm))
                    stats[arm] = (F.STATS["inplace"], F.STATS["copy_appends"])
            med = {arm: statistics.median(v) for arm, v in times.items()}
            print(f"ctx {ctx:6d} layers {a.layers} fused {int(fused)}: ms/token old {med['old']:.2f}  sync "
                  f"{med['sync']:.2f}  overlap {med['overlap']:.2f}  (old/overlap {med['old'] / med['overlap']:.2f}x)"
                  f"  appends in-place/copy old {stats['old']} sync {stats['sync']} overlap {stats['overlap']}",
                  flush=True)


if __name__ == "__main__":
    main()
