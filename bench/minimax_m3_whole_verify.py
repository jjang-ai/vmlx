# SPDX-License-Identifier: Apache-2.0
"""Whole DSpark verify on a REAL-WIDTH synthetic MiniMax-M3.1 (a few layers): MiniMaxM3DSparkTarget.verify end to
end (graph build + evaluation, one dependent chain), per width, levers in the environment. Compare with the sum of
separately timed components (bench/minimax_m3_flat_verify.py) to see what the chain itself (glue, dependent
launches, Python graph build) adds.

Model: hidden 6144, 64 x 128 heads, 4 KV, index 4 x 128 (MXFP8), q / k / v / o / shared / dense MLP affine 4 g64,
fp32 router + bias, JANGT routed experts (128 experts, top-4, I 3072; the bench's K mix and outlier columns), 6-bit
lm_head and embedding, layer 0 dense MLP, layers 1.. MoE, every layer sparse attention. Random weights; no bundle.

  python bench/minimax_m3_whole_verify.py [--layers 4] [--ctx 1500] [--widths 1,2,4,6,8,12,16] [--reps 6]
      [--ab VMLX_M3_EXACT_FLAT=1]   (an arm with this env lever set, interleaved per rep with the levers as given)
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimax_m3_verify_parts as VP  # noqa: E402


def build(n_layers: int, seed: int = 0):
    from mlx.utils import tree_flatten

    from vmlx_engine.models.minimax_m3 import minimax_m3 as M

    tc = {"model_type": "minimax_m3", "hidden_size": 6144, "num_hidden_layers": n_layers, "intermediate_size": 3072,
          "dense_intermediate_size": 12288, "shared_intermediate_size": 3072, "num_attention_heads": 64,
          "num_key_value_heads": 4, "head_dim": 128, "rotary_dim": 64, "rope_theta": 1e7, "rms_norm_eps": 1e-6,
          "vocab_size": 200064, "num_local_experts": 128, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
          "norm_topk_prob": True, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
          "moe_layer_freq": [0] + [1] * (n_layers - 1),
          "sparse_attention_config": {"sparse_num_index_heads": 4, "sparse_index_dim": 128, "sparse_block_size": 128,
                                      "sparse_topk_blocks": 16, "sparse_local_block": 1,
                                      "sparse_attention_freq": [1] * n_layers}}
    model = M.Model(M.ModelArgs.from_dict({"model_type": "minimax_m3_vl", "text_config": tc}))
    rng = np.random.default_rng(seed)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if "switch_mlp" in k:
            continue
        g = rng.standard_normal(v.shape, dtype=np.float32)
        if k.endswith("norm.weight"):
            g *= 0.1; dt = mx.bfloat16
        elif k.endswith("e_score_correction_bias"):
            g *= 0.05; dt = mx.float32
        elif k.endswith(".mlp.gate.weight"):
            g *= 0.02; dt = mx.float32
        else:
            g *= 1.5 / np.sqrt(v.shape[-1]); dt = mx.bfloat16
        w = mx.array(g).astype(dt)
        mx.eval(w)
        weights.append((k, w))
        del g
    model.load_weights(weights, strict=False)
    del weights
    routed = VP.build_routed(n_layers - 1)
    for li in range(1, n_layers):
        model.model.layers[li].mlp.switch_mlp = routed[li - 1].switch_mlp

    def plan(path, module):
        if path.endswith("mlp.gate") or "switch_mlp" in path:
            return False
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if "index_" in path:
            return {"mode": "mxfp8", "group_size": 32, "bits": 8}
        if path == "lm_head" or path.endswith("embed_tokens"):
            return {"group_size": 64, "bits": 6}
        return {"group_size": 64, "bits": 4}

    nn.quantize(model, class_predicate=plan)
    mx.eval(model.parameters())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=1500)
    ap.add_argument("--widths", default="1,2,4,6,8,12,16")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--ab", default="", help="NAME=VALUE: a second arm with this env lever, interleaved per rep")
    a = ap.parse_args()
    ab = tuple(a.ab.split("=", 1)) if a.ab else None
    prev = os.environ.get(ab[0]) if ab else None

    def arm(on: bool):
        if not ab:
            return
        if on:
            os.environ[ab[0]] = ab[1]
        elif prev is None:
            os.environ.pop(ab[0], None)
        else:
            os.environ[ab[0]] = prev
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    model = build(a.layers)
    t = MiniMaxM3DSparkTarget(model)
    ids = np.random.default_rng(1).integers(0, 200000, a.ctx + 32).tolist()
    cache = t.make_cache()
    mx.eval(t.prefill(mx.array([ids[:a.ctx]]), cache, (), 2048)[0])
    res = {}
    arms = ("off", "on") if ab else ("off",)
    for k in [int(w) for w in a.widths.split(",")]:
        toks = mx.array([ids[a.ctx:a.ctx + k]])
        outs = {}
        for name in arms:
            arm(name == "on")
            for _ in range(2):
                outs[name] = t.verify(toks, cache, ())[0]
                mx.eval(outs[name])
                t.rollback(cache, None, 0, k)
        same = all(bool(mx.array_equal(outs[n], outs["off"]).item()) for n in arms)
        tot = {n: [] for n in arms}
        bld = {n: [] for n in arms}
        for rep in range(a.reps):
            for name in (arms if rep % 2 == 0 else arms[::-1]):
                arm(name == "on")
                mx.synchronize()
                t0 = time.perf_counter()
                lg = t.verify(toks, cache, ())[0]
                t1 = time.perf_counter()
                mx.eval(lg)
                t2 = time.perf_counter()
                t.rollback(cache, None, 0, k)
                tot[name].append((t2 - t0) * 1e3)
                bld[name].append((t1 - t0) * 1e3)
        arm(False)
        res[k] = {n: (statistics.median(tot[n]), min(tot[n]), statistics.median(bld[n])) for n in arms}
        print(f"W={k:2d}: " + " | ".join(
            f"{n} verify {res[k][n][0]:7.2f} ms (min {res[k][n][1]:7.2f}, build {res[k][n][2]:5.2f}) = "
            f"{res[k][n][0] / res[1][n][0] if 1 in res else 0:.2f}x W1" for n in arms)
            + (f" | logits equal {same}" if ab else "") + f" [{a.layers} layers]", flush=True)


if __name__ == "__main__":
    main()
