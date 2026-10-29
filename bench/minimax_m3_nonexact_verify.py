# SPDX-License-Identifier: Apache-2.0
"""REFERENCE ONLY (not shipped): a NON-exact MiniMax-M3 DSpark verify through the stock multi-row paths.

The exact verify gives every row a one-row decode step's arithmetic. This measures what dropping that costs in
quality and buys in speed, so the exact kernels can be judged against it:

  --part cost      W-cost curve at M3.1 width (synthetic weights, the bench/minimax_m3_flat_verify.py builders):
                   the model's own multi-row forward per component -- attention (Attention.__call__ with the causal
                   mask: MLX's multi-row quantized matmuls + the multi-row MSA path), MoE block (SparseMoeBlock:
                   router matmul, group_expert_select, JTMixedSwitchGLU.routed, shared expert), lm_head, dense MLPs.
  --part identity  how often greedy DSpark with the stock verify leaves plain greedy decoding (tiny synthetic M3 with
                   M3.1's formats, tests/test_minimax_m3_fused_attn.py's model; drafts = the AR continuation with
                   every fifth token wrong): per prompt, the first diverging token; and per verify row, the logit
                   gap to the decode step and whether the argmax differs. A random tiny model has many near-ties, so
                   its divergence rate is an upper bound for the real bundle's.

  python bench/minimax_m3_nonexact_verify.py --part cost --rows 1,2,4,6,8,12,16
  python bench/minimax_m3_nonexact_verify.py --part identity --prompts 12 --tokens 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def part_cost(rows_list, rounds, out):
    import minimax_m3_flat_verify as FV
    from vmlx_engine.models.minimax_m3 import minimax_m3 as M

    FV.ARMS.clear()
    FV.ARMS["stock"] = {"VMLX_M3_FUSED_ATTN": None, "VMLX_M3_FUSED_MOE": None, "VMLX_JANGT_FAST": "1",
                        "VMLX_M3_FLAT_DENSE": None}
    layers = FV.AB.make_layers(8, seed=11)
    caches = [FV.AB.make_cache(1500, 50 + i) for i in range(8)]
    blocks = FV.VP.build_moe(4)
    head = FV.VP._affine(FV.D, FV.VOCAB, 6, seed=7)
    dense = [FV._MLP(FV.VP._affine(FV.D, 12288, 4, seed=40 + li), FV.VP._affine(FV.D, 12288, 4, seed=50 + li),
                     FV.VP._affine(12288, FV.D, 4, seed=60 + li)) for li in range(3)]
    mx.eval(head.parameters())
    res = {}
    for rows in rows_list:
        x = FV.VP._x(rows, FV.D, rows)
        xi = FV.VP._x(rows, 12288, rows + 1)

        def make():
            def attn():
                outs = []
                for L, c in zip(layers, caches):
                    mask = M.create_attention_mask(x, c)
                    outs.append(x + L.self_attn(L.input_layernorm(x), mask, c))

                def post():
                    for c in caches:
                        c.trim(rows)
                return outs, post

            def moe():
                return [x + b(x) for b in blocks]

            def hd():
                return [head(x)]

            def dn():
                return [m.down_proj(M.swiglu(m.up_proj(x), m.gate_proj(x), 1.702, 7.0)) for m in dense]

            return {"attn": attn, "moe": moe, "head": hd, "dense3": dn}

        t = FV.interleaved_arms(make, rounds)["stock"]
        r = {"attn_60": round(t["attn"] * 60 / 8, 2), "moe_57": round(t["moe"] * 57 / 4, 2),
             "head": round(t["head"], 2), "dense3": round(t["dense3"], 2)}
        r["total"] = round(sum(r.values()), 2)
        res[rows] = r
        print(f"stock W={rows:2d}: attention {r['attn_60']:7.2f} MoE {r['moe_57']:7.2f} head {r['head']:5.2f} "
              f"dense {r['dense3']:5.2f} total {r['total']:7.2f} ms = {r['total'] / res[rows_list[0]]['total']:.2f}x",
              flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))


class StockVerifyTarget:
    """Wraps MiniMaxM3DSparkTarget: verify = the model's own multi-row forward (what a prompt chunk runs)."""

    def __init__(self, inner):
        self._t = inner

    def __getattr__(self, name):
        return getattr(self._t, name)

    def verify(self, tokens, cache, taps):
        logits, feats = self._t._stock(tokens, cache, taps, logits=True)
        return logits[0], feats, None


def part_identity(n_prompts, n_tokens, out):
    import os

    from tests import test_minimax_m3_dspark as TD
    from tests import test_minimax_m3_fused_attn as TA
    from vmlx_engine.dspark import generate_ar, generate_dspark, load_drafter
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget
    import tempfile

    for k in ("VMLX_M3_FUSED_ATTN", "VMLX_M3_FUSED_MOE", "VMLX_M3_FLAT_DENSE"):
        os.environ.pop(k, None)                   # stock decode; the exact verify = row-exact per-row arithmetic
    exact = MiniMaxM3DSparkTarget(TA._model())
    stock = StockVerifyTarget(exact)
    ddir = TD._save_release_drafter(Path(tempfile.mkdtemp()) / "dspark")
    res = {"prompts": [], "rows": {"n": 0, "argmax_differs": 0, "max_abs_dlogit": []}}
    for p in range(n_prompts):
        prompt = TD._ids(300 + 37 * p, 100 + p)
        ref = generate_ar(exact, prompt, n_tokens, set(), prefill_step=256)
        outs = {}
        for name, tgt in (("exact", exact), ("stock", stock)):
            drafter = TD._Scripted(load_drafter(ddir), prompt + ref, TD._every_fifth)
            outs[name] = generate_dspark(tgt, drafter, prompt, n_tokens, set(), prefill_step=256,
                                         max_draft=2 * TD.DRAFT_BLOCK, ar_fallback=TD._WidthOnly())
        first = {k: next((i for i, (a, b) in enumerate(zip(ref, v)) if a != b), None) for k, v in outs.items()}
        res["prompts"].append({"prompt_len": len(prompt), "first_divergence": first})
        # per-row logits: one 8-row stock verify vs 8 decode steps along the reference
        rows = mx.array([ref[:8]])
        c1, c2 = exact.make_cache(), exact.make_cache()
        exact.prefill(mx.array([prompt]), c1, (), 256)
        exact.prefill(mx.array([prompt]), c2, (), 256)
        lg = stock.verify(rows, c1, ())[0]
        steps = [exact.decode(rows[0, r:r + 1], c2) for r in range(8)]
        for r in range(8):
            a, b = lg[r].astype(mx.float32), steps[r].astype(mx.float32)
            res["rows"]["n"] += 1
            res["rows"]["argmax_differs"] += int(int(mx.argmax(a)) != int(mx.argmax(b)))
            res["rows"]["max_abs_dlogit"].append(float(mx.abs(a - b).max()))
        print(f"prompt {p}: first divergence from AR: exact {first['exact']}, stock {first['stock']}", flush=True)
    d = res["rows"]["max_abs_dlogit"]
    res["summary"] = {
        "prompts": n_prompts, "tokens": n_tokens,
        "stock_runs_diverged": sum(1 for x in res["prompts"] if x["first_divergence"]["stock"] is not None),
        "exact_runs_diverged": sum(1 for x in res["prompts"] if x["first_divergence"]["exact"] is not None),
        "median_first_divergence_stock": statistics.median(
            [x["first_divergence"]["stock"] for x in res["prompts"] if x["first_divergence"]["stock"] is not None]
            or [-1]),
        "row_argmax_differs": f"{res['rows']['argmax_differs']} / {res['rows']['n']}",
        "row_max_abs_dlogit_median": round(statistics.median(d), 4), "row_max_abs_dlogit_max": round(max(d), 4),
    }
    print(json.dumps(res["summary"], indent=1), flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="cost", choices=("cost", "identity"))
    ap.add_argument("--rows", default="1,2,4,6,8,12,16")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.part == "cost":
        part_cost([int(r) for r in a.rows.split(",")], a.rounds, a.out)
    else:
        part_identity(a.prompts, a.tokens, a.out)


if __name__ == "__main__":
    main()
