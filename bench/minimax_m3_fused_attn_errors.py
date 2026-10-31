# SPDX-License-Identifier: Apache-2.0
"""Per-kernel error of the MiniMax-M3 fused attention (VMLX_M3_FUSED_ATTN) at M3.1 shapes, synthetic weights.

For each of the five launches: max |error| / max |reference| against a float64 numpy reference of the same math on
the fused path's own inputs, and the same for today's unfused ops on that step (tests/test_minimax_m3_fused_attn.py
``kernel_errors``). K3 also reports the share of (row, KV group) pairs whose top-k block set equals the reference's;
K5 also the error in bf16 ulps of the result. No real model is loaded.

Usage: python bench/minimax_m3_fused_attn_errors.py [--hidden 6144] [--cases 700x1,3001x5,2040x16,9000x8]
"""

from __future__ import annotations

import argparse

from tests.test_minimax_m3_fused_attn import Layer, kernel_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--cases", default="700x1,3001x1,3001x5,2040x16,9000x8,40000x4")
    a = ap.parse_args()
    layer = Layer(hidden=a.hidden)
    for case in a.cases.split(","):
        ctx, rows = (int(v) for v in case.split("x"))
        errs = kernel_errors(layer, ctx, rows)
        print(f"--- hidden {a.hidden}, {ctx} keys before the rows, {rows} rows")
        for name, (fused, today) in errs.items():
            print(f"  {name:36s} fused {fused:.3e}   today's ops {today:.3e}")


if __name__ == "__main__":
    main()
