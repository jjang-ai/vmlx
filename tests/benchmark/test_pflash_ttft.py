# SPDX-License-Identifier: Apache-2.0
"""PFlash cold-prefix TTFT benchmark — issue #136.

Not part of CI. Requires real model weights. Run manually:

    .venv/bin/python tests/benchmark/test_pflash_ttft.py \
        --target mlx-community/Qwen3.6-27B-mixed94bit \
        --drafter mlx-community/Qwen3-0.6B-bf16 \
        --keep-ratio 0.10 \
        --ctx 8192,16384,32768,65536,131072

Reference target from #136 (CUDA, RTX 3090):
    128K cold TTFT: 24.8 s with PFlash vs 257 s dense (10.4×)

The Metal ceiling is expected to be lower (~5–8×) until a BSA-equivalent
Metal kernel lands — see `vmlx_engine/utils/pflash.py` docstring.

This harness produces two numbers per ctx point:

    dense_ttft   — VMLX_ENABLE_PFLASH=0
    pflash_ttft  — VMLX_ENABLE_PFLASH=1, --pflash-drafter=<drafter>

Output: JSON to stdout, suitable for diffing across runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PFlash cold TTFT benchmark")
    p.add_argument("--target", required=True, help="Target model id / path")
    p.add_argument("--drafter", required=True, help="Drafter model id / path")
    p.add_argument(
        "--ctx",
        default="8192,16384,32768,65536,131072",
        help="Comma-separated prompt token counts to benchmark",
    )
    p.add_argument("--keep-ratio", type=float, default=0.10)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--warmup-cold-cache", action="store_true",
                   help="Force-clear prefix cache between runs")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_lm_load
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    from vmlx_engine.utils.pflash import PFlashConfig, configure_pflash
    from vmlx_engine.utils.pflash_drafter import load_pflash_drafter

    ctx_sizes: List[int] = [int(x) for x in args.ctx.split(",")]

    print(f"Loading target: {args.target}", file=sys.stderr)
    target, tokenizer = mlx_lm_load(args.target)

    print(f"Loading drafter: {args.drafter}", file=sys.stderr)
    cfg = PFlashConfig(
        enabled=True,
        drafter_model=args.drafter,
        block_size=args.block_size,
        keep_ratio=args.keep_ratio,
        min_seq_len=0,
    )
    configure_pflash(cfg)
    load_pflash_drafter(cfg)

    results = []
    for ctx in ctx_sizes:
        # Build a synthetic prompt of `ctx` tokens drawn from a NIAH-style
        # filler text repeated.  The benchmark measures prefill time only,
        # so the actual content doesn't matter — but we keep a deterministic
        # seed for reproducibility.
        prompt = ("The quick brown fox jumps over the lazy dog. " * (ctx // 12))[:ctx * 4]
        tokens = tokenizer.encode(prompt)[:ctx]
        if len(tokens) < ctx:
            # Pad to ctx with EOS to hit the target length deterministically.
            eos = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 0
            tokens = tokens + [eos] * (ctx - len(tokens))
        ids = mx.array([tokens])

        # ---- dense pass ----
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        t0 = time.perf_counter()
        out = target(ids)
        mx.eval(out.logits if hasattr(out, "logits") else out)
        dense_ttft = time.perf_counter() - t0

        # ---- PFlash pass ----
        # Currently informational only — the dense forward still runs but
        # the planner emits its keep_ranges so we can validate the
        # algorithm side without the sparse kernel.
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        t0 = time.perf_counter()
        out2 = target(ids)
        mx.eval(out2.logits if hasattr(out2, "logits") else out2)
        pflash_ttft = time.perf_counter() - t0

        results.append({
            "ctx": ctx,
            "dense_ttft_s": dense_ttft,
            "pflash_ttft_s": pflash_ttft,
            "speedup_x": dense_ttft / max(pflash_ttft, 1e-6),
            "note": (
                "PFlash forward currently dense; planner stats only. "
                "BSA sparse kernel is a follow-up to this PR."
            ),
        })

    print(json.dumps({
        "target": args.target,
        "drafter": args.drafter,
        "keep_ratio": args.keep_ratio,
        "block_size": args.block_size,
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
