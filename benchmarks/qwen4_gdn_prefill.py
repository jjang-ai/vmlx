"""Paired, synchronized component gate; not an end-to-end throughput claim."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update as stock

from vmlx_engine.metal.qwen4_gdn_blocked_prefill import (
    qwen4_blocked_gated_delta_update as candidate,
)


def inputs(rows, dtype, batch=1):
    mx.random.seed(2718 + rows)
    # CRACK: Hk=16, Hv=48, Dk=Dv=128. Normalize q/k as the model does.
    q = mx.random.normal((batch, rows, 16, 128)).astype(dtype)
    k = mx.random.normal(q.shape).astype(dtype)
    q = (128**-1) * mx.fast.rms_norm(q, None, 1e-6)
    k = (128**-0.5) * mx.fast.rms_norm(k, None, 1e-6)
    v = mx.random.normal((batch, rows, 48, 128)).astype(dtype)
    a = mx.random.normal((batch, rows, 48)).astype(dtype)
    b = mx.random.normal(a.shape).astype(dtype)
    a_log = mx.zeros((48,), dtype=mx.float32)
    dt = mx.zeros((48,), dtype=dtype)
    state = (mx.random.normal((batch, 48, 128, 128)) * 0.1).astype(mx.float32)
    args = (q, k, v, a, b, a_log, dt, state)
    mx.eval(args)
    return args


def errors(got, want):
    return [
        float(mx.max(mx.abs(g.astype(mx.float32) - w.astype(mx.float32))).item())
        for g, w in zip(got, want)
    ]


def run(rows, dtype, trials):
    args = inputs(rows, dtype)
    os.environ["VMLX_QWEN4_GDN_BLOCKED_PREFILL"] = "1"
    ref = stock(*args)
    got = candidate(*args)
    mx.eval(ref, got)
    err = errors(got, ref)
    # Match the stock lane reduction order for both output and state.
    for g, w in zip(got, ref):
        assert bool(mx.array_equal(g, w)), (rows, str(dtype), err)
    split = rows // 2

    def part(start, end, state):
        return candidate(*(x[:, start:end] for x in args[:5]), *args[5:7], state)

    first = part(0, split, args[-1])
    second = part(split, rows, first[1])
    mx.eval(first, second)
    resumed = (mx.concatenate([first[0], second[0]], axis=1), second[1])
    for g, w in zip(resumed, got):
        assert bool(mx.array_equal(g, w)), ("resume", rows)
    times = {"stock": [], "blocked": []}
    for trial in range(trials):
        order = [("stock", stock), ("blocked", candidate)]
        if trial % 2:
            order.reverse()
        for name, fn in order:
            mx.synchronize()
            start = time.perf_counter()
            result = fn(*args)
            mx.eval(result)
            times[name].append((time.perf_counter() - start) * 1000)
    med = {k: statistics.median(v) for k, v in times.items()}
    return dict(
        rows=rows,
        dtype=str(dtype),
        max_abs_error=err,
        resume_max_abs_error=errors(resumed, got),
        ms=times,
        median_ms=med,
        speedup=med["stock"] / med["blocked"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=7)
    p.add_argument("--rows", default="32,256,1024,4096")
    args = p.parse_args()
    results = []
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        for rows in map(int, args.rows.split(",")):
            result = run(rows, dtype, args.trials)
            results.append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(
                json.dumps(
                    {
                        "device": mx.device_info(),
                        "scope": "synthetic GDN component only; resident service may contend",
                        "results": results,
                    },
                    indent=2,
                )
            )
