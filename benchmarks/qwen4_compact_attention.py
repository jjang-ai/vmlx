"""Exact stock-order QSA verification and paired component timing."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

from vmlx_engine.metal.qwen4_verify_sdpa import qwen4_verify_sdpa


def inputs(context, rows, dtype, batch=1):
    mx.random.seed(context + rows)
    q = mx.random.normal((batch, 24, rows, 256)).astype(dtype)
    k = mx.random.normal((batch, 2, context, 256)).astype(dtype)
    v = mx.random.normal(k.shape).astype(dtype)
    positions = mx.arange(context - rows, context)
    blocks = mx.arange(context // 4)
    complete = blocks[None, :] < ((positions + 1) // 4)[:, None]
    scores = mx.random.uniform(shape=(batch, rows, context // 4))
    scores = mx.where(complete[None], scores, -1e9)
    top = mx.argpartition(-scores, kth=511, axis=-1)[..., :512]
    keep = mx.put_along_axis(
        mx.zeros(scores.shape, dtype=mx.bool_), top, mx.array(True), axis=-1
    )
    keep = keep & complete[None]
    keep = mx.repeat(keep, 4, axis=-1)
    if context % 4:
        keep = mx.concatenate(
            [keep, mx.zeros((batch, rows, context % 4), dtype=mx.bool_)], axis=-1
        )
    tokens = mx.arange(context)
    keep = keep | (tokens[None, None, :] >= (((positions + 1) // 4) * 4)[None, :, None])
    keep = keep & (tokens[None, None, :] <= positions[None, :, None])
    mask = mx.where(keep[:, None], 0.0, -float("inf")).astype(dtype)
    mx.eval(q, k, v, mask)
    return q, k, v, mask


def run(context, rows, dtype, trials):
    os.environ["VMLX_QWEN4_VERIFY_SDPA"] = "1"
    q, k, v, mask = inputs(context, rows, dtype)

    def stock():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=mask)

    def candidate():
        return qwen4_verify_sdpa(q, k, v, mask, scale=0.0625, selected_token_bound=2051, selected_four_token_block_bound=513)

    ref, got = stock(), candidate()
    assert got is not None
    mx.eval(ref, got)
    assert bool(mx.array_equal(ref, got))
    times = {"stock": [], "candidate": []}
    for _ in range(2):
        mx.eval(stock(), candidate())
    for trial in range(trials):
        order = [("stock", stock), ("candidate", candidate)]
        if trial % 2:
            order.reverse()
        for name, function in order:
            mx.synchronize()
            started = time.perf_counter()
            mx.eval(function())
            times[name].append(1000 * (time.perf_counter() - started))
    medians = {name: statistics.median(values) for name, values in times.items()}
    return dict(
        context=context,
        rows=rows,
        dtype=str(dtype),
        exact=True,
        ms=times,
        median_ms=medians,
        speedup=medians["stock"] / medians["candidate"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--rows", default="3,4")
    args = parser.parse_args()
    results = []
    for dtype in (mx.float16, mx.bfloat16):
        for context in (8192, 16384, 32768):
            for rows in map(int, args.rows.split(",")):
                result = run(context, rows, dtype, args.trials)
                results.append(result)
                print(json.dumps(result), flush=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "scope": "synthetic component; run without competing inference",
                            "device": mx.device_info(),
                            "results": results,
                        },
                        indent=2,
                    )
                )
