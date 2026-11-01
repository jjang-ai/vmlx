"""Exact sparse QSA QK kernels and paired component timings across dispatch sizes."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

from vmlx_engine.metal import qwen4_qsa_sparse_kernels as kernels
from vmlx_engine.metal.qwen4_qsa_direct_prefill import (
    direct_attention,
    direct_ready,
    selected_block_buffer,
)


def run(rows, context, dtype, trials, *, stock_path="legacy", candidate="materialized"):
    offset = context - rows
    mx.random.seed(context + rows)
    q = mx.random.normal((1, 24, rows, 256)).astype(dtype)
    k = mx.random.normal((1, 2, context, 256)).astype(dtype)
    v = mx.random.normal(k.shape).astype(dtype)
    complete = mx.arange(offset + 1, context + 1) // 4
    scores = mx.random.uniform(shape=(rows, context // 4))
    scores = mx.where(
        mx.arange(context // 4)[None] < complete[:, None], scores, -mx.inf
    )
    ids = mx.sort(mx.argpartition(-scores, kth=511, axis=-1)[:, :512], axis=-1).astype(
        mx.int32
    )
    valid = ids < complete[:, None]
    blocks = mx.put_along_axis(
        mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
    )
    keep = mx.repeat(blocks, 4, axis=-1)
    if context % 4:
        keep = mx.concatenate(
            [keep, mx.zeros((rows, context % 4), dtype=mx.bool_)], axis=-1
        )
    tokens = mx.arange(context)[None]
    positions = mx.arange(offset, context)[:, None]
    mask = mx.where(
        (keep | (tokens >= complete[:, None] * 4)) & (tokens <= positions), 0, -mx.inf
    ).astype(dtype)[None, None]
    mx.eval(q, k, v, ids, valid, mask)

    def stock():
        if stock_path == "runtime":
            from vmlx_engine.metal.qwen4_prefill_sdpa import qwen4_prefill_sdpa

            out = qwen4_prefill_sdpa(q, k, v, mask, scale=0.0625)
            if out is not None:
                return out
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=mask)

    def direct():
        if candidate == "online":
            # The fused online-softmax kernel; its contract is the current
            # fused SDPA (use --stock-path runtime), not the unfused oracle.
            selected = selected_block_buffer(ids, valid, pos_start=offset)
            return kernels.sparse_online_attention(q, k, v, selected, offset, 0.0625)
        return direct_attention(
            q, k, v, ids, valid, pos_start=offset, total_tokens=context, scale=0.0625
        )

    ref, got = stock(), direct()
    mx.eval(ref, got)
    assert bool(mx.all(mx.isfinite(ref))) and bool(mx.all(mx.isfinite(got)))
    exact = bool(mx.array_equal(ref, got))
    # The legacy oracle is separate QK/softmax/PV; current bulk serving uses
    # force_fused SDPA. Never present legacy equality as current-path parity.
    if stock_path == "legacy":
        assert exact
    diff = got.astype(mx.float32) - ref.astype(mx.float32)
    numerical = {
        "exact": exact,
        "max_abs": float(mx.max(mx.abs(diff))),
        "rms": float(mx.sqrt(mx.mean(mx.square(diff)))),
        "reference_rms": float(mx.sqrt(mx.mean(mx.square(ref.astype(mx.float32))))),
    }
    times = {"stock": [], "direct": []}
    for _ in range(2):
        mx.eval(stock(), direct())
    for trial in range(trials):
        order = [("stock", stock), ("direct", direct)]
        if trial % 2:
            order.reverse()
        for name, function in order:
            mx.synchronize()
            started = time.perf_counter()
            mx.eval(function())
            times[name].append(1000 * (time.perf_counter() - started))
    medians = {name: statistics.median(values) for name, values in times.items()}
    return dict(
        dtype=str(dtype),
        rows=rows,
        context=context,
        stock_path=stock_path,
        candidate=candidate,
        exact=exact,
        numerical=numerical,
        router_eligible=rows >= 256 and context >= 8192,
        ms=times,
        median_ms=medians,
        speedup=medians["stock"] / medians["direct"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--rows", default="256,1024,4096")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--stock-path", choices=("legacy", "runtime"), default="legacy")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--candidate", choices=("materialized", "online"), default="materialized")
    args = parser.parse_args()
    os.environ["VMLX_QWEN4_PREFILL_DIRECT"] = "1"
    mx.set_cache_limit(512 * 1024**2)
    assert direct_ready(), "sparse-score lane unavailable on this device/runtime"
    results = []
    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16}
    for dtype_name in args.dtypes.split(","):
        dtype = dtype_map[dtype_name]
        for rows in map(int, args.rows.split(",")):
            for context in map(int, args.contexts.split(",")):
                result = run(rows, context, dtype, args.trials,
                             stock_path=args.stock_path, candidate=args.candidate)
                results.append(result)
                print(json.dumps(result), flush=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "scope": "component diagnostic only; numerical differences are not acceptance",
                            "kernel_math": kernels.MATH_ABI,
                            "results": results,
                        },
                        indent=2,
                    )
                )
                mx.clear_cache()
