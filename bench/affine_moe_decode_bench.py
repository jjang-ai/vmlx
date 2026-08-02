#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Affine MoE decode kernel microbench.

Synthetic benchmark — does NOT load a model. Exercises the fused
dequant+matvec kernels from vmlx_engine.metal.affine_moe_decode against
MLX's stock gather_qmm baseline, using DeepSeek-V4-Flash JANG shapes.

Usage:
    python bench/affine_moe_decode_bench.py [--experts 256] [--k 6] [--iters 200]

Reports per-projection latency and total MoE-layer estimate.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import mlx.core as mx

from vmlx_engine.metal.affine_moe_decode import AffineMoEDecodeManager


def _sync(*values: Any) -> None:
    mx.eval(*values)
    mx.synchronize()


def _time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _sync(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        _sync(fn())
    return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def _make_affine_weights(
    n_experts: int, out_dim: int, in_dim: int, bits: int, group_size: int
) -> tuple[mx.array, mx.array, mx.array]:
    """Create synthetic affine-quantized weights matching MLX layout."""
    n_groups = in_dim // group_size
    vals_per_u32 = 32 // bits
    packed_cols = in_dim // vals_per_u32
    code_mask = (1 << bits) - 1

    weights = mx.random.uniform(
        shape=(n_experts, out_dim, packed_cols), low=0, high=2**32 - 1
    ).astype(mx.uint32)
    # Mask to valid code range
    weights = weights & code_mask
    scales = mx.random.uniform(shape=(n_experts, out_dim, n_groups), low=0.001, high=0.1)
    biases = mx.random.uniform(shape=(n_experts, out_dim, n_groups), low=-0.05, high=0.05)
    return weights.astype(mx.float16).astype(mx.uint32), scales.astype(mx.float16), biases.astype(mx.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--k", type=int, default=6, help="Top-k experts per token")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--inter", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    E, k, hidden, inter = args.experts, args.k, args.hidden, args.inter

    print(f"Affine MoE decode microbench")
    print(f"  experts={E}  k={k}  hidden={hidden}  inter={inter}")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print()

    mgr = AffineMoEDecodeManager(n_chunks=2, threadgroup_y=128)

    # DSV4-Flash shapes: gate/up are hidden→inter (2-bit g64 or 3-bit g64),
    # down is inter→hidden (2-bit g32)
    gate_w, gate_s, gate_b = _make_affine_weights(E, inter, hidden, 3, 64)
    up_w, up_s, up_b = _make_affine_weights(E, inter, hidden, 2, 64)
    down_w, down_s, down_b = _make_affine_weights(E, hidden, inter, 2, 32)

    x = mx.random.normal(shape=(hidden,)).astype(mx.float16)
    expert_ids = mx.arange(k).astype(mx.uint16)

    # Warm up kernel compilation
    _ = mgr.run_projection(x, gate_w, gate_s, gate_b, expert_ids, hidden, inter, hidden // 64, k, 3, 64)
    _sync(_)

    # Benchmark each projection
    def bench_gate():
        return mgr.run_projection(x, gate_w, gate_s, gate_b, expert_ids, hidden, inter, hidden // 64, k, 3, 64)

    def bench_up():
        return mgr.run_projection(x, up_w, up_s, up_b, expert_ids, hidden, inter, hidden // 64, k, 2, 64)

    def bench_down():
        act = mx.random.normal(shape=(k, inter)).astype(mx.float16)
        return mgr.run_projection(act, down_w, down_s, down_b, expert_ids, inter, hidden, inter // 32, k, 2, 32)

    gate_ms = _time_ms(bench_gate, warmup=args.warmup, iters=args.iters)
    up_ms = _time_ms(bench_up, warmup=args.warmup, iters=args.iters)
    down_ms = _time_ms(bench_down, warmup=args.warmup, iters=args.iters)

    total = gate_ms + up_ms + down_ms
    print(f"  gate_proj (3-bit g64): {gate_ms:7.3f} ms")
    print(f"  up_proj   (2-bit g64): {up_ms:7.3f} ms")
    print(f"  down_proj (2-bit g32): {down_ms:7.3f} ms")
    print(f"  ─────────────────────────────────")
    print(f"  total MoE layer:       {total:7.3f} ms")
    print()

    # Estimate full model throughput (43 layers, MoE ~62% of decode time)
    est_moe_per_token = total * 43
    est_total_per_token = est_moe_per_token / 0.62
    est_tps = 1000.0 / est_total_per_token
    print(f"  Estimated (43 layers, MoE=62% of decode):")
    print(f"    MoE per token:  {est_moe_per_token:.1f} ms")
    print(f"    Total per token: {est_total_per_token:.1f} ms")
    print(f"    Throughput:      {est_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
