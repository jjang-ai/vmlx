#!/usr/bin/env python3
"""PLD step overhead micro-benchmark (no model forward).

Measures pure Python + MLX bookkeeping cost of _step_speculative using
an instant-return mock model. Isolates overhead from model forward time.

Usage:
    python tests/benchmark/bench_pld_step_overhead.py
    python tests/benchmark/bench_pld_step_overhead.py --batch-size 4 --k 4 --layers 32
    python tests/benchmark/bench_pld_step_overhead.py --iterations 1000
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import mlx.core as mx


# ---------------------------------------------------------------------------
# Mock fixtures (adapted from test_mllm_pld_tq_and_invariants.py)
# ---------------------------------------------------------------------------


class _InstantModel:
    """Model that returns pre-computed logits instantly."""

    def __init__(self, V: int = 100):
        self.V = V
        self._call_count = 0

    def __call__(self, input_tokens, cache=None):
        B, T = input_tokens.shape
        logits = mx.random.normal(shape=(B, T, self.V))
        self._call_count += 1
        if cache:
            for c in cache:
                if hasattr(c, "offset") and isinstance(c.offset, mx.array):
                    c.offset = c.offset + T
        return logits


class _FakeKVLayer:
    def __init__(self, B, max_seq, offsets=None, head_dim=64, n_heads=2):
        self.keys = mx.zeros((B, n_heads, max_seq, head_dim))
        self.values = mx.zeros((B, n_heads, max_seq, head_dim))
        self.offset = mx.array(offsets if offsets is not None else [max_seq] * B)

    def is_trimmable(self) -> bool:
        return True


class _FakeReq:
    def __init__(self):
        self.last_token = 5
        self.num_tokens = 50
        self.output_tokens = list(range(50))
        self.input_ids = mx.array(list(range(100)))
        self.max_tokens = 512
        self.scratch_extra_tokens = None
        self._pld_ngram_index = None
        self._cached_prompt_token_ids = None


class _FakeBatch:
    def __init__(self, requests, cache, y):
        self.requests = requests
        self.cache = cache
        self.y = y
        self.logprobs = [None] * len(requests)


def _make_gen(language_model, B):
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    Gen = MLLMBatchGenerator
    gen = Gen.__new__(Gen)
    gen.language_model = language_model
    gen._pld_spec_enabled = True
    gen._pld_excluded_token_ids = None
    gen._spec_batched_steps = 0
    gen._spec_batched_tokens = 0
    gen._spec_batched_acceptance_ema = 0.0
    gen._spec_batched_accept_histogram = {}
    gen._spec_batched_min_acceptance = 0.30
    gen._spec_batched_warmup_steps = 20
    gen._spec_batched_probe_interval = 200
    gen._spec_batched_cooldown = 0
    gen._spec_batched_cooldown_count = 0
    gen._spec_batched_debug_remaining = 0
    gen._pld_replay_attempts = 0
    gen._pld_replay_emitted = 0
    gen._pld_replay_failures = 0
    gen._pld_replay_enabled = True
    gen._is_hybrid = False

    def _fallback_step(input_tokens, cache):
        n = input_tokens.shape[0] if hasattr(input_tokens, "shape") else 1
        return mx.zeros((n,), dtype=mx.int32), [None] * n
    gen._step = _fallback_step
    return gen


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------


def bench_fn(fn, iterations: int, warmup: int = 50):
    """Time a function, return (median_ms, p95_ms, min_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    mn = times[0]
    return median, p95, mn


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------


def bench_accept_reject(B: int, K: int, iterations: int):
    """Time argmax + tolist + Python accept loop on pre-computed logits."""
    V = 100
    logits = mx.random.normal(shape=(B, K + 1, V))
    mx.eval(logits)
    drafts = [[j for j in range(K)] for _ in range(B)]

    def run():
        predicted = mx.argmax(logits, axis=-1)
        predicted_list = predicted.tolist()
        n_accept_list = []
        for i in range(B):
            n = 0
            for j in range(K):
                if predicted_list[i][j] == drafts[i][j]:
                    n += 1
                else:
                    break
            n_accept_list.append(n)
        return n_accept_list

    return bench_fn(run, iterations)


def bench_kv_rewind_vectorized(B: int, L: int, iterations: int):
    """Time vectorized offset rewind across L layers."""
    layers = [_FakeKVLayer(B, 512, offsets=[256] * B) for _ in range(L)]
    shortfalls = [2] * B
    shortfall_arr = mx.array(shortfalls)

    def run():
        for layer in layers:
            layer.offset = mx.maximum(layer.offset - shortfall_arr, 0)
        # Force eval to measure actual compute
        mx.eval(layers[0].offset)

    return bench_fn(run, iterations)


def bench_kv_rewind_scalar(B: int, L: int, iterations: int):
    """Time old scalar offset rewind (tolist/list-comp/mx.array) for comparison."""
    layers = [_FakeKVLayer(B, 512, offsets=[256] * B) for _ in range(L)]
    shortfalls = [2] * B

    def run():
        for layer in layers:
            cur = layer.offset.tolist()
            if not isinstance(cur, list):
                cur = [cur]
            new_off = [max(0, int(cur[i]) - shortfalls[i]) for i in range(B)]
            layer.offset = mx.array(new_off)

    return bench_fn(run, iterations)


def bench_kv_writeback_inplace(B: int, L: int, iterations: int):
    """Time in-place KV writeback (new Opt 2 path)."""
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    H, D, max_seq = 2, 64, 512
    layers = [_FakeKVLayer(B, max_seq, offsets=[256] * B, head_dim=D, n_heads=H)
              for _ in range(L)]
    solo = _FakeKVLayer(1, 10, offsets=[10], head_dim=D, n_heads=H)
    solo.keys = mx.random.normal(shape=(1, H, 10, D))
    solo.values = mx.random.normal(shape=(1, H, 10, D))
    mx.eval(solo.keys, solo.values)

    def run():
        for layer in layers:
            MLLMBatchGenerator._writeback_kv_row(layer, solo, row_idx=0)

    if B > 1:
        return bench_fn(run, iterations)
    return (0.0, 0.0, 0.0)  # B=1 uses direct replacement, not writeback


def bench_total_step(B: int, K: int, L: int, iterations: int):
    """Time full _step_speculative with instant model (total overhead)."""
    model = _InstantModel(V=100)
    gen = _make_gen(model, B)
    reqs = [_FakeReq() for _ in range(B)]
    cache = [_FakeKVLayer(B, 512, offsets=[256] * B) for _ in range(L)]
    y = mx.array([5] * B, dtype=mx.uint32)
    batch = _FakeBatch(reqs, cache, y)

    # _step_speculative needs drafts — it will fall back to _step if none found.
    # For total overhead, we measure the fallback path (still exercises guard
    # checks, TQ detection, prefill check, draft gather attempt).
    def run():
        gen._step_speculative(batch, K=K)

    return bench_fn(run, iterations)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PLD step overhead benchmark")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: run B=1 and B=4)")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--layers", type=int, default=32)
    args = parser.parse_args()

    batch_sizes = [args.batch_size] if args.batch_size else [1, 4]

    print("PLD Step Overhead Benchmark")
    print("=" * 60)
    print(f"Config: K={args.k}, L={args.layers}, iterations={args.iterations}")
    print()

    header = f"{'Component':<30}"
    for B in batch_sizes:
        header += f"  B={B} med(ms)  B={B} p95(ms)"
    print(header)
    print("─" * (30 + len(batch_sizes) * 26))

    components = [
        ("Accept/reject", lambda B: bench_accept_reject(B, args.k, args.iterations)),
        ("KV rewind (vectorized)", lambda B: bench_kv_rewind_vectorized(B, args.layers, args.iterations)),
        ("KV rewind (scalar, old)", lambda B: bench_kv_rewind_scalar(B, args.layers, args.iterations)),
        ("KV writeback (B>1 only)", lambda B: bench_kv_writeback_inplace(B, args.layers, args.iterations)),
        ("Total step (w/ fallback)", lambda B: bench_total_step(B, args.k, args.layers, args.iterations)),
    ]

    for name, fn in components:
        row = f"{name:<30}"
        for B in batch_sizes:
            med, p95, mn = fn(B)
            row += f"  {med:>9.3f}    {p95:>9.3f}"
        print(row)

    print()
    print("Note: Model forward time excluded (instant mock).")
    print("      Real-world step = overhead + model_forward (~30ms at 30 tok/s).")


if __name__ == "__main__":
    main()
