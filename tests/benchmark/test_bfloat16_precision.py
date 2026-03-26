#!/usr/bin/env python3
"""Quantify precision loss from bfloat16 -> float16 -> bfloat16 round-trip.

This measures the actual impact of the dtype cast used in block disk store
serialization, where bfloat16 KV cache data is cast to float16 (because
safetensors doesn't support bfloat16) and restored on load.

Usage: python tests/benchmark/test_bfloat16_precision.py
"""
import mlx.core as mx
import numpy as np


def measure_roundtrip_loss(data_bf16):
    """Measure loss from bfloat16 -> float16 -> bfloat16."""
    data_f16 = data_bf16.astype(mx.float16)
    data_restored = data_f16.astype(mx.bfloat16)

    original = data_bf16.astype(mx.float32)
    restored = data_restored.astype(mx.float32)
    diff = mx.abs(original - restored)

    # noqa comments: mx.eval is MLX tensor materialization, not Python eval
    mx.eval(original, restored, diff)  # noqa: S307

    max_abs_err = float(mx.max(diff).item())
    mean_abs_err = float(mx.mean(diff).item())

    abs_orig = mx.abs(original)
    nonzero_mask = abs_orig > 1e-10
    mx.eval(nonzero_mask)  # noqa: S307
    if float(mx.sum(nonzero_mask).item()) > 0:
        rel_err = mx.where(nonzero_mask, diff / abs_orig, mx.zeros_like(diff))
        mx.eval(rel_err)  # noqa: S307
        max_rel_err = float(mx.max(rel_err).item())
        mean_rel_err = float(mx.mean(rel_err).item())
    else:
        max_rel_err = 0.0
        mean_rel_err = 0.0

    changed = float(mx.sum(diff > 0).item())
    total = float(data_bf16.size)

    f16_max = 65504.0
    overflow = float(mx.sum(mx.abs(original) > f16_max).item())

    return {
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "max_rel_error": max_rel_err,
        "mean_rel_error": mean_rel_err,
        "pct_changed": 100.0 * changed / total if total > 0 else 0,
        "overflow_count": int(overflow),
        "total_elements": int(total),
    }


def main():
    print("=" * 70)
    print("  bfloat16 -> float16 -> bfloat16 Precision Loss Analysis")
    print("  (for block disk cache serialization)")
    print("=" * 70)

    # ---- Test 1: Random normal (typical KV cache distribution) ----
    print("\n1. Random normal distribution (mean=0, std=1)")
    print("   Simulates typical KV cache attention values")
    data = mx.random.normal((1, 2, 64, 256)).astype(mx.bfloat16)
    mx.eval(data)  # noqa: S307
    r = measure_roundtrip_loss(data)
    print(f"   Elements:       {r['total_elements']:,}")
    print(f"   Max abs error:  {r['max_abs_error']:.2e}")
    print(f"   Mean abs error: {r['mean_abs_error']:.2e}")
    print(f"   Max rel error:  {r['max_rel_error']:.6f} ({r['max_rel_error']*100:.4f}%)")
    print(f"   Mean rel error: {r['mean_rel_error']:.6f} ({r['mean_rel_error']*100:.4f}%)")
    print(f"   Values changed: {r['pct_changed']:.1f}%")
    print(f"   Overflow (>65504): {r['overflow_count']}")

    # ---- Test 2: Larger values (stress test) ----
    print("\n2. Wider distribution (mean=0, std=10)")
    print("   Stress test for larger attention values")
    data = (mx.random.normal((1, 2, 64, 256)) * 10).astype(mx.bfloat16)
    mx.eval(data)  # noqa: S307
    r = measure_roundtrip_loss(data)
    print(f"   Elements:       {r['total_elements']:,}")
    print(f"   Max abs error:  {r['max_abs_error']:.2e}")
    print(f"   Mean abs error: {r['mean_abs_error']:.2e}")
    print(f"   Max rel error:  {r['max_rel_error']:.6f} ({r['max_rel_error']*100:.4f}%)")
    print(f"   Mean rel error: {r['mean_rel_error']:.6f} ({r['mean_rel_error']*100:.4f}%)")
    print(f"   Values changed: {r['pct_changed']:.1f}%")
    print(f"   Overflow (>65504): {r['overflow_count']}")

    # ---- Test 3: Full-scale KV cache (realistic size) ----
    print("\n3. Full-scale KV block (1, 2, 64, 256) x 10 layers")
    print("   Matches Qwen3.5-35B-A3B KV cache dimensions")
    total_r = {
        "max_abs_error": 0, "mean_abs_error": 0, "max_rel_error": 0,
        "mean_rel_error": 0, "pct_changed": 0, "overflow_count": 0,
        "total_elements": 0,
    }
    for _ in range(10):
        keys = mx.random.normal((1, 2, 64, 256)).astype(mx.bfloat16)
        values = mx.random.normal((1, 2, 64, 256)).astype(mx.bfloat16)
        mx.eval(keys, values)  # noqa: S307
        for data in [keys, values]:
            r = measure_roundtrip_loss(data)
            total_r["max_abs_error"] = max(total_r["max_abs_error"], r["max_abs_error"])
            total_r["max_rel_error"] = max(total_r["max_rel_error"], r["max_rel_error"])
            total_r["mean_abs_error"] += r["mean_abs_error"]
            total_r["mean_rel_error"] += r["mean_rel_error"]
            total_r["pct_changed"] += r["pct_changed"]
            total_r["overflow_count"] += r["overflow_count"]
            total_r["total_elements"] += r["total_elements"]
    n = 20  # 10 layers x 2 (keys+values)
    total_r["mean_abs_error"] /= n
    total_r["mean_rel_error"] /= n
    total_r["pct_changed"] /= n
    print(f"   Total elements: {total_r['total_elements']:,}")
    print(f"   Max abs error:  {total_r['max_abs_error']:.2e}")
    print(f"   Mean abs error: {total_r['mean_abs_error']:.2e}")
    print(f"   Max rel error:  {total_r['max_rel_error']:.6f} ({total_r['max_rel_error']*100:.4f}%)")
    print(f"   Mean rel error: {total_r['mean_rel_error']:.6f} ({total_r['mean_rel_error']*100:.4f}%)")
    print(f"   Values changed: {total_r['pct_changed']:.1f}%")
    print(f"   Overflow (>65504): {total_r['overflow_count']}")

    # ---- Explanation ----
    print("\n" + "=" * 70)
    print("  Analysis")
    print("=" * 70)
    print("""
  bfloat16: 1 sign + 8 exponent + 7 mantissa bits (same range as float32)
  float16:  1 sign + 5 exponent + 10 mantissa bits (range +/-65504)

  Round-trip path: bfloat16 -> float16 -> bfloat16

  The round-trip is LOSSLESS for values in the float16 range because:
  - bfloat16 has 7 mantissa bits, float16 has 10
  - All 7 mantissa bits of bfloat16 are preserved in float16's 10 bits
  - Converting back truncates the extra 3 bits (which were zero)

  The ONLY loss occurs for values outside float16's range:
  - |value| > 65504: overflows to +/-inf in float16
  - Very small subnormals: may underflow

  For KV cache data (typically |values| < 10), this is lossless.
""")


if __name__ == "__main__":
    main()
