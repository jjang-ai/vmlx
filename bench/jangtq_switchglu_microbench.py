#!/usr/bin/env python3
"""Synthetic JANGTQ SwitchGLU decode-kernel microbench.

This does not load a model. It exercises the same MiniMax-shaped decode helpers
used by `jang_tools.load_jangtq`:

    hidden rotate -> fused gate/up/SwiGLU -> intermediate rotate -> down gather

The benchmark is intentionally narrow. It is for ranking kernel work before a
large MiniMax run, not for claiming end-to-end model speed or coherence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from jang_tools.turboquant.fused_gate_up_kernel import (
    make_fused_gate_up_swiglu_decode,
)
from jang_tools.turboquant.gather_tq_kernel import make_gather_tq_decode_per_row
from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal


def _packed_cols(in_features: int, bits: int) -> int:
    vals_per_u32 = 32 // bits
    return (in_features + vals_per_u32 - 1) // vals_per_u32


def _sync(*values: Any) -> None:
    mx.eval(*values)
    mx.synchronize()


def _time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> tuple[float, Any]:
    last = None
    for _ in range(warmup):
        last = fn()
        _sync(last)
    t0 = time.perf_counter()
    for _ in range(iters):
        last = fn()
        _sync(last)
    elapsed = time.perf_counter() - t0
    return elapsed * 1000.0 / max(1, iters), last


def _make_inputs(
    *,
    experts: int,
    hidden_features: int,
    intermediate_features: int,
    bits: int,
    down_bits: int,
    top_k: int,
) -> dict[str, Any]:
    gate_cols = _packed_cols(hidden_features, bits)
    down_cols = _packed_cols(intermediate_features, down_bits)
    cb_gate = mx.array(
        [math.sin(i + 1) * 0.02 for i in range(1 << bits)], dtype=mx.float32
    )
    cb_down = mx.array(
        [math.cos(i + 1) * 0.02 for i in range(1 << down_bits)], dtype=mx.float32
    )
    return {
        "x": mx.ones((1, hidden_features), dtype=mx.float32),
        "signs_hidden": mx.ones((hidden_features,), dtype=mx.float32),
        "signs_inter": mx.ones((intermediate_features,), dtype=mx.float32),
        "idx": mx.arange(top_k, dtype=mx.uint32) % experts,
        "packed_gate": mx.zeros(
            (experts, intermediate_features, gate_cols), dtype=mx.uint32
        ),
        "packed_up": mx.zeros(
            (experts, intermediate_features, gate_cols), dtype=mx.uint32
        ),
        "packed_down": mx.zeros(
            (experts, hidden_features, down_cols), dtype=mx.uint32
        ),
        "norms_gate": mx.ones((experts, intermediate_features), dtype=mx.float16),
        "norms_up": mx.ones((experts, intermediate_features), dtype=mx.float16),
        "norms_down": mx.ones((experts, hidden_features), dtype=mx.float16),
        "cb_gate": cb_gate,
        "cb_down": cb_down,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden-features", type=int, default=3072)
    parser.add_argument("--intermediate-features", type=int, default=1536)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--down-bits", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--artifact", default="")
    args = parser.parse_args()

    data = _make_inputs(
        experts=args.experts,
        hidden_features=args.hidden_features,
        intermediate_features=args.intermediate_features,
        bits=args.bits,
        down_bits=args.down_bits,
        top_k=args.top_k,
    )
    _sync(*data.values())

    fused_gu = make_fused_gate_up_swiglu_decode(
        args.hidden_features, args.intermediate_features, args.bits, args.top_k
    )
    gather_dn = make_gather_tq_decode_per_row(
        args.intermediate_features,
        args.hidden_features,
        args.down_bits,
        args.top_k,
    )

    def hidden_rotate():
        return hadamard_rotate_metal(data["x"], data["signs_hidden"])

    x_rot = hidden_rotate()
    _sync(x_rot)

    def gate_up():
        return fused_gu(
            x_rot,
            data["packed_gate"],
            data["norms_gate"],
            data["packed_up"],
            data["norms_up"],
            data["cb_gate"],
            data["idx"],
        )

    x_act = gate_up()
    _sync(x_act)

    def inter_rotate():
        return hadamard_rotate_metal(x_act, data["signs_inter"])

    x_act_rot = inter_rotate()
    _sync(x_act_rot)

    def down_gather():
        return gather_dn(
            x_act_rot,
            data["packed_down"],
            data["norms_down"],
            data["cb_down"],
            data["idx"],
        )

    def full_split():
        xr = hadamard_rotate_metal(data["x"], data["signs_hidden"])
        xa = fused_gu(
            xr,
            data["packed_gate"],
            data["norms_gate"],
            data["packed_up"],
            data["norms_up"],
            data["cb_gate"],
            data["idx"],
        )
        xar = hadamard_rotate_metal(xa, data["signs_inter"])
        return gather_dn(
            xar,
            data["packed_down"],
            data["norms_down"],
            data["cb_down"],
            data["idx"],
        )

    compiled_full = mx.compile(full_split)

    rows: list[dict[str, Any]] = []
    for name, fn in (
        ("hidden_rotate", hidden_rotate),
        ("fused_gate_up_swiglu", gate_up),
        ("intermediate_rotate", inter_rotate),
        ("down_gather", down_gather),
        ("full_split_compiled", compiled_full),
    ):
        mean_ms, out = _time_ms(fn, warmup=args.warmup, iters=args.iters)
        rows.append({"name": name, "mean_ms": mean_ms, "shape": tuple(out.shape)})

    report = {
        "schema": "jangtq_switchglu_microbench.v1",
        "settings": vars(args),
        "rows": rows,
        "notes": [
            "Synthetic arrays; no model load and no coherence claim.",
            "Use for relative kernel targeting before a large MiniMax bench.",
        ],
    }

    print("component                 mean_ms    output_shape")
    for row in rows:
        print(f"{row['name']:<25} {row['mean_ms']:>8.3f}    {row['shape']}")

    if args.artifact:
        out_path = Path(args.artifact)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
