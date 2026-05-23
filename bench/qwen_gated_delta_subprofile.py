#!/usr/bin/env python3
"""Qwen text-loader vs VLM-language GatedDelta submodule profiler.

This diagnostic measures raw forward passes over fixed token IDs. It does not
generate tokens and rejects sampler/generation kwargs. The purpose is to
localize Qwen3.5/3.6 hybrid SSM prompt-processing gaps below the scheduler,
especially when the same artifact is fast through mlx-lm's text copy and slow
through mlx-vlm's language copy.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from qwen_forward_path_ab import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_VLM_MODEL,
    assert_no_generation_kwargs,
    encode_tokens,
    forward_target,
    load_text_model,
    load_vlm_model,
    make_cache,
    route_summary,
    supports_return_logits,
)


TIMED_OPS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "conv1d",
    "norm",
    "out_proj",
)


class TimedModule:
    def __init__(
        self,
        *,
        route: str,
        layer_index: int,
        op_name: str,
        module: Any,
        records: list[dict[str, Any]],
    ) -> None:
        self.route = route
        self.layer_index = layer_index
        self.op_name = op_name
        self.module = module
        self.records = records

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import mlx.core as mx

        mx.synchronize()
        started = time.perf_counter()
        out = self.module(*args, **kwargs)
        mx.eval(out)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        self.records.append(
            {
                "route": self.route,
                "layer": self.layer_index,
                "op": self.op_name,
                "elapsed_s": elapsed,
            }
        )
        return out


def _layers(model: Any) -> Any:
    target, _ = forward_target(model)
    if hasattr(target, "model"):
        return target.model.layers
    return target.layers


def install_timers(
    route: str,
    model: Any,
) -> tuple[list[dict[str, Any]], list[tuple[Any, str, Any]]]:
    records: list[dict[str, Any]] = []
    restorers: list[tuple[Any, str, Any]] = []
    for index, layer in enumerate(_layers(model)):
        if not bool(getattr(layer, "is_linear", False)):
            continue
        gated_delta = getattr(layer, "linear_attn", None)
        if gated_delta is None:
            continue
        for op_name in TIMED_OPS:
            if not hasattr(gated_delta, op_name):
                continue
            original = getattr(gated_delta, op_name)
            setattr(
                gated_delta,
                op_name,
                TimedModule(
                    route=route,
                    layer_index=index,
                    op_name=op_name,
                    module=original,
                    records=records,
                ),
            )
            restorers.append((gated_delta, op_name, original))
    return records, restorers


def restore_timers(restorers: list[tuple[Any, str, Any]]) -> None:
    for obj, name, original in restorers:
        setattr(obj, name, original)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_op: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "total_s": 0.0})
    for record in records:
        op = str(record.get("op") or "unknown")
        bucket = by_op[op]
        bucket["count"] += 1
        bucket["total_s"] = round(
            float(bucket["total_s"]) + float(record.get("elapsed_s") or 0.0),
            12,
        )

    summary: dict[str, dict[str, Any]] = {}
    for op, bucket in sorted(by_op.items()):
        count = int(bucket["count"])
        total = float(bucket["total_s"])
        summary[op] = {
            "count": count,
            "total_s": round(total, 9),
            "avg_s": round(total / count, 9) if count else 0.0,
        }
    return summary


def raw_forward(model: Any, tokens: list[int]) -> float:
    import mlx.core as mx

    target, _ = forward_target(model)
    cache = make_cache(target)
    input_ids = mx.array([tokens], dtype=mx.int32)
    kwargs: dict[str, Any] = {"cache": cache}
    if supports_return_logits(target):
        kwargs["return_logits"] = False
    assert_no_generation_kwargs(kwargs)

    mx.synchronize()
    started = time.perf_counter()
    out = target(input_ids, **kwargs)
    mx.eval(out)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    del cache, input_ids, out
    return elapsed


def profile_route(route: str, model: Any, tokens: list[int]) -> dict[str, Any]:
    records, restorers = install_timers(route, model)
    try:
        elapsed = raw_forward(model, tokens)
    finally:
        restore_timers(restorers)
    return {
        "route": route,
        "route_summary": route_summary(model),
        "prompt_tokens": len(tokens),
        "elapsed_s": elapsed,
        "pp_tok_s": len(tokens) / elapsed if elapsed > 0 else 0.0,
        "summary": summarize_records(records),
        "records": records,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    text_model, tokenizer = load_text_model(args.model)
    tokens = encode_tokens(tokenizer, args.prompt, int(args.tokens))
    text_result = profile_route("text_loader", text_model, tokens)
    del text_model
    gc.collect()

    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass

    vlm_model, _processor = load_vlm_model(args.model)
    vlm_result = profile_route("vlm_language", vlm_model, tokens)
    result = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "generation_defaults_policy": (
            "raw-forward-only; no sampler or generation kwargs are accepted"
        ),
        "text_loader": text_result,
        "vlm_language": vlm_result,
        "comparison": {
            "text_pp_tok_s": text_result["pp_tok_s"],
            "vlm_pp_tok_s": vlm_result["pp_tok_s"],
            "text_over_vlm": (
                text_result["pp_tok_s"] / vlm_result["pp_tok_s"]
                if vlm_result["pp_tok_s"] > 0
                else None
            ),
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
