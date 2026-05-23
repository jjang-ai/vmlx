#!/usr/bin/env python3
"""Layer-level Qwen text-loader vs VLM-language forward profiler.

This is diagnostic-only. It runs raw forward passes over fixed token IDs and
does not generate, sample, or set generation defaults.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
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
    GENERATION_KEYS,
    assert_no_generation_kwargs,
    class_name,
    encode_tokens,
    forward_target,
    load_text_model,
    load_vlm_model,
    make_cache,
    route_summary,
    supports_return_logits,
)


class LayerProxy:
    def __init__(
        self,
        *,
        route: str,
        index: int,
        layer: Any,
        records: list[dict[str, Any]],
    ) -> None:
        self.route = route
        self.index = index
        self.layer = layer
        self.records = records

    def __getattr__(self, name: str) -> Any:
        return getattr(self.layer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import mlx.core as mx

        mx.synchronize()
        started = time.perf_counter()
        out = self.layer(*args, **kwargs)
        mx.eval(out)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        self.records.append(
            {
                "route": self.route,
                "index": self.index,
                "kind": layer_kind(self.layer),
                "layer_class": class_name(self.layer),
                "elapsed_s": elapsed,
            }
        )
        return out


def layer_kind(layer: Any) -> str:
    return "linear" if bool(getattr(layer, "is_linear", False)) else "attention"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    total = 0.0
    for record in records:
        elapsed = round(float(record.get("elapsed_s") or 0.0), 12)
        total += elapsed
        kind = str(record.get("kind") or "unknown")
        bucket = by_kind.setdefault(kind, {"count": 0, "total_s": 0.0})
        bucket["count"] += 1
        bucket["total_s"] = round(float(bucket["total_s"]) + elapsed, 12)
    return {"total_s": round(total, 12), "by_kind": by_kind}


def wrap_layers(model: Any, route: str) -> tuple[list[Any], list[dict[str, Any]]]:
    target, _ = forward_target(model)
    layers = target.model.layers if hasattr(target, "model") else target.layers
    original_layers = list(layers)
    records: list[dict[str, Any]] = []
    for index, layer in enumerate(original_layers):
        layers[index] = LayerProxy(route=route, index=index, layer=layer, records=records)
    return original_layers, records


def restore_layers(model: Any, original_layers: list[Any]) -> None:
    target, _ = forward_target(model)
    layers = target.model.layers if hasattr(target, "model") else target.layers
    for index, layer in enumerate(original_layers):
        layers[index] = layer


def raw_forward(model: Any, tokens: list[int]) -> None:
    import mlx.core as mx

    target, _ = forward_target(model)
    cache = make_cache(target)
    input_ids = mx.array([tokens], dtype=mx.int32)
    kwargs: dict[str, Any] = {"cache": cache}
    if supports_return_logits(target):
        kwargs["return_logits"] = False
    assert_no_generation_kwargs(kwargs)
    out = target(input_ids, **kwargs)
    mx.eval(out)
    mx.synchronize()


def profile_loaded(route: str, model: Any, tokens: list[int]) -> dict[str, Any]:
    original_layers, records = wrap_layers(model, route)
    try:
        started = time.perf_counter()
        raw_forward(model, tokens)
        total_s = time.perf_counter() - started
    finally:
        restore_layers(model, original_layers)
    return {
        "route": route,
        "route_summary": route_summary(model),
        "prompt_tokens": len(tokens),
        "total_s": total_s,
        "pp_tok_s": len(tokens) / total_s if total_s > 0 else 0.0,
        "summary": summarize_records(records),
        "layers": records,
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
    text_result = profile_loaded("text_loader", text_model, tokens)
    del text_model
    gc.collect()

    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass

    vlm_model, _processor = load_vlm_model(args.model)
    vlm_result = profile_loaded("vlm_language", vlm_model, tokens)
    result = {
        "model": args.model,
        "generation_defaults_policy": "raw-forward-only; no sampler or generation kwargs are accepted",
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
