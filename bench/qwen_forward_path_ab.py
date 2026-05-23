#!/usr/bin/env python3
"""Raw Qwen text-loader vs VLM-language forward microbench.

This diagnostic intentionally does not generate tokens and does not set sampler
or generation defaults. It measures model forward calls over the same token IDs
so the Qwen PP gap can be isolated below the OpenAI/API/scheduler layer.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TEXT_MODEL = "/Users/example/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK"
DEFAULT_VLM_MODEL = "/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
DEFAULT_PROMPT = (
    "This is a raw prompt-processing timing probe for Qwen. "
    "Repeat the context naturally without generating a response. "
)

GENERATION_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "max_tokens",
    "max_new_tokens",
    "enable_thinking",
    "reasoning_effort",
}


def assert_no_generation_kwargs(kwargs: dict[str, Any]) -> None:
    bad = sorted(set(kwargs) & GENERATION_KEYS)
    if bad:
        raise ValueError(
            "qwen_forward_path_ab measures raw forward only; generation/sampler "
            f"kwargs are forbidden: {', '.join(bad)}"
        )


def class_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def forward_target(model: Any) -> tuple[Any, bool]:
    target = getattr(model, "language_model", None)
    if target is not None:
        return target, True
    return model, False


def route_summary(model: Any) -> dict[str, Any]:
    target, used_language_attr = forward_target(model)
    return {
        "model_class": class_name(model),
        "forward_class": class_name(target),
        "uses_language_model_attr": used_language_attr,
        "force_text_rope_1d": bool(getattr(target, "_vmlx_force_text_rope_1d", False)),
        "supports_return_logits": supports_return_logits(target),
        "has_mtp_head": bool(getattr(target, "mtp", None) is not None),
        "has_make_cache": callable(getattr(target, "make_cache", None)),
    }


def supports_return_logits(model: Any) -> bool:
    call = getattr(model, "__call__", None)
    if call is None:
        return False
    try:
        sig = inspect.signature(call)
    except (TypeError, ValueError):
        return False
    return "return_logits" in sig.parameters


def encode_tokens(tokenizer: Any, prompt: str, target_tokens: int) -> list[int]:
    text = prompt
    tokens = tokenizer.encode(text)
    while len(tokens) < target_tokens:
        text += prompt
        tokens = tokenizer.encode(text)
    return [int(token) for token in tokens[:target_tokens]]


def make_cache(model: Any) -> Any:
    if callable(getattr(model, "make_cache", None)):
        return model.make_cache()
    try:
        from mlx_lm.models.cache import KVCache
    except Exception as exc:  # pragma: no cover - live diagnostic fallback
        raise RuntimeError("model lacks make_cache() and KVCache fallback failed") from exc
    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError(f"cannot infer cache layers for {class_name(model)}")
    return [KVCache() for _ in layers]


def call_forward(model: Any, input_ids: Any, cache: Any) -> Any:
    kwargs: dict[str, Any] = {"cache": cache}
    if supports_return_logits(model):
        kwargs["return_logits"] = False
    assert_no_generation_kwargs(kwargs)
    return model(input_ids, **kwargs)


def timed_forward(model: Any, tokens: list[int], repeats: int) -> dict[str, Any]:
    import mlx.core as mx

    target, _ = forward_target(model)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        cache = make_cache(target)
        input_ids = mx.array([tokens], dtype=mx.int32)
        mx.synchronize()
        t0 = time.perf_counter()
        output = call_forward(target, input_ids, cache)
        arrays = [output]
        if hasattr(output, "logits"):
            arrays = [output.logits]
        elif isinstance(output, (tuple, list)):
            arrays = [item for item in output if hasattr(item, "shape")]
        mx.eval(*arrays)
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del cache, input_ids, output, arrays
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        gc.collect()
    best = min(times)
    return {
        "prompt_tokens": len(tokens),
        "times_s": times,
        "best_s": best,
        "best_pp_tok_s": len(tokens) / best if best > 0 else 0.0,
    }


def load_text_model(path: str) -> tuple[Any, Any]:
    from vmlx_engine.utils.tokenizer import load_model_with_fallback

    return load_model_with_fallback(path)


def load_vlm_model(path: str) -> tuple[Any, Any]:
    from vmlx_engine.models.mllm import MLXMultimodalLM

    instance = MLXMultimodalLM(path)
    instance.load()
    return instance.model, instance.processor


def bench_loaded(
    *,
    path: str,
    model: Any,
    token_lists: list[list[int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    route = route_summary(model)
    rows = []
    for tokens in token_lists:
        rows.append(timed_forward(model, tokens, args.repeats))
    return {
        "path": path,
        "route": route,
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results: dict[str, Any] = {
        "started_at": started,
        "generation_defaults_policy": "raw-forward-only; no sampler or generation kwargs are accepted",
        "text_model": args.text_model,
        "vlm_model": args.vlm_model,
        "tokens": args.tokens,
        "repeats": args.repeats,
        "rows": {},
    }

    text_model, text_tokenizer = load_text_model(args.text_model)
    token_lists = [
        encode_tokens(text_tokenizer, args.prompt, int(target_tokens))
        for target_tokens in args.tokens
    ]
    results["rows"]["text_loader"] = bench_loaded(
        path=args.text_model,
        model=text_model,
        token_lists=token_lists,
        args=args,
    )
    del text_model
    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass

    vlm_model, _vlm_processor = load_vlm_model(args.vlm_model)
    results["rows"]["vlm_language"] = bench_loaded(
        path=args.vlm_model,
        model=vlm_model,
        token_lists=token_lists,
        args=args,
    )

    text_rows = results["rows"].get("text_loader", {}).get("rows") or []
    vlm_rows = results["rows"].get("vlm_language", {}).get("rows") or []
    ratios = []
    for text_row, vlm_row in zip(text_rows, vlm_rows):
        text_pp = float(text_row.get("best_pp_tok_s") or 0.0)
        vlm_pp = float(vlm_row.get("best_pp_tok_s") or 0.0)
        ratios.append(
            {
                "prompt_tokens": text_row.get("prompt_tokens"),
                "text_pp_tok_s": text_pp,
                "vlm_pp_tok_s": vlm_pp,
                "text_over_vlm": text_pp / vlm_pp if vlm_pp > 0 else None,
            }
        )
    results["comparison"] = ratios

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
