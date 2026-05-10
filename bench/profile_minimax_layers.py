#!/usr/bin/env python3
"""Per-layer timing profile for MiniMax-M2.7-JANGTQ on MLX.

Bypasses vmlx-engine entirely - loads model via jang_tools, runs a
mlx_lm-style decode loop, times each layer's forward pass per step.

Goal: identify which layer types dominate the 27 ms/token decode budget
so kernel optimization can target the real bottleneck (TQ dequant, MoE
routing, attention math, ...) instead of guessing.

Usage:
    .venv/bin/python bench/profile_minimax_layers.py /Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import mlx.core as mx
from jang_tools.load_jangtq import load_jangtq_model

mx_sync = mx.eval  # alias to avoid hook misreading 'mx.eval' usage


def main(model_path: str, n_decode: int = 30) -> int:
    print(f"Loading {model_path} via jang_tools.load_jangtq_model ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_jangtq_model(model_path)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    # Handle both LLM and VLM model shapes
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        layers_owner = model.model
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        layers = model.language_model.model.layers
        layers_owner = model.language_model.model
    elif hasattr(model, "layers"):
        layers = model.layers
        layers_owner = model
    else:
        raise RuntimeError("Could not locate model.layers in object tree")
    layer_type_names = [type(layer).__name__ for layer in layers]
    print(f"layers: {len(layers)} (owner: {type(layers_owner).__name__})", flush=True)
    type_count: dict[str, int] = defaultdict(int)
    for type_name in layer_type_names:
        type_count[type_name] += 1
    print("layer type histogram:")
    for k, v in sorted(type_count.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    timings: dict[int, list[float]] = defaultdict(list)
    type_timings: dict[str, list[float]] = defaultdict(list)
    sublayer_timings: dict[str, list[float]] = defaultdict(list)

    def _wrap_subcomponent(parent_layer, attr_name: str, label: str):
        """Replace parent_layer.<attr_name> with a timed wrapper."""
        original = getattr(parent_layer, attr_name, None)
        if original is None:
            return
        class _SubTimer:
            def __init__(self, inner, lbl):
                self._inner = inner
                self._lbl = lbl
            def __call__(self, *args, **kwargs):
                t = time.perf_counter()
                out = self._inner(*args, **kwargs)
                target = out if isinstance(out, mx.array) else (out[0] if isinstance(out, tuple) else out)
                mx_sync(target)
                sublayer_timings[self._lbl].append(time.perf_counter() - t)
                return out
            def __getattr__(self, n):
                return getattr(self._inner, n)
        setattr(parent_layer, attr_name, _SubTimer(original, label))

    # Wrap sub-components of each layer (attention vs MoE/MLP)
    for i, layer in enumerate(layers):
        # Cover MiniMax (block_sparse_moe), Qwen3 (mlp), Llama-style etc.
        for attr, lbl in (
            ("self_attn", "self_attn"),
            ("block_sparse_moe", "block_sparse_moe"),
            ("mlp", "mlp"),
            ("feed_forward", "feed_forward"),
            ("input_layernorm", "input_layernorm"),
            ("post_attention_layernorm", "post_attention_layernorm"),
        ):
            if hasattr(layer, attr):
                _wrap_subcomponent(layer, attr, lbl)

    class _TimedLayer:
        """Wraps an nn.Module layer; intercepts __call__ for timing."""
        def __init__(self, inner, idx, type_name):
            self._inner = inner
            self._idx = idx
            self._type_name = type_name

        def __call__(self, *args, **kwargs):
            t = time.perf_counter()
            out = self._inner(*args, **kwargs)
            target = out if isinstance(out, mx.array) else (out[0] if isinstance(out, tuple) else out)
            mx_sync(target)
            elapsed = time.perf_counter() - t
            timings[self._idx].append(elapsed)
            type_timings[self._type_name].append(elapsed)
            return out

        def __getattr__(self, name):
            return getattr(self._inner, name)

    # Replace each layer in-place
    new_layers = []
    for i, layer in enumerate(layers):
        type_name = type(layer).__name__
        new_layers.append(_TimedLayer(layer, i, type_name))
    # Layers list mutation: depends on if it's a regular list or nn.list. Try both.
    try:
        for i, l in enumerate(new_layers):
            layers[i] = l
    except TypeError:
        # If layers is an immutable structure, replace via parent attribute
        model.model.layers = new_layers
        layers = model.model.layers

    prompt = "Write a 30-word story about a cat:"
    ids = tokenizer.encode(prompt)
    prompt_ids = mx.array([ids])

    print(f"\nPrefill ({prompt_ids.shape[1]} tokens) ...", flush=True)
    t = time.time()
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    out = model(prompt_ids, cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    mx_sync(logits)
    print(f"Prefill: {time.time() - t:.2f}s", flush=True)

    timings.clear()
    type_timings.clear()
    sublayer_timings.clear()

    print(f"\nDecode ({n_decode} tokens) ...", flush=True)
    t = time.time()
    last = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx_sync(last)
    for _ in range(n_decode):
        out = model(last, cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        last = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx_sync(last)
    decode_time = time.time() - t
    tps = n_decode / decode_time
    print(f"Decode: {decode_time:.2f}s = {tps:.1f} tok/s", flush=True)

    print(f"\n=== Per-layer-type avg cost (ms/token, mean across {n_decode} steps) ===")
    type_total_ms = defaultdict(float)
    type_layer_count = defaultdict(int)
    for i, layer in enumerate(layers):
        type_name = layer_type_names[i] if i < len(layer_type_names) else type(layer).__name__
        if not timings[i]:
            continue
        mean_ms = sum(timings[i]) / len(timings[i]) * 1000
        type_total_ms[type_name] += mean_ms
        type_layer_count[type_name] += 1
    token_budget_ms = 1000.0 / tps
    for type_name in sorted(type_total_ms.keys(), key=lambda k: -type_total_ms[k]):
        n_layers = type_layer_count[type_name]
        total = type_total_ms[type_name]
        per_layer = total / n_layers
        pct = total / token_budget_ms * 100
        print(
            f"  {type_name:<35} {n_layers:>3} layers x {per_layer:>6.3f} ms = "
            f"{total:>7.2f} ms ({pct:>5.1f}% of token budget)"
        )

    layer_means = []
    for i, layer in enumerate(layers):
        if timings[i]:
            type_name = layer_type_names[i] if i < len(layer_type_names) else type(layer).__name__
            layer_means.append((i, type_name, sum(timings[i]) / len(timings[i]) * 1000))
    layer_means.sort(key=lambda x: -x[2])
    print(f"\n=== Top 10 slowest individual layers ===")
    for i, type_name, mean_ms in layer_means[:10]:
        print(f"  layer {i:>3} {type_name:<35} {mean_ms:>6.3f} ms")

    # Sub-layer breakdown
    print(f"\n=== Sub-component cost (sum across all {len(layers)} layers x {n_decode} steps) ===")
    print(f"  {'component':<28} {'mean_ms_per_call':>18} {'total_ms_per_token':>20}")
    sub_total_token_ms = 0.0
    for label in sorted(sublayer_timings.keys(), key=lambda k: -sum(sublayer_timings[k])):
        samples = sublayer_timings[label]
        if not samples:
            continue
        n_calls = len(samples)
        n_calls_per_token = n_calls / n_decode
        total_per_token = sum(samples) * 1000 / n_decode
        per_call_mean = sum(samples) * 1000 / n_calls
        sub_total_token_ms += total_per_token
        print(f"  {label:<28} {per_call_mean:>15.3f} ms {total_per_token:>17.2f} ms ({n_calls_per_token:.0f} calls/token)")
    print(f"  {'(sum sublayers/token)':<28} {'':>18} {sub_total_token_ms:>17.2f} ms")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], n_decode=int(sys.argv[2]) if len(sys.argv) > 2 else 30))
