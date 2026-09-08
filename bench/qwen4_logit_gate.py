"""Same-weight, teacher-forced full-model Qwen4 optimization qualification.

Requires exclusive ownership of model memory; do not run beside a server.
Each option is compared with the unchanged math path on identical tokens.
Gates are fixed before execution: mean KL <= .01, maximum row KL <= .05,
and logit RMS <= .1. Top-1 agreement and absolute error are also reported.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLAGS = [
    "VMLX_QWEN4_GDN_BLOCKED_PREFILL",
    "VMLX_QWEN4_VERIFY_SDPA",
    "VMLX_QWEN4_PREFILL_DIRECT",
    "VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contexts", default="1024,4096,8192,32768")
    p.add_argument("--short-final-segments", default="1,8")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    from types import SimpleNamespace

    import mlx.core as mx
    import numpy as np

    from vmlx_engine.models.mllm import MLXMultimodalLM
    from vmlx_engine.utils.qwen4_prefill_checkpoints import (
        coalesce_qwen4_prefill_checkpoints,
    )

    for flag in FLAGS:
        os.environ[flag] = "0"
    os.environ["VMLX_NATIVE_MTP_PROMPT_PRIMING"] = "0"
    mx.set_cache_limit(2 * 1024**3)
    print("Loading one full model for teacher-forced comparison", flush=True)
    instance = MLXMultimodalLM(a.model, enable_cache=False)
    instance.load()
    lm = getattr(instance.model, "language_model", instance.model)
    tokenizer = getattr(instance.processor, "tokenizer", instance.processor)
    assert hasattr(lm, "make_cache")
    text = "A bounded cache stores keys and values, evicts the oldest entry, and updates recency after a lookup. "
    seed_tokens = tokenizer.encode(text, add_special_tokens=False)
    continuation = tokenizer.encode(
        "The cache keeps recently used entries and removes older entries when capacity is reached.",
        add_special_tokens=False,
    )[:16]
    rows = []

    def last_logits(ids, cache, last=False):
        hidden = lm(
            ids, cache=cache, return_logits=False, prefill_last_logits_only=last
        )
        hidden = hidden[:, -1:]
        logits = (
            lm.model.embed_tokens.as_linear(hidden)
            if lm.args.tie_word_embeddings
            else lm.lm_head(hidden)
        )
        mx.eval(logits)
        return logits

    def forward(tokens, flags, tail=1):
        for flag in FLAGS:
            os.environ[flag] = str(int(flag in flags))
        cache = lm.make_cache()
        ids = mx.array([tokens], dtype=mx.int32)
        length = len(tokens)
        if length <= 4112:
            boundaries = ((length - tail) // 64 * 64, length - tail)
            if FLAGS[3] in flags:
                g = SimpleNamespace(
                    _model_type="qwen4_exp_text",
                    _hybrid_kv_positions=[
                        i
                        for i, c in enumerate(cache)
                        if type(c).__name__ != "ArraysCache"
                    ],
                    _maybe_capture_clean_ssm_boundary=lambda *args: True,
                )
                out = coalesce_qwen4_prefill_checkpoints(
                    g,
                    SimpleNamespace(_cached_tokens=0, request_id="logit-gate"),
                    lm,
                    ids,
                    cache,
                    tokens,
                    boundaries,
                    lambda start, end, active_cache=cache: {"cache": active_cache},
                )
                logits = out.logits
                mx.eval(logits)
            else:
                start = 0
                for end in (*boundaries, length):
                    logits = last_logits(ids[:, start:end], cache)
                    start = end
        else:
            for start in range(0, length, 4096):
                logits = last_logits(ids[:, start : start + 4096], cache)
        outputs = [np.asarray(logits.astype(mx.float32))[0]]
        # Four rows exercise the MTP verification attention shape on fixed,
        # teacher-forced continuations, without divergent generated history.
        for start in range(0, len(continuation), 4):
            out = lm(mx.array([continuation[start : start + 4]]), cache=cache).logits
            mx.eval(out)
            outputs.append(np.asarray(out.astype(mx.float32))[0])
        result = np.concatenate(outputs, axis=0)
        del cache, ids, logits, outputs
        gc.collect()
        mx.clear_cache()
        return result

    cases = [
        (context, tail)
        for context in map(int, a.contexts.split(","))
        for tail in (map(int, a.short_final_segments.split(",")) if context <= 4096 else (1,))
    ]
    for context, tail in cases:
        length = context + tail - 1
        tokens = (seed_tokens * (length // len(seed_tokens) + 1))[:length]
        print("Reference", context, flush=True)
        ref = forward(tokens, [], tail)
        for label, flags in [
            ("gdn", [FLAGS[0]]),
            ("verify", [FLAGS[1]]),
            ("direct", [FLAGS[2]]),
            ("checkpoints", [FLAGS[3]]),
            ("combined", FLAGS),
        ]:
            if label == "checkpoints" and context > 4096:
                continue
            start = time.monotonic()
            got = forward(tokens, flags, tail)

            def logsoftmax(x):
                x = x.astype(np.float64)
                x -= np.max(x, axis=-1, keepdims=True)
                return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

            lp, lq = logsoftmax(ref), logsoftmax(got)
            kl = (np.exp(lp) * (lp - lq)).sum(axis=-1)
            error = got - ref
            row = {
                "context": context,
                "actual_tokens": length,
                "final_segment_tokens": tail,
                "arm": label,
                "rows": len(ref),
                "mean_kl": float(np.mean(kl)),
                "max_kl": float(np.max(kl)),
                "logit_rms": float(np.sqrt(np.mean(error**2))),
                "max_abs_logit_error": float(np.max(np.abs(error))),
                "top1_agreement": float(np.mean(ref.argmax(-1) == got.argmax(-1))),
                "wall_s": time.monotonic() - start,
            }
            row["passed"] = (
                row["mean_kl"] <= 0.01
                and row["max_kl"] <= 0.05
                and row["logit_rms"] <= 0.1
            )
            rows.append(row)
            a.output.write_text(
                json.dumps(
                    {
                        "gates": {"mean_kl": 0.01, "max_kl": 0.05, "logit_rms": 0.1},
                        "rows": rows,
                        "all_passed": all(r["passed"] for r in rows),
                    },
                    indent=2,
                )
            )
            print(json.dumps(row), flush=True)
    print("Logit gate complete", flush=True)
    return 0 if rows and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
