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
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLAGS = [
    "VMLX_QWEN4_GDN_BLOCKED_PREFILL",
    "VMLX_QWEN4_VERIFY_SDPA",
    "VMLX_QWEN4_PREFILL_DIRECT",
    "VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS",
    "VMLX_QWEN4_ALIGNED_MOE_PREFILL",
]
SPARSE_FUSED_FLAG = "VMLX_QWEN4_SPARSE_FUSED_PREFILL"


@contextmanager
def observe_verifier(module):
    """Observe actual dispatch without materializing tensors or changing math."""
    original = module.qwen4_verify_sdpa
    observed = {"dispatches": 0, "shapes": []}

    def call(q, k, v, mask, **kwargs):
        result = original(q, k, v, mask, **kwargs)
        if result is not None:
            observed["dispatches"] += 1
            shape = dict(rows=q.shape[2], context=k.shape[2], dtype=str(q.dtype))
            if shape not in observed["shapes"] and len(observed["shapes"]) < 32:
                observed["shapes"].append(shape)
        return result

    module.qwen4_verify_sdpa = call
    try:
        yield observed
    finally:
        module.qwen4_verify_sdpa = original


def qualification_passes(row, *, require_exact=False):
    return bool(
        row["mean_kl"] <= 0.01
        and row["max_kl"] <= 0.05
        and row["logit_rms"] <= 0.1
        and (not require_exact or row["exact_logits"])
        and (
            row["arm"] != "verify"
            or row["context"] < 8192
            or row["verifier_dispatches"] > 0
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contexts", default="1024,4096,8192,32768")
    p.add_argument("--short-final-segments", default="1,8")
    p.add_argument("--arms", default="gdn,verify,direct,checkpoints,aligned_moe,combined")
    p.add_argument("--prefill-step-size", type=int, default=4096)
    p.add_argument("--continuation-step-size", type=int, default=4)
    p.add_argument("--require-exact", action="store_true",
                   help="Also reject any unequal full-model logits")
    p.add_argument("--attention-fixture", type=Path,
                   help="For sparse_oracle, save the first real attention inputs for component diagnosis")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    from types import SimpleNamespace

    import mlx.core as mx
    import numpy as np

    from vmlx_engine.models.mllm import MLXMultimodalLM
    from vmlx_engine.models.qwen4_exp import language as language_impl
    from vmlx_engine.utils.qwen4_prefill_checkpoints import (
        coalesce_qwen4_prefill_checkpoints,
    )

    for flag in [*FLAGS, SPARSE_FUSED_FLAG]:
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
    attention_rows = []

    def stock_from_blocks(q, k, v, ids, valid, *, pos_start, total_tokens, scale):
        """Causal control: new selected-block route, unchanged MLX consumer.

        Only used by the diagnostic arm below, never by the product. Comparing
        both consumers on identical real tensors separates arithmetic errors
        from changes introduced by block selection or surrounding graph work.
        """
        from vmlx_engine.metal import qwen4_sparse_fused_prefill as sparse_impl
        positions = mx.arange(pos_start, total_tokens)
        complete = (positions + 1) // 4
        chosen = mx.put_along_axis(
            mx.zeros((q.shape[2], total_tokens // 4), dtype=mx.bool_), ids,
            valid, axis=-1,
        )
        keep = mx.concatenate([
            mx.repeat(chosen, 4, axis=-1),
            mx.zeros((q.shape[2], total_tokens % 4), dtype=mx.bool_),
        ], axis=-1)
        tokens = mx.arange(total_tokens)[None]
        mask = mx.where(
            (keep | (tokens >= complete[:, None] * 4))
            & (tokens <= positions[:, None]), 0, -mx.inf,
        ).astype(q.dtype)[None, None]
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                  mask=mask, force_fused=True)
        if total_tokens >= sparse_impl.MIN_CONTEXT:
            selected = sparse_impl.native.qsa_prefill_direct_topk_buffer(
                ids, valid, pos_start=pos_start)
            got = sparse_impl.native._EXT.qwen4_qsa_sparse_gqa_attention_nax(
                q, k, v, selected, scale, pos_start)
            diff = got.astype(mx.float32) - ref.astype(mx.float32)
            rms = mx.sqrt(mx.mean(mx.square(diff)))
            reference_rms = mx.sqrt(mx.mean(mx.square(ref.astype(mx.float32))))
            max_abs = mx.max(mx.abs(diff))
            mx.eval(rms, reference_rms, max_abs)
            if not attention_rows and a.attention_fixture:
                a.attention_fixture.parent.mkdir(parents=True, exist_ok=True)
                mx.save_safetensors(str(a.attention_fixture),
                                    dict(q=q, k=k, v=v, ids=ids, valid=valid,
                                         mask=mask, ref=ref, got=got))
            measured = dict(context=total_tokens, rows=q.shape[2],
                            dispatch=len(attention_rows), max_abs=float(max_abs),
                            rms=float(rms), reference_rms=float(reference_rms))
            attention_rows.append(measured)
            print("ATTENTION " + json.dumps(measured), flush=True)
        return ref

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
        for flag in [*FLAGS, SPARSE_FUSED_FLAG]:
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
            for start in range(0, length, a.prefill_step_size):
                logits = last_logits(ids[:, start : start + a.prefill_step_size], cache)
        outputs = [np.asarray(logits.astype(mx.float32))[0]]
        # Four rows exercise the MTP verification attention shape on fixed,
        # teacher-forced continuations, without divergent generated history.
        for start in range(0, len(continuation), a.continuation_step_size):
            out = lm(mx.array([continuation[start : start + a.continuation_step_size]]), cache=cache).logits
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
            ("repeat", []),
            ("gdn", [FLAGS[0]]),
            ("verify", [FLAGS[1]]),
            ("direct", [FLAGS[2]]),
            ("checkpoints", [FLAGS[3]]),
            ("aligned_moe", [FLAGS[4]]),
            ("combined", FLAGS),
            ("sparse_fused", [SPARSE_FUSED_FLAG]),
            ("sparse_oracle", [SPARSE_FUSED_FLAG]),
        ]:
            if label not in a.arms.split(","):
                continue
            if label == "checkpoints" and context > 4096:
                continue
            from vmlx_engine.metal import qwen4_aligned_moe_prefill as aligned_impl
            from vmlx_engine.metal import qwen4_sparse_fused_prefill as sparse_impl
            before_dispatch = aligned_impl.DISPATCH_COUNT
            before_sparse = sparse_impl.DISPATCH_COUNT
            start = time.monotonic()
            original_call = sparse_impl._call
            try:
                if label == "sparse_oracle":
                    sparse_impl._call = stock_from_blocks
                with observe_verifier(language_impl) as verifier:
                    got = forward(tokens, flags, tail)
            finally:
                sparse_impl._call = original_call

            dispatches = aligned_impl.DISPATCH_COUNT - before_dispatch
            sparse_dispatches = sparse_impl.DISPATCH_COUNT - before_sparse
            if label in {"sparse_fused", "sparse_oracle"}:
                assert sparse_dispatches > 0, "sparse fused path did not execute"
            if context == 1024 and label in ("aligned_moe", "combined"):
                assert dispatches > 0, "aligned MoE guard silently bypassed the eligible case"

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
                "aligned_moe_dispatches": dispatches,
                "sparse_fused_dispatches": sparse_dispatches,
                "verifier_dispatches": verifier["dispatches"],
                "verifier_shapes": verifier["shapes"],
                "exact_logits": bool(np.array_equal(ref, got)),
                "prefill_step_size": a.prefill_step_size,
                "continuation_step_size": a.continuation_step_size,
                "mean_kl": float(np.mean(kl)),
                "max_kl": float(np.max(kl)),
                "logit_rms": float(np.sqrt(np.mean(error**2))),
                "max_abs_logit_error": float(np.max(np.abs(error))),
                "top1_agreement": float(np.mean(ref.argmax(-1) == got.argmax(-1))),
                "wall_s": time.monotonic() - start,
            }
            row["passed"] = qualification_passes(row, require_exact=a.require_exact)
            rows.append(row)
            a.output.write_text(
                json.dumps(
                    {
                        "gates": {"mean_kl": 0.01, "max_kl": 0.05, "logit_rms": 0.1,
                                  "require_exact": a.require_exact,
                                  "verify_dispatch_required_min_context": 8192},
                        "rows": rows,
                        "attention_diagnostics": attention_rows,
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
