#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose multi-token vs sequential single-token forward divergence.

PR #150's MLLM batched PLD `_step_speculative` calls
`self.language_model(verify_input, cache=cache)` with shape (B, K+1).
Live byte-equality validation (PR #171) showed output drifts vs PLD-off
on smolvlm. This script pins WHERE the divergence comes from by directly
comparing logits between:

  - Path 1 (multi-token, what _step_speculative does)
  - Path 2 (sequential K+1 single-token, what standalone _step does K+1 times)

If logits differ → the multi-token forward is the bug. Fix: replace with
sequential K+1 forwards in _step_speculative.

Usage:
    python tests/benchmark/diagnose_multi_token_forward.py \\
        --model mlx-community/SmolVLM-Instruct-bf16

Exit code: 0 if logits match within tolerance, 1 if they diverge.
"""

from __future__ import annotations

import argparse
import copy
import sys

import mlx.core as mx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mlx-community/SmolVLM-Instruct-bf16",
        help="HuggingFace model name (must be pre-cached)",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog. It then ran away.",
        help="Prefill prompt (will be tokenized + processed)",
    )
    parser.add_argument(
        "--n-drafts",
        type=int,
        default=3,
        help="Number of draft tokens to verify (K+1 in PLD terms)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Max-abs logit-difference tolerance (FP noise allowance)",
    )
    args = parser.parse_args()

    # Load via mlx_vlm (smolvlm is a VLM)
    print(f"Loading {args.model} ...")
    try:
        from mlx_vlm import load as load_vlm
        model, processor = load_vlm(args.model)
        tokenizer = getattr(processor, "tokenizer", processor)
        language_model = getattr(model, "language_model", model)
    except Exception as e:
        print(f"FAIL: model load: {e}", file=sys.stderr)
        return 1

    # Tokenize prompt
    if hasattr(tokenizer, "encode"):
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        prompt_ids = tokenizer(args.prompt).input_ids
    prompt_ids = list(prompt_ids) if not isinstance(prompt_ids, list) else prompt_ids
    print(f"Prompt: {len(prompt_ids)} tokens")

    # Make a fresh cache for prefill
    if hasattr(language_model, "make_cache"):
        cache_prefill = language_model.make_cache()
    else:
        from mlx_lm.models.cache import KVCache
        n_layers = len(language_model.layers) if hasattr(language_model, "layers") else 1
        cache_prefill = [KVCache() for _ in range(n_layers)]

    # Prefill
    prefill_input = mx.array([prompt_ids])
    _ = language_model(prefill_input, cache=cache_prefill)
    mx.eval(cache_prefill[0].keys if hasattr(cache_prefill[0], "keys") else cache_prefill[0])
    # Confirm offset
    sample_offset = (
        cache_prefill[0].offset
        if hasattr(cache_prefill[0], "offset") else None
    )
    print(f"Prefill done; cache[0].offset = {sample_offset}")

    # Choose K+1 draft tokens (use a known pattern; tokens 100-103 are usually
    # ASCII / safe for most tokenizers)
    drafts = list(range(100, 100 + args.n_drafts))
    print(f"Draft tokens: {drafts}")

    # Path 1: multi-token forward
    cache_M = copy.deepcopy(cache_prefill)
    multi_input = mx.array([drafts])
    out_M = language_model(multi_input, cache=cache_M)
    logits_M = out_M.logits if hasattr(out_M, "logits") else out_M
    mx.eval(logits_M)
    print(f"Multi-token forward: logits shape = {logits_M.shape}")

    # Path 2: sequential single-token forwards
    cache_S = copy.deepcopy(cache_prefill)
    seq_logits = []
    for t in drafts:
        out_S = language_model(mx.array([[t]]), cache=cache_S)
        logits_t = out_S.logits if hasattr(out_S, "logits") else out_S
        mx.eval(logits_t)
        seq_logits.append(logits_t[:, -1:, :])
    logits_S = mx.concatenate(seq_logits, axis=1)
    print(f"Sequential forward: logits shape = {logits_S.shape}")

    # Compare position-by-position
    print("\n--- per-position comparison ---")
    all_pass = True
    for j in range(args.n_drafts):
        diff = mx.abs(logits_M[:, j, :] - logits_S[:, j, :]).max().item()
        argmax_M = int(mx.argmax(logits_M[:, j, :], axis=-1).item())
        argmax_S = int(mx.argmax(logits_S[:, j, :], axis=-1).item())
        argmax_match = argmax_M == argmax_S
        within_tol = diff < args.tol
        status = "PASS" if (within_tol and argmax_match) else "FAIL"
        all_pass = all_pass and within_tol and argmax_match
        print(
            f"  pos {j}: max_diff={diff:.6e}  "
            f"argmax_M={argmax_M}  argmax_S={argmax_S}  "
            f"argmax_match={argmax_match}  within_tol={within_tol}  [{status}]"
        )

    # Cache state comparison
    print("\n--- cache state comparison ---")
    if hasattr(cache_M[0], "offset") and hasattr(cache_S[0], "offset"):
        off_M = cache_M[0].offset
        off_S = cache_S[0].offset
        print(f"  cache[0].offset: M={off_M}  S={off_S}")
        if hasattr(cache_M[0], "keys") and cache_M[0].keys is not None:
            # Compare keys at the last new position
            try:
                last_M = cache_M[0].keys[..., -1:, :]
                last_S = cache_S[0].keys[..., -1:, :]
                key_diff = mx.abs(last_M - last_S).max().item()
                print(f"  cache[0].keys last position max_diff = {key_diff:.6e}")
            except Exception as e:
                print(f"  (couldn't compare keys: {e})")

    print(f"\nVerdict: {'PASS' if all_pass else 'FAIL'} (multi vs sequential)")
    if not all_pass:
        print(
            "\nInterpretation:\n"
            "  Multi-token forward and sequential forward DIFFER.\n"
            "  This is the root cause of MLLM PLD output drift.\n"
            "  Fix: in MLLMBatchGenerator._step_speculative, replace the\n"
            "  single multi-token forward with K+1 sequential single-token\n"
            "  forwards. K+1× slower but byte-equal to standalone decode."
        )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
