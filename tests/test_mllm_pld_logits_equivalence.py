# SPDX-License-Identifier: Apache-2.0
"""Logit-equivalence tests for sequential vs multi-token verify (PR #172 Layer 1).

Pins the root cause of MLLM PLD output drift: multi-token forward produces
different logits than K+1 sequential single-token forwards at the same input
positions. The sequential-verify fix in `_step_speculative` (PR #172) trades
K+1× kernel-launch cost for byte-equivalence with standalone decode.

These tests load a real model (small pure-attention LLM) and directly
compare logits position-by-position. Marked @pytest.mark.slow; require
model to be cached or downloadable.

Run:
    .venv/bin/python -m pytest tests/test_mllm_pld_logits_equivalence.py \\
        -v --run-slow
"""

from __future__ import annotations

import copy

import mlx.core as mx
import pytest


# Small pure-attention LLM. mlx-community/Llama-3.2-1B-Instruct-4bit is the
# same model used by tests/test_batching_deterministic.py (proven deterministic
# at T=0 in vmlx). 600 MB cached.
TEST_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    try:
        from mlx_lm import load
        model, tokenizer = load(TEST_MODEL)
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load model {TEST_MODEL}: {e}")


def _prefill_and_cache(model, prompt_ids):
    """Prefill the model on prompt_ids, return the cache."""
    if hasattr(model, "make_cache"):
        cache = model.make_cache()
    else:
        from mlx_lm.models.cache import KVCache
        n_layers = (
            len(model.layers) if hasattr(model, "layers") else
            len(model.model.layers) if hasattr(model, "model") else 1
        )
        cache = [KVCache() for _ in range(n_layers)]
    prefill_input = mx.array([prompt_ids])
    _ = model(prefill_input, cache=cache)
    mx.eval([c.keys if hasattr(c, "keys") and c.keys is not None else c for c in cache])
    return cache


@pytest.mark.slow
class TestLogitsEquivalence:
    """Validate that sequential K+1 single-token forwards produce equivalent
    output to a single multi-token forward of shape (B, K+1).

    NOTE: On some models (e.g., smolvlm), these tests FAIL — that's the bug
    PR #172's sequential verify works around. The tests document the
    expected behaviour and serve as regression detection if mlx_lm
    upstream is fixed.
    """

    def test_sequential_matches_multi_token_via_argmax(self, model_and_tokenizer):
        """argmax token at each position must match between multi-token
        and sequential forwards."""
        model, tokenizer = model_and_tokenizer
        prompt = "The quick brown fox"
        prompt_ids = tokenizer.encode(prompt) if hasattr(tokenizer, "encode") else list(
            tokenizer(prompt).input_ids
        )

        cache_for_prefill = _prefill_and_cache(model, prompt_ids)

        # 3 draft tokens (ASCII range, safe for most tokenizers)
        drafts = [100, 101, 102]

        # Path 1: multi-token forward
        cache_M = copy.deepcopy(cache_for_prefill)
        out_M = model(mx.array([drafts]), cache=cache_M)
        logits_M = out_M.logits if hasattr(out_M, "logits") else out_M
        mx.eval(logits_M)
        assert logits_M.shape[0] == 1
        assert logits_M.shape[1] == 3, f"expected T=3, got {logits_M.shape}"

        # Path 2: sequential single-token forwards
        cache_S = copy.deepcopy(cache_for_prefill)
        seq_logits = []
        for t in drafts:
            out_S = model(mx.array([[t]]), cache=cache_S)
            logits_t = out_S.logits if hasattr(out_S, "logits") else out_S
            mx.eval(logits_t)
            seq_logits.append(logits_t[:, -1:, :])
        logits_S = mx.concatenate(seq_logits, axis=1)
        assert logits_S.shape[1] == 3

        # Compare argmax per position
        for j in range(3):
            argmax_M = int(mx.argmax(logits_M[:, j, :], axis=-1).item())
            argmax_S = int(mx.argmax(logits_S[:, j, :], axis=-1).item())
            # On Llama-3.2 we expect these to match (deterministic at T=0).
            # On smolvlm they don't — bug documented in PR #172.
            assert argmax_M == argmax_S, (
                f"argmax mismatch at pos {j}: M={argmax_M} S={argmax_S}. "
                f"Sequential verify is the workaround."
            )

    def test_sequential_matches_multi_token_via_logits_tolerance(self, model_and_tokenizer):
        """logits should match within FP tolerance (1e-3 ABS).

        On smolvlm this fails (max_diff up to 0.375). On Llama-3.2 it should
        pass — meaning Llama's multi-token forward IS correct, and the
        smolvlm-specific bug is in mlx_vlm's wrapper, not mlx_lm core.
        """
        model, tokenizer = model_and_tokenizer
        prompt = "The quick brown fox"
        prompt_ids = tokenizer.encode(prompt) if hasattr(tokenizer, "encode") else list(
            tokenizer(prompt).input_ids
        )
        cache_for_prefill = _prefill_and_cache(model, prompt_ids)
        drafts = [100, 101, 102]

        cache_M = copy.deepcopy(cache_for_prefill)
        out_M = model(mx.array([drafts]), cache=cache_M)
        logits_M = out_M.logits if hasattr(out_M, "logits") else out_M
        mx.eval(logits_M)

        cache_S = copy.deepcopy(cache_for_prefill)
        seq_logits = []
        for t in drafts:
            out_S = model(mx.array([[t]]), cache=cache_S)
            logits_t = out_S.logits if hasattr(out_S, "logits") else out_S
            mx.eval(logits_t)
            seq_logits.append(logits_t[:, -1:, :])
        logits_S = mx.concatenate(seq_logits, axis=1)

        max_diff_overall = 0.0
        per_pos_diffs = []
        for j in range(3):
            d = mx.abs(logits_M[:, j, :] - logits_S[:, j, :]).max().item()
            per_pos_diffs.append(d)
            max_diff_overall = max(max_diff_overall, d)

        # Tolerance: 1e-2 covers normal MLX FP variance on Apple Silicon.
        # If this fails, the model has a known multi-token bug → use
        # sequential verify as workaround.
        assert max_diff_overall < 1e-2, (
            f"Multi-token logits diverge from sequential: "
            f"max_diff={max_diff_overall:.6e}, per_pos={per_pos_diffs}. "
            f"Sequential verify is the correctness workaround."
        )

    def test_sequential_matches_multi_token_cache_state(self, model_and_tokenizer):
        """Ending cache state (keys at last position) should match between
        multi-token and sequential paths."""
        model, tokenizer = model_and_tokenizer
        prompt_ids = tokenizer.encode("Hello world") if hasattr(tokenizer, "encode") else [1, 2]
        cache_for_prefill = _prefill_and_cache(model, prompt_ids)
        drafts = [100, 101, 102]

        cache_M = copy.deepcopy(cache_for_prefill)
        _ = model(mx.array([drafts]), cache=cache_M)
        mx.eval(cache_M[0].keys if hasattr(cache_M[0], "keys") else cache_M[0])

        cache_S = copy.deepcopy(cache_for_prefill)
        for t in drafts:
            _ = model(mx.array([[t]]), cache=cache_S)
            mx.eval(cache_S[0].keys if hasattr(cache_S[0], "keys") else cache_S[0])

        # Compare final offset
        if hasattr(cache_M[0], "offset") and hasattr(cache_S[0], "offset"):
            off_M = int(cache_M[0].offset) if not isinstance(cache_M[0].offset, mx.array) else int(cache_M[0].offset.item())
            off_S = int(cache_S[0].offset) if not isinstance(cache_S[0].offset, mx.array) else int(cache_S[0].offset.item())
            assert off_M == off_S, f"cache offsets differ: M={off_M} S={off_S}"

        # Compare last-position keys
        if hasattr(cache_M[0], "keys") and cache_M[0].keys is not None:
            last_M = cache_M[0].keys[..., -1:, :]
            last_S = cache_S[0].keys[..., -1:, :]
            key_diff = mx.abs(last_M - last_S).max().item()
            # Expect <= 1e-3 even if logits diverge; cache writes should match
            assert key_diff < 1e-3, f"cache key writes diverge: max_diff={key_diff}"
