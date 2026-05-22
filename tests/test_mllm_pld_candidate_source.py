# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMBatchGenerator's PLD candidate-source helper (issue #134).

The generator-side PLD lookup uses per-request NgramIndex over the request's
full token sequence (prompt + output). The Scheduler's V0.6 special-token
filter is propagated here via `configure_pld_spec(excluded_token_ids=...)`.

Tests exercise the helper logic via a minimal fake MLLMBatchRequest with the
shape `_pld_drafts_for_request` reads (input_ids, output_tokens, lazy index).

Run:
    .venv/bin/python -m pytest tests/test_mllm_pld_candidate_source.py -v
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Minimal fake request — duck-typed for _pld_drafts_for_request
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for MLLMBatchRequest exposing only the fields
    `_pld_drafts_for_request` reads."""

    def __init__(
        self,
        input_ids: Optional[mx.array] = None,
        output_tokens: Optional[List[int]] = None,
    ):
        self.input_ids = input_ids
        self.output_tokens = output_tokens if output_tokens is not None else []
        self._pld_ngram_index = None
        self._cached_prompt_token_ids = None


# ---------------------------------------------------------------------------
# Test fixture: minimal MLLMBatchGenerator method via class proxy
# ---------------------------------------------------------------------------


class _MethodProxy:
    """Calls MLLMBatchGenerator instance methods as if we constructed one.

    Building a real MLLMBatchGenerator requires a model + processor. For unit
    tests of pure-logic helpers we attach the relevant state attributes
    directly and reuse the method via descriptor protocol.
    """

    @classmethod
    def make(cls, pld_enabled: bool = True, excluded_token_ids=None):
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        # Bypass __init__ by allocating an instance and setting only what
        # _pld_drafts_for_request reads.
        instance = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        instance._pld_spec_enabled = pld_enabled
        instance._pld_excluded_token_ids = excluded_token_ids
        return instance


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------


def test_pld_disabled_returns_empty():
    gen = _MethodProxy.make(pld_enabled=False)
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, 4, 5, 1, 2]),
        output_tokens=[],
    )
    drafts = gen._pld_drafts_for_request(req, K=3, seed_token=None)
    assert drafts == []


def test_seed_token_included_in_lookup_no_match():
    """seed_token appends to full_tokens for n-gram query (live-fix #1)."""
    gen = _MethodProxy.make(pld_enabled=True)
    req = _FakeRequest(input_ids=mx.array([1, 2, 3, 4, 5]), output_tokens=[])
    drafts = gen._pld_drafts_for_request(req, K=3, seed_token=1)
    # With seed: full = [1,2,3,4,5,1]. Query last 2 = [5,1] — no earlier match.
    assert drafts == []


def test_seed_token_completes_query_for_match():
    """Seed appended creates the query that finds a prompt repetition match."""
    gen = _MethodProxy.make(pld_enabled=True)
    # Prompt: [1,2,3,4,5,1,2,3]. Seed: 4. Full: [1,2,3,4,5,1,2,3,4].
    # Trigram [2,3,4] match at idx 1; drafts after = [5,1,2].
    req = _FakeRequest(input_ids=mx.array([1, 2, 3, 4, 5, 1, 2, 3]), output_tokens=[])
    drafts = gen._pld_drafts_for_request(req, K=3, seed_token=4)
    assert drafts == [5, 1, 2]


def test_k_zero_returns_empty():
    gen = _MethodProxy.make(pld_enabled=True)
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, 4, 5, 1, 2]),
        output_tokens=[],
    )
    drafts = gen._pld_drafts_for_request(req, K=0)
    assert drafts == []


# ---------------------------------------------------------------------------
# Prompt-only lookups
# ---------------------------------------------------------------------------


def test_lookup_against_prompt_only():
    """N-gram match found within prompt tokens; output empty."""
    gen = _MethodProxy.make(pld_enabled=True)
    # Bigram [1, 2] at idx 0; query is last 2 of [1,2,3,4,5,1,2]
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, 4, 5, 1, 2]),
        output_tokens=[],
    )
    drafts = gen._pld_drafts_for_request(req, K=3)
    # Drafts after [1,2] at idx 0: [3, 4, 5]
    assert drafts == [3, 4, 5]


def test_lookup_short_sequence_returns_empty():
    """Sequence too short for n-gram lookup (n>=3 required)."""
    gen = _MethodProxy.make(pld_enabled=True)
    req = _FakeRequest(input_ids=mx.array([1, 2]), output_tokens=[])
    drafts = gen._pld_drafts_for_request(req, K=3)
    assert drafts == []


# ---------------------------------------------------------------------------
# Prompt + output combined
# ---------------------------------------------------------------------------


def test_lookup_combines_prompt_and_output():
    """Lookup must include output tokens (PLD often matches mid-decode)."""
    gen = _MethodProxy.make(pld_enabled=True)
    # Prompt: [10, 20, 30]. Output: [40, 50, 10, 20].
    # Full seq: [10, 20, 30, 40, 50, 10, 20]. Query [10, 20] matches at idx 0.
    # Drafts after idx 0 = [30, 40, 50]
    req = _FakeRequest(
        input_ids=mx.array([10, 20, 30]),
        output_tokens=[40, 50, 10, 20],
    )
    drafts = gen._pld_drafts_for_request(req, K=3)
    assert drafts == [30, 40, 50]


# ---------------------------------------------------------------------------
# Lazy index initialization + caching
# ---------------------------------------------------------------------------


def test_lazy_index_created_on_first_call():
    """NgramIndex must be lazy-init on first call and reused after."""
    gen = _MethodProxy.make(pld_enabled=True)
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, 4, 5, 1, 2]),
        output_tokens=[],
    )
    assert req._pld_ngram_index is None
    gen._pld_drafts_for_request(req, K=3)
    first_index = req._pld_ngram_index
    assert first_index is not None

    # Second call must reuse the same index instance
    gen._pld_drafts_for_request(req, K=3)
    assert req._pld_ngram_index is first_index


def test_prompt_tokens_cached():
    """Prompt token list materialized once on first call."""
    gen = _MethodProxy.make(pld_enabled=True)
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, 4, 5, 1, 2]),
        output_tokens=[],
    )
    assert req._cached_prompt_token_ids is None
    gen._pld_drafts_for_request(req, K=3)
    assert req._cached_prompt_token_ids == [1, 2, 3, 4, 5, 1, 2]


def test_2d_input_ids_flattened():
    """input_ids may be shape (1, T) from VLM preprocessing — handle flatten."""
    gen = _MethodProxy.make(pld_enabled=True)
    req = _FakeRequest(
        input_ids=mx.array([[1, 2, 3, 4, 5, 1, 2]]),  # shape (1, 7)
        output_tokens=[],
    )
    drafts = gen._pld_drafts_for_request(req, K=3)
    assert drafts == [3, 4, 5]
    assert req._cached_prompt_token_ids == [1, 2, 3, 4, 5, 1, 2]


# ---------------------------------------------------------------------------
# Special-token filter (V0.6) — wired through configure_pld_spec
# ---------------------------------------------------------------------------


def test_excluded_token_truncates_drafts():
    """Image-pad / vision tokens in the prompt must be filtered from drafts."""
    IMAGE_PAD = 999
    gen = _MethodProxy.make(
        pld_enabled=True, excluded_token_ids={IMAGE_PAD}
    )
    # Seq: [1, 2, 3, IMAGE_PAD, 5, 1, 2]. After bigram [1,2] at idx 0:
    # drafts = [3, IMAGE_PAD, 5] → truncated at IMAGE_PAD → [3]
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, IMAGE_PAD, 5, 1, 2]),
        output_tokens=[],
    )
    drafts = gen._pld_drafts_for_request(req, K=5)
    assert drafts == [3]


def test_excluded_set_none_means_no_filter():
    gen = _MethodProxy.make(pld_enabled=True, excluded_token_ids=None)
    req = _FakeRequest(
        input_ids=mx.array([1, 2, 3, 999, 5, 1, 2]),
        output_tokens=[],
    )
    drafts = gen._pld_drafts_for_request(req, K=5)
    # No filter; whole draft slice (5 tokens) returned including 999
    assert drafts == [3, 999, 5, 1, 2]
    assert 999 in drafts


# ---------------------------------------------------------------------------
# configure_pld_spec contract
# ---------------------------------------------------------------------------


def test_configure_pld_spec_sets_enabled():
    gen = _MethodProxy.make(pld_enabled=False)
    assert gen._pld_spec_enabled is False
    # Use the real configure method via the instance
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    MLLMBatchGenerator.configure_pld_spec(gen, enabled=True)
    assert gen._pld_spec_enabled is True


def test_configure_pld_spec_copies_excluded_set():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = _MethodProxy.make(pld_enabled=False)
    excl = {1, 2, 3}
    MLLMBatchGenerator.configure_pld_spec(gen, enabled=True, excluded_token_ids=excl)
    assert gen._pld_excluded_token_ids == {1, 2, 3}
    # Mutating the caller's set must not affect the stored copy
    excl.add(99)
    assert 99 not in gen._pld_excluded_token_ids


def test_configure_pld_spec_none_excluded_clears():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = _MethodProxy.make(pld_enabled=True, excluded_token_ids={1, 2})
    MLLMBatchGenerator.configure_pld_spec(gen, enabled=True, excluded_token_ids=None)
    assert gen._pld_excluded_token_ids is None
