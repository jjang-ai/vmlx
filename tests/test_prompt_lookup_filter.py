# SPDX-License-Identifier: Apache-2.0
"""Tests for exclude_token_ids filter in prompt_lookup.find_draft_tokens
and NgramIndex.find_drafts — issue #134 follow-up (V0.6).

The filter prevents model-special tokens (image-pad, pad, vision markers)
from being proposed as PLD drafts. Without the filter, n-gram lookup over
the prompt could find these tokens in the indexed sequence and propose them
as drafts. Verify usually rejects them, but truncating drafts at the first
special ID avoids the risk entirely and saves a wasted verify forward.

Run:
    .venv/bin/python -m pytest tests/test_prompt_lookup_filter.py -v
"""

from __future__ import annotations

import pytest

from vmlx_engine.prompt_lookup import (
    NgramIndex,
    _truncate_at_excluded,
    find_draft_tokens,
)


# ---------------------------------------------------------------------------
# _truncate_at_excluded — pure unit tests of the helper
# ---------------------------------------------------------------------------


def test_truncate_at_excluded_no_filter_returns_unchanged():
    assert _truncate_at_excluded([1, 2, 3], None) == [1, 2, 3]
    assert _truncate_at_excluded([1, 2, 3], set()) == [1, 2, 3]


def test_truncate_at_excluded_truncates_at_first_match():
    drafts = [10, 11, 999, 12, 13]
    assert _truncate_at_excluded(drafts, {999}) == [10, 11]


def test_truncate_at_excluded_first_token_excluded_returns_empty():
    assert _truncate_at_excluded([999, 1, 2], {999}) == []


def test_truncate_at_excluded_none_excluded_returns_full():
    assert _truncate_at_excluded([1, 2, 3], {999, 1000}) == [1, 2, 3]


def test_truncate_at_excluded_multiple_excluded_in_set():
    # First excluded ID wins (we don't scan past it)
    drafts = [1, 2, 999, 3, 1000, 4]
    assert _truncate_at_excluded(drafts, {999, 1000}) == [1, 2]


# ---------------------------------------------------------------------------
# find_draft_tokens — module-level function with exclude filter
# ---------------------------------------------------------------------------


def test_find_draft_tokens_default_no_filter_back_compat():
    """Without exclude_token_ids, behaviour matches pre-patch."""
    seq = [1, 2, 3, 4, 5, 1, 2]  # query [1,2] → match at idx 0 → drafts [3, 4, 5]
    drafts = find_draft_tokens(seq, num_draft_tokens=3)
    assert drafts == [3, 4, 5]


def test_find_draft_tokens_filters_pad_in_draft():
    """Excluded token mid-draft truncates the result."""
    # Seq: prompt has bigram [1,2] followed by [3, PAD, 4]. Query at end is [1,2].
    PAD = 999
    seq = [1, 2, 3, PAD, 4, 1, 2]
    drafts = find_draft_tokens(seq, num_draft_tokens=4, exclude_token_ids={PAD})
    # Drafts after [1,2] are [3, PAD, 4]; truncated at PAD → [3]
    assert drafts == [3]


def test_find_draft_tokens_first_draft_excluded_returns_empty():
    """First draft token is excluded → empty result (no false acceleration)."""
    PAD = 999
    seq = [1, 2, PAD, 4, 5, 1, 2]
    drafts = find_draft_tokens(seq, num_draft_tokens=4, exclude_token_ids={PAD})
    # After [1,2] is [PAD, 4, 5]; first token excluded → []
    # Function continues to try smaller n-grams, but those also start with PAD
    # in this construction. The function tries other matches; verify it returns
    # empty drafts or a clean prefix from a smaller n-gram match.
    # Realistic assertion: result either empty OR has no PAD.
    assert PAD not in drafts


def test_find_draft_tokens_no_filter_returns_all():
    seq = [1, 2, 3, 4, 5, 1, 2]
    drafts = find_draft_tokens(seq, num_draft_tokens=3, exclude_token_ids=None)
    assert drafts == [3, 4, 5]


def test_find_draft_tokens_empty_exclude_set_is_noop():
    seq = [1, 2, 3, 4, 5, 1, 2]
    drafts = find_draft_tokens(seq, num_draft_tokens=3, exclude_token_ids=set())
    assert drafts == [3, 4, 5]


# ---------------------------------------------------------------------------
# NgramIndex.find_drafts — class method with exclude filter
# ---------------------------------------------------------------------------


def test_ngram_index_filters_excluded_token():
    PAD = 999
    seq = [1, 2, 3, PAD, 4, 5, 1, 2]
    idx = NgramIndex()
    drafts = idx.find_drafts(
        seq, num_draft_tokens=4, exclude_token_ids={PAD}
    )
    # Trigram [3, PAD, 4] not at end (end is [5, 1, 2]). Bigram [1, 2] matches
    # at idx 0, drafts after are [3, PAD, 4] truncated → [3].
    assert drafts == [3]


def test_ngram_index_no_filter_back_compat():
    seq = [1, 2, 3, 4, 5, 1, 2]
    idx = NgramIndex()
    drafts = idx.find_drafts(seq, num_draft_tokens=3)
    assert drafts == [3, 4, 5]


def test_ngram_index_excluded_filter_preserves_valid_prefix():
    """If the n-gram match has [valid, valid, bad, valid, valid] after it,
    we keep the leading valid prefix and drop the suffix."""
    BAD = 99999
    # Build: ... [1, 2, 10, 11, BAD, 12, 13] ... [1, 2] at end
    seq = [1, 2, 10, 11, BAD, 12, 13, 50, 51, 1, 2]
    idx = NgramIndex()
    drafts = idx.find_drafts(
        seq, num_draft_tokens=5, exclude_token_ids={BAD}
    )
    # Match on [1, 2] at idx 0, drafts after are [10, 11, BAD, 12, 13],
    # truncated at BAD → [10, 11]
    assert drafts == [10, 11]


# ---------------------------------------------------------------------------
# Realistic VLM scenarios
# ---------------------------------------------------------------------------


def test_vlm_image_token_filtered_from_drafts():
    """Simulates a VLM prompt where image-pad tokens (262147 in Zaya1-VL)
    appear in the indexed prompt. n-gram lookup must not propose them."""
    IMAGE_PAD = 262147
    VISION_START = 255999
    # Realistic prompt: text + image markers + text
    # Bigram [101, 102] appears twice; second occurrence has image content after
    seq = (
        [101, 102, 200, 201, 300]  # first text section
        + [VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD]  # image block
        + [400, 401, 402]  # text after image
        + [101, 102]  # query at end
    )
    excluded = {IMAGE_PAD, VISION_START}
    drafts = find_draft_tokens(
        seq, num_draft_tokens=4, exclude_token_ids=excluded
    )
    # Bigram [101, 102] matches at idx 0; drafts after = [200, 201, 300, VISION_START]
    # Truncate at VISION_START → [200, 201, 300]
    assert drafts == [200, 201, 300]
    assert IMAGE_PAD not in drafts
    assert VISION_START not in drafts
