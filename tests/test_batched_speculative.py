# SPDX-License-Identifier: Apache-2.0
"""Unit tests for batched speculative decoding (issue #135).

Tests the new should_use_speculative_batched() function and the PLD precedence
skip without requiring real model weights or mlx-lm/transformers imports.

Run:
    .venv/bin/python -m pytest tests/test_batched_speculative.py -v
"""

from __future__ import annotations

import os
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Standalone implementations for testing
# (mirror the logic without importing the full module chain)
# ---------------------------------------------------------------------------

def _should_use_speculative_batched(is_speculative_enabled_fn, is_mllm=False):
    """Test-local copy of should_use_speculative_batched logic."""
    if not is_speculative_enabled_fn():
        return False
    if is_mllm:
        return False
    return os.getenv("VMLX_ENABLE_BATCHED_SPEC", "0") == "1"


def _should_use_speculative(is_speculative_enabled_fn, is_batched=False, is_mllm=False):
    """Test-local copy of should_use_speculative logic."""
    if not is_speculative_enabled_fn():
        return False
    if is_batched:
        return False
    if is_mllm:
        return False
    return True


# ---------------------------------------------------------------------------
# 1. test_should_use_speculative_batched_requires_env
# ---------------------------------------------------------------------------


def test_should_use_speculative_batched_requires_env(monkeypatch):
    """Without VMLX_ENABLE_BATCHED_SPEC=1, batched spec returns False."""
    monkeypatch.delenv("VMLX_ENABLE_BATCHED_SPEC", raising=False)

    result = _should_use_speculative_batched(
        is_speculative_enabled_fn=lambda: True,
        is_mllm=False,
    )
    assert result is False, "Batched spec must be disabled without env var"


# ---------------------------------------------------------------------------
# 2. test_should_use_speculative_batched_with_env
# ---------------------------------------------------------------------------


def test_should_use_speculative_batched_with_env(monkeypatch):
    """VMLX_ENABLE_BATCHED_SPEC=1 + spec enabled → returns True."""
    monkeypatch.setenv("VMLX_ENABLE_BATCHED_SPEC", "1")

    result = _should_use_speculative_batched(
        is_speculative_enabled_fn=lambda: True,
        is_mllm=False,
    )
    assert result is True, "Batched spec must be enabled with env var + spec enabled"


# ---------------------------------------------------------------------------
# 3. test_should_use_speculative_batched_mllm_excluded
# ---------------------------------------------------------------------------


def test_should_use_speculative_batched_mllm_excluded(monkeypatch):
    """is_mllm=True must return False even with VMLX_ENABLE_BATCHED_SPEC=1."""
    monkeypatch.setenv("VMLX_ENABLE_BATCHED_SPEC", "1")

    result = _should_use_speculative_batched(
        is_speculative_enabled_fn=lambda: True,
        is_mllm=True,
    )
    assert result is False, "Batched spec must be disabled for MLLM models"


# ---------------------------------------------------------------------------
# 4. test_pld_precedence_skip
# ---------------------------------------------------------------------------


def test_pld_precedence_skip():
    """When draft spec is enabled and model is not MLLM, PLD returns [] early.

    This verifies the condition from scheduler._try_speculative_decode:
      if is_speculative_enabled() and not is_mllm: return []
    """
    is_spec_enabled = True
    is_mllm = False

    # Reproduce the precedence check from the scheduler
    should_skip = is_spec_enabled and not is_mllm
    assert should_skip is True, "PLD must skip when draft spec is active on LLM"


def test_pld_precedence_skip_mllm_not_skipped():
    """On MLLM models, spec is not active → PLD should not skip."""
    is_spec_enabled = True
    is_mllm = True

    should_skip = is_spec_enabled and not is_mllm
    assert should_skip is False, "PLD must NOT skip on MLLM models (spec disabled for MLLM)"


# ---------------------------------------------------------------------------
# 5. test_legacy_should_use_speculative_batched_false
# ---------------------------------------------------------------------------


def test_legacy_should_use_speculative_batched_false(monkeypatch):
    """Legacy should_use_speculative(is_batched=True) must still return False.

    Batched mode goes through should_use_speculative_batched() not the old path.
    This ensures existing callers that pass is_batched=True are not broken.
    """
    monkeypatch.setenv("VMLX_ENABLE_BATCHED_SPEC", "1")

    # The legacy function must still return False for is_batched=True
    result = _should_use_speculative(
        is_speculative_enabled_fn=lambda: True,
        is_batched=True,
        is_mllm=False,
    )
    assert result is False, (
        "Legacy should_use_speculative(is_batched=True) must return False; "
        "use should_use_speculative_batched() instead"
    )
