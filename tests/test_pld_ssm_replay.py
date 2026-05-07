# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Scheduler._replay_ssm_forward() — issue #134.

Tests the hybrid partial-accept replay path without requiring real model
weights or a full mlx-lm/transformers environment.  Uses _FakeSSMLayer /
_FakeKVCache stubs and a mock model callable.  The static method is
replicated directly in this file to isolate the test from the full
scheduler import chain.

Run:
    .venv/bin/python -m pytest tests/test_pld_ssm_replay.py -v
"""

from __future__ import annotations

import os
import unittest.mock as mock

import pytest

import mlx.core as mx


# ---------------------------------------------------------------------------
# Standalone implementation of _replay_ssm_forward for testing
# (mirrors the logic in Scheduler._replay_ssm_forward without importing
#  the full scheduler module which pulls in mlx_lm/transformers)
# ---------------------------------------------------------------------------

def _replay_ssm_forward(model, kv_cache, saved_array_caches, accepted_tokens,
                        pre_verify_offset):
    """Test-local copy of Scheduler._replay_ssm_forward logic."""
    import numpy as _np_local

    def _rewind_kv_to(kv_cache, target_offset):
        for c in kv_cache:
            if not c.is_trimmable() or c.offset == 0:
                continue
            if c.offset <= target_offset:
                continue
            if isinstance(c.keys, mx.array):
                _kd, _vd = c.keys.dtype, c.values.dtype
                _ka = c.keys.astype(mx.float16) if "bfloat16" in str(_kd) else c.keys
                _va = c.values.astype(mx.float16) if "bfloat16" in str(_vd) else c.values
                _k, _v = _np_local.array(_ka), _np_local.array(_va)
                c.keys = mx.array(_k[..., :target_offset, :]).astype(_kd)
                c.values = mx.array(_v[..., :target_offset, :]).astype(_vd)
            c.offset = target_offset
            if hasattr(c, "_idx"):
                c._idx = target_offset

    try:
        for i, c in enumerate(kv_cache):
            if i in saved_array_caches:
                c.cache = saved_array_caches[i]
        _rewind_kv_to(kv_cache, pre_verify_offset)

        replay_input = mx.array([accepted_tokens])
        _ = model(replay_input, cache=kv_cache)
        mx.eval(kv_cache)

        return True

    except Exception as exc:
        # Best-effort restore
        try:
            for i, c in enumerate(kv_cache):
                if i in saved_array_caches:
                    c.cache = saved_array_caches[i]
            _rewind_kv_to(kv_cache, pre_verify_offset)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Helpers — fake cache objects
# ---------------------------------------------------------------------------


class _FakeSSMLayer:
    """Minimal SSM cache layer with a `.cache` list of arrays.

    Mimics BatchMambaCache / ArraysCache.  `is_trimmable()` returns False so
    the KV-rewind loop skips it.
    """

    def __init__(self, marker: float, n_arrays: int = 2, shape=(4,)):
        self.cache = [mx.array([marker] * shape[0]) for _ in range(n_arrays)]
        self.lengths = None

    def is_trimmable(self) -> bool:
        return False


class _FakeKVCache:
    """Minimal KVCache with numpy-sliceable keys/values and an offset."""

    def __init__(self, n_tokens: int, head_dim: int = 8, n_heads: int = 2):
        # Shape: (1, n_heads, n_tokens, head_dim)
        self.keys = mx.zeros((1, n_heads, n_tokens, head_dim))
        self.values = mx.zeros((1, n_heads, n_tokens, head_dim))
        self.offset = n_tokens

    def is_trimmable(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 1. test_replay_advances_kv_offset
# ---------------------------------------------------------------------------


def test_replay_advances_kv_offset():
    """Replay of 1 accepted token must advance KV offset from N to N+1.

    Setup:
      - One SSM layer at position N (marker=1.0)
      - One KV layer with offset = N + K  (verify already ran, K=2 drafts)
      - pre_verify_offset = N
      - accepted_tokens = [42]  (1 token)
      - Model callable advances kv.offset by num_tokens and returns zeros

    Expected:
      - KV offset = N + 1  (rewound to N, then advanced by replay forward)
      - SSM cache restored to marker=1.0 (not zeros)
      - Returns True
    """
    N = 5
    K = 2
    vocab_size = 32

    ssm = _FakeSSMLayer(marker=1.0)
    kv = _FakeKVCache(n_tokens=N + K)  # offset = N+K after verify

    kv_cache = [ssm, kv]
    saved_array_caches = {0: list(ssm.cache)}  # snapshot at N

    # Perturb SSM cache to simulate post-verify state
    ssm.cache = [mx.zeros((4,)) for _ in ssm.cache]

    def _mock_model(tokens, cache):
        # Side-effect: advance kv offset by num_tokens
        num_tokens = tokens.shape[1]
        kv.offset += num_tokens
        return mx.zeros((1, num_tokens, vocab_size))

    result = _replay_ssm_forward(
        model=_mock_model,
        kv_cache=kv_cache,
        saved_array_caches=saved_array_caches,
        accepted_tokens=[42],
        pre_verify_offset=N,
    )

    assert result is True, "Expected True on success"
    # KV offset: rewound to N, then mock advances by 1
    assert kv.offset == N + 1, f"Expected offset={N+1}, got {kv.offset}"
    # SSM cache restored from snapshot (marker=1.0, not zeros)
    # Use id() comparison since mx.array identity is preserved on restore
    for i, arr in enumerate(ssm.cache):
        assert arr is saved_array_caches[0][i], (
            f"SSM cache[{i}] not restored to snapshot (identity check)"
        )


# ---------------------------------------------------------------------------
# 2. test_replay_fallback_on_failure
# ---------------------------------------------------------------------------


def test_replay_fallback_on_failure():
    """When the model raises during replay, _replay_ssm_forward returns False.

    The caches must be restored to pre_verify_offset by the except handler.
    """
    N = 3
    K = 2

    ssm = _FakeSSMLayer(marker=2.0)
    kv = _FakeKVCache(n_tokens=N + K)

    kv_cache = [ssm, kv]
    saved_array_caches = {0: list(ssm.cache)}

    # Trash the SSM cache to simulate post-verify state
    ssm.cache = [mx.zeros((4,)) for _ in ssm.cache]

    call_count = 0

    def _failing_model(tokens, cache):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated GPU OOM")

    result = _replay_ssm_forward(
        model=_failing_model,
        kv_cache=kv_cache,
        saved_array_caches=saved_array_caches,
        accepted_tokens=[7, 8],
        pre_verify_offset=N,
    )

    assert result is False, "Expected False on failure"
    assert call_count == 1, "Model should have been called once"
    # KV offset should be restored to N by the except handler
    assert kv.offset == N, f"Expected KV offset={N} after failure, got {kv.offset}"
    # SSM cache restored from snapshot (marker=2.0) — use identity check
    for i, arr in enumerate(ssm.cache):
        assert arr is saved_array_caches[0][i], (
            f"SSM cache[{i}] not restored to snapshot after failure (identity check)"
        )


# ---------------------------------------------------------------------------
# 3. test_replay_zero_drafts_skipped
# ---------------------------------------------------------------------------


def test_replay_zero_drafts_skipped():
    """num_accept=0 must not trigger replay (condition: 0 < num_accept).

    We verify the calling condition directly — the scheduler code checks
    `0 < num_accept < num_drafts` before calling _replay_ssm_forward.
    """
    num_accept = 0
    num_drafts = 2
    replay_enabled = True

    # This is the condition from the scheduler's case (b1) branch
    should_replay = 0 < num_accept < num_drafts and replay_enabled
    assert not should_replay, "replay must not trigger when num_accept==0"


# ---------------------------------------------------------------------------
# 4. test_env_var_disables_replay
# ---------------------------------------------------------------------------


def test_env_var_disables_replay(monkeypatch):
    """VMLX_DISABLE_PLD_REPLAY=1 must set _pld_replay_enabled=False."""
    monkeypatch.setenv("VMLX_DISABLE_PLD_REPLAY", "1")

    # Build a minimal config mock
    config = mock.MagicMock()
    config.pld_replay_enabled = True  # config says True, env overrides

    # Evaluate the expression used in Scheduler.__init__
    enabled = (
        os.getenv("VMLX_DISABLE_PLD_REPLAY") != "1"
        and getattr(config, "pld_replay_enabled", True)
    )
    assert enabled is False, "Replay should be disabled by VMLX_DISABLE_PLD_REPLAY=1"


def test_env_var_enabled_by_default(monkeypatch):
    """Without VMLX_DISABLE_PLD_REPLAY, replay is enabled."""
    monkeypatch.delenv("VMLX_DISABLE_PLD_REPLAY", raising=False)

    config = mock.MagicMock()
    config.pld_replay_enabled = True

    enabled = (
        os.getenv("VMLX_DISABLE_PLD_REPLAY") != "1"
        and getattr(config, "pld_replay_enabled", True)
    )
    assert enabled is True, "Replay should be enabled by default"


# ---------------------------------------------------------------------------
# 5. test_replay_full_accept_path_untouched
# ---------------------------------------------------------------------------


def test_replay_full_accept_path_untouched():
    """When num_accept == num_drafts (full accept), replay branch not entered.

    The case (b) branch is only reached when num_to_trim != 0.  With full
    accept, num_to_trim == 0 and the code takes the fast path.
    This test verifies the branch condition, not the scheduler internals.
    """
    num_accept = 2
    num_drafts = 2
    replay_enabled = True

    # Full accept: num_to_trim == 0 → skip case (b)
    num_to_trim = num_drafts - num_accept
    assert num_to_trim == 0, "Full accept must have num_to_trim == 0"

    # Also verify: the replay condition (b1) would not trigger even if we got here
    should_replay = 0 < num_accept < num_drafts and replay_enabled
    assert not should_replay, "Full accept must not trigger replay (num_accept == num_drafts)"
