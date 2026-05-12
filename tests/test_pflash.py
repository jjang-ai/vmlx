# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PFlash sparse-prefill helpers — issue #136.

Algorithm-only: no real drafter/target weights, no MLX model load. The
drafter forward is mocked with a synthetic logits matrix so we exercise:

    pflash_score_blocks → pflash_select_top_k → pflash_block_ranges
                                              → plan_sparse_prefill

Run:
    .venv/bin/python -m pytest tests/test_pflash.py -v
"""

from __future__ import annotations

import os

import pytest

import mlx.core as mx

from vmlx_engine.utils.pflash import (
    PFlashConfig,
    pflash_block_ranges,
    pflash_score_blocks,
    pflash_select_top_k,
    plan_sparse_prefill,
)


def test_pflash_config_validates_keep_ratio():
    with pytest.raises(ValueError):
        PFlashConfig(enabled=True, keep_ratio=0.0)
    with pytest.raises(ValueError):
        PFlashConfig(enabled=True, keep_ratio=1.1)
    PFlashConfig(enabled=True, keep_ratio=0.5)


def test_pflash_config_validates_block_size():
    with pytest.raises(ValueError):
        PFlashConfig(enabled=True, block_size=0)


def test_score_blocks_returns_entropy():
    # 3 blocks, vocab=4.  Block 0 is uniform (max entropy), block 2 is
    # one-hot (min entropy), block 1 is in between.
    logits = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0],    # uniform → high entropy
            [2.0, 1.0, 0.5, 0.0],    # skewed
            [10.0, 0.0, 0.0, 0.0],   # one-hot → low entropy
        ]
    )
    scores = pflash_score_blocks(logits, block_size=64)
    assert scores.shape == (3,)
    s = scores.tolist()
    # Block 0 must be the highest-entropy block, block 2 the lowest.
    assert s[0] > s[1] > s[2]


def test_score_blocks_rejects_wrong_shape():
    with pytest.raises(ValueError):
        pflash_score_blocks(mx.array([1.0, 2.0]), block_size=64)


def test_select_top_k_basic():
    scores = mx.array([0.1, 0.9, 0.5, 0.2, 0.8])
    mask = pflash_select_top_k(scores, keep_ratio=0.4)
    m = mask.tolist()
    # 0.4 * 5 = 2 → keep top 2: indices 1 (0.9) and 4 (0.8).
    assert sum(m) == 2
    assert m[1] is True
    assert m[4] is True


def test_select_top_k_pins_head_and_tail():
    # head=1, tail=1, middle=3.  keep_ratio=0.0 inside the middle, but head
    # and tail are unconditionally kept.
    scores = mx.array([0.1, 0.0, 0.0, 0.0, 0.5])
    mask = pflash_select_top_k(
        scores, keep_ratio=0.0, always_keep_head_blocks=1, always_keep_tail_blocks=1
    )
    m = mask.tolist()
    assert m[0] is True
    assert m[-1] is True
    assert sum(m) == 2


def test_select_top_k_keep_all():
    scores = mx.array([0.1, 0.9, 0.5])
    mask = pflash_select_top_k(scores, keep_ratio=1.0)
    assert mask.tolist() == [True, True, True]


def test_block_ranges_coalesces_adjacent():
    # blocks: keep 0-1, skip 2, keep 3-4
    mask = mx.array([True, True, False, True, True], dtype=mx.bool_)
    ranges = pflash_block_ranges(mask, block_size=10, seq_len=50)
    assert ranges == [(0, 20), (30, 50)]


def test_block_ranges_trailing_partial_block():
    # Last block is partial (only 5 of 10 tokens).
    mask = mx.array([True, True], dtype=mx.bool_)
    ranges = pflash_block_ranges(mask, block_size=10, seq_len=15)
    assert ranges == [(0, 15)]


def test_block_ranges_empty_mask():
    mask = mx.array([False, False, False], dtype=mx.bool_)
    ranges = pflash_block_ranges(mask, block_size=10, seq_len=30)
    assert ranges == []


def test_plan_sparse_prefill_end_to_end():
    # Build a synthetic 12-block, vocab=8 drafter that returns near-one-hot
    # in the middle blocks (low entropy → low score → dropped) and uniform
    # in blocks 4-7 (high entropy → high score → kept).
    def score_fn(start, end):
        rows = []
        for b in range(12):
            if 4 <= b <= 7:
                rows.append(mx.zeros((8,)))  # uniform-like → high entropy
            else:
                row = [-10.0] * 8
                row[0] = 10.0
                rows.append(mx.array(row))    # near one-hot → low entropy
        return mx.stack(rows, axis=0)

    cfg = PFlashConfig(
        enabled=True,
        drafter_model="stub",
        block_size=4,
        keep_ratio=0.5,
        min_seq_len=0,
        always_keep_head=0,
        always_keep_tail=0,
    )
    plan = plan_sparse_prefill(score_fn, seq_len=48, config=cfg)
    assert plan.total_blocks == 12
    # 0.5 * 12 = 6 blocks kept — the 4 high-entropy middle plus 2 more
    assert plan.kept_blocks == 6
    # Coverage = 24 / 48 = 0.5
    assert abs(plan.coverage() - 0.5) < 1e-6


def test_plan_sparse_prefill_empty_prompt():
    def score_fn(start, end):
        return mx.zeros((0, 8))

    cfg = PFlashConfig(enabled=True, drafter_model="stub", block_size=4)
    plan = plan_sparse_prefill(score_fn, seq_len=0, config=cfg)
    assert plan.total_blocks == 0
    assert plan.kept_blocks == 0
    assert plan.keep_ranges == []


def test_should_activate_pflash_requires_drafter(monkeypatch):
    from vmlx_engine.utils import pflash as pflash_mod

    cfg = PFlashConfig(
        enabled=True, drafter_model="stub", min_seq_len=100
    )
    pflash_mod.configure_pflash(cfg)
    # Drafter not loaded → not active even if seq_len qualifies.
    assert pflash_mod.should_activate_pflash(seq_len=10_000) is False
    # Below min_seq_len → never active.
    assert pflash_mod.should_activate_pflash(seq_len=50) is False


def test_should_activate_pflash_with_drafter(monkeypatch):
    from vmlx_engine.utils import pflash as pflash_mod
    from vmlx_engine.utils import pflash_drafter as drafter_mod

    cfg = PFlashConfig(
        enabled=True, drafter_model="stub", min_seq_len=100
    )
    pflash_mod.configure_pflash(cfg)
    monkeypatch.setattr(drafter_mod, "_drafter_model", object())
    try:
        assert pflash_mod.should_activate_pflash(seq_len=10_000) is True
        assert pflash_mod.should_activate_pflash(seq_len=50) is False
    finally:
        monkeypatch.setattr(drafter_mod, "_drafter_model", None)


def test_env_config_disabled_by_default(monkeypatch):
    from vmlx_engine.utils.pflash import config_from_env

    for name in (
        "VMLX_ENABLE_PFLASH",
        "VMLX_PFLASH_DRAFTER",
        "VMLX_PFLASH_BLOCK_SIZE",
        "VMLX_PFLASH_KEEP_RATIO",
        "VMLX_PFLASH_MIN_SEQ_LEN",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = config_from_env()
    assert cfg.enabled is False
    assert cfg.block_size == 256
    assert cfg.keep_ratio == 0.10
    assert cfg.min_seq_len == 8192


def test_env_config_enables(monkeypatch):
    from vmlx_engine.utils.pflash import config_from_env

    monkeypatch.setenv("VMLX_ENABLE_PFLASH", "1")
    monkeypatch.setenv("VMLX_PFLASH_DRAFTER", "mlx-community/Qwen3-0.6B-bf16")
    monkeypatch.setenv("VMLX_PFLASH_KEEP_RATIO", "0.05")
    monkeypatch.setenv("VMLX_PFLASH_MIN_SEQ_LEN", "32768")
    cfg = config_from_env()
    assert cfg.enabled is True
    assert cfg.drafter_model == "mlx-community/Qwen3-0.6B-bf16"
    assert cfg.keep_ratio == 0.05
    assert cfg.min_seq_len == 32768
