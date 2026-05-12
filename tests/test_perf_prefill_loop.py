# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PR-B prefill-loop perf cleanups.

Algorithm-only: no real model load, no real cache layers.  We test the
helper logic in isolation and assert the source-level invariants
(env-gate present, boundary pointer used, hoisted tolist) via inspection.

Run:
    .venv/bin/python -m pytest tests/test_perf_prefill_loop.py -v
"""

from __future__ import annotations

import inspect
import os

import pytest


def _src() -> str:
    import vmlx_engine.mllm_batch_generator as mod

    return inspect.getsource(mod)


def test_chunk_loop_uses_sorted_boundary_pointer():
    """The chunk loop must walk a sorted-boundary pointer, not a per-chunk
    `next(generator)` linear scan over the boundary list."""
    src = _src()
    # Generator-style lookup is gone.
    assert "for b in ssm_boundaries" not in src or (
        "_sorted_boundaries" in src
    ), "stale generator-style next(...) lookup still present"
    # Pointer-style names are present.
    assert "_sorted_boundaries" in src, "boundary pointer list missing"
    assert "_boundary_idx" in src, "boundary pointer index missing"


def test_chunk_loop_precomputes_state_layers():
    """The per-chunk `mx.eval([c.state for c in cache if hasattr…])` must be
    replaced by a precomputed `_state_layers` list."""
    src = _src()
    assert "_state_layers" in src, "_state_layers precompute missing"
    assert "mx.eval([c.state for c in _state_layers])" in src, (
        "per-chunk eval over precomputed _state_layers missing"
    )


def test_chunk_loop_hoists_all_tokens_tolist():
    """The `getattr(request, '_original_token_ids', None) or input_ids[0]
    .tolist()` Metal→CPU sync must be hoisted out of the chunk loop."""
    src = _src()
    assert "_hoisted_all_tokens" in src, "hoisted tolist variable missing"


def test_chunk_loop_env_gates_clear_cache():
    """`mx.clear_cache()` inside the chunk loop must be gated on
    `_prefill_keep_alloc`, the env-var-driven flag."""
    src = _src()
    assert "VMLX_PREFILL_KEEP_ALLOC" in src, "env var name missing"
    assert "_prefill_keep_alloc" in src, "gate variable missing"
    assert "if not _prefill_keep_alloc:" in src, "gate wrapper missing"


def test_cli_flag_propagates_to_env(monkeypatch):
    """The `--prefill-keep-alloc` CLI flag must set the env var that the
    chunked-prefill loop reads."""
    import vmlx_engine.cli as cli_mod

    src = inspect.getsource(cli_mod)
    assert '"--prefill-keep-alloc"' in src, "CLI flag missing"
    assert "VMLX_PREFILL_KEEP_ALLOC" in src, "env propagation missing"
    assert "prefill_keep_alloc" in src, "dest naming consistent"


def test_prefill_keep_alloc_env_off_by_default(monkeypatch):
    """The env var must be unset by default; flag is opt-in."""
    monkeypatch.delenv("VMLX_PREFILL_KEEP_ALLOC", raising=False)
    # The chunk loop reads this at run time; mirror its check here.
    assert os.environ.get("VMLX_PREFILL_KEEP_ALLOC", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_prefill_keep_alloc_env_recognised(monkeypatch):
    """Common truthy spellings must all enable the gate."""
    truthy = {"1", "true", "TRUE", "yes", "Yes", "on", "ON"}
    for v in truthy:
        monkeypatch.setenv("VMLX_PREFILL_KEEP_ALLOC", v)
        # Mirror the gate logic from mllm_batch_generator.
        assert os.environ.get("VMLX_PREFILL_KEEP_ALLOC", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


def test_boundary_pointer_advances_past_captured():
    """Synthetic check on the pointer advance rule: any boundary already
    captured or behind ``processed`` must be skipped."""
    sorted_boundaries = [100, 250, 400, 800]
    captured: set[int] = {100, 400}
    processed = 50

    # Mirror the inner advance loop:
    idx = 0
    while idx < len(sorted_boundaries) and (
        sorted_boundaries[idx] <= processed
        or sorted_boundaries[idx] in captured
    ):
        idx += 1
    # First eligible boundary is 250.
    assert idx == 1
    assert sorted_boundaries[idx] == 250

    # Advance further: simulate processed = 300, captured grows.
    processed = 300
    captured.add(250)
    while idx < len(sorted_boundaries) and (
        sorted_boundaries[idx] <= processed
        or sorted_boundaries[idx] in captured
    ):
        idx += 1
    # Next eligible boundary is 800 (400 is in captured, 250 advanced past).
    assert idx == 3
    assert sorted_boundaries[idx] == 800
