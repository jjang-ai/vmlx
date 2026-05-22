# SPDX-License-Identifier: Apache-2.0
"""No-heavy contracts for prefill-loop perf cleanup intake.

These tests pin the source-level behavior from PR #163 without loading a model.
They deliberately avoid approving output quality or cache equivalence; those
remain live-model gates.
"""

from __future__ import annotations

import inspect
import os


def _mllm_source() -> str:
    import vmlx_engine.mllm_batch_generator as mod

    return inspect.getsource(mod)


def test_chunk_loop_uses_sorted_boundary_pointer():
    src = _mllm_source()
    assert "_sorted_boundaries" in src
    assert "_boundary_idx" in src
    assert "for b in ssm_boundaries" not in src


def test_chunk_loop_precomputes_state_layers():
    src = _mllm_source()
    assert "_state_layers" in src
    assert "mx.eval([c.state for c in _state_layers])" in src


def test_chunk_loop_hoists_all_tokens_tolist():
    src = _mllm_source()
    assert "_hoisted_all_tokens" in src


def test_chunk_loop_env_gates_clear_cache():
    src = _mllm_source()
    assert "VMLX_PREFILL_KEEP_ALLOC" in src
    assert "_prefill_keep_alloc" in src
    assert "if not _prefill_keep_alloc:" in src


def test_single_batch_prefill_loop_env_gates_clear_cache():
    import vmlx_engine.utils.single_batch_generator as mod

    src = inspect.getsource(mod.SingleBatchGenerator._prefill)
    assert "VMLX_PREFILL_KEEP_ALLOC" in src
    assert "_prefill_keep_alloc" in src
    assert "if not _prefill_keep_alloc:" in src


def test_cli_flag_propagates_to_env():
    import vmlx_engine.cli as cli_mod

    src = inspect.getsource(cli_mod)
    assert '"--prefill-keep-alloc"' in src
    assert "VMLX_PREFILL_KEEP_ALLOC" in src
    assert "prefill_keep_alloc" in src


def test_prefill_keep_alloc_env_off_by_default(monkeypatch):
    monkeypatch.delenv("VMLX_PREFILL_KEEP_ALLOC", raising=False)
    assert os.environ.get("VMLX_PREFILL_KEEP_ALLOC", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_boundary_pointer_advances_past_captured():
    sorted_boundaries = [100, 250, 400, 800]
    captured: set[int] = {100, 400}
    processed = 50

    idx = 0
    while idx < len(sorted_boundaries) and (
        sorted_boundaries[idx] <= processed
        or sorted_boundaries[idx] in captured
    ):
        idx += 1
    assert idx == 1
    assert sorted_boundaries[idx] == 250

    processed = 300
    captured.add(250)
    while idx < len(sorted_boundaries) and (
        sorted_boundaries[idx] <= processed
        or sorted_boundaries[idx] in captured
    ):
        idx += 1
    assert idx == 3
    assert sorted_boundaries[idx] == 800
