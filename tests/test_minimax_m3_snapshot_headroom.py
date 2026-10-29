# SPDX-License-Identifier: Apache-2.0
"""The prompt-boundary snapshot (a full copy of the prompt's cache, taken after prefill so the prefix cache can store
it) is skipped when the copy does not fit in the free Metal working set; the request is still served.

On the real M3.1 JANGHT bundle (95.5 GiB with its drafter, 107.5 GiB working set) a 65k-token prompt's 8.2 GiB copy
raised [METAL] Insufficient Memory inside the snapshot and failed the request (2026-10-28). Tiny synthetic M3 of
tests/test_minimax_m3_dspark.py; the working-set numbers are injected."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

import vmlx_engine.utils.single_batch_generator as sbg  # noqa: E402
from tests.test_minimax_m3_dspark import _ids, _model  # noqa: E402
from tests.test_minimax_m3_dspark_serving import _bundle, _drive, _greedy  # noqa: E402,F401


def _working_set(monkeypatch, active, maximum):
    monkeypatch.setattr(sbg, "get_effective_metal_working_set_bytes", lambda _mx=None: (active, maximum))


@pytest.mark.parametrize("lane", ["plain", "dspark"])
def test_snapshot_taken_with_headroom_and_skipped_without(tmp_path, monkeypatch, lane):
    from vmlx_engine.dspark.runtime import load_dspark_runtime
    from vmlx_engine.utils.dspark_text_generator import DSparkTextGenerator

    model = _model(seed=3)
    if lane == "dspark":
        object.__setattr__(model, "_vmlx_dspark_runtime", load_dspark_runtime(model, _bundle(tmp_path)))
    cls = DSparkTextGenerator if lane == "dspark" else sbg.SingleBatchGenerator
    prompt = _ids(645, 17)

    _working_set(monkeypatch, active=1 << 30, maximum=64 << 30)          # plenty of room
    gen = cls(model, sampler=_greedy(), max_tokens=12, prefill_step_size=256, stop_tokens=[])
    _, ref_tokens, ref_finish, snap = _drive(gen, [prompt], max_tokens=12)
    assert snap is not None and gen.prompt_snapshot_last_reason == "captured"
    assert gen.prompt_snapshot_last_estimated_bytes > 0

    est = gen.prompt_snapshot_last_estimated_bytes
    maximum = 64 << 30
    active = int(maximum * 0.98) - est // 2                              # headroom = half the copy
    _working_set(monkeypatch, active=active, maximum=maximum)
    gen = cls(model, sampler=_greedy(), max_tokens=12, prefill_step_size=256, stop_tokens=[])
    _, tokens, finish, snap = _drive(gen, [prompt], max_tokens=12)
    assert snap is None
    assert gen.prompt_snapshot_last_reason == "metal_headroom" and gen.prompt_snapshot_headroom_skips == 1
    assert 0 < gen.prompt_snapshot_last_headroom_bytes < est
    # the request itself is served unchanged
    assert tokens == ref_tokens and finish == ref_finish == "length"
    if lane == "dspark":
        object.__setattr__(model, "_vmlx_dspark_runtime", None)
