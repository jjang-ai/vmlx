# SPDX-License-Identifier: Apache-2.0
"""Adapted no-heavy contracts for PR #162 safe perf wins.

The submitted sampler-helper hoist is intentionally not included here because
current vMLX routes through ``vmlx_engine.sampling``. These tests cover the
cache/index pieces that are safe to adapt without changing sampler semantics.
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from vmlx_engine.mllm_batch_generator import MLLMBatch


def _make_batch(uids):
    n = len(uids)
    return MLLMBatch(
        uids=list(uids),
        request_ids=[f"r{u}" for u in uids],
        y=mx.zeros((n,), dtype=mx.int32),
        logprobs=[mx.zeros((1,)) for _ in uids],
        max_tokens=[0] * n,
        num_tokens=[0] * n,
        cache=[],
        requests=[object() for _ in uids],
    )


def test_index_of_basic():
    batch = _make_batch([10, 20, 30])
    assert batch.index_of(10) == 0
    assert batch.index_of(20) == 1
    assert batch.index_of(30) == 2


def test_has_uid_basic():
    batch = _make_batch([10, 20, 30])
    assert batch.has_uid(10) is True
    assert batch.has_uid(99) is False


def test_index_of_after_filter_invalidates_cache():
    batch = _make_batch([10, 20, 30])
    assert batch.index_of(20) == 1
    batch.filter([0, 2])
    assert batch.has_uid(20) is False
    assert batch.index_of(10) == 0
    assert batch.index_of(30) == 1


def test_index_of_after_extend_invalidates_cache():
    batch = _make_batch([10, 20])
    other = _make_batch([30, 40])
    assert batch.index_of(20) == 1
    batch.extend(other)
    assert batch.has_uid(40) is True
    assert batch.index_of(10) == 0
    assert batch.index_of(40) == 3


def test_scheduler_pld_uses_batch_uid_lookup_helpers_when_available():
    import vmlx_engine.scheduler as mod

    src = inspect.getsource(mod.Scheduler)
    assert "ab.has_uid(uid)" in src
    assert "ab.index_of(uid)" in src


def test_ssm_clone_states_drops_force_materialize_multiplier():
    import vmlx_engine.utils.ssm_companion_cache as mod

    src = inspect.getsource(mod.SSMCompanionCache._clone_states)
    assert "mx.array(c.lengths) * 1" not in src
    assert "mx.array(c.lengths)" in src
    assert "_mx_materialize(c.lengths)" in src


def test_ssm_disk_store_no_deepcopy_import_or_post_load_copy():
    import vmlx_engine.utils.ssm_companion_disk_store as mod

    src = inspect.getsource(mod)
    assert "from copy import deepcopy" not in src
    assert "deepcopy(s)" not in src
