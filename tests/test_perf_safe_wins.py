# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PR-A safe perf wins.

Coverage:
- `MLLMBatch.index_of` / `has_uid` O(1) lookup; invalidated by
  ``filter`` and ``extend``.
- `mlx_lm.sample_utils` symbols importable at module scope from
  `vmlx_engine.mllm_batch_generator`.
- `ssm_companion_cache._clone_states` no longer multiplies `lengths` by
  one (drops the no-op Metal kernel).
- `ssm_companion_disk_store` no longer imports `copy.deepcopy` (post-load
  deep-copy removed; `_clone_states` upstream handles isolation).

Run:
    .venv/bin/python -m pytest tests/test_perf_safe_wins.py -v
"""

from __future__ import annotations

import importlib
import inspect

import mlx.core as mx

from vmlx_engine.mllm_batch_generator import MLLMBatch


def _make_batch(uids):
    """Build a minimal MLLMBatch for index-of testing.

    The ``requests`` list must be the same length as ``uids`` so that
    :meth:`MLLMBatch.filter` can index every parallel list cleanly.
    """
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
    assert batch.has_uid(20) is True
    assert batch.has_uid(99) is False


def test_index_of_after_filter_invalidates_cache():
    batch = _make_batch([10, 20, 30])
    # Prime the cache.
    assert batch.index_of(20) == 1
    # Filter keeps uids[0] and uids[2] → new positions 0 and 1.
    batch.filter([0, 2])
    assert batch.has_uid(10) is True
    assert batch.has_uid(20) is False  # filtered out
    assert batch.has_uid(30) is True
    assert batch.index_of(10) == 0
    assert batch.index_of(30) == 1


def test_index_of_after_extend_invalidates_cache():
    batch = _make_batch([10, 20])
    other = _make_batch([30, 40])
    # Prime cache for first batch.
    assert batch.index_of(20) == 1
    batch.extend(other)
    assert batch.has_uid(30) is True
    assert batch.has_uid(40) is True
    assert batch.index_of(40) == 3
    # Order preserved: original entries keep their positions.
    assert batch.index_of(10) == 0
    assert batch.index_of(20) == 1


def test_sample_helpers_imported_at_module_scope():
    """The sampler helpers must live at module scope to skip the per-request
    `from mlx_lm.sample_utils import make_sampler` round-trip."""
    mod = importlib.import_module("vmlx_engine.mllm_batch_generator")
    assert hasattr(mod, "make_sampler"), "make_sampler not hoisted"
    assert hasattr(mod, "make_logits_processors"), "make_logits_processors not hoisted"
    # Belt and suspenders: ensure no callsite still does the *isolated* local
    # imports (`make_sampler` alone or `make_logits_processors` alone).  We
    # accept the consolidated module-level form `make_logits_processors,
    # make_sampler` which is what the hoist produced.
    src = inspect.getsource(mod)
    assert "from mlx_lm.sample_utils import make_sampler\n" not in src, (
        "stale local `from mlx_lm.sample_utils import make_sampler` remains"
    )
    assert "from mlx_lm.sample_utils import make_logits_processors\n" not in src, (
        "stale local `from mlx_lm.sample_utils import make_logits_processors` remains"
    )


def test_ssm_clone_states_drops_force_materialize_multiplier():
    """The `c.lengths = mx.array(c.lengths) * 1` no-op multiply must be gone."""
    import vmlx_engine.utils.ssm_companion_cache as mod
    src = inspect.getsource(mod._SSMCompanionCacheBase._clone_states) if hasattr(mod, "_SSMCompanionCacheBase") else inspect.getsource(mod)
    assert "mx.array(c.lengths) * 1" not in src, (
        "lengths force-materialize multiply still present"
    )
    # New form should still materialize.
    assert "mx.array(c.lengths)" in src
    assert "_mx_materialize(c.lengths)" in src


def test_ssm_disk_store_no_deepcopy_import():
    """`ssm_companion_disk_store` should no longer import `deepcopy` — the
    L1 fetch path's `_clone_states` already isolates the returned states.
    """
    import vmlx_engine.utils.ssm_companion_disk_store as mod
    src = inspect.getsource(mod)
    assert "from copy import deepcopy" not in src, "stale deepcopy import"
    assert "deepcopy(s)" not in src, "stale per-state deepcopy call"
