# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PR-C sampler + async pipelining perf cleanups.

Covers:
- `_batch_shares_sampler_params` fast-path predicate (shared / mixed /
  repetition-penalty / empty).
- `single_batch_generator._clone_array` uses `1 * value` (no `zeros_like`).
- `single_batch_generator` decode step uses `mx.async_eval` for the
  token + logprobs pair.

Run:
    .venv/bin/python -m pytest tests/test_perf_sampler_async.py -v
"""

from __future__ import annotations

import inspect
import types

import pytest


def _make_req(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0):
    """Build a duck-typed request that satisfies the helper's getattr probes."""
    req = types.SimpleNamespace()
    req.temperature = temperature
    req.top_p = top_p
    req.top_k = top_k
    req.min_p = min_p
    req.repetition_penalty = repetition_penalty
    return req


def test_shared_params_fast_path_all_equal():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    reqs = [_make_req() for _ in range(4)]
    assert _batch_shares_sampler_params(reqs) is True


def test_shared_params_fast_path_one_differs():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    reqs = [_make_req() for _ in range(3)]
    reqs.append(_make_req(temperature=0.7))
    assert _batch_shares_sampler_params(reqs) is False


def test_shared_params_fast_path_repetition_penalty_disables():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    # All same params, but one has a non-trivial rep penalty.
    reqs = [_make_req() for _ in range(3)]
    reqs.append(_make_req(repetition_penalty=1.2))
    assert _batch_shares_sampler_params(reqs) is False


def test_shared_params_fast_path_first_request_repetition_penalty():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    # First request has rep penalty — short-circuits before scanning rest.
    reqs = [
        _make_req(repetition_penalty=1.1),
        _make_req(),
        _make_req(),
    ]
    assert _batch_shares_sampler_params(reqs) is False


def test_shared_params_fast_path_empty_returns_false():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    assert _batch_shares_sampler_params([]) is False


def test_shared_params_fast_path_repetition_penalty_none_is_ok():
    """`repetition_penalty == None` (unset) is treated as 1.0 → fast-path ok."""
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    reqs = [_make_req(repetition_penalty=None) for _ in range(2)]
    assert _batch_shares_sampler_params(reqs) is True


def test_single_batch_clone_array_uses_single_mul():
    """`_clone_array` should use `1 * value` instead of the
    `value + mx.zeros_like(value)` two-op pattern."""
    import vmlx_engine.utils.single_batch_generator as mod

    src = inspect.getsource(mod._cls_clone_array if hasattr(mod, "_cls_clone_array") else mod)
    # Stale form (the code expression, not the explanatory comment):
    # if `value + mx.zeros_like(value)` survives, the stale add+alloc lives
    # on.  The explanatory comment is allowed; only the executable
    # expression is checked.
    assert "cloned = value + mx.zeros_like(value)" not in src, (
        "stale `value + mx.zeros_like(value)` add+alloc in _clone_array"
    )
    # New form present.
    assert "cloned = 1 * value" in src


def test_single_batch_uses_async_eval_for_token_pair():
    """The decode step's `mx.eval(current_token); mx.eval(current_logprobs)`
    pair must be a single `mx.async_eval(current_token, current_logprobs)`."""
    import vmlx_engine.utils.single_batch_generator as mod

    src = inspect.getsource(mod)
    assert "mx.async_eval(current_token, current_logprobs)" in src, (
        "async_eval pair missing"
    )
    # The standalone `mx.eval(current_logprobs)` line should be gone.
    assert "mx.eval(current_logprobs)" not in src, (
        "stale standalone mx.eval(current_logprobs) remains"
    )
