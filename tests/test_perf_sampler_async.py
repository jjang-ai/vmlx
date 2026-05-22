# SPDX-License-Identifier: Apache-2.0
"""Contracts for the safe PR #164 sampler/clone cleanup subset."""

from __future__ import annotations

import inspect
import types


def _make_req(
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    repetition_penalty=1.0,
):
    return types.SimpleNamespace(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
    )


def test_shared_params_fast_path_all_equal():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    assert _batch_shares_sampler_params([_make_req(), _make_req()]) is True


def test_shared_params_fast_path_one_differs():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    assert _batch_shares_sampler_params([_make_req(), _make_req(top_p=0.9)]) is False


def test_shared_params_fast_path_repetition_penalty_disables():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    assert (
        _batch_shares_sampler_params([_make_req(), _make_req(repetition_penalty=1.1)])
        is False
    )


def test_shared_params_fast_path_empty_returns_false():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    assert _batch_shares_sampler_params([]) is False


def test_shared_params_fast_path_repetition_penalty_none_is_ok():
    from vmlx_engine.mllm_batch_generator import _batch_shares_sampler_params

    assert (
        _batch_shares_sampler_params(
            [_make_req(repetition_penalty=None), _make_req(repetition_penalty=None)]
        )
        is True
    )


def test_shared_params_fast_path_uses_request_sampler_not_generator_default():
    """Shared requests can differ from the generator default sampler.

    The fast path may batch the logits, but it must still use the sampler built
    from the shared request parameters. Calling ``self.sampler(logits)`` here
    would silently ignore per-request temperature/top-p/top-k/min-p overrides.
    """
    import vmlx_engine.mllm_batch_generator as mod

    source = inspect.getsource(mod.MLLMBatchGenerator._step)
    assert "_batch_shares_sampler_params(batch.requests)" in source
    assert "shared_sampler = self._make_request_sampler(batch.requests[0])" in source
    assert "sampled = shared_sampler(logits)" in source
    assert "sampled = self.sampler(logits)" in source


def test_single_batch_clone_array_uses_single_mul():
    import vmlx_engine.utils.single_batch_generator as mod

    source = inspect.getsource(mod.SingleBatchGenerator._clone_array)
    assert "cloned = value + mx.zeros_like(value)" not in source
    assert "cloned = 1 * value" in source


def test_single_batch_stream_eval_hunk_not_applied_without_stream_audit():
    """Do not copy PR #164's async-eval hunk over current stream rehoming."""
    import vmlx_engine.utils.single_batch_generator as mod

    source = inspect.getsource(mod.SingleBatchGenerator._yield_current_and_schedule_next)
    assert "current_token = self._eval_on_stream(current_token)[0]" in source
    assert "mx.async_eval(current_token, current_logprobs)" not in source
