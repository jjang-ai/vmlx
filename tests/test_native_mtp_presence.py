from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.native_mtp_presence import NativeMTPPresence
from vmlx_engine.utils.token_logits_processors import make_openai_token_penalty_processor


@pytest.mark.parametrize("presence", [1.5, -0.5])
@pytest.mark.parametrize("temperature", [0., .7])
def test_native_presence_requests_remain_eligible(monkeypatch, presence, temperature):
    import vmlx_engine.mllm_batch_generator as mbg

    monkeypatch.setattr(mbg, "_NATIVE_MTP_STOCHASTIC_ACCEPT", True)
    generator = mbg.MLLMBatchGenerator.__new__(mbg.MLLMBatchGenerator)
    generator._model_type = "qwen4_exp"
    generator.language_model = SimpleNamespace(
        mtp=object(), make_mtp_cache=lambda: [], mtp_forward=lambda *a, **kw: None,
    )
    request = mbg.MLLMBatchRequest(
        uid=0, request_id="presence", prompt="", max_tokens=16,
        temperature=temperature, presence_penalty=presence,
    )
    request._original_token_ids = [0]
    assert generator._native_mtp_disabled_reason_for_request(request) is None


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("presence", [1.5, -0.5])
def test_causal_rows_match_ar_including_rejected_and_duplicate_proposals(dtype, presence):
    bias = {0: .2, 1: -.4}
    transform = NativeMTPPresence(presence, bias)
    logits = mx.array([[8., 7., 6., 5.]] * 4, dtype=dtype)
    for history, proposals in [
        ([0, 1, 0], [2, 2, 3]),
        ([0, 1, 0, 1], [3, 2, 3]),  # previous proposals rejected
        ([3], [3, 0, 1]),  # history reset, not an append
    ]:
        actual = transform(logits, history, [mx.array([t]) for t in proposals])
        expected = []
        for i in range(4):
            processor = make_openai_token_penalty_processor(
                presence_penalty=presence, logit_bias=bias
            )
            expected.append(processor(history + proposals[:i], logits[i]))
        assert bool(mx.array_equal(actual, mx.stack(expected)))


def test_presence_sampler_preserves_target_law_and_separate_proposal_law():
    from vmlx_engine.mllm_batch_generator import (
        MLLMBatchGenerator, _native_mtp_sample_one, _native_mtp_sample_rows,
    )
    from vmlx_engine.native_mtp_acceptance import accept_lp_for

    request = SimpleNamespace(
        temperature=.7, top_p=.8, top_k=3, min_p=0.,
        repetition_penalty=1., frequency_penalty=0., presence_penalty=1.5,
        logit_bias=None, enable_thinking=False,
        _original_token_ids=[0], output_tokens=[1],
    )
    wrapper = MLLMBatchGenerator._make_request_sampler(
        SimpleNamespace(_model_type="qwen4_exp"), request
    )
    base = wrapper._vmlx_mtp_base_sampler
    logits = mx.array([[4., 3.5, 3., 2.5]] * 3)
    processed = wrapper._vmlx_mtp_history_logits(
        logits, request.output_tokens, [mx.array([2]), mx.array([3])]
    )
    _, target_lps, _ = _native_mtp_sample_rows(processed, base)
    for i, context in enumerate([[0, 1], [0, 1, 2], [0, 1, 2, 3]]):
        processor = make_openai_token_penalty_processor(presence_penalty=1.5)
        expected = processor(context, logits[i:i+1])
        _, expected_lp = _native_mtp_sample_one(expected, base)
        assert bool(mx.array_equal(target_lps[i], expected_lp))
        assert bool(mx.array_equal(
            accept_lp_for(wrapper, target_lps[i]),
            accept_lp_for(base, expected_lp),
        ))
    _, proposal_lp = _native_mtp_sample_one(
        logits[:1], wrapper._vmlx_mtp_proposal_sampler
    )
    _, unmodified_lp = _native_mtp_sample_one(logits[:1], base)
    assert bool(mx.array_equal(proposal_lp, unmodified_lp))
    assert not bool(mx.array_equal(proposal_lp, target_lps[0]))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_presence_keeps_incremental_bias_rounding_across_seed_and_rejection(dtype):
    raw = mx.zeros((1, 4), dtype=dtype)
    bias = {1: 1.49, 2: -.49}
    processor = make_openai_token_penalty_processor(
        presence_penalty=1.5, logit_bias=bias
    )
    processor([0], raw)  # actual first-token AR prefill
    native = NativeMTPPresence(1.5, bias, processor=processor)
    reference = make_openai_token_penalty_processor(
        presence_penalty=1.5, logit_bias=bias
    )
    reference([0], raw)
    assert bool(mx.array_equal(native(raw, [0, 1]), reference([0, 1], raw)))
    matrix = mx.broadcast_to(raw, (3, 4))
    actual = native(matrix, [0, 1], [mx.array([2]), mx.array([2])])
    expected = mx.concatenate([
        reference([0, 1], raw),
        reference([0, 1, 2], raw),
        reference([0, 1, 2, 2], raw),
    ])
    assert bool(mx.array_equal(actual, expected))
    # Both proposals rejected; their adjustment must not contaminate AR's
    # shared committed state when a different correction is appended.
    reference = make_openai_token_penalty_processor(
        presence_penalty=1.5, logit_bias=bias
    )
    reference([0], raw)
    reference([0, 1], raw)
    assert bool(mx.array_equal(
        native(raw, [0, 1, 3]), reference([0, 1, 3], raw)
    ))


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("fused", [False, True])
def test_presence_seed_verify_rejection_and_queue_match_sequential_target(monkeypatch, depth, fused):
    import vmlx_engine.mllm_batch_generator as mbg

    monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", str(depth))  # diagnostic fixture only
    monkeypatch.setattr(mbg, "_NATIVE_MTP_FUSED_SYNC", fused)
    raw = mx.array([[4., 3.9, 3.8, 3.7]])

    class Cache:
        def __init__(self):
            self.tokens = []

        def is_trimmable(self):
            return True

        def trim(self, count):
            if count:
                self.tokens = self.tokens[:-count]

    class Model:
        mtp = object()

        def make_mtp_cache(self):
            return [Cache()]

        def __call__(self, tokens, cache=None, return_hidden=False, **kwargs):
            ids = tokens.reshape(-1).tolist()
            if cache:
                cache[0].tokens.extend(ids)
            logits = mx.broadcast_to(raw, (1, len(ids), 4))
            hidden = mx.ones((1, len(ids), 4))
            return (logits, hidden) if return_hidden else logits

        def mtp_forward(self, hidden, tokens, cache, return_hidden=False):
            logits = raw[None]
            return (logits, mx.ones((1, 1, 4))) if return_hidden else logits

    request = mbg.MLLMBatchRequest(
        uid=0, request_id="presence-cycle", prompt="", max_tokens=9,
        temperature=0., presence_penalty=1.5,
    )
    request.input_ids = mx.array([0])
    request._original_token_ids = [0]
    generator = mbg.MLLMBatchGenerator.__new__(mbg.MLLMBatchGenerator)
    generator._model_type = "qwen4_exp"
    generator.language_model = Model()
    generator.model = SimpleNamespace(language_model=generator.language_model)
    generator.unprocessed_requests = []
    generator.completion_batch_size = 1
    generator.stop_tokens = set()
    generator._prefill_errors = []
    generator._stats = mbg.MLLMBatchStats()
    first, lp = mbg._sample_mllm_prefill_logits(
        raw, generator._make_request_sampler(request)
    )
    generator.active_batch = mbg.MLLMBatch(
        uids=[0], request_ids=[request.request_id], y=first, logprobs=[lp],
        max_tokens=[request.max_tokens], num_tokens=[0],
        cache=[Cache()], requests=[request],
    )
    assert generator._seed_native_mtp_from_prefill(
        request, generator.active_batch.cache, first, [lp]
    )
    state = request._native_mtp_state
    responses = [generator._next()[0] for _ in range(request.max_tokens)]
    assert all(r.finish_reason != "error" for r in responses)
    actual = [r.token for r in responses]
    expected = []
    processor = make_openai_token_penalty_processor(presence_penalty=1.5)
    for _ in range(request.max_tokens):
        expected.append(int(mx.argmax(processor([0] + expected, raw)).item()))
    assert actual == expected == [1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert state.stats.cycles > 0
    assert state.stats.rejects > 0
