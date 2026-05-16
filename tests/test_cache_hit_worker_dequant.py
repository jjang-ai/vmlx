# SPDX-License-Identifier: Apache-2.0
"""Cache-hit dequantization must stay on the scheduler worker thread."""

from __future__ import annotations

import mlx.core as mx

from vmlx_engine.request import Request, SamplingParams
from vmlx_engine.scheduler import Scheduler, SchedulerConfig


class _TinyCache:
    pass


class _TinyModel:
    def make_cache(self):
        return [_TinyCache()]

    def __call__(self, input_ids, cache):
        batch, seq_len = input_ids.shape
        return mx.zeros((batch, seq_len, 8), dtype=mx.float32)


class _TinyTokenizer:
    clean_up_tokenization_spaces = False

    def decode(self, tokens):
        return "".join(str(int(t)) for t in tokens)


class _FakeMemoryCache:
    def __init__(self, cache):
        self.cache = cache

    def fetch(self, tokens):
        return self.cache, [tokens[-1]]


def _request():
    request = Request(
        request_id="r-cache-hit",
        prompt=[10, 11, 12],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )
    request.prompt_token_ids = [10, 11, 12]
    request.num_prompt_tokens = 3
    return request


class TestCacheHitWorkerDequant:
    def test_memory_cache_q4_hit_defers_dequant_until_worker_schedule(self):
        quantized_cache = ["quantized-cache"]
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                enable_prefix_cache=False,
                use_memory_aware_cache=False,
                kv_cache_quantization="none",
            ),
        )
        scheduler.memory_aware_cache = _FakeMemoryCache(quantized_cache)
        scheduler._kv_cache_bits = 4

        def _must_not_run_on_add_request(_cache):
            raise AssertionError("dequantization ran before scheduler worker")

        scheduler._dequantize_cache_for_use = _must_not_run_on_add_request

        request = _request()
        scheduler.add_request(request)

        assert request.prompt_cache is quantized_cache
        assert request._prompt_cache_needs_worker_dequant is True

    def test_worker_schedule_dequantizes_flagged_memory_cache_hit(self):
        quantized_cache = ["quantized-cache"]
        dequantized_cache = [_TinyCache()]
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
                use_memory_aware_cache=False,
                kv_cache_quantization="none",
            ),
        )
        scheduler.memory_aware_cache = _FakeMemoryCache(quantized_cache)
        scheduler._kv_cache_bits = 4
        scheduler._validate_cache = lambda _cache: True
        scheduler._dequantize_cache_for_use = lambda cache: (
            dequantized_cache if cache is quantized_cache else cache
        )

        request = _request()
        scheduler.add_request(request)
        scheduled = scheduler._schedule_waiting()

        assert scheduled == [request]
        assert request.prompt_cache is dequantized_cache
        assert request._prompt_cache_needs_worker_dequant is False
        assert scheduler.batch_generator._unprocessed[0].cache is dequantized_cache
