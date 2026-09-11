"""Reject doomed media-tail hits before companion reconstruction/derivation."""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


@pytest.mark.parametrize(
    "family,tokens,hit,declines",
    [
        ("qwen4_exp", [10, 20, 99, 40], 2, True),
        ("qwen4_exp", [10, 99, 30, 40], 3, False),
        ("qwen4_exp", [10, 20, 30, 40], 2, False),
        ("qwen3_5", [10, 20, 99, 40], 2, False),
        ("qwen3_5_moe", [10, 20, 99, 40], 2, False),
    ],
)
def test_early_media_admission_preserves_supported_and_complete_hits(family, tokens, hit, declines):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._model_type = family
    gen.block_aware_cache = MagicMock()
    gen._ssm_state_cache = MagicMock()
    gen._tokens_contain_media_placeholders = lambda ids: 99 in ids
    req = MLLMBatchRequest(uid=1, request_id="early-media", prompt="x")
    req._original_token_ids = tokens
    req._gen_prefix_tokens = [50]
    req.input_ids = mx.array([tokens + [50]])
    req.pixel_values = mx.ones((1, 3, 2, 2))
    pixels = req.pixel_values
    req._cache_execution = {"attempted_cached_tokens": hit}

    assert gen._decline_unsupported_hybrid_media_tail(req, tokens, hit) is declines
    gen._ssm_state_cache.fetch.assert_not_called()
    gen.block_aware_cache.reconstruct_cache.assert_not_called()
    assert req.pixel_values is pixels
    assert req.input_ids.tolist() == [tokens + [50]]
    if declines:
        gen.block_aware_cache.release_cache.assert_called_once_with(req.request_id)
        gen.block_aware_cache.adjust_cache_hit_credit.assert_called_once_with(req.request_id, accepted_tokens=0)
        assert req._cache_execution["cache_outcome"] == "discarded"
        assert req._cache_execution["attempted_cached_tokens"] == hit
        assert req._cache_execution["cached_tokens"] == 0
        assert req._cache_execution["fallback_reason"] == "media_placeholders_in_uncached_tail"
    else:
        gen.block_aware_cache.release_cache.assert_not_called()
