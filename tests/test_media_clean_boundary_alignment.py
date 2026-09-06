# SPDX-License-Identifier: Apache-2.0
"""The media clean boundary must be BLOCK-ALIGNED, or nothing ever pairs.

Measured live on Qwen3.8-27B VL (M5 Max, SSD-only tier, block_size 64), before
this fix:

    turn 2 stores: KV 7607 tokens, SSM companion at 7607
    turn 3 fetches: paged cache hit 117 blocks = 7488 tokens
    turn 3 asks for a companion at 7488 -> nothing (it is at 7607)
    "VLM prefix cache MISS: 7488 KV blocks found but no SSM companion
     state - full prefill required"

The paged chain can only ever be MATCHED on block boundaries, so a companion
stored at an unaligned N-1 is invisible to every future turn: a found
7,488-token hit was discarded on turn 3, again on turn 4, and TTFT climbed
35s -> 60s -> 85s until the prompt tripped a hard prefill guard and the
conversation was dead.

Giving up at most block_size-1 tokens of stored prefix buys all of that back.
"""

import inspect
from types import SimpleNamespace

from vmlx_engine.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchResponse,
)


class _Gen:
    """Bare generator exposing only the alignment helper under test."""

    def __init__(self, block_size=64):
        self.gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        from types import SimpleNamespace

        self.gen.block_aware_cache = SimpleNamespace(block_size=block_size)

    def aligned(self, boundary):
        return self.gen._ssm_block_aligned_boundary(boundary)


def test_the_helper_floors_to_a_block_and_steps_back_when_exact():
    g = _Gen(64)
    # 7607 is the real N-1 from the live failure. 7607 // 64 = 118 -> 7552.
    assert g.aligned(7607) == 7552
    # An EXACTLY aligned boundary steps back one block, so a future request
    # that diverges inside the final block still finds a checkpoint.
    assert g.aligned(7552) == 7488
    # Nothing usable below one block.
    assert g.aligned(64) == 0
    assert g.aligned(10) == 0


def test_media_kv_only_miss_teaches_the_next_clean_boundary():
    """A learned KV-only boundary must win once it covers all media.

    The live Qwen tool continuation matched 4,672 KV tokens, but the media
    lane stored its companion only at the later 6,144-token terminal boundary.
    Text-only prefills already honor ``_ssm_required_checkpoint_tokens``;
    media prefills need the same repair when the learned boundary is
    block-aligned and lies strictly after every media placeholder run.
    """
    generator = _Gen(64).gen
    generator._media_placeholder_token_ids = lambda: {99}
    request = SimpleNamespace(_ssm_required_checkpoint_tokens=4672)
    tokens = [1] * 4000 + [99] * 128 + [2] * 2200

    assert generator._media_clean_cache_boundary_for(request, tokens) == 4672


def test_media_required_boundary_inside_a_placeholder_run_moves_before_the_run():
    """A learned boundary that cuts THROUGH a placeholder run can never be
    restored (the tail would need partial media), so the store no longer
    targets it: it moves to the aligned boundary before the run. Seen live on
    Flash-Next (762-token variants) and 27B ("the 72-token tail still contains
    media placeholders" → hit declined) before the change."""
    generator = _Gen(64).gen
    generator._media_placeholder_token_ids = lambda: {99}
    request = SimpleNamespace(_ssm_required_checkpoint_tokens=4032)
    tokens = [1] * 4000 + [99] * 128 + [2] * 2200

    assert generator._media_clean_cache_boundary_for(request, tokens) == 3968


def test_media_required_boundary_between_whole_media_items_is_kept_for_non_qwen_families():
    generator = _Gen(64).gen
    generator._media_placeholder_token_ids = lambda: {99}
    request = SimpleNamespace(_ssm_required_checkpoint_tokens=4160)
    # image A (4000..4128), text, image B (4200..4328): 4160 sits between whole items
    tokens = [1] * 4000 + [99] * 128 + [3] * 72 + [99] * 128 + [2] * 2200

    assert generator._media_clean_cache_boundary_for(request, tokens) == 4160


def test_in_media_clean_prefill_encodes_full_media_then_forwards_exact_prefix():
    """The repair must never feed full pixels to truncated placeholders."""
    import mlx.core as mx

    calls = []

    class FakeFeatures:
        def __init__(self):
            self.inputs_embeds = mx.arange(30).reshape(1, 10, 3)

        def to_dict(self):
            return {"inputs_embeds": self.inputs_embeds}

    class FakeModel:
        def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
            calls.append(("embed", input_ids.tolist(), pixel_values))
            language._position_ids = mx.arange(30).reshape(3, 1, 10)
            return FakeFeatures()

    class FakeLanguage:
        _rope_deltas = "main-rope"
        _position_ids = "main-position"

        def make_cache(self):
            return []

        def __call__(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            **kwargs,
        ):
            calls.append(
                (
                    "lm",
                    inputs.tolist(),
                    inputs_embeds.tolist(),
                    kwargs["position_ids"].tolist(),
                )
            )
            return mx.array([[0.0]])

    language = FakeLanguage()
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.language_model = language
    generator.model = FakeModel()
    generator._model_type = "qwen3_5"
    generator._cache_model = None
    generator._media_placeholder_token_ids = lambda: {3}
    generator._media_prefill_chunk_tokens = lambda _seq_len: 4
    pixel_marker = object()
    full_tokens = [0, 1, 2, 3, 3, 3, 3, 7, 8, 9]
    request = SimpleNamespace(
        request_id="qwen-in-media-repair",
        input_ids=mx.array([full_tokens]),
        _original_token_ids=full_tokens,
        pixel_values=pixel_marker,
        video_pixel_values=None,
        image_grid_thw=mx.array([[1, 1, 4]]),
        video_grid_thw=None,
        attention_mask=None,
        extra_kwargs={},
    )

    # Placeholder run is [3, 7); boundary 6 is deliberately inside it.
    result = generator._prefill_for_clean_media_prefix_cache(
        request, full_tokens[:6]
    )

    assert result == []
    assert calls[0] == ("embed", [full_tokens], pixel_marker)
    assert [entry[1] for entry in calls[1:]] == [
        [[0, 1, 2, 3]],
        [[3, 3]],
    ]
    assert [entry[2] for entry in calls[1:]] == [
        [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]],
        [[[12, 13, 14], [15, 16, 17]]],
    ]
    assert language._rope_deltas == "main-rope"
    assert language._position_ids == "main-position"


def test_before_media_clean_prefill_also_uses_full_conditioned_embeddings():
    """The live 4,352 hit ended before 580 later media placeholders."""
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.language_model = object()
    generator._media_placeholder_token_ids = lambda: {99}
    observed = {}
    generator._prefill_for_exact_media_embedding_prefix_cache = (
        lambda request, tokens, full_tokens: observed.update(
            tokens=tokens, full_tokens=full_tokens
        ) or "exact"
    )
    request = SimpleNamespace(
        _original_token_ids=[1] * 5 + [99] * 3 + [2] * 2
    )

    result = generator._prefill_for_clean_media_prefix_cache(
        request, [1] * 4
    )

    assert result == "exact"
    assert observed == {
        "tokens": [1] * 4,
        "full_tokens": [1] * 5 + [99] * 3 + [2] * 2,
    }


def test_qwen_hybrid_media_tail_admits_only_a_pure_text_prefix():
    import mlx.core as mx

    class Wrapper:
        def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
            return None

    class Language:
        def __call__(self, inputs, inputs_embeds=None, cache=None, **kwargs):
            return None

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator._model_type = "qwen3_5"
    generator.model = Wrapper()
    generator.language_model = Language()
    generator._media_safe_capture_limit = lambda _tokens: 5
    generator._media_prefix_cache_allowed = lambda _request, _tokens: True
    request = SimpleNamespace(input_ids=mx.array([list(range(12))]))

    result = generator._prepare_qwen_hybrid_media_tail_for_cache_hit(
        request, list(range(10)), 4
    )

    assert result == {
        "kind": "qwen_hybrid_conditioned_media_tail",
        "conditioned_full_tokens": 12,
        "conditioned_tail_tokens": 8,
    }
    assert request._qwen_media_tail_cached_tokens == 4
    assert request._qwen_media_tail_full_input_ids.tolist() == [list(range(12))]
    assert generator._prepare_qwen_hybrid_media_tail_for_cache_hit(
        request, list(range(10)), 6
    ) is None


def test_qwen_hybrid_media_tail_forwards_conditioned_suffix_over_native_cache():
    import mlx.core as mx

    calls = []

    class Features:
        def __init__(self):
            self.inputs_embeds = mx.arange(30).reshape(1, 10, 3)

        def to_dict(self):
            return {"inputs_embeds": self.inputs_embeds}

    class Wrapper:
        def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
            calls.append(("embed", input_ids.tolist(), pixel_values))
            language._position_ids = mx.arange(30).reshape(3, 1, 10)
            return Features()

    class Language:
        _position_ids = None

        def __call__(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            **kwargs,
        ):
            calls.append(
                (
                    "lm",
                    inputs.tolist(),
                    inputs_embeds.tolist(),
                    kwargs["position_ids"].tolist(),
                )
            )
            return SimpleNamespace(logits=mx.array([[1.0]]))

    language = Language()
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.model = Wrapper()
    generator.language_model = language
    generator._media_prefill_chunk_tokens = lambda _seq_len: 4
    full_ids = mx.array([list(range(10))])
    request = SimpleNamespace(
        request_id="qwen-tail",
        _qwen_media_tail_full_input_ids=full_ids,
        _qwen_media_tail_cached_tokens=4,
        _cached_tokens=4,
        input_ids=full_ids[:, 4:],
        pixel_values=object(),
        image_grid_thw=object(),
        video_pixel_values=None,
        video_grid_thw=None,
    )
    cache = []

    output = generator._run_qwen_conditioned_media_tail(
        request,
        request.input_ids,
        cache,
        {"pixel_values": request.pixel_values, "cache": cache},
    )

    assert output.logits.tolist() == [[1.0]]
    assert calls[0][0:2] == ("embed", [list(range(10))])
    assert [entry[1] for entry in calls[1:]] == [
        [[4, 5, 6, 7]],
        [[8, 9]],
    ]
    assert [entry[2] for entry in calls[1:]] == [
        [[[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]],
        [[[24, 25, 26], [27, 28, 29]]],
    ]
    assert request.pixel_values is None
    assert not hasattr(request, "_qwen_media_tail_cached_tokens")


def test_capture_uses_the_aligned_length_not_n_minus_1():
    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    assert "_clean_media_len = self._media_clean_cache_boundary_for(" in src, (
        "the clean media capture is back on the unaligned N-1 boundary"
    )
    assert "_media_tokens[:_clean_media_len]" in src, (
        "the prefill still runs over N-1 rather than the aligned prefix"
    )
    assert "req._media_clean_prefix_len = _clean_media_len" in src


def test_the_companion_store_uses_the_same_length_as_the_capture():
    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    idx = src.index("stored clean media SSM")
    window = src[max(0, idx - 1400) : idx]
    assert '_media_clean_prefix_len' in window, (
        "the companion is stored at a different length than the cache "
        "actually covers — it would claim state it does not have"
    )


def test_the_response_carries_a_separate_clean_store_key():
    """usage.prompt_tokens is derived from prompt_token_ids.

    The scheduler sets request.num_prompt_tokens = len(prompt_token_ids), and
    that becomes the user-visible usage.prompt_tokens. Shortening it to the
    aligned length would misreport every media request, so the aligned key
    travels on its own field.
    """
    assert "clean_store_token_ids" in MLLMBatchResponse.__dataclass_fields__
    assert (
        MLLMBatchResponse.__dataclass_fields__["clean_store_token_ids"].default
        is None
    )

    src = inspect.getsource(MLLMBatchGenerator._next)
    assert "clean_store_token_ids=_clean_store_tokens" in src
    assert "_orig[: _clean_len + 1]" in src, (
        "the store key must be aligned+1 so the N-1 payload contract holds"
    )


def test_the_scheduler_prefers_the_clean_store_key():
    import vmlx_engine.mllm_scheduler as sched

    src = inspect.getsource(sched)
    occurrences = src.count('"clean_store_token_ids", None)')
    assert occurrences >= 2, (
        "both _extracted_tokens assignment sites must prefer the aligned key; "
        "fixing one of two is the default failure mode here (found %d)"
        % occurrences
    )


def test_a_hit_that_does_not_cover_the_media_span_is_declined():
    """Dropping pixel_values is only safe when the hit COVERS the image.

    A warm media turn is cheap precisely because the hybrid hit path sets
    req.pixel_values = None -- the vision tower is skipped since the image's
    KV is already in the restored prefix. That reasoning holds only if the
    cached prefix actually contains the image. If the tail still carries
    placeholders, forwarding it with no pixels embeds them as ordinary text
    tokens and the model answers fluently about an image it never saw. Nothing
    raises, nothing logs, and the answer looks fine.

    So the hit is refused in that case. A full prefill is slow; a confidently
    wrong answer is worse.
    """
    import inspect

    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    # The current implementation uses a media-tail admission helper rather
    # than the old local `_tail_has_media` variable. A paged hit may clear
    # processor payloads only after the tail is proven media-free or a
    # family-specific whole-item re-encode is admitted; otherwise it must
    # discard the hit and cold-prefill.
    assert "_tokens_contain_media_placeholders" in src
    assert "_prepare_muse_media_tail_for_cache_hit" in src
    assert "_prepare_dots3_media_tail_for_cache_hit" in src
    assert "_prepare_qwen_hybrid_media_tail_for_cache_hit" in src
    assert 'reason="media_placeholders_in_uncached_tail"' in src
    assert "_discard_request_cache_hit" in src
    assert "_clear_mllm_request_media_payloads(req)" in src
