# SPDX-License-Identifier: Apache-2.0
"""Media-expanded prefill must chunk, so a long VL chat stops hitting a wall.

The one-shot forward pushed the WHOLE media-expanded prompt through the
language model in a single command buffer. Measured on an M5 Max: a
28,483-token prompt returned kIOGPUCommandBufferCallbackErrorOutOfMemory with
89GB free -- the problem is one enormous allocation, not total memory. Since a
chat client re-sends the image every turn, the prompt only grows, so a VL
conversation dies and never recovers.

The vision tower needs the whole image; the language model does not need the
whole sequence. Every wrapper here already exposes that seam.
"""

import os

import pytest

from vmlx_engine.mllm_batch_generator import (
    _MEDIA_PREFILL_CHUNK_FLOOR,
    _MEDIA_PREFILL_CHUNK_MIN_SEQ,
    _media_chunk_boundaries,
    _media_embed_kwarg_name,
    _media_placeholder_runs,
    _named_params,
)


class TestCapabilityDetection:
    """`**kwargs` is not support. It only looks like support."""

    def test_kwargs_absorption_does_not_count_as_a_named_parameter(self):
        def swallows(a, **kwargs):
            pass

        assert _named_params(swallows) == {"a"}
        assert "inputs_embeds" not in _named_params(swallows)

    def test_detects_both_spellings(self):
        class _A:
            def __call__(self, inputs, inputs_embeds=None, cache=None):
                pass

        class _B:
            def __call__(self, inputs, input_embeddings=None, cache=None):
                pass

        assert _media_embed_kwarg_name(_A()) == "inputs_embeds"
        assert _media_embed_kwarg_name(_B()) == "input_embeddings"

    def test_a_model_that_only_swallows_kwargs_is_not_chunkable(self):
        """Several wrappers here take position_ids into **kwargs and drop it.

        Treating that as support would build a chunked prefill on a model
        that ignores the per-chunk positions and returns confident garbage.
        """

        class _Swallower:
            def __call__(self, inputs, cache=None, **kwargs):
                pass

        assert _media_embed_kwarg_name(_Swallower()) is None

    def test_none_model_is_not_chunkable(self):
        assert _media_embed_kwarg_name(None) is None


class TestChunkBoundaries:
    def test_plain_split_when_there_is_no_media_run(self):
        assert _media_chunk_boundaries(10, 4, []) == [4, 8, 10]

    def test_a_boundary_inside_a_media_run_snaps_to_the_run_start(self):
        """Harmless today, not necessarily tomorrow.

        Post-merge every family here builds masks from the cache offset, so a
        mid-run split matches the one-shot result. It stops matching the
        moment a family builds a mask from whole-sequence image geometry --
        gemma4's config already asks for bidirectional vision attention even
        though the MLX language model does not implement it, and qwen3_vl's
        deepstack injection is keyed to visual rows in the current window.
        Snapping costs nothing, so it is not worth being clever about.
        """
        # run occupies [5, 12); a naive split at 8 would land inside it
        bounds = _media_chunk_boundaries(20, 8, [(5, 12)])
        assert 8 not in bounds
        assert bounds[0] == 5

    def test_snapping_always_makes_forward_progress(self):
        """A run starting at the current position must not stall the loop."""
        bounds = _media_chunk_boundaries(30, 4, [(0, 25)])
        assert bounds == sorted(set(bounds))
        assert bounds[-1] == 30
        assert all(b > 0 for b in bounds)
        # strictly increasing => the loop terminates
        assert all(b2 > b1 for b1, b2 in zip(bounds, bounds[1:]))

    def test_a_run_covering_the_whole_sequence_still_terminates(self):
        assert _media_chunk_boundaries(16, 4, [(0, 16)])[-1] == 16

    def test_degenerate_inputs(self):
        assert _media_chunk_boundaries(0, 8, []) == [0]
        assert _media_chunk_boundaries(10, 0, []) == [10]


class TestPlaceholderRuns:
    def test_finds_half_open_spans(self):
        ids = [1, 2, 99, 99, 99, 3, 4, 99, 5]
        assert _media_placeholder_runs(ids, {99}) == [(2, 5), (7, 8)]

    def test_run_touching_the_end(self):
        assert _media_placeholder_runs([1, 99, 99], {99}) == [(1, 3)]

    def test_no_media_ids_means_no_runs(self):
        assert _media_placeholder_runs([1, 2, 3], set()) == []
        assert _media_placeholder_runs(None, {99}) == []


class TestChunkSizing:
    def test_the_floor_is_large_on_purpose(self):
        """A SMALLER CHUNK DOES NOT REDUCE WEIGHT STREAMING -- IT MULTIPLIES IT.

        The chunk bounds only the terms that scale with it; the weights are
        re-read in full on every chunk. dots3 restreams ~85GB of expert
        weights per chunk, so a 64-token chunk paid that 32x more often than a
        2048-token one. Anyone tempted to shrink this to "be safe" is making
        long prompts slower, not safer.
        """
        assert _MEDIA_PREFILL_CHUNK_FLOOR >= 4096

    def test_short_media_prompts_stay_one_shot(self):
        """One-shot reads the weights once and was never the failing shape."""
        assert _MEDIA_PREFILL_CHUNK_MIN_SEQ >= _MEDIA_PREFILL_CHUNK_FLOOR


class _OneShotModel:
    """Callable stand-in for a VLM wrapper. `__call__` must live on the TYPE.

    A SimpleNamespace with a `__call__` attribute is NOT callable -- Python
    looks dunders up on the type, not the instance -- which is exactly how the
    first version of these tests managed to "fail" against working code.
    """

    def __init__(self, calls, **attrs):
        self._calls = calls
        for key, value in attrs.items():
            setattr(self, key, value)

    def __call__(self, ids, **kwargs):
        self._calls.append("one-shot")
        return "out"


class _EmbedsLM:
    def __call__(self, inputs, inputs_embeds=None, cache=None):
        return None


class _NoEmbedsLM:
    def __call__(self, inputs, cache=None, **kwargs):
        return None


class TestMediaForwardFallbacks:
    def test_bounded_glm_materializes_each_chunk_and_keeps_guard(self, monkeypatch):
        from types import SimpleNamespace
        import vmlx_engine.mllm_batch_generator as mllm

        events = []
        class Logits:
            def __getitem__(self, key):
                return self
        class LM:
            model_type = "glm5_next"
            def __call__(self, ids, inputs_embeds=None, cache=None):
                events.append("forward")
                return Logits()
        gen = self._gen(_OneShotModel([]), LM())
        gen._tight_memory_prefill_drain = True
        gen._media_placeholder_token_ids = lambda: set()
        gen.model.get_input_embeddings = lambda ids, **kw: SimpleNamespace(
            inputs_embeds=_FakeIds(2500)
        )
        monkeypatch.setenv("VMLX_GLM5_BOUNDED_MEDIA_PREFILL", "1")
        monkeypatch.setattr(mllm, "get_effective_metal_working_set_bytes", lambda mx: (100, 1000))
        monkeypatch.setattr(mllm, "hybrid_chunk_valve_check", lambda *a, **kw: events.append("guard"))
        monkeypatch.setattr(mllm, "prefill_valve_enabled", lambda: True)
        monkeypatch.setattr(mllm, "_materialize_prefill_cache_state", lambda c: events.append("state"))
        monkeypatch.setattr(mllm.mx, "eval", lambda *x: events.append("eval"))
        monkeypatch.setattr(mllm.mx, "clear_cache", lambda: events.append("clear"))
        monkeypatch.setattr(mllm.mx, "reset_peak_memory", lambda: None)
        monkeypatch.setattr(mllm.mx, "get_peak_memory", lambda: 200)
        gen._media_forward(SimpleNamespace(request_id="glm"), _FakeIds(2500), 2500, [object()], {})
        assert events.count("forward") == 3
        assert events.count("guard") == events.count("state") == 3
        assert events[0] == "eval"
        for i, event in enumerate(events):
            if event == "forward":
                assert events[i-1] == "guard" and events[i+1] == "state"

    def _gen(self, model, lm):
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.prefill_step_size = 2048
        gen.model = model
        gen.language_model = lm
        return gen

    def _run(self, gen):
        from types import SimpleNamespace

        return gen._media_forward(
            SimpleNamespace(request_id="r"),
            _FakeIds(30000),
            30000,
            [object()],
            {},
        )

    def test_falls_back_to_one_shot_without_an_embeddings_api(self):
        calls = []
        gen = self._gen(_OneShotModel(calls), _NoEmbedsLM())
        assert self._run(gen) == "out"
        assert calls == ["one-shot"]

    def test_env_kill_switch_forces_one_shot(self, monkeypatch):
        monkeypatch.setenv("VMLX_DISABLE_MEDIA_CHUNKED_PREFILL", "1")
        calls = []
        gen = self._gen(_OneShotModel(calls), _EmbedsLM())
        self._run(gen)
        assert calls == ["one-shot"]

    def test_no_chunked_prefill_protects_spans_it_does_not_forbid_chunking(self):
        """gemma4 sets this from a config that DEFAULTS to "vision".

        Treating it as an absolute kill switch made every gemma4 media prompt
        one-shot, and an 80,611-token conversation then died on
        `[metal::malloc] Attempting to allocate 207,940,266,272 bytes` against
        an 86.6GB cap -- every turn after it failed. What the flag protects is
        vision spans, and the chunker already keeps runs whole, so the intent
        is satisfiable without refusing to chunk.
        """
        from types import SimpleNamespace

        calls = []
        gen = self._gen(
            _OneShotModel(calls, no_chunked_prefill=True), _EmbedsLM()
        )
        gen.model.get_input_embeddings = lambda ids, **kw: SimpleNamespace(
            inputs_embeds=_FakeIds(30000)
        )
        gen._media_placeholder_token_ids = lambda: set()
        self._run(gen)
        assert calls == [], (
            "no_chunked_prefill still forces one-shot; gemma4 media prompts "
            "will keep dying on an oversized single allocation"
        )

    def test_short_prompts_stay_one_shot_even_when_chunkable(self):
        """One-shot reads the weights once; it was never the failing shape."""
        from types import SimpleNamespace

        calls = []
        gen = self._gen(_OneShotModel(calls), _EmbedsLM())
        gen.model.get_input_embeddings = lambda ids, **kw: SimpleNamespace(
            inputs_embeds=_FakeIds(1000)
        )
        gen._media_forward(
            SimpleNamespace(request_id="r"), _FakeIds(1000), 1000,
            [object()], {},
        )
        assert calls == ["one-shot"]

    def test_a_failing_embedding_merge_falls_back_instead_of_erroring(self):
        from types import SimpleNamespace

        def _boom(ids, **kw):
            raise RuntimeError("merge exploded")

        calls = []
        gen = self._gen(_OneShotModel(calls), _EmbedsLM())
        gen.model.get_input_embeddings = _boom
        assert self._run(gen) == "out"
        assert calls == ["one-shot"]

    def test_final_chunk_logits_are_realized_before_transients_clear(
        self, monkeypatch
    ):
        """The returned final logits must not reference a cleared Metal resource."""
        from types import SimpleNamespace

        import vmlx_engine.mllm_batch_generator as mllm

        events = []

        class _LazyLogits:
            def __getitem__(self, item):
                events.append("last-token-slice")
                return self

        class _ChunkableLM:
            def __call__(self, inputs, inputs_embeds=None, cache=None):
                events.append("forward")
                return _LazyLogits()

        gen = self._gen(_OneShotModel([]), _ChunkableLM())
        gen.model.get_input_embeddings = lambda ids, **kw: SimpleNamespace(
            inputs_embeds=_FakeIds(9000)
        )
        gen._media_placeholder_token_ids = lambda: set()
        gen._media_prefill_chunk_tokens = lambda seq_len: 4096
        monkeypatch.setattr(mllm.mx, "eval", lambda value: events.append("eval"))
        monkeypatch.setattr(
            mllm.mx, "clear_cache", lambda: events.append("clear-cache")
        )

        result = gen._media_forward(
            SimpleNamespace(request_id="media-last-logits"),
            _FakeIds(9000),
            9000,
            [object()],
            {},
        )

        assert isinstance(result, _LazyLogits)
        assert events.count("forward") == 3
        assert events.count("eval") == 1
        assert events[-3:] == ["last-token-slice", "eval", "clear-cache"]

    def test_final_chunk_unwraps_language_model_output_before_slicing(
        self, monkeypatch
    ):
        """VLM language forwards return a wrapper whose logits are sliceable."""
        from types import SimpleNamespace

        import vmlx_engine.mllm_batch_generator as mllm

        events = []

        class _LazyLogits:
            def __getitem__(self, item):
                events.append("last-token-slice")
                return self

        class _WrappedLM:
            def __call__(self, inputs, inputs_embeds=None, cache=None):
                return SimpleNamespace(logits=_LazyLogits())

        gen = self._gen(_OneShotModel([]), _WrappedLM())
        gen.model.get_input_embeddings = lambda ids, **kw: SimpleNamespace(
            inputs_embeds=_FakeIds(9000)
        )
        gen._media_placeholder_token_ids = lambda: set()
        gen._media_prefill_chunk_tokens = lambda seq_len: 4096
        monkeypatch.setattr(mllm.mx, "eval", lambda value: events.append("eval"))
        monkeypatch.setattr(
            mllm.mx, "clear_cache", lambda: events.append("clear-cache")
        )

        result = gen._media_forward(
            SimpleNamespace(request_id="media-wrapped-logits"),
            _FakeIds(9000),
            9000,
            [object()],
            {},
        )

        assert isinstance(result, _LazyLogits)
        assert events.count("last-token-slice") == 1
        assert events.count("eval") == 1
        assert events[-3:] == ["last-token-slice", "eval", "clear-cache"]


class _FakeIds:
    """Minimal stand-in for an mx.array of token ids."""

    def __init__(self, n):
        self._n = n
        self.ndim = 2

    def __getitem__(self, item):
        return self

    def tolist(self):
        return list(range(self._n))


class TestMediaSpansAreNeverSplit:
    """The invariant the wrapper flag actually cares about."""

    def test_no_boundary_lands_inside_a_run(self):
        runs = [(100, 900), (1500, 4200), (9000, 9100)]
        bounds = _media_chunk_boundaries(12000, 4096, runs)
        for end in bounds[:-1]:
            for rs, re_ in runs:
                assert not (rs < end < re_), (
                    "boundary %d splits media run [%d, %d)" % (end, rs, re_)
                )

    def test_a_run_longer_than_the_chunk_is_kept_whole(self):
        """A 6000-token image with a 4096 chunk must not be cut in half."""
        runs = [(200, 6200)]
        bounds = _media_chunk_boundaries(9000, 4096, runs)
        for end in bounds[:-1]:
            assert not (200 < end < 6200), "oversized run was split at %d" % end
        assert bounds[-1] == 9000
        assert all(b2 > b1 for b1, b2 in zip(bounds, bounds[1:]))


class TestNativeCleanMediaBoundary:
    """The clean media boundary is a chunk edge of the MAIN forward and the recurrent layers are snapshotted there,
    replacing the auxiliary truncated forward whose recurrent state differed numerically from the main pass (live:
    cached vs uncached wording changed on media prompts only; text stores slice the main pass and were identical)."""

    class _SSM:
        def __init__(self, arr): self.cache = [arr]

    class _KV:
        offset = 0

    class _LM:
        def __init__(self): self.spans = []
        def __call__(self, inputs, inputs_embeds=None, cache=None, mask=None, position_ids=None):
            import mlx.core as mx
            self.spans.append(int(inputs_embeds.shape[1]))
            # the recurrent state advances with every chunk (so the snapshot at the boundary must differ from the end)
            cache[1].cache[0] = cache[1].cache[0] + mx.array(float(inputs_embeds.shape[1]))
            return mx.zeros((1, int(inputs_embeds.shape[1]), 4))

    class _Model:
        def __init__(self, n): self.n = n
        def get_input_embeddings(self, ids, **kwargs):
            import mlx.core as mx
            from types import SimpleNamespace
            return SimpleNamespace(inputs_embeds=mx.zeros((1, self.n, 8)))
        def __call__(self, ids, **kwargs):
            raise AssertionError("one-shot forward must not run when a native boundary applies")

    def _gen(self, n, tokens, hybrid=True):
        import mlx.core as mx
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.prefill_step_size = 2048
        gen.model = self._Model(n); gen.language_model = self._LM()
        gen._is_hybrid = hybrid; gen._ssm_state_cache = object(); gen._hybrid_kv_positions = [0]
        gen._model_type = "qwen3_5"; gen.block_aware_cache = SimpleNamespace(block_size=64)
        gen._media_placeholder_token_ids = lambda: {99}
        gen._media_prefix_cache_allowed = lambda req, toks: True
        lm = gen.language_model
        # the language model's position ids for the merged sequence
        lm._position_ids = mx.broadcast_to(mx.arange(n)[None, None, :], (3, 1, n))
        cache = [self._KV(), self._SSM(mx.array(0.0))]
        req = SimpleNamespace(request_id="r", _original_token_ids=tokens, _cached_tokens=0)
        return gen, cache, req

    def test_splits_at_the_boundary_and_snapshots_the_recurrent_state_there(self):
        import mlx.core as mx
        tokens = [1] * 4 + [99] * 1760 + [2] * 8  # cache-key tokens: 1772, media 4..1764, exact N-1 = 1771
        n = len(tokens) + 7  # the processed input carries a 7-token generation-prompt suffix the key excludes (live 1772 vs 1779)
        gen, cache, req = self._gen(n, tokens)
        from tests.test_media_chunked_prefill import _FakeIds
        out = gen._media_forward(req, _FakeIds(n), n, cache, {})
        assert gen.language_model.spans == [1771, 8]
        assert req._media_clean_prefix_len == 1771 and req._media_clean_native is True
        snap = req._media_clean_prefix_cache
        assert snap[0] is cache[0]  # KV layers are not cloned (the paged store slices the main cache)
        assert snap[1] is not cache[1]
        assert float(snap[1].cache[0].item()) == 1771.0  # state at the boundary, not at the end (1779)
        assert float(cache[1].cache[0].item()) == 1779.0

    def test_warm_requests_and_non_hybrid_models_keep_the_previous_behavior(self):
        tokens = [1] * 4 + [99] * 1760 + [2] * 8
        gen, cache, req = self._gen(len(tokens), tokens, hybrid=False)
        assert gen._native_media_clean_boundary(req, len(tokens), cache) == 0
        gen, cache, req = self._gen(len(tokens), tokens)
        req._cached_tokens = 1771
        assert gen._native_media_clean_boundary(req, len(tokens), cache) == 0
        # media running to the very end: no boundary after it -> 0 (the store's pre-media fallback is not a native snapshot)
        gen, cache, req = self._gen(1764, [1] * 4 + [99] * 1760)
        assert gen._native_media_clean_boundary(req, 1764, cache) == 0

    def test_conditioned_warm_tail_snapshots_new_boundary_not_restored_boundary(self):
        tokens = [1] * 4 + [99] * 1760 + [2] * 8
        gen, cache, req = self._gen(len(tokens) + 7, tokens)
        req._cached_tokens = 64
        assert gen._native_media_clean_boundary(
            req, len(tokens) + 7, cache, allow_conditioned_tail=True
        ) == 1771
        req._cached_tokens = 1771
        assert gen._native_media_clean_boundary(
            req, len(tokens) + 7, cache, allow_conditioned_tail=True
        ) == 0


class TestMediaHitTextTailKeepsMRoPEPositions:
    """First divergent computation between cached and uncached video answers (fingerprints on the box): the warm tail,
    text after a fully cached video, was forwarded with absolute text positions (1771..1778) while the cold pass gave
    the same tokens their mRoPE positions (~31..38). KV, recurrent state, embeddings and decode were identical."""

    class _LM:
        def get_rope_index(self, ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
            import mlx.core as mx
            n = int(ids.shape[1])
            # a compressed video: positions after it continue from 40, not from the token count
            pos = mx.array([[list(range(4)) + [4 + i // 100 for i in range(1760)] + list(range(40, 40 + n - 1764))]] * 3)
            return pos, mx.array([[40 - 1764]])

    def _gen(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.language_model = self._LM()
        return gen

    def test_hit_records_the_tail_positions_from_the_full_prompt(self):
        import mlx.core as mx
        from types import SimpleNamespace
        gen = self._gen()
        full = mx.array([[1] * 4 + [99] * 1760 + [2] * 15])  # 1779 processed ids
        req = SimpleNamespace(request_id="r", image_grid_thw=None, video_grid_thw=mx.array([[8, 22, 40]]))
        assert gen._mrope_tail_position_ids(req, full, 1771) is True
        tail = req._mrope_tail_position_ids
        assert tuple(tail.shape) == (3, 1, 8)
        assert tail[0, 0, :].tolist() == list(range(47, 55))  # continues from the media's extent, not from 1771
        # the text-path prefill prefers those positions for exactly that tail, absolute positions otherwise
        cache = [SimpleNamespace(offset=1771)]
        got = gen._text_prefill_position_ids(req, mx.zeros((1, 8)), cache, SimpleNamespace())
        assert got is tail
        other = gen._text_prefill_position_ids(SimpleNamespace(), mx.zeros((1, 8)), cache, SimpleNamespace())
        assert other[0, 0, :].tolist() == list(range(1771, 1779))

    def test_no_rope_index_or_bad_shapes_leave_the_request_untouched(self):
        import mlx.core as mx
        from types import SimpleNamespace
        gen = self._gen(); gen.language_model = SimpleNamespace()
        req = SimpleNamespace(request_id="r")
        assert gen._mrope_tail_position_ids(req, mx.zeros((1, 10)), 4) is False
        assert not hasattr(req, "_mrope_tail_position_ids")
        gen = self._gen()
        assert gen._mrope_tail_position_ids(SimpleNamespace(request_id="r", image_grid_thw=None, video_grid_thw=None), mx.zeros((1, 10)), 10) is False


def test_the_media_split_does_not_depend_on_the_prefix_cache_flag():
    """A request that may not store (skip_prefix_cache) still prefills in the same two pieces as a storing request,
    so its answer matches; only the snapshot is skipped. Live: single-pass vs split prefill gave 45 vs 34 tokens."""
    import mlx.core as mx
    from types import SimpleNamespace
    T = TestNativeCleanMediaBoundary
    tokens = [1] * 4 + [99] * 1760 + [2] * 8; n = len(tokens) + 7
    gen, cache, req = T()._gen(n, tokens)
    gen._media_prefix_cache_allowed = lambda r, t: False  # e.g. skip_prefix_cache
    out = gen._media_forward(req, _FakeIds(n), n, cache, {})
    assert gen.language_model.spans == [1771, 8]          # split taken
    assert not hasattr(req, "_media_clean_prefix_cache")  # no snapshot stored
    assert req._media_clean_snapshot_allowed is False
