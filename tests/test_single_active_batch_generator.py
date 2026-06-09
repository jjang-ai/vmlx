import numpy as np
import mlx.core as mx
from types import SimpleNamespace

from vmlx_engine.request import Request, RequestStatus, SamplingParams
from vmlx_engine.scheduler import Scheduler, SchedulerConfig


class _TinyCache:
    def __init__(self):
        self.tokens = []
        self.offset = 0

    @property
    def state(self):
        return (mx.array([self.offset], dtype=mx.int32),)

    @property
    def nbytes(self):
        return 0

    def extract(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self

    def filter(self, keep):
        if list(keep) != [0]:
            raise NotImplementedError("tiny cache only supports one sequence")

    def prepare(self, *args, **kwargs):
        return None

    def finalize(self):
        return None


class _TinyModel:
    def __init__(self):
        self.calls = []

    def make_cache(self):
        return [_TinyCache()]

    def __call__(self, input_ids, cache):
        tokens = np.array(input_ids).reshape(-1).astype(int).tolist()
        cache[0].tokens.extend(tokens)
        cache[0].offset += len(tokens)
        self.calls.append(tokens)
        batch, seq_len = input_ids.shape
        logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
        logits = logits + mx.array([0, 1, 2, 7, 3, 2, 1, 0], dtype=mx.float32)
        return logits


class _TinyTokenizer:
    clean_up_tokenization_spaces = False

    def decode(self, tokens):
        return "".join(str(int(t)) for t in tokens)


class TestSingleActiveBatchGenerator:
    def test_scheduler_uses_owned_single_active_generator_for_max_one(self):
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
            ),
        )

        generator = scheduler._create_batch_generator(SamplingParams())

        assert generator.__class__.__module__ == "vmlx_engine.utils.single_batch_generator"
        assert generator.__class__.__name__ == "SingleBatchGenerator"

    def test_single_active_generator_uses_concrete_stream_not_thread_default(self, monkeypatch):
        """Single-active cache-hit replay must not depend on the thread-default
        Metal stream.  The app runs scheduler.step() through an executor, and
        MLX default streams are thread-local; using Stream(gpu,0) here can fail
        with "There is no Stream(gpu, 0) in current thread" on cache-hit decode.
        """
        from vmlx_engine.utils import single_batch_generator as single

        fake_device = object()
        fake_streams = [object(), object()]
        new_stream_calls = []

        monkeypatch.setattr(single.mx, "default_device", lambda: fake_device)

        def fake_new_stream(device):
            new_stream_calls.append(device)
            return fake_streams[len(new_stream_calls) - 1]

        def fail_default_stream(device):
            raise AssertionError("SingleBatchGenerator must not use default_stream")

        monkeypatch.setattr(single.mx, "new_stream", fake_new_stream)
        monkeypatch.setattr(single.mx, "default_stream", fail_default_stream)

        generator = single.SingleBatchGenerator(model=_TinyModel())

        assert generator._stream is fake_streams[0]
        generator._refresh_thread_stream()
        assert generator._stream is fake_streams[1]
        assert new_stream_calls == [fake_device, fake_device]

    def test_single_active_generator_keeps_max_kv_size_for_plain_kv_model(
        self, monkeypatch
    ):
        from vmlx_engine.utils import single_batch_generator as single

        captured = {}

        def fake_make_prompt_cache(model, max_kv_size=None):
            captured["max_kv_size"] = max_kv_size
            return ["cache"]

        monkeypatch.setattr(single.mlx_cache, "make_prompt_cache", fake_make_prompt_cache)

        model = _TinyModel()
        model.config = SimpleNamespace(model_type="llama")
        generator = single.SingleBatchGenerator(model=model, max_kv_size=128)

        assert generator._make_new_cache() == ["cache"]
        assert captured["max_kv_size"] == 128

    def test_single_active_generator_suppresses_generic_max_kv_for_mixed_cache_families(
        self, monkeypatch
    ):
        from vmlx_engine.utils import single_batch_generator as single

        captured = []

        def fake_make_prompt_cache(model, max_kv_size=None):
            captured.append((model.config.model_type, max_kv_size))
            return ["cache"]

        monkeypatch.setattr(single.mlx_cache, "make_prompt_cache", fake_make_prompt_cache)

        for model_type in ("gemma4", "mimo_v2", "qwen3_5_moe"):
            model = _TinyModel()
            model.config = SimpleNamespace(model_type=model_type)
            generator = single.SingleBatchGenerator(model=model, max_kv_size=128)

            assert generator._make_new_cache() == ["cache"]

        assert captured == [
            ("gemma4", None),
            ("mimo_v2", None),
            ("qwen3_5_moe", None),
        ]

    def test_single_active_generator_suppresses_generic_max_kv_for_mixed_layer_types(
        self, monkeypatch
    ):
        from vmlx_engine.utils import single_batch_generator as single

        captured = {}

        def fake_make_prompt_cache(model, max_kv_size=None):
            captured["max_kv_size"] = max_kv_size
            return ["cache"]

        monkeypatch.setattr(single.mlx_cache, "make_prompt_cache", fake_make_prompt_cache)

        model = _TinyModel()
        model.config = SimpleNamespace(
            model_type="future_family",
            layer_types=["sliding_attention", "full_attention"],
        )
        generator = single.SingleBatchGenerator(model=model, max_kv_size=128)

        assert generator._make_new_cache() == ["cache"]
        assert captured["max_kv_size"] is None

    def test_single_active_generator_materializes_sample_without_async_eval(
        self, monkeypatch
    ):
        """The single-active path must materialize sampled tokens on its owned
        stream.  Cache-hit replay can carry restored KV graphs from an earlier
        request; mx.async_eval() may resolve those graphs through a missing
        thread-local Stream(gpu,0).
        """
        from vmlx_engine.utils import single_batch_generator as single

        generator = single.SingleBatchGenerator(model=_TinyModel())

        def sampler(logits):
            return mx.array([3], dtype=mx.int32)

        sampler._vmlx_accepts_logits = True

        def fail_async_eval(*args, **kwargs):
            raise AssertionError("SingleBatchGenerator must not use async_eval")

        eval_calls = []
        original_eval = single.mx.eval

        def recording_eval(*args, **kwargs):
            eval_calls.append(len(args))
            return original_eval(*args, **kwargs)

        monkeypatch.setattr(single.mx, "async_eval", fail_async_eval)
        monkeypatch.setattr(single.mx, "eval", recording_eval)

        generator.insert([[11]], max_tokens=[1], samplers=[sampler])
        prompt_responses, generation_responses = generator.next()

        assert generation_responses == []
        assert prompt_responses[0].token == 3
        assert prompt_responses[0].finish_reason == "length"
        assert eval_calls == [1]

    def test_eval_on_stream_rehomes_values_inside_owned_stream(self, monkeypatch):
        """Rebinding has to happen inside the generator stream too.

        Building the rebound graph outside the stream context can still attach
        it to MLX's thread-default stream; cache-hit replay then fails when
        mx.eval materializes that graph on the worker thread.
        """
        from vmlx_engine.utils import single_batch_generator as single

        active_stream = {"value": False}

        class FakeStreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                active_stream["value"] = True

            def __exit__(self, exc_type, exc, tb):
                active_stream["value"] = False

        monkeypatch.setattr(single.mx, "stream", lambda stream: FakeStreamContext(stream))

        generator = single.SingleBatchGenerator(model=_TinyModel(), stream=object())
        token = object()

        def assert_rehome_inside_stream(value):
            assert active_stream["value"]
            return value

        def assert_eval_inside_stream(*values):
            assert active_stream["value"]

        monkeypatch.setattr(generator, "_rehome_on_stream", assert_rehome_inside_stream)
        monkeypatch.setattr(single.mx, "eval", assert_eval_inside_stream)

        assert generator._eval_on_stream(token) == (token,)

    def test_single_active_prefill_runs_model_inside_owned_stream(self, monkeypatch):
        """Cold prefill must use the same concrete stream as decode.

        The app runs scheduler steps on a worker thread.  If prefill forwards
        happen on a thread-default stream while decode samples on the owned
        stream, hybrid/JANGTQ cache state can diverge from the direct
        mlx-lm generation path.
        """
        from vmlx_engine.utils import single_batch_generator as single

        active_stream = {"value": False}

        class FakeStreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                active_stream["value"] = True

            def __exit__(self, exc_type, exc, tb):
                active_stream["value"] = False

        class StreamAssertingModel(_TinyModel):
            def __call__(self, input_ids, cache):
                assert active_stream["value"]
                return super().__call__(input_ids, cache)

        monkeypatch.setattr(single.mx, "stream", lambda stream: FakeStreamContext(stream))
        monkeypatch.setattr(single.mx, "eval", lambda *values: None)
        monkeypatch.setattr(single.mx, "synchronize", lambda *args, **kwargs: None)
        monkeypatch.setattr(single.mx, "clear_cache", lambda: None, raising=False)

        generator = single.SingleBatchGenerator(
            model=StreamAssertingModel(),
            stream=object(),
            prefill_step_size=2,
        )

        generator.insert([[11, 12, 13]], max_tokens=[1])
        generator.next()

    def test_scheduler_keeps_mlx_lm_batch_generator_for_real_multi_sequence(self):
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=2,
                enable_prefix_cache=False,
            ),
        )

        generator = scheduler._create_batch_generator(SamplingParams())

        assert generator.__class__.__module__ == "mlx_lm.generate"
        assert generator.__class__.__name__ == "BatchGenerator"

    def test_single_active_generator_preserves_native_cache_object(self):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        model = _TinyModel()
        native_cache = [_TinyCache()]
        generator = SingleBatchGenerator(
            model=model,
            max_tokens=2,
            stop_tokens={9},
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )

        [uid] = generator.insert([[11, 12]], caches=[native_cache], max_tokens=[2])
        prompt_responses, generation_responses = generator.next()

        assert uid == 0
        assert generation_responses == []
        assert len(prompt_responses) == 1
        assert prompt_responses[0].token == 3
        assert prompt_responses[0].prompt_cache[0] is native_cache[0]
        assert native_cache[0].tokens == [11, 12, 3]
        # Regression: response.all_tokens must include the final prompt token
        # before the sampled token (the cache has consumed it).
        assert prompt_responses[0].all_tokens == [11, 12, 3]
        # extract_cache() context metadata must agree with all_tokens.
        extracted = generator.extract_cache([uid])
        _cache, context = extracted[uid]
        assert context == [11, 12, 3]

    def test_single_active_generator_next_generated_does_not_infinite_loop_when_prompt_finishes(self):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        generator = SingleBatchGenerator(
            model=_TinyModel(),
            max_tokens=1,  # forces length-cap on first sampled token
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        generator.insert([[11, 12]], max_tokens=[1])

        # First call returns the prompt response (which has finish_reason=length),
        # so the active request is cleared. next_generated() must return [] on
        # the follow-up call instead of looping forever.
        first = generator.next_generated()
        assert first == []
        second = generator.next_generated()
        assert second == []

    def test_single_active_generator_finishes_on_stop_token(self):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        generator = SingleBatchGenerator(
            model=_TinyModel(),
            max_tokens=8,
            stop_tokens={3},
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )

        generator.insert([[1, 2]], max_tokens=[8])
        prompt_responses, _ = generator.next()

        assert prompt_responses[0].token == 3
        assert prompt_responses[0].finish_reason == "stop"

    def test_scheduler_step_uses_single_active_generator_end_to_end(self):
        request = Request(
            request_id="r1",
            prompt=[1, 2],
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
            ),
        )

        scheduler.add_request(request)
        output = scheduler.step()

        assert scheduler.batch_generator.__class__.__name__ == "SingleBatchGenerator"
        stats = scheduler.get_stats()
        assert stats["engine_path"] == "single_active"
        assert stats["batch_generator"]["single_active_decode"] is True
        assert output.outputs[0].new_token_ids == [3]
        assert output.outputs[0].finish_reason == "length"
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_single_active_scheduler_serializes_second_request(self):
        first = Request(
            request_id="r1",
            prompt=[1, 2],
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        )
        second = Request(
            request_id="r2",
            prompt=[4, 5],
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        )
        for request in (first, second):
            request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt)

        scheduler = Scheduler(
            _TinyModel(),
            tokenizer=_TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
            ),
        )

        scheduler.add_request(first)
        scheduler.add_request(second)
        first_output = scheduler.step()
        second_output = scheduler.step()

        assert scheduler.batch_generator.__class__.__name__ == "SingleBatchGenerator"
        assert [out.request_id for out in first_output.outputs] == ["r1"]
        assert [out.request_id for out in second_output.outputs] == ["r2"]
        assert first.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert second.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_single_active_generator_preserves_cached_prefix_in_resumed_context(self):
        """Cache-hit insertion: scheduler hands a warm cache + cached prefix as
        all_tokens; only the tail tokens are processed but response.all_tokens
        must continue from the full cached prefix (not start fresh).

        Without this, scheduler.append_to_cache + extract_cache round-trips
        would silently lose the cached prefix on the first decode response.
        """
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        model = _TinyModel()
        # Simulate a warm cache that already saw 5 prefix tokens.
        warm_cache = [_TinyCache()]
        warm_cache[0].tokens = [101, 102, 103, 104, 105]
        warm_cache[0].offset = 5
        cached_prefix = [101, 102, 103, 104, 105]
        tail_tokens = [201, 202]

        generator = SingleBatchGenerator(
            model=model,
            max_tokens=1,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        [uid] = generator.insert(
            [tail_tokens],
            caches=[warm_cache],
            all_tokens=[cached_prefix],
            max_tokens=[1],
        )

        prompt_responses, _ = generator.next()
        assert len(prompt_responses) == 1
        # Sampled token from _TinyModel is 3 (argmax of fixed logits vector).
        assert prompt_responses[0].token == 3
        # Critical: response.all_tokens must = cached_prefix + tail + sampled.
        # If SingleBatchGenerator dropped the prefix, this would be [201,202,3].
        assert prompt_responses[0].all_tokens == [101, 102, 103, 104, 105, 201, 202, 3]
        # Native cache must have consumed only the tail forwards (prefix
        # tokens were already in the cache; the sampled output token is not
        # fed back when max_tokens=1 short-circuits the lookahead).
        assert warm_cache[0].tokens == [101, 102, 103, 104, 105, 201, 202]
        # Model received only the tail tokens, never re-processed the prefix.
        flat_calls = [t for call in model.calls for t in call]
        assert flat_calls == [201, 202]

    def test_single_active_generator_skips_token_buffer_without_processors(
        self, monkeypatch
    ):
        from vmlx_engine.utils import single_batch_generator as single

        calls = []

        class RecordingTokenBuffer:
            def __init__(self, tokens):
                self.tokens = list(tokens)

            def update_and_fetch(self, tokens):
                calls.append(tokens)
                return tokens

        monkeypatch.setattr(single, "TokenBuffer", RecordingTokenBuffer)

        generator = single.SingleBatchGenerator(
            model=_TinyModel(),
            max_tokens=2,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        generator.insert([[11, 12]], max_tokens=[2], logits_processors=[[]])

        prompt_responses, _ = generator.next()
        assert prompt_responses[0].token == 3
        _, generation_responses = generator.next()
        assert generation_responses[0].token == 3
        assert calls == []

    def test_single_active_generator_keeps_token_buffer_for_processors(
        self, monkeypatch
    ):
        from vmlx_engine.utils import single_batch_generator as single

        calls = []

        class RecordingTokenBuffer:
            def __init__(self, tokens):
                self.tokens = list(tokens)

            def update_and_fetch(self, tokens):
                calls.append(tokens)
                return tokens

        def passthrough_processor(_context, logits):
            return logits

        monkeypatch.setattr(single, "TokenBuffer", RecordingTokenBuffer)

        generator = single.SingleBatchGenerator(
            model=_TinyModel(),
            max_tokens=1,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        generator.insert(
            [[11, 12]],
            max_tokens=[1],
            logits_processors=[[passthrough_processor]],
        )

        prompt_responses, _ = generator.next()
        assert prompt_responses[0].token == 3
        assert len(calls) == 2
