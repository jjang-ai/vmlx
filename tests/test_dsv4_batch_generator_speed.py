import mlx.core as mx


def test_dsv4_sample_skips_logprobs_for_logits_sampler(monkeypatch):
    from vmlx_engine.utils import dsv4_batch_generator as mod

    gen = mod.DSV4BatchGenerator.__new__(mod.DSV4BatchGenerator)
    gen.fallback_sampler = None

    def sampler(logits):
        return mx.argmax(logits, axis=-1)

    sampler._vmlx_accepts_logits = True

    def fail_logsumexp(*args, **kwargs):
        raise AssertionError("logsumexp should not run on default greedy path")

    monkeypatch.setattr(mod.mx, "logsumexp", fail_logsumexp)

    sampled, logprobs = gen._sample(
        mx.array([[0.0, 2.0, 1.0]]),
        sampler,
        processors=[],
        recent_tokens=[],
        capture_logprobs=False,
    )

    mx.eval(sampled)
    assert int(sampled.item()) == 1
    assert logprobs is None


def test_dsv4_sample_preserves_logprobs_when_requested():
    from vmlx_engine.utils import dsv4_batch_generator as mod

    gen = mod.DSV4BatchGenerator.__new__(mod.DSV4BatchGenerator)
    gen.fallback_sampler = None

    def sampler(logits):
        return mx.argmax(logits, axis=-1)

    sampler._vmlx_accepts_logits = True

    sampled, logprobs = gen._sample(
        mx.array([[0.0, 2.0, 1.0]]),
        sampler,
        processors=[],
        recent_tokens=[],
        capture_logprobs=True,
    )

    mx.eval(sampled, logprobs)
    assert int(sampled.item()) == 1
    assert logprobs is not None
    assert tuple(logprobs.shape) == (1, 3)


def test_dsv4_logprob_capture_registry_controls_uid():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator
    from vmlx_engine.utils.mamba_cache import (
        register_generation_logprobs,
        unregister_generation_logprobs,
    )

    model = object()
    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen.model = model

    assert gen._should_capture_logprobs(7) is False
    register_generation_logprobs(model, 7)
    try:
        assert gen._should_capture_logprobs(7) is True
        assert gen._should_capture_logprobs(8) is False
    finally:
        unregister_generation_logprobs(model, 7)


def test_dsv4_sampled_token_materialization_does_not_double_sync():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    class Sampled:
        def tolist(self):
            return [42]

    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen._stream = mx.default_stream(mx.default_device())

    def fail_sync():
        raise AssertionError("_sampled_token_id should rely on scalar materialization")

    gen._sync = fail_sync

    assert gen._sampled_token_id(Sampled()) == 42


def test_dsv4_prefill_realizes_last_logits_before_clearing_transients(monkeypatch):
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator
    from vmlx_engine.utils import dsv4_batch_generator as mod

    events = []
    logits = mx.array([[[0.0, 1.0]]])

    class FakeModel:
        def __call__(self, tokens, cache):
            events.append(("model", tuple(tokens.shape)))
            return logits

    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen.model = FakeModel()
    gen.prefill_step_size = 2048
    gen._sync = lambda: events.append(("sync", None))

    def record_eval(value):
        assert value.shape == (1, 2)
        events.append(("eval", tuple(value.shape)))

    def record_clear_cache():
        events.append(("clear_cache", None))

    monkeypatch.setattr(mod.mx, "eval", record_eval)
    monkeypatch.setattr(mod.mx, "clear_cache", record_clear_cache)

    result = gen._prefill_last_logits([1, 2, 3], cache=[])

    assert result.shape == (1, 2)
    assert events.index(("eval", (1, 2))) < events.index(("clear_cache", None))


def test_dsv4_warmup_realizes_model_forward_before_sync(monkeypatch):
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator
    from vmlx_engine.utils import dsv4_batch_generator as mod

    events = []
    logits = mx.array([[[0.0, 1.0]]])

    class FakeModel:
        def __call__(self, tokens, cache):
            events.append(("model", tuple(tokens.shape), cache))
            return logits

    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen.model = FakeModel()
    gen._requests = []
    gen._warmed_up = False
    gen._stream = mx.default_stream(mx.default_device())
    gen._device = mx.default_device()
    gen._make_new_cache = lambda: ["warm-cache"]
    gen._refresh_thread_stream = lambda: None
    gen._sync = lambda: events.append(("sync", None, None))

    def record_eval(value):
        assert value is logits
        events.append(("eval", tuple(value.shape), None))

    monkeypatch.setattr(mod.mx, "eval", record_eval)

    prompt_resps, gen_resps = gen.next()

    assert prompt_resps == []
    assert gen_resps == []
    assert gen._warmed_up is True
    assert events == [
        ("model", (1, 1), ["warm-cache"]),
        ("eval", (1, 1, 2), None),
        ("sync", None, None),
    ]


def test_dsv4_cache_hit_tail_prefill_uses_cross_thread_safe_realization():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator, _Request

    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen._requests = [
        _Request(
            uid=11,
            prompt_tokens=[101],
            context_tokens=[1, 2, 101],
            cache=[object()],
            max_tokens=4,
        )
    ]
    gen._stream = mx.default_stream(mx.default_device())
    gen._warmed_up = True
    gen.stop_tokens = set()
    gen._refresh_thread_stream = lambda: None
    gen._trace_timing = lambda *args, **kwargs: None

    calls = []

    def prefill_last_logits(tokens, cache, *, realize_before_clear=True):
        calls.append((list(tokens), realize_before_clear))
        return mx.array([[0.0, 2.0]])

    gen._prefill_last_logits = prefill_last_logits
    gen._sample = lambda *args, **kwargs: (mx.array([1]), None)
    gen._sampled_token_id = lambda sampled: 1
    gen._should_capture_logprobs = lambda uid: False

    prompt_resps, gen_resps = gen.next()

    assert calls == [([101], False)]
    assert [resp.token for resp in prompt_resps] == [1]
    assert gen_resps == []
