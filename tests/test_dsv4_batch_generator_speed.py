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
