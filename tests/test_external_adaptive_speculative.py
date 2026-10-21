"""External-drafter acceptance, causal history and adaptive wall-cost regressions."""
import numpy as np
import pytest

from vmlx_engine.adaptive_speculative import AdaptiveDraftWidth, adaptive_generate_step


def test_seed_is_not_ceiling_and_acceptance_is_not_speed():
    controller = AdaptiveDraftWidth(1)
    seen = set()
    for _ in range(160):
        width = controller.width
        seen.add(width)
        controller.observe(width, width + 1, width, .01 * (1 + .2 * width))
    assert max(seen) > 1
    assert any(x["reason"] == "refresh_ar_baseline" for x in controller.transitions)
    slow = AdaptiveDraftWidth(7)
    for _ in range(80):
        width = slow.width
        slow.observe(width, width + 1, width, .01 if width == 0 else (width + 1) * .04)
    transitions = slow.snapshot()["transitions"]
    assert any(x["reason"] == "measured_ar_wins" for x in transitions)
    assert any(x["reason"] == "ar_reentry_probe" for x in transitions)
    assert slow.accepted == slow.drafted


class TokenCache:
    def __init__(self):
        self.tokens = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, len(self.tokens))
        if n:
            del self.tokens[-n:]
        return n

    @property
    def state(self):
        import mlx.core as mx
        return (mx.array(self.tokens),)


class CausalModel:
    layers = [None]

    def __init__(self, shift=0, distribution=None):
        self.shift, self.distribution = shift, distribution

    def make_cache(self):
        return [TokenCache()]

    def __call__(self, tokens, cache):
        import mlx.core as mx
        rows = []
        for token in tokens.reshape(-1).tolist():
            cache[0].tokens.append(token)
            scores = np.full(11, -50.0)
            scores[(sum(cache[0].tokens) + 1 + self.shift) % 11] = 50.0
            rows.append(scores if self.distribution is None else self.distribution)
        return mx.array(np.array(rows, dtype=np.float32))[None]


class WidthSequence(AdaptiveDraftWidth):
    def __init__(self, seed):
        super().__init__(seed)
        self.width = 3

    def observe(self, *args):
        super().observe(*args)
        self.width = [0, 0, 1, 4, 2, 3][self.cycles % 6]


def test_shape_sensitive_target_preserves_decode_decisions(monkeypatch):
    """Block arithmetic must not replace an arbitrary target's AR decisions."""
    import mlx.core as mx
    import vmlx_engine.adaptive_speculative as adaptive

    class ShapeSensitiveTarget(CausalModel):
        def __call__(self, tokens, cache):
            logits = super().__call__(tokens, cache)
            return mx.roll(logits, 1, axis=-1) if tokens.shape[1] > 1 else logits

    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    prompt = [2, 4, 3]
    expected = list(prompt)
    for _ in range(61):
        expected.append((sum(expected) + 1) % 11)
    stats = []
    rows = list(adaptive_generate_step(mx.array(prompt), ShapeSensitiveTarget(),
                CausalModel(), max_tokens=61, on_stats=stats.append))
    assert [row[0] for row in rows] == expected[len(prompt):]
    assert stats[0]["verification_mode"] == "sequential_target_decode"


@pytest.mark.parametrize("shift", [0, 1, 5])
def test_verification_rollback_and_ar_catchup_are_causal(monkeypatch, shift):
    import mlx.core as mx
    import vmlx_engine.adaptive_speculative as adaptive
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    target, draft = CausalModel(), CausalModel(shift)
    caches = target.make_cache() + draft.make_cache()
    prompt = [2, 4, 3]
    expected = list(prompt)
    for _ in range(61):
        expected.append((sum(expected) + 1) % 11)
    stats = []
    result = list(adaptive_generate_step(mx.array(prompt), target, draft,
                  max_tokens=61, prompt_cache=caches, on_stats=stats.append))
    assert [x[0] for x in result] == expected[len(prompt):]
    assert caches[0].tokens == expected[:-1]
    assert caches[1].tokens == expected[:len(caches[1].tokens)]
    assert stats[0]["delivered_tokens"] == 61
    assert len(stats[0]["width_counts"]) >= 4


def test_close_mid_batch_drops_every_undelivered_cache_token(monkeypatch):
    import mlx.core as mx
    import vmlx_engine.adaptive_speculative as adaptive
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    model = CausalModel()
    caches = model.make_cache() + model.make_cache()
    stats = []
    generator = adaptive_generate_step(mx.array([2, 4]), model, model,
                                      prompt_cache=caches, on_stats=stats.append)
    first = next(generator)[0]
    generator.close()
    assert first == 7
    assert caches[0].tokens == [2, 4]
    assert caches[1].tokens == [2, 4]
    assert stats[0]["delivered_tokens"] == stats[0]["delivered_draft_tokens"] == 1


def test_processors_see_only_causal_target_contexts(monkeypatch):
    import mlx.core as mx
    import vmlx_engine.adaptive_speculative as adaptive
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    calls = []

    def processor(tokens, logits):
        calls.append(tokens.tolist())
        forced = len(tokens) % 11
        return mx.where(mx.arange(11) == forced, 50.0, -50.0)[None]

    prompt = [2, 4]
    result = list(adaptive_generate_step(mx.array(prompt), CausalModel(), CausalModel(),
                  max_tokens=21, logits_processors=[processor]))
    tokens = [x[0] for x in result]
    assert tokens == [i % 11 for i in range(1, 22)]
    assert calls == [prompt[-1:] + tokens[:i] for i in range(21)]


def test_processor_history_matches_upstream_ar_after_prefill(monkeypatch):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    import vmlx_engine.adaptive_speculative as adaptive
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    from vmlx_engine.utils.token_logits_processors import make_openai_token_penalty_processor
    prompt = mx.array([2, 4, 3, 5, 3, 3])

    def processors():
        return [make_openai_token_penalty_processor(
            presence_penalty=1.5, frequency_penalty=.3, logit_bias={"3": 2})]

    reference = list(generate_step(prompt, CausalModel(), max_tokens=21,
                     prefill_step_size=2, logits_processors=processors()))
    actual = list(adaptive_generate_step(prompt, CausalModel(), CausalModel(5),
                  max_tokens=21, prefill_step_size=2, logits_processors=processors()))
    assert [int(x[0]) for x in actual] == [int(x[0]) for x in reference]
    for a, b in zip(actual, reference):
        assert mx.array_equal(a[1], b[1]).item()


def test_sampled_tokens_follow_target_not_draft_distribution(monkeypatch):
    import mlx.core as mx
    import vmlx_engine.adaptive_speculative as adaptive
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    mx.random.seed(184)
    target = CausalModel(distribution=np.log([.7, .3] + [1e-20] * 9))
    draft = CausalModel(distribution=np.log([.05, .95] + [1e-20] * 9))
    out = list(adaptive_generate_step(mx.array([2]), target, draft, max_tokens=1200,
               sampler=lambda lp: mx.random.categorical(lp)))
    fraction = sum(x[0] == 0 for x in out) / len(out)
    assert .65 < fraction < .75


def test_rejects_nontrimmable_draft_before_prefill():
    import mlx.core as mx
    caches = [TokenCache(), TokenCache()]
    caches[1].is_trimmable = lambda: False
    with pytest.raises(ValueError, match="trimmable target and draft"):
        list(adaptive_generate_step(mx.array([2]), CausalModel(), CausalModel(),
                                    prompt_cache=caches))
    assert not caches[0].tokens and not caches[1].tokens


@pytest.mark.parametrize("method", ["generate", "stream_generate"])
def test_both_llm_wrappers_use_adaptive_and_reject_mismatched_tokenizers(monkeypatch, method):
    from types import SimpleNamespace
    import vmlx_engine.speculative as spec
    import vmlx_engine.adaptive_speculative as adaptive
    from vmlx_engine.models.llm import MLXLanguageModel
    wrapper = MLXLanguageModel("target")
    wrapper._loaded = True
    wrapper.model = object()
    wrapper.tokenizer = SimpleNamespace(encode=lambda _: [4])
    monkeypatch.setattr(wrapper, "_create_sampler", lambda *a, **k: None)
    monkeypatch.setattr(spec, "should_use_speculative", lambda **k: True)
    draft = object()
    monkeypatch.setattr(spec, "get_draft_model", lambda: draft)
    monkeypatch.setattr(spec, "get_num_draft_tokens", lambda: 7)
    monkeypatch.setattr(spec, "validate_draft_tokenizer", lambda _: True)
    calls = []

    def stream(model, tokenizer, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(text="answer", token=4, finish_reason="stop")

    monkeypatch.setattr(adaptive, "adaptive_stream_generate", stream)
    result = getattr(wrapper, method)("prompt")
    if method == "stream_generate":
        result = list(result)
    assert calls[0]["draft_model"] is draft
    assert calls[0]["num_draft_tokens"] == 7  # Seed forwarded, actual controller tested separately.
    assert callable(calls[0]["on_stats"])
    monkeypatch.setattr(spec, "validate_draft_tokenizer", lambda _: False)
    with pytest.raises(ValueError, match="tokenizers are incompatible"):
        result = getattr(wrapper, method)("prompt")
        if method == "stream_generate":
            list(result)
    assert len(calls) == 1


def test_stream_eos_closes_verifier_and_preserves_final_event(monkeypatch):
    import contextlib
    import importlib
    import mlx.core as mx
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    import vmlx_engine.adaptive_speculative as adaptive
    generation = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(generation, "wired_limit", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({str(i): i for i in range(11)}, unk_token="0")),
        eos_token="7", unk_token="0")
    model = CausalModel()
    caches = model.make_cache() + model.make_cache()
    stats = []
    result = list(adaptive.adaptive_stream_generate(model, tokenizer, mx.array([2, 4]),
                  draft_model=model, prompt_cache=caches, max_tokens=20, on_stats=stats.append))
    assert len(result) == 1 and result[0].finish_reason == "stop"
    assert result[0].text == ""
    assert caches[0].tokens == caches[1].tokens == [2, 4]
    assert stats[0]["delivered_tokens"] == 1


def test_actual_mlx_kv_cache_rewinds_after_mismatch_and_ar_catchup(monkeypatch):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    import vmlx_engine.adaptive_speculative as adaptive
    monkeypatch.setattr(adaptive, "AdaptiveDraftWidth", WidthSequence)

    class Model(CausalModel):
        def make_cache(self):
            return [KVCache()]

        def __call__(self, tokens, cache):
            values = tokens[:, None, :, None].astype(mx.float32)
            keys, _ = cache[0].update_and_fetch(values, values)
            next_ids = (mx.cumsum(keys[0, 0, :, 0])[-tokens.shape[1]:] + 1 + self.shift) % 11
            return mx.where(mx.arange(11)[None, :] == next_ids[:, None], 50.0, -50.0)[None]

    target, draft = Model(), Model(1)
    caches = target.make_cache() + draft.make_cache()
    result = list(adaptive_generate_step(mx.array([2, 4]), target, draft,
                                        max_tokens=40, prompt_cache=caches))
    history = [2, 4] + [x[0] for x in result]
    for i in range(2, len(history)):
        assert history[i] == (sum(history[:i]) + 1) % 11
    assert caches[0].offset == len(history) - 1
    for c in caches:
        assert c.keys[0, 0, :c.offset, 0].tolist() == history[:c.offset]
