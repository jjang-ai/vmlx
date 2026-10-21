"""Request-local bypass must avoid both prefix reads and snapshot writes."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest


class Detokenizer:
    last_segment = ""

    def reset(self):
        pass

    def add_token(self, token):
        pass

    def finalize(self):
        pass


class Tokenizer:
    bos_token = None
    eos_token_ids = {0}
    eos_token_id = 0
    detokenizer = Detokenizer()

    def encode(self, *args, **kwargs):
        return [1, 2, 3]

    def __call__(self, *args, **kwargs):
        return {"input_ids": [1, 2, 3]}


def forbidden(*args, **kwargs):
    raise AssertionError("bypassed request touched a prefix cache")


def test_qwen_bypass_skips_ram_ssd_snapshots_and_terminal_store(monkeypatch):
    import dflash.model_mlx as upstream
    from vmlx_engine import dflash2_runtime as df

    monkeypatch.setenv("VMLX_DFLASH2_PREFIX_REUSE", "1")
    monkeypatch.setattr(df, "_SESSION_STORE", SimpleNamespace(take_matching=forbidden, describe_misses=forbidden))
    monkeypatch.setattr(df, "_SESSION_SSD", SimpleNamespace(take_matching=forbidden, put=forbidden))
    for name in ("_store_session", "_assistant_tag_cut", "_system_end_cut", "_chunk_snapshot_cuts"):
        monkeypatch.setattr(df, name, forbidden)
    monkeypatch.setattr(upstream, "TokenizerWrapper", Tokenizer)
    monkeypatch.setattr(upstream, "_patch_model", lambda *args: None)
    monkeypatch.setattr(upstream, "make_prompt_cache", lambda *args: [SimpleNamespace(offset=0)])
    monkeypatch.setattr(upstream, "can_trim_prompt_cache", lambda cache: True)
    monkeypatch.setattr(df, "_rebase_drafter_cache", lambda *args: None)
    monkeypatch.setattr(df, "_prefill_last_logits", lambda *args: (mx.array([[[9., 0.]]]), mx.zeros((1, 3, 2)), 0))
    monkeypatch.setattr(upstream, "_make_response", lambda *args: SimpleNamespace(finish_reason=args[-1]))
    draft = SimpleNamespace(bind=lambda a: None, config=SimpleNamespace(
        target_layer_ids=[], mask_token_id=1, sliding_window=8, layer_types=["sliding_attention"]))
    result = list(df._stream_generate_resumable(object(), SimpleNamespace(), draft, Tokenizer(), "prompt",
                  block_size=5, max_tokens=3, temperature=0., top_p=1., top_k=0, use_cache=False))
    assert result[-1].finish_reason == "stop"
    assert result[-1].cached_tokens == 0
    assert result[-1].persistence_future is None
    assert result[-1].prefill_usage["tokens"] == 3


@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_dflash_bridge_preserves_request_cache_policy(monkeypatch, glm, use_cache):
    import dflash.model_mlx as upstream
    from vmlx_engine import dflash2_runtime as df, glm_dflash2 as gd
    monkeypatch.setattr(gd, "is_glm5_next", lambda model: glm)
    monkeypatch.setattr(upstream, "wired_limit", lambda *args: nullcontext())
    monkeypatch.setattr(df, "_adapter_for", lambda model: model)
    route = Mock(return_value=iter([]))
    if glm:
        monkeypatch.setattr(gd, "stream_glm_dflash2", route)
    else:
        monkeypatch.setattr(df, "_stream_generate_resumable", route)
    list(df.stream_dflash2_generate(object(), object(), SimpleNamespace(config=SimpleNamespace(block_size=5)),
         "prompt", max_tokens=3, temperature=0., use_cache=use_cache))
    assert route.call_args.kwargs["use_cache"] is use_cache


def test_glm_bypass_does_not_create_or_access_prefix_store(monkeypatch):
    import mlx_lm.tokenizer_utils as tu
    from vmlx_engine import glm_dflash2 as gd
    monkeypatch.setattr(tu, "TokenizerWrapper", Tokenizer)
    monkeypatch.setattr(gd, "target_for", lambda *args: object())
    monkeypatch.setattr(gd, "prefix_store_for", forbidden)
    rounds = Mock(return_value=iter([]))
    monkeypatch.setattr(gd, "_dflash2_rounds", rounds)
    list(gd.stream_glm_dflash2(object(), Tokenizer(), SimpleNamespace(config=SimpleNamespace(target_layer_ids=[])),
         "prompt", max_tokens=3, temperature=0., use_cache=False))
    assert rounds.call_args.kwargs["prefix_store"] is None


def test_dspark_bypass_skips_lookup_snapshot_and_turn_store(monkeypatch):
    from vmlx_engine.models import mllm
    from vmlx_engine import dspark
    from vmlx_engine.dspark import session
    import mlx_vlm.tokenizer_utils as tu
    for name in ("lookup", "snapshot_cuts", "store"):
        monkeypatch.setattr(session, name, forbidden)
    monkeypatch.setattr(tu, "make_streaming_detokenizer", lambda p: Detokenizer())
    monkeypatch.setattr(mllm, "_MaybeVLMStream", nullcontext)
    target = SimpleNamespace(make_cache=lambda: [], set_media=lambda **kw: None, rope_delta=0)
    drafter = SimpleNamespace(layers=[], context_caches=forbidden)
    model = mllm.MLXMultimodalLM.__new__(mllm.MLXMultimodalLM)
    model.model_name = "fixture"
    model.processor = Tokenizer()
    model._dspark = dict(target=target, drafter=drafter, max_draft=4, cost_per_row=1.)

    def rounds(*args, **kw):
        assert kw["resume"] is None and kw["snapshot_cuts"] == []
        kw["session"].update(forwarded=[1, 2, 3], target_cache=[object()])
        yield [0]

    monkeypatch.setattr(dspark, "iter_dspark", rounds)
    result = list(model._stream_dspark("prompt", 3, 0., {"use_cache": False}))
    assert result[-1].cached_tokens == 0 and result[-1].persistence_future is None


def test_mimo_bypass_skips_lookup_snapshot_and_turn_store(monkeypatch):
    import mlx_lm.tokenizer_utils as tu
    from vmlx_engine.models import mllm
    from vmlx_engine import mimo_session as session, mimo_dflash as md
    for name in ("lookup", "snapshot_cuts", "store"):
        monkeypatch.setattr(session, name, forbidden)
    monkeypatch.setattr(tu, "TokenizerWrapper", Tokenizer)
    target = SimpleNamespace(model=SimpleNamespace(make_cache=lambda: []))
    monkeypatch.setattr(md, "_lane_for", lambda *args: {"target": target})
    model = mllm.MLXMultimodalLM.__new__(mllm.MLXMultimodalLM)
    model.model_name, model.model, model.config = "fixture", object(), {}
    model.processor = lambda *args, **kw: {"input_ids": mx.array([[1, 2, 3]])}
    # The processor and tokenizer are distinct in production.
    model.processor.tokenizer = Tokenizer()

    def rounds(*args, **kw):
        assert kw["resume"] is None and kw["snapshot_cuts"] == []
        kw["session"].update(forwarded=3, target_cache=[object()], taps=None)
        yield [151645], 0

    monkeypatch.setattr(md, "mimo_rounds", rounds)
    result = list(model._stream_mimo_dflash("prompt", 3, 0., {"use_cache": False},
                  draft=(object(), object()), images=[], videos=[], audio=[]))
    assert result[-1].cached_tokens == 0 and result[-1].persistence_future is None
