"""Non-streamed MLLM requests on a DFlash2 server take the streaming route (I-25).

MLLM.chat() is a plain mlx-vlm generate with no DFlash2 decode, session store or
media plan, so before this every non-streamed request on a 27B DFlash2 server
ran AR and re-prefilled the whole prompt.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vmlx_engine.engine import simple as simple_mod
from vmlx_engine.engine.simple import SimpleEngine


class _Model:
    def __init__(self):
        self.stream_kwargs = None

    def chat(self, **_kwargs):
        raise AssertionError("DFlash2 non-streamed chat must not use MLLM.chat")

    def stream_chat(self, **kwargs):
        self.stream_kwargs = kwargs
        usage = {"tokens": 7, "seconds": 0.01, "scope": "model_prefill_and_prompt_state", "path": "dflash2"}
        for text, done in (("Bl", False), ("ue.", True)):
            yield SimpleNamespace(
                text=text,
                finish_reason="stop" if done else None,
                prompt_tokens=2595,
                completion_tokens=2 if done else 1,
                cached_tokens=2588,
                cache_detail="dflash2",
                prefill_usage=usage,
                persistence_future=None,
            )


def _engine(model):
    engine = SimpleEngine.__new__(SimpleEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._model = model
    engine._generation_lock = asyncio.Lock()

    async def _run_model_call(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    engine._run_model_call = _run_model_call
    return engine


def test_dflash2_nonstreamed_mllm_chat_drains_the_stream_route(monkeypatch):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)
    model = _Model()
    out = asyncio.run(
        _engine(model).chat(
            [{"role": "user", "content": "What color?"}],
            max_tokens=20,
            temperature=0.0,
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    )
    assert out.text == "Blue."
    assert out.finish_reason == "stop"
    assert (out.prompt_tokens, out.completion_tokens, out.cached_tokens) == (2595, 2, 2588)
    assert out.prefill_usage["scope"] == "model_prefill_and_prompt_state"
    assert model.stream_kwargs["enable_thinking"] is False
    assert model.stream_kwargs["chat_template_kwargs"] == {"enable_thinking": False}


def test_non_dflash2_mllm_chat_keeps_the_native_route(monkeypatch):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: False)
    with pytest.raises(AssertionError, match="must not use MLLM.chat"):
        asyncio.run(_engine(_Model()).chat([{"role": "user", "content": "hi"}], max_tokens=4))


@pytest.mark.parametrize("bypass", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_cache_bypass_is_request_local_on_chat(monkeypatch, bypass, stream):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)
    model = _Model()
    engine = _engine(model)

    async def run():
        if stream:
            return [x async for x in engine.stream_chat([], _bypass_prefix_cache=bypass)]
        return await engine.chat([], _bypass_prefix_cache=bypass)

    asyncio.run(run())
    assert "_bypass_prefix_cache" not in model.stream_kwargs
    assert model.stream_kwargs.get("use_cache", True) is (not bypass)
    # A following request must not inherit the bypass.
    asyncio.run(engine.chat([]))
    assert model.stream_kwargs.get("use_cache", True) is True


@pytest.mark.parametrize("is_mllm", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_generate_translates_bypass_only_for_mllm(is_mllm, stream):
    calls = []
    result = SimpleNamespace(text="ready", finish_reason="stop", prompt_tokens=3,
                             completion_tokens=1, tokens=[7])

    def generate(**kwargs):
        calls.append(kwargs)
        return result

    def stream_generate(**kwargs):
        calls.append(kwargs)
        yield result

    model = SimpleNamespace(generate=generate, stream_generate=stream_generate)
    engine = _engine(model)
    engine._is_mllm = is_mllm

    async def run():
        if stream:
            return [x async for x in engine.stream_generate("prompt", _bypass_prefix_cache=True)]
        return await engine.generate("prompt", _bypass_prefix_cache=True)

    asyncio.run(run())
    assert "_bypass_prefix_cache" not in calls[0]
    if is_mllm:
        assert calls[0]["use_cache"] is False
    else:
        assert "use_cache" not in calls[0]
