"""Exercise request cleanup before and after recurrent rollback capture starts."""

from contextlib import nullcontext
from types import SimpleNamespace
from weakref import ref

import pytest

from vmlx_engine import dflash2_runtime as runtime


class RecordedState:
    pass


class TargetLayers:
    layers = ()

    def __call__(self, inputs, *, gdn_sink, **kwargs):
        if gdn_sink is not None:
            gdn_sink.append(RecordedState())
        return inputs

    @staticmethod
    def norm(array):
        return array


@pytest.fixture
def route(monkeypatch):
    import dflash.model_mlx as upstream
    import mlx.core as mx

    rollbacks = []
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=TargetLayers(),
        lm_head=lambda array: array,
        rollback_speculative_cache=lambda cache, states, accepted, block: rollbacks.append(len(states)),
    ))
    draft = SimpleNamespace(config=SimpleNamespace(block_size=8))
    store = runtime._DFlash2SessionStore()
    checkpoint = {"model_key": (id(model), id(draft)), "tokens": [1], "cache_len": 1}
    store.put(checkpoint)
    monkeypatch.setattr(runtime, "_SESSION_STORE", store)
    monkeypatch.setattr(upstream, "wired_limit", lambda *args: nullcontext())
    adapter = runtime._adapter_for(model)
    hooks = [None]
    adapter._hidden_states = hooks

    def run():
        return runtime.stream_dflash2_generate(
            model, None, draft, "example", max_tokens=4, temperature=0
        )

    return SimpleNamespace(
        adapter=adapter, hooks=hooks, rows=mx.zeros((1, 3, 1)), run=run,
        store=store, checkpoint=checkpoint, rollbacks=rollbacks,
    )


def test_rollback_captures_only_the_latest_requested_forward(route):
    adapter = route.adapter
    adapter(route.rows)
    assert not adapter.gdn_states
    capture = runtime._VLMGDNStateCapture(adapter)
    adapter(route.rows)
    previous = ref(adapter.gdn_states[0])
    adapter(route.rows)
    assert previous() is None
    capture.rollback([], accepted=1, trim=1)
    assert route.rollbacks == [1]
    capture.close()
    assert not adapter.gdn_states
    adapter(route.rows)
    assert not adapter.gdn_states


@pytest.mark.parametrize("exit_mode", ["success", "disconnect", "prefill_error"])
def test_public_route_releases_owned_handles_and_keeps_completed_cache(route, monkeypatch, exit_mode):
    observed = []

    def generate(*args, **kwargs):
        route.hooks[0] = RecordedState()
        observed.append(ref(route.hooks[0]))
        route.adapter.media_embeds = RecordedState()
        observed.append(ref(route.adapter.media_embeds))
        route.adapter.media_positions = RecordedState()
        observed.append(ref(route.adapter.media_positions))
        route.adapter.rope_delta = -3
        if exit_mode == "prefill_error":
            raise RuntimeError("prefill setup failed")
        runtime._VLMGDNStateCapture(route.adapter)
        route.adapter.gdn_states.append(RecordedState())
        observed.append(ref(route.adapter.gdn_states[0]))
        yield "first"
        yield "last"

    monkeypatch.setattr(runtime, "_stream_generate_resumable", generate)
    gen = route.run()
    if exit_mode == "disconnect":
        assert next(gen) == "first"
        gen.close()
    elif exit_mode == "prefill_error":
        with pytest.raises(RuntimeError, match="prefill setup failed"):
            next(gen)
    else:
        assert list(gen) == ["first", "last"]

    assert all(state() is None for state in observed)
    assert route.adapter._hidden_states is route.hooks
    assert route.hooks == [None]
    assert route.adapter.rope_delta == 0
    assert not route.adapter.gdn_states
    assert route.store.take_matching(route.checkpoint["model_key"], [1, 2]) is route.checkpoint
