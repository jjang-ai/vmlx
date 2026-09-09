import json
import logging
from types import SimpleNamespace

import pytest

from vmlx_engine.image_progress import observed_image_call


class Registry:
    def __init__(self):
        self.before_loop, self.in_loop, self.after_loop, self.interrupt = [], [], [], []

    def register(self, callback):
        for name in ("before_loop", "in_loop", "after_loop", "interrupt"):
            if hasattr(callback, "call_" + name):
                getattr(self, name).append(callback)


def records(caplog):
    return [json.loads(r.getMessage().split("IMAGEJOB ", 1)[1])
            for r in caplog.records if r.getMessage().startswith("IMAGEJOB ")]


def test_callbacks_log_checkpoints_without_changing_arguments_or_result(caplog):
    registry = Registry()
    sentinel = object()
    registry.before_loop.append(sentinel)
    kwargs = dict(prompt="private prompt", image_paths=["private source"], seed=42,
                  num_inference_steps=2, width=512, height=512)
    def generate(**received):
        assert received == kwargs
        registry.before_loop[-1].call_before_loop()
        for t in range(2):
            registry.in_loop[-1].call_in_loop(t=t)
        registry.after_loop[-1].call_after_loop()
        return sentinel
    model = SimpleNamespace(callbacks=registry, generate_image=generate)
    with caplog.at_level(logging.INFO):
        result, trace = observed_image_call(model, model_name="qwen-image-edit",
                                            model_class="QwenImageEdit", **kwargs)
    assert result is sentinel
    events = records(caplog)
    assert [e["phase"] for e in events] == ["model_call_started", "before_denoise_loop",
             "denoise_checkpoint", "denoise_checkpoint", "after_denoise_loop", "model_call_returned"]
    assert all(e["job_id"] == trace.job_id for e in events)
    assert [e["step_index"] for e in events if "step_index" in e] == [0, 1]
    assert not events[2]["completion_confirmed"]
    assert "private" not in caplog.text
    assert registry.before_loop == [sentinel]
    assert registry.in_loop == registry.after_loop == registry.interrupt == []


@pytest.mark.parametrize("error", [ValueError("bad shape"), KeyboardInterrupt()])
def test_failure_keeps_traceback_and_removes_only_owned_callbacks(error, caplog):
    registry = Registry()
    def fail(**kwargs):
        raise error
    with caplog.at_level(logging.INFO), pytest.raises(type(error)):
        observed_image_call(SimpleNamespace(callbacks=registry, generate_image=fail),
                            model_name="test", model_class="test", seed=1)
    assert records(caplog)[-1]["phase"] == "model_call_failed"
    assert caplog.records[-1].exc_info[1] is error
    assert registry.before_loop == registry.in_loop == registry.after_loop == registry.interrupt == []


def test_no_callback_capability_is_explicit_and_not_fabricated(caplog):
    model = SimpleNamespace(generate_image=lambda **kwargs: "image")
    with caplog.at_level(logging.INFO):
        result, _ = observed_image_call(model, model_name="legacy", model_class="legacy")
    assert result == "image"
    events = records(caplog)
    assert events[0]["step_callbacks"] is False
    assert [e["phase"] for e in events] == ["model_call_started", "model_call_returned"]
