"""Observable mflux calls without changing model math or forcing GPU evaluation."""
import json
import logging
import time
import uuid
from .image_requests import current_image_request

logger = logging.getLogger(__name__)


class ImageCallTrace:
    def __init__(self, *, model, model_class, steps, seed):
        self.job_id = "img_" + uuid.uuid4().hex
        self.started = time.monotonic()
        self.request = current_image_request.get()
        self.fields = dict(model=model, model_class=model_class, requested_steps=steps, seed=seed,
                           request_id=self.request.request_id if self.request else None)

    def check_cancelled(self):
        if self.request:
            self.request.check()

    def event(self, phase, **fields):
        record = dict(self.fields, job_id=self.job_id, phase=phase,
                      elapsed_seconds=round(time.monotonic() - self.started, 3), **fields)
        logger.info("IMAGEJOB %s", json.dumps(record, separators=(",", ":")))

    def call_before_loop(self, **kwargs):
        self.check_cancelled()
        self.event("before_denoise_loop")

    def call_in_loop(self, *, t, **kwargs):
        self.check_cancelled()
        # In Qwen this callback precedes mx.eval(latents). It is a checkpoint,
        # not proof that step t's GPU work has completed. Never add an eval here.
        self.event("denoise_checkpoint", step_index=int(t), completion_confirmed=False)

    def call_after_loop(self, **kwargs):
        self.check_cancelled()
        self.event("after_denoise_loop")

    def call_interrupt(self, *, t, **kwargs):
        self.event("interrupted", step_index=int(t))


def observed_image_call(model, *, model_name, model_class, **kwargs):
    trace = ImageCallTrace(model=model_name, model_class=model_class,
                           steps=kwargs.get("num_inference_steps"), seed=kwargs.get("seed"))
    registry = getattr(model, "callbacks", None)
    lists = [getattr(registry, name, None) for name in ("before_loop", "in_loop", "after_loop", "interrupt")]
    observable = all(isinstance(items, list) for items in lists) and callable(getattr(registry, "register", None))
    trace.event("model_call_started", step_callbacks=observable,
                width=kwargs.get("width"), height=kwargs.get("height"))
    try:
        trace.check_cancelled()
        if observable:
            registry.register(trace)
        result = model.generate_image(**kwargs)
        trace.check_cancelled()
        trace.event("model_call_returned")
        return result, trace
    except BaseException:
        # Record cancellation/interrupt too, without converting its semantics.
        logger.exception("IMAGEJOB %s", json.dumps(dict(trace.fields, job_id=trace.job_id,
                         phase="model_call_failed", elapsed_seconds=round(time.monotonic()-trace.started, 3))))
        raise
    finally:
        # This trace owns only its own subscribers; preserve every other callback.
        for items in lists:
            if isinstance(items, list):
                items[:] = [item for item in items if item is not trace]
