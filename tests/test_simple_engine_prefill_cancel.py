"""A client that drops during a long MLLM prefill must stop that prefill at its next chunk boundary, even though the
follow-up request takes the generation lock first and resets the engine-wide abort flag (MiMo DFlash lane 2026-10-16:
a dropped 75k-token prompt kept prefilling, the follow-up waited 104 s). SimpleEngine keeps a per-request cancel event,
set by abort_request and by the stream's own cleanup, that the cooperative cancel points read."""
import asyncio
import time

import pytest

from vmlx_engine.engine.simple import SimpleEngine
from vmlx_engine.models.mllm import MLLMOutput
from vmlx_engine.utils.generation_cancellation import raise_if_generation_cancelled

CHUNKS, CHUNK_S = 40, 0.05


class SlowPrefillMllm:
    def __init__(self):
        self.done = []

    def stream_chat(self, messages, max_tokens=8, temperature=0.0, top_p=1.0, **kwargs):
        tag = messages[-1]["content"]
        for i in range(CHUNKS if tag == "long" else 1):      # a long prompt: 40 prefill chunks of 50 ms
            raise_if_generation_cancelled()
            time.sleep(CHUNK_S)
            self.done.append((tag, i))
        yield MLLMOutput(text="ok", finish_reason="stop", prompt_tokens=1, completion_tokens=1)


@pytest.mark.asyncio
async def test_dropped_long_prefill_stops_and_the_follow_up_is_served_promptly():
    engine = SimpleEngine("prefill-cancel-test")
    fake = SlowPrefillMllm()
    engine._is_mllm, engine._loaded, engine._model = True, True, fake

    async def consume(content, rid):
        return [o async for o in engine.stream_chat(messages=[{"role": "user", "content": content}], max_tokens=4,
                                                    temperature=0.0, request_id=rid)]

    first = asyncio.create_task(consume("long", "req-long"))
    await asyncio.sleep(0.3)                                  # a few chunks into the prefill
    first.cancel()                                            # the client is gone (Starlette cancels the consumer)
    with pytest.raises(asyncio.CancelledError):
        await first
    await engine.abort_request("req-long")                    # the server's abort for the dropped stream
    t0 = time.perf_counter()
    out = await consume("short", "req-short")                 # immediate follow-up
    waited = time.perf_counter() - t0
    long_done = sum(1 for tag, _ in fake.done if tag == "long")
    assert out and out[-1].finished
    assert long_done < CHUNKS // 2, long_done                 # stopped at a chunk boundary, not after 40 chunks
    assert waited < 0.5 * CHUNKS * CHUNK_S, waited            # not queued behind the whole abandoned prefill
    assert not engine._cancel_events                          # no leaked per-request events
