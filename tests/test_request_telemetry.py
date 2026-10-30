"""/health.last_request: one producer (vmlx_engine.request_telemetry) for every lane and API route.

Contract with the Perf panel (panel/src/renderer/src/components/sessions/perfPanelHelpers.ts lastRequestItems):
the per-lane records below are written to tests/fixtures/last_request/<lane>.json, and the panel test renders the
SAME files. A renamed or dropped key fails here (fixture mismatch) and there (panel reads the old path).
"""
import ast
import asyncio
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from vmlx_engine import request_telemetry as RT
from vmlx_engine.engine.base import GenerationOutput

FIXTURES = Path(__file__).parent / "fixtures" / "last_request"
VOLATILE = ("finished_at", "seq")


def _out(t, completion, *, finished=False, **kw):
    return GenerationOutput(text="", completion_tokens=completion, finished=finished,
                            finish_reason="stop" if finished else None, generated_at=t, **kw)


def _lane_outputs(lane):
    """Output sequences shaped like each lane's real stream (timestamps on the perf_counter clock)."""
    t0 = 1000.0
    receipt = {"tokens": 4096, "seconds": 2.0, "scope": "model_prefill_and_prompt_state"}
    if lane == "batched-llm":          # SingleBatchGenerator receipt, block-disk prefix hit
        return [_out(t0 + 0.5, 1, prompt_tokens=8192, cached_tokens=4096, cache_detail="block-disk",
                     prefill_usage=receipt),
                _out(t0 + 1.5, 51), _out(t0 + 2.5, 101, finished=True, prompt_tokens=8192, cached_tokens=4096,
                                           cache_detail="block-disk", prefill_usage=receipt)]
    if lane == "batched-mllm":         # cold VLM prompt (no hit)
        return [_out(t0 + 2.0, 1, prompt_tokens=4096, prefill_usage=dict(receipt, path="mllm")),
                _out(t0 + 4.0, 81, finished=True, prompt_tokens=4096, prefill_usage=dict(receipt, path="mllm"))]
    if lane == "batched-mllm-mtp":     # Qwen Flash-Next native MTP (record attached by the scheduler at finish)
        mtp = RT.mtp_spec_stats({"request_id": "r", "cycles": 70, "drafted_tokens": 140, "accepted_tokens": 98,
                                 "acceptance_rate": 0.7, "final_depth": 2, "depth_acceptance_rates":
                                 {"d1": 0.8, "d2": 0.6, "d3": None}, "finish_reason": "stop", "fallback_reason": None})
        r = dict(receipt, tokens=108, seconds=0.38, path="mllm")
        return [_out(t0 + 0.63, 1, prompt_tokens=6344, cached_tokens=6236, cache_detail="block-disk+ssm",
                     prefill_usage=r),
                _out(t0 + 4.89, 127, finished=True, prompt_tokens=6344, cached_tokens=6236,
                     cache_detail="block-disk+ssm", prefill_usage=r, spec_stats=mtp)]
    if lane == "simple-dflash2-qwen":  # DFlash2 with AR bursts, session-store hit
        spec = {"kind": "dflash2", "lane": "qwen", "cycles": 60, "emitted_per_cycle": 3.293, "drafted": 600,
                "accepted": 138, "acceptance": 0.23, "widths": {"5": 11, "8": 37, "16": 10}, "ar_tokens": 192,
                "generation_tps": 28.27}
        r = dict(receipt, tokens=49, seconds=0.12, path="dflash2")
        return [_out(t0 + 0.3, 1, prompt_tokens=4485, cached_tokens=4436, prefill_usage=r),
                _out(t0 + 10.3, 381, finished=True, prompt_tokens=4485, cached_tokens=4436, prefill_usage=r,
                     cache_detail="dflash2-session", spec_stats=spec)]
    if lane == "simple-dflash2-glm":   # GLM lane, prefix-store hit, adaptive AR
        spec = {"kind": "dflash2", "lane": "glm", "cycles": 40, "emitted_per_cycle": 2.1, "drafted": 160,
                "accepted": 44, "acceptance": 0.275, "widths": {"4": 40}, "ar_tokens": 64, "generation_tps": 38.2}
        r = dict(receipt, tokens=359, seconds=1.55, path="glm_dflash2")
        return [_out(t0 + 1.66, 1, prompt_tokens=4272, cached_tokens=3913, prefill_usage=r,
                     cache_detail="glm-dflash2-prefix"),
                _out(t0 + 2.4, 29, finished=True, prompt_tokens=4272, cached_tokens=3913, prefill_usage=r,
                     cache_detail="glm-dflash2-prefix", spec_stats=spec)]
    if lane == "simple-llm-no-receipt":  # mlx-lm SimpleEngine path: no engine prefill receipt
        return [_out(t0 + 1.0, 1, prompt_tokens=1000), _out(t0 + 2.0, 31, finished=True, prompt_tokens=1000)]
    raise AssertionError(lane)


LANES = ["batched-llm", "batched-mllm", "batched-mllm-mtp", "simple-dflash2-qwen", "simple-dflash2-glm",
         "simple-llm-no-receipt"]


class _Engine:
    is_mllm = False

    def __init__(self, outputs):
        self.outputs = outputs
        self.closed = False

    async def stream_chat(self, *args, **kwargs):
        try:
            for out in self.outputs:
                yield out
        finally:
            self.closed = True

    async def chat(self, *args, **kwargs):          # SimpleEngine.chat drains its own stream_chat
        final = None
        async for out in self.stream_chat():
            final = out
        return final

    async def stream_generate(self, *args, **kwargs):
        for out in self.outputs:
            yield out

    async def generate(self, *args, **kwargs):
        return self.outputs[-1]


SimpleEngine = type("SimpleEngine", (_Engine,), {})
BatchedEngine = type("BatchedEngine", (_Engine,), {})


def _record(lane, monkeypatch):
    RT.reset()
    outputs = _lane_outputs(lane)
    cls = SimpleEngine if lane.startswith("simple") else BatchedEngine
    engine = RT.instrument(cls(outputs))
    if lane.startswith("batched-mllm"):
        engine.is_mllm = True
    clock = iter([1000.0])
    monkeypatch.setattr(RT.time, "perf_counter", lambda: next(clock, 1000.0))

    async def run():
        return [o async for o in engine.stream_chat()]

    assert len(asyncio.run(run())) == len(outputs)
    record = RT.last()
    return {k: v for k, v in record.items() if k not in VOLATILE}


@pytest.mark.parametrize("lane", LANES)
def test_each_lane_record_matches_the_shared_panel_fixture(lane, monkeypatch):
    record = _record(lane, monkeypatch)
    path = FIXTURES / f"{lane}.json"
    expected = json.loads(path.read_text())
    assert record == expected, f"{lane}: update {path} AND the panel test if this change is intended"


def test_batched_hit_record_has_every_panel_figure(monkeypatch):
    r = _record("batched-llm", monkeypatch)
    assert r["ttft_s"] == 0.5 and r["decode"] == {"tokens": 100, "seconds": 2.0, "tps": 50.0}
    assert r["prefill"]["tps"] == 2048.0 and r["cached_prompt_tps"] == 16384.0
    assert r["cache_detail"] == "block-disk" and r["finish_reason"] == "stop" and r["aborted"] is False


def test_early_close_records_an_aborted_request_and_closes_the_engine_stream():
    RT.reset()
    engine = RT.instrument(SimpleEngine(_lane_outputs("batched-llm")))

    async def run():
        stream = engine.stream_chat()
        await stream.__anext__()
        await stream.aclose()                        # client disconnect

    asyncio.run(run())
    assert engine.closed is True                     # the engine's own finally (request abort) ran immediately
    assert RT.last()["aborted"] is True and RT.last()["finish_reason"] is None


def test_chat_draining_its_stream_keeps_the_stream_record(monkeypatch):
    RT.reset()
    engine = RT.instrument(SimpleEngine(_lane_outputs("simple-dflash2-qwen")))
    asyncio.run(engine.chat())
    record = RT.last()
    assert record["method"] == "stream_chat" and record["decode"] is not None and record["seq"] == 1


def test_non_stream_generate_records_counts_without_timings():
    RT.reset()
    engine = RT.instrument(BatchedEngine(_lane_outputs("batched-mllm")))
    asyncio.run(engine.generate())
    record = RT.last()
    assert record["method"] == "generate" and record["completion_tokens"] == 81 and record["decode"] is None


def test_instrument_is_idempotent():
    engine = SimpleEngine([])
    RT.instrument(engine)
    first = engine.stream_chat
    RT.instrument(engine)
    assert engine.stream_chat is first


def test_load_model_instruments_every_engine_it_creates():
    import vmlx_engine.server as server

    src = textwrap.dedent(inspect.getsource(server.load_model))
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "_request_telemetry.instrument"]
    assert len(calls) == 2                            # distributed early return + local engines


@pytest.mark.asyncio
async def test_health_and_cache_stats_publish_the_record(monkeypatch):
    import vmlx_engine.server as server
    import vmlx_engine.speculative as speculative

    RT.reset()
    engine = RT.instrument(SimpleEngine(_lane_outputs("simple-llm-no-receipt")))
    async for _ in engine.stream_chat():
        pass
    engine.get_cache_stats = lambda: None
    engine.get_stats = lambda: {"engine_type": "simple"}
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_get_scheduler", lambda: None)
    monkeypatch.setattr(speculative, "is_dflash2_enabled", lambda: False)
    payload = await server.cache_stats()
    assert payload["last_request"]["completion_tokens"] == 31


def test_mtp_record_is_attached_only_to_its_own_request(monkeypatch):
    """Text lane: the MTP lane keeps ONE process-global record keyed by generator uid."""
    from vmlx_engine.text_mtp import lane_stats
    from vmlx_engine.scheduler import _native_mtp_spec_stats_for

    monkeypatch.setattr(lane_stats, "_last", {"request_id": "7", "acceptance_rate": 0.5, "drafted_tokens": 10,
                                                      "accepted_tokens": 5, "cycles": 6, "final_depth": 1})
    assert _native_mtp_spec_stats_for(7)["acceptance"] == 0.5
    assert _native_mtp_spec_stats_for(8) is None          # another request's record never leaks
    monkeypatch.setattr(lane_stats, "_last", None)
    assert _native_mtp_spec_stats_for(7) is None


def test_mllm_scheduler_attaches_the_finished_requests_mtp_record():
    src = textwrap.dedent(inspect.getsource(__import__("vmlx_engine.mllm_scheduler", fromlist=["x"]).MLLMScheduler
                                            ._process_batch_responses))
    assert 'str(_mtp.get("request_id")) == str(request_id)' in src
    assert "output.spec_stats = mtp_spec_stats(_mtp)" in src


def test_mtp_phase_scope_survives_projection_without_mutating_record():
    record = {"request_id": "phase-row", "scope": "last_native_mtp_phase",
              "cycles": 2, "drafted_tokens": 4, "accepted_tokens": 3,
              "acceptance_rate": 0.75, "finish_reason": "ar_calibration"}
    before = dict(record)
    projected = RT.mtp_spec_stats(record)
    assert projected["scope"] == "last_native_mtp_phase"
    assert projected["cycles"] == 2
    assert projected["finish_reason"] == "ar_calibration"
    assert record == before


def test_last_request_finish_and_timing_are_not_mtp_phase_finish(monkeypatch):
    monkeypatch.setattr(RT, "_LAST", None)
    monkeypatch.setattr(RT, "_SEQ", 0)
    phase = RT.mtp_spec_stats({"scope": "last_native_mtp_phase", "cycles": 2,
                              "finish_reason": "ar_calibration", "accepted_tokens": 3})
    tracker = RT._Tracker("stream_chat", "batched", "mllm")
    tracker.start = 1000.0
    tracker.observe(_out(1001.0, 1))
    tracker.observe(_out(1002.0, 51, finished=True, spec_stats=phase))
    tracker.record()
    rec = RT.last()
    assert rec["finish_reason"] == "stop"
    assert rec["completion_tokens"] == 51
    assert rec["speculative"]["scope"] == "last_native_mtp_phase"
    assert rec["speculative"]["finish_reason"] == "ar_calibration"
    assert rec["decode"]["tokens"] == 50
