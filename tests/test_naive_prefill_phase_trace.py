"""Execute producer methods without MLX; phase tracing cannot change boundaries."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from test_cache_cleanup_phase_timing import ROOT, load_functions


class Array:
    def __getitem__(self, key):
        return self


def owner(monkeypatch, enabled=True, family="naive_n05_flash", fail=None):
    monkeypatch.setenv("VMLX_NAIVE_PREFILL_PHASE_TRACE", "1" if enabled else "0")
    events, logs, clocks = [], [], []

    def clock():
        clocks.append(1)
        return len(clocks) / 1000

    def event(name):
        events.append(name)
        if fail == name:
            raise RuntimeError(name)

    array = Array()
    req = SimpleNamespace(
        uid=91,
        context_tokens=[10, 11],
        prompt_tokens=[1, 2, 3, 4],
        cache=[SimpleNamespace(state="native")],
        logits_processors=[],
        prefill_started=None,
        pixel_values=None,
        pixel_values_videos=None,
        gen_prompt_len=0,
    )

    class Model:
        model_type = family

        def __call__(self, *args, **kwargs):
            event("final_model")
            return array

    def evaluate(*args):
        event("final_eval")
        return args

    def sample(*args, **kwargs):
        event("sample")
        return array, None

    def snapshot(req):
        event("snapshot")
        return ["detached"]

    ns = dict(
        os=os,
        QuantizedEmbedding=type("QuantizedEmbedding", (), {}),
        time=SimpleNamespace(perf_counter=clock),
        logger=SimpleNamespace(info=lambda *args: logs.append(args)),
        mx=SimpleNamespace(
            uint32="uint32",
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
            eval=lambda state: event("state_eval"),
            clear_cache=lambda: event("clear"),
            get_active_memory=lambda: 10,
            get_peak_memory=lambda: 20,
            reset_peak_memory=lambda: event("reset_peak"),
        ),
        _prefill_keep_alloc_enabled=lambda: False,
        _prefill_valve_enabled=lambda: True,
        get_effective_metal_working_set_bytes=lambda mx: (0, 100),
        _prefill_valve_min_margin_bytes=lambda: 1,
        _prefill_valve_check=lambda *args, **kwargs: event("valve"),
        _cold_prefill_tail_split=lambda n: 0,
    )
    load_functions(
        ROOT / "utils/single_batch_generator.py",
        {
            "_naive_prefill_phase_trace_enabled",
            "_naive_prefill_attention_geometry",
            "_log_naive_prefill_phase",
            "_prefill",
            "_start_request",
            "_compute_next_from_input_array",
        },
        ns,
    )
    obj = SimpleNamespace(
        model=Model(),
        prefill_step_size=2,
        _stream_context=nullcontext,
        _model_call=lambda chunk, req: event(("chunk", tuple(chunk))),
        _sync=lambda: event("sync"),
        _decode_trace=False,
        _rehome_on_stream=lambda x: x,
        _eval_on_stream=evaluate,
        _can_overlap_decode=lambda req: False,
        _sample_from_logits=sample,
        _needs_affine2_sync=lambda cache: False,
        _refresh_thread_stream=lambda: event("refresh"),
        _cache_uses_m3_msa=lambda cache: False,
        _cache_uses_openpangu=lambda cache: False,
        _cache_uses_glm5_next=lambda cache: False,
        _clone_naive_prompt_snapshot=snapshot,
        _yield_current_and_schedule_next=lambda req, **kwargs: "response",
    )
    obj._prefill = lambda tokens, req: ns["_prefill"](obj, tokens, req)
    obj._compute_next_from_input = lambda req, token: ns[
        "_compute_next_from_input_array"
    ](obj, req, array)
    return ns, obj, req, events, logs, clocks


@pytest.mark.parametrize("enabled", [False, True])
def test_connected_prefill_keeps_existing_completion_and_snapshot_order(
    monkeypatch, enabled
):
    ns, obj, req, events, logs, clocks = owner(monkeypatch, enabled)
    assert ns["_start_request"](obj, req) == "response"
    assert events == [
        "refresh",
        "valve",
        "reset_peak",
        ("chunk", (1, 2)),
        "state_eval",
        "sync",
        "clear",
        "valve",
        "reset_peak",
        ("chunk", (3,)),
        "state_eval",
        "sync",
        "clear",
        "snapshot",
        "final_model",
        "final_eval",
        "sample",
    ]
    assert req.context_tokens == [10, 11, 1, 2, 3, 4]
    assert req.prompt_cache_snapshot == ["detached"]
    assert req.next_token_materialized and req.prefill_started is None
    if enabled:
        assert [row[2] for row in logs] == [
            "chunk_completed",
            "chunk_completed",
            "prompt_snapshot_completed",
            "prompt_logits_completed",
        ]
        assert [row[3]["prior_context_tokens"] for row in logs] == [2, 4, 5, 5]
        assert [row[3]["query_tokens"] for row in logs if "query_tokens" in row[3]] == [
            2,
            1,
            1,
        ]
        for row in logs[:2]:
            phases = row[3]
            assert phases["total_ms"] == pytest.approx(
                sum(
                    phases[k]
                    for k in (
                        "valve_and_peak_reset_ms",
                        "model_and_native_state_ms",
                        "sync_ms",
                        "peak_read_ms",
                        "clear_ms",
                    )
                )
            )
        assert logs[-1][3]["full_prefill_ms"] == req.prefill_usage["seconds"] * 1000
    else:
        assert logs == []
        assert len(clocks) == 2  # Only the pre-existing aggregate producer clock.


@pytest.mark.parametrize(
    "enabled,family", [(False, "naive_n05_flash"), (True, "other")]
)
def test_chunk_has_no_clock_when_disabled_or_other_family(monkeypatch, enabled, family):
    ns, obj, req, events, logs, clocks = owner(monkeypatch, enabled, family)
    ns["_prefill"](obj, [1, 2], req)
    assert clocks == [] and logs == []
    assert events.count("sync") == 1


@pytest.mark.parametrize("failure", [("chunk", (1, 2)), "state_eval", "sync", "clear"])
def test_failed_chunk_never_emits_completed_stamp(monkeypatch, failure):
    ns, obj, req, events, logs, clocks = owner(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError):
        ns["_prefill"](obj, [1, 2], req)
    assert logs == []


@pytest.mark.parametrize("failure", ["snapshot", "final_eval"])
def test_failed_snapshot_or_final_logits_never_claims_completion(monkeypatch, failure):
    ns, obj, req, events, logs, clocks = owner(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError):
        ns["_start_request"](obj, req)
    assert "prompt_logits_completed" not in [row[2] for row in logs]
    if failure == "snapshot":
        assert "prompt_snapshot_completed" not in [row[2] for row in logs]
    assert "sample" not in events


def test_logger_failure_does_not_change_success(monkeypatch):
    ns, obj, req, events, logs, clocks = owner(monkeypatch)

    def broken(*args):
        raise RuntimeError("logger unavailable")

    ns["logger"].info = broken
    assert ns["_start_request"](obj, req) == "response"
    assert events[-1] == "sample"
