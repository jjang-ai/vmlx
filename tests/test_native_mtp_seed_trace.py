import json
import logging

import pytest

from vmlx_engine import native_mtp_seed_trace as trace


@pytest.mark.parametrize("value", [None, "0", "false", "invalid"])
def test_disabled_trace_does_not_read_clock(monkeypatch, value):
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_SEED_TRACE", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_SEED_TRACE", raising=False)
    if value is not None:
        monkeypatch.setenv("VMLX_NATIVE_MTP_SEED_TRACE", value)

    def forbidden_clock():
        raise AssertionError("disabled trace read the clock")

    monkeypatch.setattr(trace.time, "perf_counter", forbidden_clock)
    assert trace.start_native_mtp_seed_trace() is None


@pytest.mark.parametrize("key", ["VMLX_NATIVE_MTP_SEED_TRACE", "VMLINUX_NATIVE_MTP_SEED_TRACE"])
def test_trace_preserves_stage_sum_and_labels_async_work(monkeypatch, caplog, key):
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_SEED_TRACE", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_SEED_TRACE", raising=False)
    monkeypatch.setenv(key, "true")
    clock = iter([1.0, 1.010, 1.012, 1.030])
    monkeypatch.setattr(trace.time, "perf_counter", lambda: next(clock))
    probe = trace.start_native_mtp_seed_trace()
    probe.mark("pending_token_read")
    probe.mark("drain")
    probe.mark("target_forward_sample")
    with caplog.at_level(logging.INFO):
        probe.emit(logging.getLogger(__name__), request_id="trace-owner")
    record = json.loads(caplog.records[-1].args[0])
    assert record["stages_ms"] == pytest.approx({
        "pending_token_read": 10.0, "drain": 2.0, "target_forward_sample": 18.0,
    })
    assert record["total_ms"] == pytest.approx(sum(record["stages_ms"].values()))
    assert record["clock"] == "host_wall_no_added_sync"
    assert record["async_work_may_be_charged_to_later_waits"] is True
    assert record["request_id"] == "trace-owner"
