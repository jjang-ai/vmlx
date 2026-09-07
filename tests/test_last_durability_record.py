"""Cache/Perf display audit (2026-09-07): the Cache panel showed the latest cache
execution but no per-generation save/fence, and the MTP cards showed no
request scope. The engine now keeps the last generation's terminal durability
fence as data (``last_durability``) and the MTP record carries its policy and
configured depth."""

from types import SimpleNamespace

from vmlx_engine.mllm_scheduler import _record_last_durability


def test_last_durability_record_carries_request_wait_and_ledger_outcome():
    stats = SimpleNamespace()
    _record_last_durability(stats, "chatcmpl-abc", 283.5314, True, {"outcome": "stored", "detail": "paged 48 layers, blocks=17", "retained_tokens": 1088, "durable": True})
    rec = stats.last_durability
    assert rec["request_id"] == "chatcmpl-abc" and rec["wait_ms"] == 283.531 and rec["waited"] is True
    assert rec["cache_outcome"] == "stored" and rec["retained_tokens"] == 1088 and rec["durable"] is True
    assert rec["detail"].startswith("paged 48 layers") and rec["at"] > 0


def test_last_durability_unknown_outcome_is_named_not_zeroed():
    stats = SimpleNamespace()
    _record_last_durability(stats, "resp_1", 0.2, False, {"outcome": "unknown", "detail": "cleanup completed; no store outcome recorded", "retained_tokens": None})
    rec = stats.last_durability
    assert rec["cache_outcome"] == "unknown" and rec["retained_tokens"] is None and rec["waited"] is False
    # no stats object: nothing recorded, nothing raised
    _record_last_durability(None, "x", 1.0, True, {})


def test_batch_stats_publish_last_durability_and_mtp_scope_fields():
    from vmlx_engine.mllm_batch_generator import MLLMBatchStats, MLLMNativeMTPStats

    d = MLLMBatchStats().to_dict()
    assert "last_durability" in d and d["last_durability"] is None
    m = MLLMNativeMTPStats()
    m.depth_policy = "fixed"; m.configured_depth = 3
    md = m.to_dict(request_id="chatcmpl-1", finish_reason="stop", final_depth=1)
    assert md["policy"] == "fixed" and md["configured_depth"] == 3 and md["request_id"] == "chatcmpl-1" and md["final_depth"] == 1
    md2 = MLLMNativeMTPStats().to_dict(request_id="r", finish_reason="stop", final_depth=2)
    assert md2["policy"] is None and md2["configured_depth"] is None  # unknown stays unknown, never 0
    from vmlx_engine import server
    src = open(server.__file__).read()
    assert '        "last_durability",\n        "last_native_mtp",' in src
