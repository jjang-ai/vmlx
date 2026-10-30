"""Progress publication must not count a request or a decode cycle twice."""
def test_repeated_ar_progress_only_adds_new_work(monkeypatch):
    from vmlx_engine.text_mtp import lane_stats
    totals = {key: 0 for key in lane_stats.TOTALS}
    monkeypatch.setattr(lane_stats, "TOTALS", totals)
    monkeypatch.setattr(lane_stats, "_last", None)
    stats = lane_stats.LaneStats(cycles=12, draft_tokens_proposed=24,
                           draft_tokens_accepted=16)
    lane_stats.publish("one-request", stats, "fallback_to_ar")
    for _ in range(4):
        lane_stats.publish("one-request", stats, "ar")
    assert totals["requests"] == 1
    assert totals["cycles"] == 12
    assert totals["drafted_tokens"] == 24
    assert totals["accepted_tokens"] == 16
    stats.cycles += 2
    stats.draft_tokens_proposed += 4
    stats.draft_tokens_accepted += 3
    lane_stats.publish("one-request", stats, "stop")
    assert totals["requests"] == 1
    assert totals["cycles"] == 14
    assert totals["drafted_tokens"] == 28
    assert totals["accepted_tokens"] == 19


def test_reentry_phase_adds_work_not_another_request(monkeypatch):
    from vmlx_engine.text_mtp import lane_stats
    totals = {key: 0 for key in lane_stats.TOTALS}
    monkeypatch.setattr(lane_stats, "TOTALS", totals)
    monkeypatch.setattr(lane_stats, "_last", None)
    first = lane_stats.LaneStats(cycles=20, draft_tokens_proposed=40,
                           draft_tokens_accepted=30)
    lane_stats.publish("a", first, "fallback_to_ar")
    retry = lane_stats.LaneStats(cycles=12, draft_tokens_proposed=12,
                           draft_tokens_accepted=10, request_counted=True)
    for reason in ("fallback_to_ar", "ar", "ar", "length"):
        lane_stats.publish("a", retry, reason)
    assert totals["requests"] == 1
    assert totals["cycles"] == 32
    assert totals["drafted_tokens"] == 52
    assert totals["accepted_tokens"] == 40
    lane_stats.publish("b", lane_stats.LaneStats(cycles=2), "stop")
    assert totals["requests"] == 2 and totals["cycles"] == 34
