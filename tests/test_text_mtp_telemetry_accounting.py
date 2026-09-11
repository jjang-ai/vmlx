"""Progress publication must not count a request or a decode cycle twice."""
def test_repeated_ar_progress_only_adds_new_work(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    totals = {key: 0 for key in lane._NATIVE_MTP_TOTALS}
    monkeypatch.setattr(lane, "_NATIVE_MTP_TOTALS", totals)
    monkeypatch.setattr(lane, "_LAST_NATIVE_MTP", None)
    stats = lane._MtpStats(cycles=12, draft_tokens_proposed=24,
                           draft_tokens_accepted=16)
    lane._publish_native_mtp_stats("one-request", stats, "fallback_to_ar")
    for _ in range(4):
        lane._publish_native_mtp_stats("one-request", stats, "ar")
    assert totals["requests"] == 1
    assert totals["cycles"] == 12
    assert totals["drafted_tokens"] == 24
    assert totals["accepted_tokens"] == 16
    stats.cycles += 2
    stats.draft_tokens_proposed += 4
    stats.draft_tokens_accepted += 3
    lane._publish_native_mtp_stats("one-request", stats, "stop")
    assert totals["requests"] == 1
    assert totals["cycles"] == 14
    assert totals["drafted_tokens"] == 28
    assert totals["accepted_tokens"] == 19


def test_reentry_phase_adds_work_not_another_request(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    totals = {key: 0 for key in lane._NATIVE_MTP_TOTALS}
    monkeypatch.setattr(lane, "_NATIVE_MTP_TOTALS", totals)
    monkeypatch.setattr(lane, "_LAST_NATIVE_MTP", None)
    first = lane._MtpStats(cycles=20, draft_tokens_proposed=40,
                           draft_tokens_accepted=30)
    lane._publish_native_mtp_stats("a", first, "fallback_to_ar")
    retry = lane._MtpStats(cycles=12, draft_tokens_proposed=12,
                           draft_tokens_accepted=10, request_counted=True)
    for reason in ("fallback_to_ar", "ar", "ar", "length"):
        lane._publish_native_mtp_stats("a", retry, reason)
    assert totals["requests"] == 1
    assert totals["cycles"] == 32
    assert totals["drafted_tokens"] == 52
    assert totals["accepted_tokens"] == 40
    lane._publish_native_mtp_stats("b", lane._MtpStats(cycles=2), "stop")
    assert totals["requests"] == 2 and totals["cycles"] == 34
