"""Terminal persistence outcome: the durability barrier reports what the store sites recorded, never 'persisted' by default."""
from pathlib import Path

from vmlx_engine.persistence_outcome import TerminalPersistenceLedger, format_outcome


def test_unknown_by_default_and_take_clears():
    l = TerminalPersistenceLedger()
    e = l.take("r1")
    assert e["outcome"] == "unknown" and "no store outcome recorded" in e["detail"]
    l.record("r2", "stored", "paged", retained_tokens=407)
    assert l.peek("r2")["retained_tokens"] == 407
    assert l.take("r2")["outcome"] == "stored"
    assert l.take("r2")["outcome"] == "unknown"


def test_precedence_stored_refused_failed_never_downgrade_to_skipped():
    l = TerminalPersistenceLedger()
    l.record("r", "stored", "paged", retained_tokens=10)
    l.record("r", "skipped", "legacy tier had nothing")
    assert l.take("r")["outcome"] == "stored"
    l.record("f", "failed", "paged: boom")
    l.record("f", "stored", "legacy")
    assert l.take("f")["outcome"] == "failed"
    l.record("s", "skipped", "empty")
    l.record("s", "already_durable", "covers 5/5", retained_tokens=5)
    assert l.take("s")["outcome"] == "already_durable"
    l.record("x", "not-an-outcome", "ignored")
    assert l.take("x")["outcome"] == "unknown"


def test_format_outcome_carries_tokens_and_durability():
    txt = format_outcome({"outcome": "stored", "detail": "legacy prefix cache (L1 only)", "retained_tokens": 12, "durable": False})
    assert txt.startswith("cache_outcome=stored") and "retained_tokens=12" in txt and "durable=false" in txt
    assert format_outcome({"outcome": "unknown", "detail": "", "retained_tokens": None, "durable": None}) == "cache_outcome=unknown"


def test_no_lane_claims_persistence_without_a_recorded_outcome():
    root = Path(__file__).resolve().parents[1] / "vmlx_engine"
    for name in ("engine_core.py", "mllm_scheduler.py"):
        src = (root / name).read_text()
        assert "cache_persisted=true" not in src, name
        assert "Terminal durability barrier" in src and "_format_persistence_outcome" in src, name
    for name in ("scheduler.py", "mllm_scheduler.py"):
        src = (root / name).read_text()
        for outcome in ("\"stored\"", "\"skipped\"", "\"failed\""):
            assert f"_PERSIST.record(request_id, {outcome}" in src, (name, outcome)


def test_media_policy_skip_records_its_own_reason_not_the_generic_empty_cache():
    """A media-bearing prompt on a family outside the media prefix-cache
    allow-list nulls the extracted handle on purpose. That skip must be
    recorded as the family policy it is; before, the nulled handle fell
    through to the generic "resolved extracted cache is empty" outcome, which
    read like a lost cache in the terminal ledger."""
    import inspect
    import vmlx_engine.mllm_scheduler as m
    src = inspect.getsource(m)
    policy = src.index('"media context: prefix reuse not allowed for this family"')
    generic = src.index('_PERSIST.record(request_id, "skipped", "resolved extracted cache is empty")')
    assert policy < generic
    # the generic outcome is guarded by the flag the policy skip sets
    assert "if cache_blocks is None and _media_skip_recorded:" in src
    assert src.count("_media_skip_recorded = True") == 1
