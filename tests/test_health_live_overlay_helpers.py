# SPDX-License-Identifier: Apache-2.0
"""Pins for the cached-health live-overlay helpers (campaign #181 rows 88/90).

The /health snapshot cache serves a pre-request payload while any request is
running or lingering; these helpers supply the LIVE records the cached branch
overlays. Attribute-read-only discipline — nothing here may call get_stats."""

from types import SimpleNamespace

from vmlx_engine.server import (
    _live_batch_generator_request_records,
    _live_last_cache_execution,
    _live_ssm_prefix_lookup,
)


def test_live_batch_generator_records_are_copied_without_get_stats():
    records = {
        "last_cache_execution": {"request_id": "r1"},
        "last_native_mtp": {"request_id": "r1", "accepted_tokens": 3},
        "last_native_mtp_skip": None,
        # the terminal durability fence record (request-exact Cache panel, 50331002)
        "last_durability": {"request_id": "r1", "wait_ms": 12.5, "cache_outcome": "stored"},
    }
    stats = SimpleNamespace(**records)
    scheduler = SimpleNamespace(
        batch_generator=SimpleNamespace(_stats=stats)
    )
    observed = _live_batch_generator_request_records(scheduler)
    assert observed == records
    assert observed["last_native_mtp"] is not stats.last_native_mtp
    assert observed["last_durability"] is not stats.last_durability
    assert _live_batch_generator_request_records(None) is None


def test_live_lce_prefers_generator_record_then_admission():
    gen_stats = SimpleNamespace(
        last_cache_execution={"request_id": "gen", "cache_outcome": "hit"}
    )
    scheduler = SimpleNamespace(
        batch_generator=SimpleNamespace(_stats=gen_stats),
        _admission_cache_execution={"request_id": "adm"},
    )
    assert _live_last_cache_execution(scheduler)["request_id"] == "gen"
    gen_stats.last_cache_execution = None
    assert _live_last_cache_execution(scheduler)["request_id"] == "adm"


def test_live_lce_falls_back_to_text_scheduler_record_and_none():
    scheduler = SimpleNamespace(
        batch_generator=None,
        _last_cache_execution={"request_id": "txt"},
    )
    assert _live_last_cache_execution(scheduler)["request_id"] == "txt"
    assert _live_last_cache_execution(None) is None
    assert (
        _live_last_cache_execution(SimpleNamespace(batch_generator=None))
        is None
    )


def test_live_ssm_lookup_resolves_scheduler_then_generator_cache():
    lookup = {"request_id": "r1", "matched": True}
    scheduler = SimpleNamespace(
        _ssm_state_cache=SimpleNamespace(last_prefix_lookup=lookup)
    )
    assert _live_ssm_prefix_lookup(scheduler) == lookup
    scheduler2 = SimpleNamespace(
        _ssm_state_cache=None,
        batch_generator=SimpleNamespace(
            _ssm_state_cache=SimpleNamespace(last_prefix_lookup=lookup)
        ),
    )
    assert _live_ssm_prefix_lookup(scheduler2) == lookup
    assert _live_ssm_prefix_lookup(None) is None


def test_companion_overlay_is_not_gated_on_prior_key_presence():
    """Row 94: a cached snapshot taken BEFORE the request legitimately has no
    last_prefix_lookup key; gating the overlay on prior presence suppressed
    the bound live lookup for exactly the rows that need it (r4 refault: lce
    bound with a bound embedded lookup, companion field null)."""
    from pathlib import Path
    import re

    src = (
        Path(__file__).resolve().parents[1] / "vmlx_engine" / "server.py"
    ).read_text(encoding="utf-8")
    anchor = src.index("Same staleness applies to the SSM companion")
    window = src[anchor : anchor + 2400]
    assert re.search(
        r"if isinstance\(_companion_block, dict\):", window
    ), "companion overlay guard changed shape"
    assert '"last_prefix_lookup" in' not in window.split(
        "_live_lookup = None"
    )[0], (
        "the companion overlay is again gated on the cached snapshot already "
        "having a lookup key — pre-request snapshots lack it and the bound "
        "live lookup gets suppressed"
    )
