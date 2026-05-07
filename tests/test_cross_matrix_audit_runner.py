"""Regression tests for the live production-family audit harness.

These tests keep the harness from producing false-positive PASS rows when a
model mentions the expected token inside reasoning but never emits visible
content.
"""

from tests.cross_matrix.run_production_family_audit import (
    ROWS,
    cache_exact_hit_required,
    extract_anthropic_text_and_stop,
    extract_ollama_visible_text_and_stop,
    is_non_length_stop,
    normalize_short_answer,
    simple_loop_score,
    static_audit,
)


def test_anthropic_exact_probe_ignores_reasoning_blocks():
    resp = {
        "content": [
            {"type": "thinking", "thinking": "I should answer anthropic-ok"},
            {"type": "text", "text": ""},
        ],
        "stop_reason": "max_tokens",
    }

    text, stop = extract_anthropic_text_and_stop(resp)

    assert normalize_short_answer(text).lower() != "anthropic-ok"
    assert not is_non_length_stop(stop)


def test_anthropic_exact_probe_reads_visible_text_only():
    resp = {
        "content": [
            {"type": "thinking", "thinking": "ignore me"},
            {"type": "text", "text": "\nanthropic-ok\n"},
        ],
        "stop_reason": "end_turn",
    }

    text, stop = extract_anthropic_text_and_stop(resp)

    assert normalize_short_answer(text).lower() == "anthropic-ok"
    assert is_non_length_stop(stop)


def test_ollama_exact_probe_ignores_thinking_field():
    resp = {
        "message": {
            "content": "",
            "thinking": "The answer should be ollama-ok",
        },
        "done_reason": "length",
    }

    text, stop = extract_ollama_visible_text_and_stop(resp)

    assert normalize_short_answer(text).lower() != "ollama-ok"
    assert not is_non_length_stop(stop)


def test_ollama_exact_probe_reads_visible_content():
    resp = {
        "message": {
            "content": "ollama-ok",
            "thinking": "reasoning",
        },
        "done_reason": "stop",
    }

    text, stop = extract_ollama_visible_text_and_stop(resp)

    assert normalize_short_answer(text).lower() == "ollama-ok"
    assert is_non_length_stop(stop)


def test_short_answer_normalizer_accepts_terminal_punctuation_only():
    assert normalize_short_answer("Paris.").lower() == "paris"
    assert normalize_short_answer('"Paris!"').lower() == "paris"
    assert normalize_short_answer("Paris. Explanation").lower() != "paris"


def test_zaya_rows_are_present_and_marked_cca():
    rows = {row.id: row for row in ROWS}

    assert rows["zaya_jangtq2"].cache_profile == "zaya_cca"
    assert rows["zaya_jangtq4"].cache_profile == "zaya_cca"
    assert rows["zaya_mxfp4"].cache_profile == "zaya_cca"


def test_zaya_static_audit_exposes_cache_subtype_when_local_bundle_exists():
    row = next(row for row in ROWS if row.id == "zaya_jangtq2")
    static = static_audit(row)

    if not static["exists"]:
        return
    assert static["family_expected"] == "zaya"
    assert static["jang"]["cache_subtype"] == "zaya_cca"
    assert static["registry"]["cache_subtype"] == "zaya_cca"


def test_zaya_cca_rows_do_not_run_generic_exact_hit_cache_probe():
    rows = {row.id: row for row in ROWS}

    assert not cache_exact_hit_required(rows["zaya_jangtq2"])
    assert not cache_exact_hit_required(rows["zaya_jangtq4"])
    assert not cache_exact_hit_required(rows["zaya_mxfp4"])
    assert cache_exact_hit_required(rows["dsv4_tq"])


def test_loop_score_catches_no_space_cjk_and_emoji_repetition():
    assert simple_loop_score("音苷苷和音诺族的对策" * 80) >= 0.25
    assert simple_loop_score("👀" * 200) >= 0.25
    assert simple_loop_score("state " * 80) >= 0.25
    assert simple_loop_score("Paris is the capital of France.") < 0.25
