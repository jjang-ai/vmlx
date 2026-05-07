"""Regression tests for the live production-family audit harness.

These tests keep the harness from producing false-positive PASS rows when a
model mentions the expected token inside reasoning but never emits visible
content.
"""

from tests.cross_matrix.run_production_family_audit import (
    extract_anthropic_text_and_stop,
    extract_ollama_visible_text_and_stop,
    is_non_length_stop,
    normalize_short_answer,
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
