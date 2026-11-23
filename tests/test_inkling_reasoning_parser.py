# SPDX-License-Identifier: Apache-2.0
"""Inkling splits channels, it does not close tags.

Contract read from the shipping bundle
(`Inkling-Small-JANG-AWQ-90GB/jang_config.json`, `chat.reasoning`):

    thinking_channel_token   <|content_thinking|>
    visible_channel_token    <|content_text|>
    message_end_token        <|end_message|>
    mechanism                effort_scalar   (NOT boolean enable_thinking)

`<|content_thinking|>` has no closer — `<|content_text|>` is what ends it. Every
other reasoning family here uses a matched pair, so registering Inkling on a
think-tag parser would mis-split every response, and the failure would only
appear on a real bundle.
"""

from __future__ import annotations

import pytest

from vmlx_engine.reasoning import get_parser, list_parsers


@pytest.fixture
def parser():
    return get_parser("inkling")()


def test_registered_under_its_own_name():
    assert "inkling" in list_parsers()


@pytest.mark.parametrize(
    "output, reasoning, content",
    [
        (
            "<|content_thinking|>weigh it up<|content_text|>The answer.<|end_message|>",
            "weigh it up",
            "The answer.",
        ),
        # effort "none" (0.0): the model may open straight on the visible channel
        ("<|content_text|>Direct answer.<|end_message|>", None, "Direct answer."),
        # Thinking that never switches to visible — a reasoning-only turn, which
        # is what the never-empty answer pass exists to catch.
        ("<|content_thinking|>ran out of budget<|end_message|>", "ran out of budget", None),
        # No markers at all: it is ALL visible. Treating a bare prefix as
        # thought would swallow the answer.
        ("plain answer with no channels", None, "plain answer with no channels"),
        # Text before the first marker is visible, not thought.
        (
            "preamble<|content_thinking|>thought<|content_text|>rest<|end_message|>",
            "thought",
            "preamblerest",
        ),
    ],
)
def test_channel_split(parser, output, reasoning, content):
    got_reasoning, got_content = parser.extract_reasoning(output)
    assert got_reasoning == reasoning
    assert got_content == content


def test_no_marker_ever_reaches_rendered_text(parser):
    output = (
        "<|content_thinking|>a<|content_text|>b<|end_message|>"
    )
    reasoning, content = parser.extract_reasoning(output)
    for marker in (
        "<|content_thinking|>",
        "<|content_text|>",
        "<|end_message|>",
        "<|message_model|>",
    ):
        assert marker not in (reasoning or "")
        assert marker not in (content or "")


def test_streaming_routes_each_delta_to_the_open_channel(parser):
    """Feed it token-by-token the way the server does."""
    chunks = [
        "<|content_thinking|>", "let me ", "check", "<|content_text|>",
        "Tokyo", ".", "<|end_message|>",
    ]
    reasoning, content = "", ""
    seen = ""
    for chunk in chunks:
        previous, seen = seen, seen + chunk
        delta = parser.extract_reasoning_streaming(previous, seen, chunk)
        if delta is None:
            continue
        reasoning += delta.reasoning_content or ""
        content += delta.content or ""
    assert reasoning == "let me check"
    assert content == "Tokyo."


def test_streaming_survives_a_marker_split_across_deltas(parser):
    """A marker arriving in pieces must not be routed as text.

    The channel is recomputed from the full text each delta precisely so a
    fragment like "<|content_" cannot be mistaken for content.
    """
    chunks = ["<|content_thinking|>", "idea", "<|content_", "text|>", "Answer"]
    reasoning, content = "", ""
    seen = ""
    for chunk in chunks:
        previous, seen = seen, seen + chunk
        delta = parser.extract_reasoning_streaming(previous, seen, chunk)
        if delta is None:
            continue
        reasoning += delta.reasoning_content or ""
        content += delta.content or ""
    assert content == "Answer", f"content was {content!r}"
    assert "Answer" not in reasoning


def _stream(parser, text, chunk_size=1):
    """Feed `text` through the streaming API in fixed-size chunks."""
    reasoning, content, seen = "", "", ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        previous, seen = seen, seen + chunk
        delta = parser.extract_reasoning_streaming(previous, seen, chunk)
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
    return reasoning, content


def test_char_by_char_never_publishes_a_half_arrived_marker(parser):
    """The strictest case: one character per delta.

    Each marker spends several deltas looking like ordinary text. Because the
    emitted counters only move forward, anything published while a marker is
    half-arrived can never be retracted — it just appears in the user's chat.
    """
    reasoning, content = _stream(
        parser, "<|content_thinking|>weigh<|content_text|>Answer<|end_message|>"
    )
    assert reasoning == "weigh"
    assert content == "Answer"
    for marker in ("<|content", "<|end", "|>", "thinking|", "text|"):
        assert marker not in reasoning
        assert marker not in content


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 64])
def test_streaming_agrees_with_oneshot_at_every_chunk_size(parser, chunk_size):
    text = "<|content_thinking|>step one\nstep two<|content_text|>Final\n\nanswer.<|end_message|>"
    streamed_reasoning, streamed_content = _stream(parser, text, chunk_size)
    oneshot_reasoning, oneshot_content = parser.extract_reasoning(text)
    assert streamed_reasoning.strip() == (oneshot_reasoning or "").strip()
    assert streamed_content.strip() == (oneshot_content or "").strip()


def test_reset_state_clears_counters_between_requests(parser):
    """The server may hand the same instance a second request."""
    _stream(parser, "<|content_thinking|>first<|content_text|>one<|end_message|>")
    parser.reset_state(think_in_prompt=False, harmony_active=False)
    reasoning, content = _stream(
        parser, "<|content_thinking|>second<|content_text|>two<|end_message|>"
    )
    assert reasoning == "second"
    assert content == "two"


def test_inkling_mini_visible_message_after_reasoning_message(parser):
    text = (
        '<|message_model|><|content_thinking|>Final should be "19".'
        '<|end_message|><|message_model|><|content_text|>19<|end_message|>'
        '<|content_model_end_sampling|><|content_text|>ignored'
    )
    assert parser.extract_reasoning(text) == ('Final should be "19".', "19")
    assert _stream(parser, text) == ('Final should be "19".', "19")
