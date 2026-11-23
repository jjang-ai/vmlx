# SPDX-License-Identifier: Apache-2.0
"""Inkling reasoning parser — CHANNEL-SWITCHED, not tag-delimited.

Every other reasoning family in this engine wraps thought in a matched pair
(``<think>…</think>``, ``<mm:think>…</mm:think>``). Inkling does not, and
building it on the think-tag base would be wrong in a way that only shows up on
a real bundle. Its contract, read from
``Inkling-Small-JANG-AWQ-90GB/jang_config.json`` (``chat.reasoning``):

    thinking_channel_token      <|content_thinking|>
    visible_channel_token       <|content_text|>
    message_end_token           <|end_message|>
    generation_prompt_ends_with <|message_model|>
    message_field               reasoning_content

So the markers are OPENERS that switch which channel subsequent text belongs
to; there is no closer for thinking. Text runs into ``reasoning_content`` from
``<|content_thinking|>`` until either ``<|content_text|>`` switches it to
visible content or ``<|end_message|>`` ends the message. Mini may then
open a new message with its visible answer before the sampling stop token.

Two consequences worth stating, because both are easy to get wrong:

* A response may open directly on ``<|content_text|>`` with no thinking at all
  (effort 0.0 / "none"), and it may also emit thinking and then end without ever
  switching to visible — that is a reasoning-only turn, which is exactly what
  the never-empty answer pass exists to catch.
* Text BEFORE any channel marker is visible content, not reasoning. Treating a
  bare prefix as thought would swallow the answer.

Reasoning depth is a SCALAR here (``mechanism: effort_scalar``,
``control_argument: reasoning_effort``, map none 0.0 … max 0.99, default 0.9),
not a boolean — ``boolean_enable_thinking_supported`` is explicitly false. That
mapping is a separate concern from parsing and is deliberately not done here.

Created by Jinho Jang (eric@jangq.ai).
"""

from ..tool_parsers.inkling_tool_parser import TOOL_CHANNEL, read_tool_message
from .base import DeltaMessage, ReasoningParser, marker_prefix_hold_length

THINKING_CHANNEL = "<|content_thinking|>"
VISIBLE_CHANNEL = "<|content_text|>"
MESSAGE_END = "<|end_message|>"
SAMPLING_END = "<|content_model_end_sampling|>"

# A turn may contain multiple channel messages. Only SAMPLING_END closes
# generation. Hold incomplete markers across streaming chunks.
_MESSAGE_MODEL = "<|message_model|>"
_CHANNEL_MARKERS = (
    THINKING_CHANNEL, VISIBLE_CHANNEL, MESSAGE_END, SAMPLING_END,
    TOOL_CHANNEL, _MESSAGE_MODEL,
)
_MARKERS = _CHANNEL_MARKERS


def _strip_markers(text: str) -> str:
    for marker in _MARKERS:
        text = text.replace(marker, "")
    return text


def _split_channels(text: str) -> tuple[str, str]:
    """Return ``(reasoning, visible)`` for a complete Inkling message."""
    reasoning_parts: list[str] = []
    visible_parts: list[str] = []
    # Anything before the first marker is visible content, not thought.
    channel = visible_parts
    cursor = 0
    while cursor < len(text):
        next_at = len(text)
        next_marker = None
        for marker in _CHANNEL_MARKERS:
            at = text.find(marker, cursor)
            if at != -1 and at < next_at:
                next_at, next_marker = at, marker
        if channel is not None:
            channel.append(text[cursor:next_at])
        if next_marker is None or next_marker == SAMPLING_END:
            break
        cursor = next_at + len(next_marker)
        if next_marker == TOOL_CHANNEL:
            visible_parts.append(TOOL_CHANNEL)
            message = read_tool_message(text, cursor)
            if message is None:
                break
            _, end = message
            visible_parts.append(text[cursor:end])
            cursor = end
            channel = None
        elif next_marker in (MESSAGE_END, _MESSAGE_MODEL):
            channel = None
        elif next_marker == THINKING_CHANNEL:
            channel = reasoning_parts
        else:
            channel = visible_parts
    return "".join(reasoning_parts), "".join(visible_parts)


class InklingReasoningParser(ReasoningParser):
    """Channel-switched reasoning extraction for the Inkling family."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_reasoning = 0
        self._emitted_visible = 0

    @property
    def start_token(self) -> str:
        return THINKING_CHANNEL

    @property
    def end_token(self) -> str:
        # There is no closing token for the thinking channel; the visible
        # channel opener is what ends it. Reported for surfaces that ask.
        return VISIBLE_CHANNEL

    def reset_state(self, **kwargs) -> None:
        """Reset per-request counters.

        Must accept **kwargs: the server passes ``think_in_prompt`` and
        ``harmony_active`` to every reasoning parser. Neither applies to a
        channel-switched family — the channel is chosen by the model mid-stream,
        not opened by the prompt — but refusing the keywords turns every chat
        request into a 500.
        """
        self._emitted_reasoning = 0
        self._emitted_visible = 0

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        if not any(marker in model_output for marker in (
            THINKING_CHANNEL, VISIBLE_CHANNEL, TOOL_CHANNEL
        )):
            # No channel markers at all: the whole thing is visible content.
            cleaned = _strip_markers(model_output)
            return None, cleaned or None
        reasoning, visible = _split_channels(model_output)
        return (reasoning.strip() or None), (visible.strip() or None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Emit only what is newly RESOLVED, routed to the open channel.

        Two things have to be true at once, and the obvious implementation gets
        the second one wrong:

        1. The channel is recomputed from the whole accumulated text, not from
           ``delta_text`` — a marker that straddles a chunk boundary would
           otherwise be routed as if it were prose.
        2. Only the STABLE prefix is interpreted. A marker arrives a character
           at a time, so ``<|content_thinking|`` is briefly indistinguishable
           from text; publishing it is unrecoverable because the emitted
           counters below only move forward. Char-by-char streaming leaked
           exactly that marker into visible content before this held it back.

        The counters are per-request state — the server builds a fresh parser
        with ``_reasoning_parser.__class__()`` for every request — so they do
        not carry across turns.
        """
        stable = current_text[: len(current_text) - marker_prefix_hold_length(current_text, _MARKERS)]
        reasoning, visible = _split_channels(stable)

        new_reasoning = reasoning[self._emitted_reasoning:]
        new_visible = visible[self._emitted_visible:]
        self._emitted_reasoning = len(reasoning)
        self._emitted_visible = len(visible)

        if not new_reasoning and not new_visible:
            return None
        # The field is `reasoning`; `reasoning_content` is a read-only
        # backward-compat property on DeltaMessage, so passing it as a kwarg
        # raises TypeError on EVERY delta.
        return DeltaMessage(
            reasoning=new_reasoning or None,
            content=new_visible or None,
        )
