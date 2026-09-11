# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Gemma 4 models.

Gemma 4 uses channel markers for reasoning/content separation:
  <|channel>thought
  ...reasoning content...
  <channel|>...final content...<turn|>

When enable_thinking=True, the model generates a thought channel before content.
When enable_thinking=False, the template injects an empty thought block
(<|channel>thought\n<channel|>) so no reasoning is produced.

The parser handles both streaming and complete extraction, including partial
marker buffering at chunk boundaries.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Channel marker tokens (from Gemma 4 tokenizer_config.json)
_SOC = "<|channel>"       # soc_token: start-of-channel
_EOC = "<channel|>"       # eoc_token: end-of-channel
_EOT = "<turn|>"          # eot_token: end-of-turn (EOS)
_THOUGHT = "thought"      # Channel name for reasoning

# Full start marker: <|channel>thought\n (newline is part of format)
_THOUGHT_START = f"{_SOC}{_THOUGHT}\n"
# End of thinking = start of content
_THOUGHT_END = _EOC

# Gemma 4 normally opens the thought channel at the start of a generation.
# During a tool-result continuation, however, the model can finish a visible
# answer and then open another channel.  The detokenizer may also consume the
# ``<|channel>`` token, leaving only a line-delimited ``thought\n`` sentinel.
# Keep this family-scoped: ordinary models can legitimately use the word
# "thought" in prose, while Gemma's native channel grammar reserves this
# standalone line.
_DEGRADED_THOUGHT_START_RE = re.compile(r"(?:^|\n)thought\n")

_PLAIN_THINKING_RE = re.compile(
    r"^\s*(?:\*\*)?Thinking Process:?(?:\*\*)?\s*",
    re.IGNORECASE,
)
_PLAIN_THINKING_PREFIXES = (
    "Thinking Process:",
    "**Thinking Process:**",
    "Thinking Process",
    "**Thinking Process**",
)
_PLAIN_FINAL_RE = re.compile(
    r"\n\s*(?:\*\*)?(?:Final Answer|Final Response|Answer|Response):(?:\*\*)?\s*",
    re.IGNORECASE,
)
_INLINE_SELF_CORRECTION_TAIL_RE = re.compile(
    r"(?P<reasoning>[\s\S]*?\*\([^)]*(?:Self-Correction|Self Correction|Refinement|Correction)[^)]*\)\*)"
    r"(?P<content>[A-Z][A-Z0-9_:-][^\n]*)\s*$",
    re.IGNORECASE,
)



def _trim_trailing_eot(text: str) -> str:
    """Drop a trailing ``<turn|>`` — including one that is still ARRIVING.

    Streaming hands the marker over one character at a time, so a whole-string
    ``replace`` or an ``endswith`` check never matches mid-flight and the EOS
    token renders in the answer as ``<turn|``. Holding the partial prefix is
    safe: the counters that drive emission only ever grow, so if the run turns
    out to be prose after all it is emitted on the delta that disambiguates it.
    """
    while text.endswith(_EOT):
        text = text[: -len(_EOT)]
    for size in range(len(_EOT) - 1, 0, -1):
        if text.endswith(_EOT[:size]):
            return text[:-size]
    return text


class Gemma4ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Gemma 4 models.

    Extracts reasoning from <|channel>thought blocks and content from
    after <channel|>.

    Example (thinking ON):
        Input: "<|channel>thought\nLet me think...\n<channel|>The answer is 42.<turn|>"
        Output: reasoning="Let me think...", content="The answer is 42."

    Example (thinking OFF — empty thought block in prompt):
        Input: "The answer is 42.<turn|>"
        Output: reasoning=None, content="The answer is 42."
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_reasoning: int = 0
        self._emitted_content: int = 0
        self._in_thought: bool = False
        self._saw_thought: bool = False
        self._saw_eoc: bool = False

    def reset_state(self, **kwargs):
        """Reset state for a new streaming request."""
        self._emitted_reasoning = 0
        self._emitted_content = 0
        self._in_thought = False
        self._saw_thought = False
        self._saw_eoc = False

    @staticmethod
    def _find_thought_start(text: str) -> tuple[int, int] | None:
        """Return ``(index, marker_length)`` for a native thought channel.

        Unlike the original start-only check, this intentionally recognizes a
        channel after visible content.  That shape is produced by real Gemma 4
        tool-result continuations and must not be streamed into ``content``.
        """
        full_idx = text.find(_SOC + _THOUGHT)
        degraded = _DEGRADED_THOUGHT_START_RE.search(text)
        degraded_idx = -1
        degraded_len = 0
        if degraded is not None:
            degraded_idx = degraded.start()
            # Preserve the separating newline as part of the visible prefix;
            # the actual marker begins after it.
            if text[degraded_idx:degraded.end()].startswith("\n"):
                degraded_idx += 1
            degraded_len = len(_THOUGHT) + 1

        if full_idx >= 0 and (degraded_idx < 0 or full_idx <= degraded_idx):
            marker_len = len(_SOC + _THOUGHT)
            if text[full_idx + marker_len:].startswith("\n"):
                marker_len += 1
            return full_idx, marker_len
        if degraded_idx >= 0:
            return degraded_idx, degraded_len
        return None

    @staticmethod
    def _join_visible_content(prefix: str, suffix: str) -> str | None:
        """Join visible rails around a thought channel without duplication."""
        prefix = prefix.strip()
        suffix = suffix.strip()
        if prefix and suffix:
            # MONOTONICITY: the caller diffs this against a counter that only
            # grows, so this join must never SHRINK as the suffix arrives.
            #
            # The dedup branches below returned `suffix` alone whenever the
            # suffix happened to begin with the prefix. Mid-stream that is a
            # coincidence, not a duplicate: with prefix "Hi" and the suffix rail
            # opening "Hi\nMore", the join briefly went "Hi\nHi" -> "Hi\nHi\nM"
            # under the old concat but "Hi" -> "Hi\nM" under the dedup, so the
            # counter desynced and one character was duplicated while another
            # was lost: "Hi\nMore" streamed as "Hi\nHore".
            #
            # Only dedup on an EXACT whole-string repeat, which cannot shrink
            # the result. Genuine prefix-repetition is a display concern for the
            # completed text, not something to resolve while bytes are still
            # arriving.
            if suffix == prefix:
                return suffix
            return f"{prefix}\n{suffix}"
        return prefix or suffix or None

    @staticmethod
    def _plain_content_before_possible_thought(text: str) -> str:
        """Return the delta-safe visible prefix before a partial channel marker.

        Streaming can split ``thought\n`` one character at a time.  If the
        parser emits ``thought`` as content before the terminating newline
        arrives, SSE cannot retract it.  Hold only the smallest suffix that can
        still become Gemma's native full marker or its line-delimited degraded
        marker.
        """
        # Trim only a TRAILING <turn|> (including one still arriving), never a
        # whole-buffer replace. The replace shrank the accumulated text under a
        # monotonic emission counter, so a literal <turn|> in the answer body
        # desynced it: "Use <turn|> to end turns." streamed as
        # "Use <turn|d turns." — the signature defect class of applying a
        # whole-string operation to a growing buffer.
        text = _trim_trailing_eot(text)
        safe_end = len(text)

        full_marker = _SOC + _THOUGHT + "\n"
        for length in range(1, len(full_marker)):
            if text.endswith(full_marker[:length]):
                safe_end = min(safe_end, len(text) - length)

        line_start = text.rfind("\n") + 1
        line_tail = text[line_start:]
        degraded_marker = _THOUGHT + "\n"
        if line_tail and degraded_marker.startswith(line_tail):
            safe_end = min(safe_end, line_start)

        return text[:safe_end].strip()

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete Gemma 4 output.

        Handles both forms:
        1. With `<|channel>` special token preserved (raw wire form):
             `<|channel>thought\\n...<channel|>...<turn|>`
        2. With `<|channel>` stripped by the detokenizer (common case
           with JANG / MLX tokenizers that treat it as a special token
           and don't emit it in the decoded string):
             `thought\\n...<channel|>...`
        """
        text = model_output

        # Strip trailing <turn|> tokens (EOS), including a partial one that is
        # still arriving — otherwise it renders in chat as `<turn|`.
        text = _trim_trailing_eot(text)

        # Streaming withholds an unfinished native thought header. Terminal
        # extraction must make the same decision: otherwise a length-limited
        # response flushes the withheld <|channel> into visible content.
        # Only native control prefixes qualify. A bare '<', the ordinary word
        # 'thought', and markers mentioned inside a sentence remain text.
        header = _SOC + _THOUGHT
        for size in range(len(header) - 1, 1, -1):
            if text.endswith(header[:size]):
                text = text[:-size]
                break

        # Detect the channel anywhere in the generation.  Tool continuations
        # can emit a complete visible answer and then start a second thought
        # rail; the prefix remains visible while the late rail stays private.
        thought_start = self._find_thought_start(text)
        if thought_start is not None:
            thought_idx, marker_len = thought_start
            visible_prefix = text[:thought_idx].strip()
            after_soc = text[thought_idx + marker_len:]
            eoc_idx = after_soc.find(_EOC)
            if eoc_idx >= 0:
                reasoning = after_soc[:eoc_idx].strip()
                visible_suffix = after_soc[eoc_idx + len(_EOC):].strip()
                while visible_suffix.endswith(_EOT):
                    visible_suffix = visible_suffix[:-len(_EOT)].strip()
                return (
                    reasoning or None,
                    self._join_visible_content(visible_prefix, visible_suffix),
                )

            # No close marker: generation ended (or is currently streaming)
            # inside the late thought rail.  Preserve any answer emitted before
            # it instead of returning a reasoning-only result.
            return after_soc.strip() or None, visible_prefix or None

        # Defensive orphan-close handling: if a bundle/template placed the
        # empty thought block in the prompt and only the channel close marker
        # leaks into generated text, strip it instead of showing it to users.
        if text.lstrip().startswith(_EOC):
            stripped = text.lstrip()
            content = stripped[len(_EOC):].strip()
            while content.endswith(_EOT):
                content = content[:-len(_EOT)].strip()
            return None, content or None

        plain_reasoning, plain_content = self._extract_plain_thinking_process(text)
        if plain_reasoning is not None:
            return plain_reasoning, plain_content

        # No thought channel — pure content (thinking was OFF)
        text = text.strip()
        return None, text if text else None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming delta."""
        if not delta_text:
            return None

        # Strip <turn|> from delta (EOS token, not content)
        clean_delta = delta_text.replace(_EOT, "")

        # Parse the full accumulated text to determine state
        reasoning_text, content_text = self._parse_accumulated(current_text)

        # Check if we just entered or left thought channel.
        # Accept both the full SOC marker AND the degraded form where
        # the detokenizer stripped `<|channel>` but left `thought\n`.
        plain_thinking_in_current = _PLAIN_THINKING_RE.match(current_text or "") is not None
        thought_in_current = (
            self._find_thought_start(current_text) is not None
            or plain_thinking_in_current
        )
        eoc_in_current = _EOC in current_text

        if thought_in_current and not self._saw_thought:
            self._saw_thought = True
            self._in_thought = True

        if eoc_in_current and self._saw_thought and not self._saw_eoc:
            self._saw_eoc = True
            self._in_thought = False

        # If we haven't seen any thought channel markers, decide whether the
        # accumulated text could still be a channel-marker prefix. If not,
        # emit as content immediately. This avoids dropping short responses
        # that finish before the legacy 18-char threshold (e.g. "BRAVO" =
        # 5 chars on /v1/messages with Anthropic-spec enable_thinking=False).
        if not self._saw_thought:
            # The two channel-prefix candidates are "<|channel>thought\n"
            # (18 chars) and degraded "thought\n" (8 chars). If accumulated
            # text is NOT a prefix of either, this is plain content — flush.
            stripped = current_text.lstrip()
            # The degraded marker is exactly ``thought\n``. A bare
            # ``startswith(_THOUGHT)`` stayed true FOREVER once an answer began
            # with the word "thought", so the hold below never released and the
            # whole reply was dropped — content='' and reasoning='' with no
            # server-side recovery (both accumulators stay empty, so the
            # late-flush and answer-pass re-arm branches never fire). Measured:
            # "thought experiments show that X." and "thoughtful reply here."
            # both streamed as NOTHING while one-shot returned them intact.
            # Hold only while the text can still BECOME the marker (a proper
            # prefix of "thought\n") or already IS it ("thought\n…").
            degraded_marker = _THOUGHT + "\n"
            could_be_channel = (
                _SOC.startswith(stripped) or stripped.startswith(_SOC)
                or degraded_marker.startswith(stripped)
                or stripped.startswith(degraded_marker)
                or any(
                    prefix.lower().startswith(stripped.lower())
                    or stripped.lower().startswith(prefix.lower())
                    for prefix in _PLAIN_THINKING_PREFIXES
                )
            )
            if could_be_channel:
                # Hold until we resolve which side of the marker we're on
                return None
            # Not a thought channel — emit ALL accumulated text as content.
            content_so_far = self._plain_content_before_possible_thought(
                current_text
            )
            if content_so_far and len(content_so_far) > self._emitted_content:
                new = content_so_far[self._emitted_content:]
                self._emitted_content = len(content_so_far)
                if new:
                    return DeltaMessage(content=new)
            return None

        # Skip the special tokens themselves
        if delta_text.strip() in (_SOC, _EOC, _EOT, _SOC + _THOUGHT,
                                   _THOUGHT, f"{_SOC}{_THOUGHT}\n"):
            return None

        # Calculate new reasoning/content since last emit
        new_reasoning = None
        new_content = None

        if reasoning_text is not None:
            if len(reasoning_text) > self._emitted_reasoning:
                new_reasoning = reasoning_text[self._emitted_reasoning:]
                self._emitted_reasoning = len(reasoning_text)
            elif len(reasoning_text) < self._emitted_reasoning:
                self._emitted_reasoning = len(reasoning_text)

        if content_text is not None:
            if len(content_text) > self._emitted_content:
                new_content = content_text[self._emitted_content:]
                self._emitted_content = len(content_text)

        if new_reasoning or new_content:
            return DeltaMessage(reasoning=new_reasoning, content=new_content)

        return None

    def _parse_accumulated(self, text: str) -> tuple[str | None, str | None]:
        """Parse accumulated text into reasoning and content portions.

        Handles both the full `<|channel>thought\\n...` marker AND the
        degraded form where the detokenizer ate the SOC token (leaving
        `thought\\n...<channel|>...`).
        """
        # Strip trailing <turn|>, including a partial one still arriving.
        text = _trim_trailing_eot(text)

        # Native or degraded marker, including a tool-continuation channel
        # that begins after already-visible content.
        thought_start = self._find_thought_start(text)
        if thought_start is not None:
            thought_idx, marker_len = thought_start
            visible_prefix = text[:thought_idx].strip()
            after_soc = text[thought_idx + marker_len:]
        # Orphan channel close: treat the post-marker tail as content. This
        # matches complete extraction and prevents streamed `<channel|>` leaks.
        elif text.lstrip().startswith(_EOC):
            stripped = text.lstrip()
            content = stripped[len(_EOC):]
            while content.endswith(_EOT):
                content = content[:-len(_EOT)]
            content = content.strip()
            return None, content or None
        else:
            plain_reasoning, plain_content = self._extract_plain_thinking_process(text)
            if plain_reasoning is not None:
                return plain_reasoning, plain_content
            return None, text.strip() if text.strip() else None

        eoc_idx = after_soc.find(_EOC)
        if eoc_idx >= 0:
            reasoning = after_soc[:eoc_idx].strip()
            visible_suffix = after_soc[eoc_idx + len(_EOC):]
            # Strip <turn|> from content
            while visible_suffix.endswith(_EOT):
                visible_suffix = visible_suffix[:-len(_EOT)]
            return (
                reasoning or None,
                self._join_visible_content(visible_prefix, visible_suffix),
            )
        else:
            # Still in thought channel — partial reasoning, no content yet
            # Don't strip trailing partial marker
            reasoning = self._strip_partial_eoc(after_soc).rstrip()
            return reasoning or None, visible_prefix or None

    @staticmethod
    def _strip_partial_eoc(text: str) -> str:
        """Strip trailing partial <channel|> marker."""
        marker = _EOC  # "<channel|>"
        for length in range(len(marker) - 1, 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length]
        return text

    @staticmethod
    def _extract_plain_thinking_process(text: str) -> tuple[str | None, str | None]:
        """Split Gemma media fallback's plain thinking dialect.

        The full Gemma chat template uses ``<|channel>thought`` markers, but the
        simple mlx-vlm media fallback can emit a native plain block starting with
        ``Thinking Process:``. Treat that leading block as reasoning only when it
        appears at the start of the model output; otherwise ordinary prose stays
        visible content.
        """
        match = _PLAIN_THINKING_RE.match(text or "")
        if not match:
            return None, None

        body = text[match.end():].strip()
        if not body:
            return None, None

        final_match = None
        for candidate in _PLAIN_FINAL_RE.finditer(body):
            final_match = candidate
        if final_match is not None:
            reasoning = body[: final_match.start()].strip()
            content = body[final_match.end():].strip()
            while content.endswith(_EOT):
                content = content[:-len(_EOT)].strip()
            return reasoning or None, content or None

        inline = _INLINE_SELF_CORRECTION_TAIL_RE.match(body)
        if inline:
            reasoning = inline.group("reasoning").strip()
            content = inline.group("content").strip()
            while content.endswith(_EOT):
                content = content[:-len(_EOT)].strip()
            return reasoning or None, content or None

        lines = [line.rstrip() for line in body.splitlines()]
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        if len(nonempty) >= 2:
            last_idx = nonempty[-1]
            last = lines[last_idx].strip()
            reasoning_prefix = "\n".join(lines[:last_idx]).strip()
            looks_like_reasoning_line = bool(
                re.match(r"^(?:\d+[\.)]|[-*•]|\*\(|\(|Self[- ]?Correction|Analyze|Verify|Formulate)\b", last, re.I)
            )
            # Only split a plain final line when it is short and does not look
            # like another reasoning bullet. Long unlabeled prose remains
            # reasoning-only until a content delimiter arrives.
            if (
                reasoning_prefix
                and last
                and not looks_like_reasoning_line
                and len(last) <= 240
            ):
                while last.endswith(_EOT):
                    last = last[:-len(_EOT)].strip()
                return reasoning_prefix or None, last or None

        return body, None
