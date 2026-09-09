# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

This module provides BaseThinkingReasoningParser, a concrete implementation
for extracting reasoning content from models that use thinking tags.

Supports four scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. Only opening tag (truncated reasoning, max_tokens hit): <think>reasoning
4. No tags: pure content
"""

from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser


def delta_safe_reasoning_marker_view(
    previous_text: str,
    current_text: str,
    markers: tuple[str, ...],
) -> tuple[str, str, str]:
    """Return accumulated views that cannot expose a split control marker.

    SSE deltas are irreversible.  When a tokenizer/engine splits ``<think>``
    or ``</think>`` across chunks, the old parser could emit ``<thi`` as
    visible content (or reasoning) before the final ``nk>`` arrived.  Hold the
    smallest trailing suffix that is still a proper prefix of any marker.  If
    later characters prove it was ordinary prose, the next safe delta releases
    the held suffix verbatim; if it completes a marker, the parser consumes the
    whole marker without ever exposing it.

    The calculation is derived only from the accumulated text, so it remains
    request-local and safe even when parser instances are reused.
    """

    usable_markers = tuple(marker for marker in markers if marker)

    def _safe_prefix(text: str) -> str:
        held = 0
        for marker in usable_markers:
            max_prefix = min(len(text), len(marker) - 1)
            for length in range(max_prefix, 0, -1):
                if text.endswith(marker[:length]):
                    held = max(held, length)
                    break
        return text[:-held] if held else text

    safe_previous = _safe_prefix(previous_text)
    safe_current = _safe_prefix(current_text)
    if safe_current.startswith(safe_previous):
        safe_delta = safe_current[len(safe_previous) :]
    else:
        # Accumulated generation text is required to be append-only.  On an
        # unexpected rewrite, emit nothing rather than replaying already-sent
        # text; downstream terminal reconciliation remains truthful.
        safe_delta = ""
    return safe_previous, safe_current, safe_delta


def _first_unquoted(text: str, marker: str) -> int:
    """Index of `marker` as real markup, skipping backtick-quoted mentions.

    Mirrors _first_unquoted_marker_pos in server.py. Only a PRECEDING backtick
    counts as quoting: testing the following character too was tried there and
    was too broad.
    """
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos < 0:
            return -1
        if pos == 0 or text[pos - 1] != "`":
            return pos
        start = pos + 1


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base parser for models using <think>...</think> style tags.

    This parser handles the common pattern where reasoning content is wrapped
    in special tags. Subclasses define the specific start and end tokens.

    Supports "implicit reasoning mode" where <think> is injected in the prompt
    and only </think> appears in the model output. This is common with AI agents
    like OpenCode that force models to reason by injecting thinking tags.

    The parser tracks state during streaming to correctly separate reasoning
    from content as tokens arrive incrementally.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._think_in_prompt = False  # Set via reset_state() when <think> is in the prompt
        self._think_in_prompt_explicit = False

    @property
    def alternate_reasoning_marker_pairs(self) -> tuple[tuple[str, str], ...]:
        """Additional private-rail spellings accepted by this parser.

        Several reasoning-capable models occasionally emit the more verbose
        ``<thinking>...</thinking>`` dialect after a canonical ``<think>`` rail
        has already closed.  That is still model-private planning, not visible
        assistant content.  Treating it as an alias here keeps the API
        reasoning/content split truthful across Chat, Responses, and the
        Electron renderer without adding a UI-only sanitizer.
        """

        return (("<thinking>", "</thinking>"),)

    @property
    def reasoning_marker_pairs(self) -> tuple[tuple[str, str], ...]:
        """All start/end marker pairs this parser treats as reasoning rails."""

        pairs: list[tuple[str, str]] = [(self.start_token, self.end_token)]
        for pair in self.alternate_reasoning_marker_pairs:
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    @property
    def reasoning_markers(self) -> tuple[str, ...]:
        return tuple(marker for pair in self.reasoning_marker_pairs for marker in pair)

    def _normalize_reasoning_markers(self, text: str) -> str:
        """Map accepted alias markers to this parser's canonical marker pair."""

        normalized = text
        for start, end in self.reasoning_marker_pairs:
            if start != self.start_token:
                normalized = normalized.replace(start, self.start_token)
            if end != self.end_token:
                normalized = normalized.replace(end, self.end_token)
        return normalized

    def _safe_normalized_marker_view(
        self,
        previous_text: str,
        current_text: str,
    ) -> tuple[str, str, str]:
        """Return delta-safe previous/current/delta after alias normalization."""

        safe_previous_raw, safe_current_raw, _ = delta_safe_reasoning_marker_view(
            previous_text,
            current_text,
            self.reasoning_markers,
        )
        safe_previous = self._normalize_reasoning_markers(safe_previous_raw)
        safe_current = self._normalize_reasoning_markers(safe_current_raw)
        if safe_current.startswith(safe_previous):
            safe_delta = safe_current[len(safe_previous) :]
        else:
            safe_delta = ""
        return safe_previous, safe_current, safe_delta

    def reset_state(self, think_in_prompt: bool = False, **kwargs):
        """Reset state for a new streaming request.

        Args:
            think_in_prompt: True if <think> was injected in the prompt
                (assistant prefix), enabling implicit reasoning mode where
                text before </think> is treated as reasoning.
        """
        self._think_in_prompt = think_in_prompt
        self._think_in_prompt_explicit = True

    def _direct_rail_without_opener(self, text: str) -> bool:
        """A known closed prompt cannot retroactively open a private rail.

        Keep standalone/unseeded implicit-close compatibility. Once the server
        explicitly seeds a direct rail, only an actual opening marker can move
        generated text into reasoning. Streaming cannot retract earlier prose.
        """
        return (
            self._think_in_prompt_explicit
            and not self._think_in_prompt
            and self.start_token not in text
        )

    def _strip_unopened_closes(self, text: str) -> str:
        # Called on normalized, marker-safe accumulated views. Preserve quoted
        # mentions and every surrounding byte, including paragraph whitespace.
        while (position := _first_unquoted(text, self.end_token)) >= 0:
            text = text[:position] + text[position + len(self.end_token):]
        return text

    def reasoning_tag_token_seqs(self, tokenizer) -> dict:
        """Return token sequences for the reasoning start/end tags.

        Encodes all accepted start/end marker pairs (text strings defined by
        each subclass plus safe aliases) using the model tokenizer. Used by the
        scheduler to build a `SequenceStateMachine` for token-level
        reasoning-boundary detection.

        Falls back to an empty result if encoding fails (e.g. tokenizer
        absent at request time, or tags become multi-segment splits the
        state machine cannot represent atomically).
        """
        if tokenizer is None:
            return {"start": [], "end": []}
        try:
            encode = getattr(tokenizer, "encode", None)
            if encode is None and hasattr(tokenizer, "tokenizer"):
                encode = getattr(tokenizer.tokenizer, "encode", None)
            if encode is None:
                return {"start": [], "end": []}
            start_seqs = []
            end_seqs = []
            for start_marker, end_marker in self.reasoning_marker_pairs:
                start_ids = encode(start_marker, add_special_tokens=False)
                end_ids = encode(end_marker, add_special_tokens=False)
                if start_ids:
                    start_seqs.append(list(start_ids))
                if end_ids:
                    end_seqs.append(list(end_ids))
        except Exception:
            return {"start": [], "end": []}
        return {"start": start_seqs, "end": end_seqs}

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles five cases:
        1. Both tags present: <think>reasoning</think>content
        2. Only closing tag: reasoning</think>content (think in prompt)
        3. Only opening tag: <think>reasoning (truncated, max_tokens hit)
        4a. No tags AND think_in_prompt=True: entire output IS reasoning
            (tokenizer ate the start tag AND output was truncated before
            </think> — common on Qwen 3.6 / MiniMax where <think> is a
            special token the detokenizer drops). Without this path the
            full reasoning prose spills into `content` on non-stream
            responses while the streaming path routes it correctly,
            producing divergent chat_vs_stream behaviour for the same
            prompt. Mirrors `extract_reasoning_streaming`'s Case 3 fallback.
        4b. No tags AND think_in_prompt=False: pure content

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        text = self._normalize_reasoning_markers(model_output)

        if self._direct_rail_without_opener(text):
            return None, self._strip_unopened_closes(text) or None

        # Case 1: Both tags present (normal case)
        if self.start_token in text and self.end_token in text:
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            pos = 0
            in_reasoning = False
            while pos < len(text):
                if in_reasoning:
                    end_idx = text.find(self.end_token, pos)
                    if end_idx < 0:
                        reasoning_parts.append(text[pos:])
                        pos = len(text)
                    else:
                        reasoning_parts.append(text[pos:end_idx])
                        pos = end_idx + len(self.end_token)
                        in_reasoning = False
                    continue

                start_idx = text.find(self.start_token, pos)
                if start_idx < 0:
                    content_parts.append(text[pos:])
                    break
                content_parts.append(text[pos:start_idx])
                pos = start_idx + len(self.start_token)
                in_reasoning = True

            reasoning = "\n".join(part.strip() for part in reasoning_parts if part.strip())
            content = "".join(content_parts).strip()
            return reasoning or None, content or None

        # Case 2: Only closing tag (think was injected in prompt).
        # Everything before </think> is reasoning.
        #
        # But a QUOTED mention is not a close. Implicit mode is a real, tested
        # contract (the template injects <think> as a special token that gets
        # eaten, so the model emits only the close), so this cannot simply be
        # gated on _think_in_prompt — that breaks implicit mode. Instead skip
        # occurrences wrapped in backticks, exactly as the suppressed-tool
        # display path does for tool markers.
        #
        # Without this, any answer that MENTIONS the tag lost its opening text
        # to the reasoning rail:
        #   "Code: `</think>` here"  ->  "` here"
        #   a fenced "foo</think>bar" -> "bar"
        # A model explaining or quoting the tag is ordinary prose, and models
        # never wrap a real emitted close in backticks.
        end_pos = _first_unquoted(text, self.end_token)
        if end_pos >= 0:
            reasoning = text[:end_pos]
            content = text[end_pos + len(self.end_token) :]
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4a: No tags but think_in_prompt (special-token-eaten path)
        if self._think_in_prompt:
            reasoning = text.strip()
            return reasoning or None, None

        # Case 4b: No tags at all - pure content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta using text-based detection.

        Handles implicit reasoning mode where <think> was in the prompt
        and only </think> appears in the output.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        previous_text, current_text, delta_text = self._safe_normalized_marker_view(
            previous_text,
            current_text,
        )
        if not delta_text:
            return None

        if self._direct_rail_without_opener(current_text):
            before = self._strip_unopened_closes(previous_text)
            after = self._strip_unopened_closes(current_text)
            content = after[len(before):] if after.startswith(before) else ""
            return DeltaMessage(content=content) if content else None

        # A delta that is ONLY a marker still carries whitespace around it, and
        # that whitespace is real answer text. Returning None dropped it: a
        # chunking where one delta is exactly "\n<think>" streamed "IntroAnswer"
        # where the one-shot path returned "Intro\n\nAnswer" — the paragraph
        # break before a mid-answer think block simply vanished. Character-wise
        # streaming never produces such a delta, which is why the fidelity suite
        # did not catch it.
        #
        # The whitespace BEFORE an opening marker belongs to the rail that was
        # active before it (content), so emit it rather than swallowing it.
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            leading = delta_text[: delta_text.index(self.start_token)]
            # This parser is stateless — the rail is derived from the text, so
            # "already reasoning" is an unclosed start token in what came before.
            already_reasoning = (
                self.start_token in previous_text
                and self.end_token not in previous_text
            )
            if leading and not already_reasoning:
                return DeltaMessage(content=leading)
            return None
        if stripped_delta == self.end_token:
            return None

        # Check token positions in text (stateless text-based detection)
        start_in_prev = self.start_token in previous_text
        start_in_current = self.start_token in current_text
        end_in_prev = self.end_token in previous_text
        end_in_delta = self.end_token in delta_text

        # Case 1: Explicit <think> found in text - standard behavior
        if start_in_current:
            return self._handle_explicit_think(
                previous_text, delta_text, start_in_prev, end_in_prev, end_in_delta
            )

        # Case 2: No <think> but </think> found - implicit reasoning mode
        # This handles when <think> was injected in the prompt
        if self.end_token in current_text:
            return self._handle_implicit_think(
                previous_text,
                delta_text,
                end_in_prev,
                end_in_delta,
            )

        # Case 3: No think tags seen yet in the output.
        # If <think> was in the prompt (implicit mode), treat as reasoning
        # until </think> appears. Otherwise default to content — most models
        # that don't output <think> are just producing normal content.
        if self._think_in_prompt:
            return DeltaMessage(reasoning=delta_text)
        return DeltaMessage(content=delta_text)

    def _handle_explicit_think(
        self,
        previous_text: str,
        delta_text: str,
        start_in_prev: bool,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> tag is explicitly in the output."""
        start_in_delta = self.start_token in delta_text
        last_start_prev = previous_text.rfind(self.start_token)
        last_end_prev = previous_text.rfind(self.end_token)
        in_explicit_reasoning = last_start_prev >= 0 and last_start_prev > last_end_prev

        if start_in_prev:
            # We're after at least one start token. Use the most recent
            # boundary, not the first one, so a later post-tool
            # <think>...</think> block is routed back to reasoning instead of
            # leaking as visible content.
            if in_explicit_reasoning and end_in_delta:
                # Transition: end token in this delta
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :].lstrip()
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            elif in_explicit_reasoning:
                # Still inside the most recent explicit reasoning block.
                return DeltaMessage(reasoning=delta_text)
            elif end_in_prev:
                # Already past a reasoning phase. A new start token in this
                # delta opens a later reasoning segment; otherwise pure content.
                if start_in_delta:
                    start_idx = delta_text.find(self.start_token)
                    pre_content = delta_text[:start_idx]
                    after_start = delta_text[start_idx + len(self.start_token) :]
                    end_idx = after_start.find(self.end_token)
                    if end_idx >= 0:
                        reasoning_part = after_start[:end_idx]
                        post_content = after_start[end_idx + len(self.end_token) :].lstrip()
                        content_part = pre_content + post_content
                    else:
                        reasoning_part = after_start
                        content_part = pre_content
                    return DeltaMessage(
                        reasoning=reasoning_part if reasoning_part else None,
                        content=content_part if content_part else None,
                    )
                content_part = self._content_after_reasoning_boundary(
                    previous_text,
                    delta_text,
                )
                return (
                    DeltaMessage(content=content_part)
                    if content_part
                    else None
                )
            else:
                # Still in reasoning phase
                return DeltaMessage(reasoning=delta_text)

        elif start_in_delta:
            # Start token is in this delta
            start_idx = delta_text.find(self.start_token)

            if end_in_delta:
                # Both tokens in this delta
                end_idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[start_idx + len(self.start_token) : end_idx]
                content_part = delta_text[end_idx + len(self.end_token) :].lstrip()
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            else:
                # Only start token - beginning of reasoning
                reasoning_part = delta_text[start_idx + len(self.start_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None
                )

        # Fallback - treat as content
        return DeltaMessage(content=delta_text)

    def _handle_implicit_think(
        self,
        previous_text: str,
        delta_text: str,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> was in prompt (only </think> in output)."""
        if end_in_delta:
            # Transition: end token in this delta
            idx = delta_text.find(self.end_token)
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len(self.end_token) :].lstrip()
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        elif end_in_prev:
            # Already past reasoning phase - pure content
            content_part = self._content_after_reasoning_boundary(
                previous_text,
                delta_text,
            )
            return DeltaMessage(content=content_part) if content_part else None
        else:
            # Still in implicit reasoning phase
            return DeltaMessage(reasoning=delta_text)

    def _content_after_reasoning_boundary(
        self,
        previous_text: str,
        delta_text: str,
    ) -> str:
        """Drop only the structural separator after the reasoning close.

        Complete extraction already applies ``strip()`` to content after the
        end marker. Streaming previously exposed the common ``\n\n`` separator
        as visible API content when the close marker and newlines arrived in
        separate deltas. Keep stripping while the accumulated post-reasoning
        content is whitespace-only, then preserve every later delta verbatim.
        """
        _, marker, previous_content = previous_text.partition(self.end_token)
        if marker and not previous_content.strip():
            return delta_text.lstrip()
        return delta_text
