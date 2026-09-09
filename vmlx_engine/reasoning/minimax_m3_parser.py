# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 reasoning parser.

M3's template-owned rail uses ``<mm:think>`` / ``</mm:think>``, but live app
testing also observed plain ``<think>`` / ``</think>`` fallback blocks when the
request asked for thinking off. Treat both as reasoning markers so internal
thought never leaks into visible assistant content or pollutes later turns.
Keep a distinct parser id so capabilities/CLI auto-config cannot drift back to
qwen3 / minimax_m2 / think_xml.

Created by Jinho Jang (eric@jangq.ai).
"""

from .base import DeltaMessage
from .think_parser import delta_safe_reasoning_marker_view
from .think_xml_parser import ThinkXmlReasoningParser


class MiniMaxM3ReasoningParser(ThinkXmlReasoningParser):
    """Reasoning parser for MiniMax-M3 think blocks."""

    _ALIASES = (
        ("<mm:think>", "</mm:think>"),
        ("<think>", "</think>"),
    )

    @property
    def start_token(self) -> str:
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        return "</mm:think>"

    def reset_state(
        self,
        think_in_prompt: bool = False,
        adaptive_mode: bool = False,
        **kwargs,
    ) -> None:
        """Reset request-local streaming state.

        M3 adaptive generation has two valid shapes with the same initial
        bytes: a marker-free direct answer, or an implicit reasoning rail that
        emits only ``</mm:think>`` before the answer. Publishing those initial
        bytes as content is irreversible, so hold them until either a real
        marker resolves the rail or the engine marks a marker-free generation
        finished.
        """
        super().reset_state(think_in_prompt=think_in_prompt, **kwargs)
        self._adaptive_stream_pending = bool(adaptive_mode and not think_in_prompt)

    def _direct_rail_without_opener(self, text: str) -> bool:
        # M3 owns an end-only implicit-rail contract for both native and alias
        # markers. Its adaptive stream holds the ambiguous initial text.
        # A false prompt seed alone does not disable that family contract.
        return False

    @property
    def adaptive_stream_pending(self) -> bool:
        """Whether the adaptive stream still has an ambiguous initial rail."""
        return bool(getattr(self, "_adaptive_stream_pending", False))

    def _matching_aliases(self, text: str) -> list[tuple[str, str]]:
        return [
            (start, end)
            for start, end in self._ALIASES
            if start in text or end in text
        ]

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract M3 reasoning from either ``<mm:think>`` or ``<think>``."""
        for start, end in self._matching_aliases(model_output):
            if start == self.start_token and end == self.end_token:
                return super().extract_reasoning(model_output)
            return _extract_with_tags(model_output, start, end)
        if getattr(self, "_think_in_prompt", False):
            reasoning = model_output.strip()
            return reasoning or None, None
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        *,
        finished: bool = False,
    ) -> DeltaMessage | None:
        """Stream M3 reasoning for both documented and fallback tags."""
        markers = tuple(marker for pair in self._ALIASES for marker in pair)
        previous_text, current_text, delta_text = delta_safe_reasoning_marker_view(
            previous_text,
            current_text,
            markers,
        )
        if not delta_text:
            return None

        if self.adaptive_stream_pending:
            if not any(marker in current_text for marker in markers):
                if not finished:
                    return None
                self._adaptive_stream_pending = False
                return DeltaMessage(content=current_text or None)

            # Nothing from the ambiguous prefix has been published yet. Parse
            # the complete safe prefix once so an end-only rail can move every
            # preceding byte to reasoning before visible content is sent.
            self._adaptive_stream_pending = False
            reasoning, content = self.extract_reasoning(current_text)
            return DeltaMessage(reasoning=reasoning, content=content)

        for start, end in self._ALIASES:
            if start in current_text or end in current_text:
                if start == self.start_token and end == self.end_token:
                    return super().extract_reasoning_streaming(
                        previous_text,
                        current_text,
                        delta_text,
                    )
                return _stream_with_tags(
                    previous_text,
                    current_text,
                    delta_text,
                    start,
                    end,
                )
        if getattr(self, "_think_in_prompt", False):
            return DeltaMessage(reasoning=delta_text)
        return DeltaMessage(content=delta_text)

    def reasoning_tag_token_seqs(self, tokenizer) -> dict:
        """Expose both M3 tag spellings to token-level boundary detection."""
        if tokenizer is None:
            return {"start": [], "end": []}
        try:
            encode = getattr(tokenizer, "encode", None)
            if encode is None and hasattr(tokenizer, "tokenizer"):
                encode = getattr(tokenizer.tokenizer, "encode", None)
            if encode is None:
                return {"start": [], "end": []}
            start = []
            end = []
            for start_tag, end_tag in self._ALIASES:
                start_ids = list(encode(start_tag, add_special_tokens=False))
                end_ids = list(encode(end_tag, add_special_tokens=False))
                if start_ids:
                    start.append(start_ids)
                if end_ids:
                    end.append(end_ids)
            return {"start": start, "end": end}
        except Exception:
            return {"start": [], "end": []}


def _extract_with_tags(text: str, start: str, end: str) -> tuple[str | None, str | None]:
    if start in text and end in text:
        _, _, after_start = text.partition(start)
        reasoning, _, content = after_start.partition(end)
        return reasoning.strip() or None, content.strip() or None
    if end in text:
        reasoning, _, content = text.partition(end)
        return reasoning.strip() or None, content.strip() or None
    if start in text:
        _, _, reasoning = text.partition(start)
        return reasoning.strip() or None, None
    return None, text


def _stream_with_tags(
    previous_text: str,
    current_text: str,
    delta_text: str,
    start: str,
    end: str,
) -> DeltaMessage | None:
    stripped_delta = delta_text.strip()
    if stripped_delta in {start, end}:
        return None

    def _content_after_close(text: str) -> str:
        """Visible text emitted after the reasoning close, so far.

        The newline(s) directly following the close are a structural boundary
        between the rail and the answer, not answer text. The canonical
        <mm:think> dialect delegates to the base parser, which strips them;
        this alias helper emitted them verbatim, so the SAME MODEL answered
        with a leading blank line on <think> and without one on <mm:think>.
        Strip only while nothing visible has been emitted yet.
        """
        if end not in text:
            return ""
        return text.rpartition(end)[2]

    start_in_prev = start in previous_text
    start_in_current = start in current_text
    end_in_prev = end in previous_text
    end_in_delta = end in delta_text

    if start_in_current:
        start_in_delta = start in delta_text
        if start_in_prev:
            if end_in_delta:
                idx = delta_text.find(end)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(end) :].lstrip()
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            if end_in_prev:
                content_part = delta_text
                if not _content_after_close(previous_text).strip():
                    content_part = content_part.lstrip()
                return DeltaMessage(content=content_part or None)
            return DeltaMessage(reasoning=delta_text)

        if start_in_delta:
            start_idx = delta_text.find(start)
            if end_in_delta:
                end_idx = delta_text.find(end)
                reasoning_part = delta_text[start_idx + len(start) : end_idx]
                content_part = delta_text[end_idx + len(end) :].lstrip()
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            reasoning_part = delta_text[start_idx + len(start) :]
            return DeltaMessage(reasoning=reasoning_part if reasoning_part else None)

    if end in current_text:
        if end_in_delta:
            idx = delta_text.find(end)
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len(end) :].lstrip()
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        if end_in_prev:
            content_part = delta_text
            if not _content_after_close(previous_text).strip():
                content_part = content_part.lstrip()
            return DeltaMessage(content=content_part or None)
        return DeltaMessage(reasoning=delta_text)

    return DeltaMessage(content=delta_text)
