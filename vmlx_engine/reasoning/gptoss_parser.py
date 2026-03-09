# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS (Harmony protocol) models.

GPT-OSS models use channel markers for reasoning/content separation:
  <|start|>assistant<|channel|>analysis<|message|>...reasoning...
  <|start|>assistant<|channel|>final<|message|>...content...

When the server injects the Harmony analysis prefix into the prompt, the
model output starts directly with reasoning text (no leading marker) and
transitions to content via <|start|>assistant<|channel|>final<|message|>.

In streaming, the output arrives as interleaved channel markers. The parser
tracks how much content has been emitted from the accumulated text to
correctly handle markers that span chunk boundaries.
"""

import re
from .base import DeltaMessage, ReasoningParser

# Channel marker tokens
_CHANNEL_TAG = "<|channel|>"
_MESSAGE_TAG = "<|message|>"
_START_TAG = "<|start|>"
_ANALYSIS_CHANNEL = "analysis"
_FINAL_CHANNEL = "final"

# Full marker strings for detection
_ANALYSIS_MARKER = f"{_CHANNEL_TAG}{_ANALYSIS_CHANNEL}{_MESSAGE_TAG}"
_FINAL_MARKER = f"{_CHANNEL_TAG}{_FINAL_CHANNEL}{_MESSAGE_TAG}"

# Regex to catch bare text Harmony markers in various garbled forms.
# The model may generate these as text tokens instead of special token IDs,
# sometimes with extra whitespace, partial pipe-bracket wrappers, </ prefix, etc.
# Order matters: match "final" before "analysis" to avoid partial overlap.
_BARE_MARKER_RE = re.compile(
    r'</?(?:\|?)?\s*(?:assistant\s*){1,3}(?:\|>?)?\s*final\s*(?:\|>?)?'
    r'|'
    r'</?(?:\|?)?\s*(?:assistant\s*){1,3}(?:\|>?)?\s*analysis\s*(?:\|>?)?',
    re.IGNORECASE,
)

# After server/client strip <|start|>, <|channel|>, <|message|>, residual
# protocol words can remain concatenated in various garbled forms.
# This regex catches all combinations including doubled words, </prefix, etc.
_PROTOCOL_RESIDUE_RE = re.compile(
    r'</?(?:assistant|analysis|final)+'  # </assistantanalysis, <assistant, etc.
    r'|'
    r'(?:assistant\s*){1,3}(?:analysis|final)'  # assistantassistantanalysis, assistant analysis
    r'|'
    r'(?:analysis|final)\s*(?:assistant\s*){1,3}',  # analysisassistant, finalassistant
    re.IGNORECASE,
)


def _bare_marker_replacer(m: re.Match) -> str:
    """Replace a bare-text marker match with proper special token format."""
    text = m.group(0).lower()
    if "final" in text:
        return f"{_START_TAG}assistant{_CHANNEL_TAG}{_FINAL_CHANNEL}{_MESSAGE_TAG}"
    return f"{_START_TAG}assistant{_CHANNEL_TAG}{_ANALYSIS_CHANNEL}{_MESSAGE_TAG}"


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS / Harmony protocol models.

    Extracts reasoning from analysis channels and content from final channels.
    Stops after the FIRST complete analysis->final cycle to prevent second-cycle
    protocol artifacts from leaking into output.
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_reasoning: int = 0
        self._emitted_content: int = 0
        self._saw_marker: bool = False
        self._fallback_emitted: int = 0
        self._harmony_active: bool = False
        self._got_final: bool = False  # True once first final channel seen

    def reset_state(self, **kwargs):
        """Reset state for a new streaming request."""
        self._emitted_reasoning = 0
        self._emitted_content = 0
        self._saw_marker = False
        self._fallback_emitted = 0
        self._harmony_active = kwargs.get("harmony_active", False)
        self._got_final = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete GPT-OSS output."""
        reasoning_parts, content_parts = self._parse_channels(model_output)

        if not reasoning_parts and not content_parts:
            return None, model_output

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        content = "\n".join(content_parts) if content_parts else None
        return reasoning or None, content or None

    def _parse_channels(self, text: str) -> tuple[list[str], list[str]]:
        """Parse channel content from text. Returns (reasoning_parts, content_parts).

        Stops after the first complete final channel to prevent second-cycle leaks.
        """
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        # Normalize bare text markers to special token format
        text = self._normalize_bare_markers(text)

        # Strip leading <|start|>assistant if present
        start_assistant = f"{_START_TAG}assistant"
        if text.startswith(start_assistant):
            text = text[len(start_assistant):]

        if _CHANNEL_TAG not in text:
            if self._harmony_active:
                # Strip trailing partial channel transition
                start_idx = text.rfind(_START_TAG)
                if start_idx >= 0:
                    text = text[:start_idx]
                # Also strip trailing "assistant" that arrives before <|channel|>
                text = re.sub(r'\s*assistant\s*$', '', text)
                safe = self._strip_partial_marker(text)
                safe = self._clean_protocol_residue(safe)
                if safe.strip():
                    reasoning_parts.append(safe.strip())
            return reasoning_parts, content_parts

        # Text before the first channel/start marker is implicit reasoning
        first_channel_idx = text.find(_CHANNEL_TAG)
        first_start_idx = text.find(_START_TAG)

        first_marker_idx = first_channel_idx
        if first_start_idx >= 0 and (first_marker_idx < 0 or first_start_idx < first_marker_idx):
            first_marker_idx = first_start_idx

        if first_marker_idx > 0:
            pre_text = self._clean_protocol_residue(text[:first_marker_idx].strip())
            if pre_text:
                reasoning_parts.append(pre_text)
            text = text[first_marker_idx:]

        # Parse channel markers — stop after first final channel
        found_final = False
        while _CHANNEL_TAG in text:
            idx = text.find(_CHANNEL_TAG)
            text = text[idx + len(_CHANNEL_TAG):]

            if text.startswith(_ANALYSIS_CHANNEL + _MESSAGE_TAG):
                if found_final:
                    break  # Second cycle — stop
                text = text[len(_ANALYSIS_CHANNEL + _MESSAGE_TAG):]
                next_marker = self._find_next_channel(text)
                if next_marker >= 0:
                    part = self._clean_protocol_residue(text[:next_marker].strip())
                    if part:
                        reasoning_parts.append(part)
                    text = text[next_marker:]
                else:
                    safe = self._strip_partial_marker(text)
                    safe = self._clean_protocol_residue(safe.strip())
                    if safe:
                        reasoning_parts.append(safe)
                    text = ""
            elif text.startswith(_FINAL_CHANNEL + _MESSAGE_TAG):
                text = text[len(_FINAL_CHANNEL + _MESSAGE_TAG):]
                found_final = True
                next_marker = self._find_next_channel(text)
                if next_marker >= 0:
                    part = self._clean_protocol_residue(text[:next_marker].strip())
                    if part:
                        content_parts.append(part)
                    # Stop here — don't process second cycle
                    break
                else:
                    safe = self._strip_partial_marker(text)
                    safe = self._clean_protocol_residue(safe.strip())
                    if safe:
                        content_parts.append(safe)
                    text = ""
            else:
                break

        return reasoning_parts, content_parts

    # Characters to buffer before concluding the model isn't using Harmony
    # protocol. The <|channel|> marker starts with "<|" (2 chars); buffer just
    # enough to detect the opening without swallowing short non-Harmony responses.
    _FALLBACK_THRESHOLD = 3

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming delta."""
        if not delta_text:
            return None

        reasoning_parts, content_parts = self._parse_channels(current_text)

        if not self._saw_marker and (reasoning_parts or content_parts):
            self._saw_marker = True

        if self._harmony_active:
            self._saw_marker = True

        # Fallback: model isn't using Harmony protocol
        if not self._saw_marker:
            if len(current_text) < self._FALLBACK_THRESHOLD:
                return None
            if len(current_text) > self._fallback_emitted:
                new_content = current_text[self._fallback_emitted:]
                self._fallback_emitted = len(current_text)
                return DeltaMessage(content=new_content)
            return None

        reasoning_so_far = "\n".join(reasoning_parts) if reasoning_parts else None
        content_so_far = "\n".join(content_parts) if content_parts else None

        new_reasoning = None
        new_content = None

        if reasoning_so_far:
            if len(reasoning_so_far) > self._emitted_reasoning:
                new_reasoning = reasoning_so_far[self._emitted_reasoning:]
                self._emitted_reasoning = len(reasoning_so_far)
            elif len(reasoning_so_far) < self._emitted_reasoning:
                # Bare marker normalization caused reasoning to shrink
                # (e.g. "assistant" was emitted then became part of a marker).
                # Reset to avoid desync.
                self._emitted_reasoning = len(reasoning_so_far)

        if content_so_far:
            if len(content_so_far) > self._emitted_content:
                new_content = content_so_far[self._emitted_content:]
                self._emitted_content = len(content_so_far)

        if new_reasoning or new_content:
            return DeltaMessage(reasoning=new_reasoning, content=new_content)

        return None

    @staticmethod
    def _normalize_bare_markers(text: str) -> str:
        """Convert bare text Harmony markers to special token format.

        Uses regex to catch various garbled forms:
        - ``assistantfinal``, ``assistantanalysis`` (standard bare text)
        - ``<|assistantfinal``, ``assistant final`` (garbled/spaced)
        - Case-insensitive matching
        """
        # Only run regex if text plausibly contains bare markers
        if "assistant" not in text.lower():
            return text
        # Don't touch text that already has proper special tokens
        if _CHANNEL_TAG in text:
            # Split on proper markers, only normalize the gaps
            parts = []
            remaining = text
            while _CHANNEL_TAG in remaining:
                idx = remaining.find(_CHANNEL_TAG)
                before = remaining[:idx]
                if "assistant" in before.lower():
                    before = _BARE_MARKER_RE.sub(_bare_marker_replacer, before)
                parts.append(before)
                # Find the end of this marker sequence
                rest = remaining[idx:]
                # Keep everything from <|channel|> onward as-is until next gap
                next_start = rest.find(_START_TAG, 1)
                if next_start >= 0:
                    parts.append(rest[:next_start])
                    remaining = rest[next_start:]
                else:
                    parts.append(rest)
                    remaining = ""
            if remaining and "assistant" in remaining.lower():
                remaining = _BARE_MARKER_RE.sub(_bare_marker_replacer, remaining)
            parts.append(remaining)
            return "".join(parts)
        # No proper markers at all — normalize the whole string
        return _BARE_MARKER_RE.sub(_bare_marker_replacer, text)

    @staticmethod
    def _clean_protocol_residue(text: str) -> str:
        """Strip residual protocol words and garbled marker fragments.

        After special tokens (<|start|>, <|channel|>, <|message|>) are consumed
        by parsing, text fragments may remain at boundaries.  Also strips
        garbled special token attempts like ``<|assistant.`` or ``<|end``.
        """
        if not text:
            return text
        # Strip garbled special token fragments: <|word. or <|word| or <|word
        text = re.sub(r'<\|?(?:assistant|analysis|final|end|start|channel|message)[^a-zA-Z]*', '', text, flags=re.IGNORECASE)
        # Strip leading/trailing protocol words
        text = re.sub(r'^(?:assistant|analysis|final)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(?:assistant|analysis|final)$', '', text, flags=re.IGNORECASE)
        # Strip concatenated protocol residue in the middle
        text = _PROTOCOL_RESIDUE_RE.sub('', text)
        return text.strip()

    @staticmethod
    def _find_next_channel(text: str) -> int:
        """Find the start of the next channel marker or <|start|> tag."""
        positions = []
        for tag in [_CHANNEL_TAG, _START_TAG]:
            idx = text.find(tag)
            if idx >= 0:
                positions.append(idx)
        return min(positions) if positions else -1

    @staticmethod
    def _strip_partial_marker(text: str) -> str:
        """Strip any trailing partial ``<|start|>`` or ``<|channel|>`` prefix."""
        for tag in (_START_TAG, _CHANNEL_TAG, _MESSAGE_TAG):
            for length in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:length]):
                    return text[:-length]
        # Also strip trailing "assistant" (partial <|start|>assistant sequence)
        if text.rstrip().endswith("assistant"):
            text = text.rstrip()
            return text[:-len("assistant")].rstrip()
        # Strip partial "assistant" at the end (e.g., "assistan", "assist")
        word = "assistant"
        for length in range(len(word) - 1, 4, -1):  # min 5 chars ("assis") to avoid stripping words like "class"
            if text.endswith(word[:length]):
                return text[:-length]
        return text
