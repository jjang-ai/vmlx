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

        # Strip trailing <turn|> tokens (EOS)
        while text.endswith(_EOT):
            text = text[:-len(_EOT)]

        # Detect thought channel. Accept both the full marker AND the
        # degraded form where the SOC token was eaten by the detokenizer
        # (the `thought\n` prefix + an `<channel|>` endmarker downstream).
        soc_thought_idx = text.find(_SOC + _THOUGHT)
        if soc_thought_idx < 0 and text.lstrip().startswith(_THOUGHT + "\n") and _EOC in text:
            # Degraded form: `thought\n...<channel|>...`
            stripped = text.lstrip()
            lead = len(text) - len(stripped)
            after_soc = stripped[len(_THOUGHT) + 1:]  # skip "thought\n"
            eoc_idx = after_soc.find(_EOC)
            if eoc_idx >= 0:
                reasoning = after_soc[:eoc_idx].strip()
                content = after_soc[eoc_idx + len(_EOC):].strip()
                while content.endswith(_EOT):
                    content = content[:-len(_EOT)].strip()
                return reasoning or None, content or None

        # Also handle the degraded form when there's NO <channel|> yet
        # (truncated mid-thought). Pull everything after `thought\n`
        # into reasoning so it doesn't spill into content.
        if soc_thought_idx < 0 and text.lstrip().startswith(_THOUGHT + "\n") and _EOC not in text:
            stripped = text.lstrip()
            after_soc = stripped[len(_THOUGHT) + 1:]
            return after_soc.strip() or None, None

        # Defensive orphan-close handling: if a bundle/template placed the
        # empty thought block in the prompt and only the channel close marker
        # leaks into generated text, strip it instead of showing it to users.
        if soc_thought_idx < 0 and text.lstrip().startswith(_EOC):
            stripped = text.lstrip()
            content = stripped[len(_EOC):].strip()
            while content.endswith(_EOT):
                content = content[:-len(_EOT)].strip()
            return None, content or None

        # Check for thought channel (full marker form)
        if soc_thought_idx >= 0:
            idx = soc_thought_idx
            # Extract after the channel marker + "thought" + optional newline
            after_soc = text[idx + len(_SOC + _THOUGHT):]
            if after_soc.startswith("\n"):
                after_soc = after_soc[1:]

            # Find the end-of-channel marker
            eoc_idx = after_soc.find(_EOC)
            if eoc_idx >= 0:
                reasoning = after_soc[:eoc_idx].strip()
                content = after_soc[eoc_idx + len(_EOC):].strip()
                # Strip any trailing <turn|> from content
                while content.endswith(_EOT):
                    content = content[:-len(_EOT)].strip()
                return reasoning or None, content or None
            else:
                # No end marker — all remaining is reasoning (truncated)
                reasoning = after_soc.strip()
                return reasoning or None, None

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
            _SOC + _THOUGHT in current_text
            or current_text.lstrip().startswith(_THOUGHT + "\n")
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
            could_be_channel = (
                _SOC.startswith(stripped) or stripped.startswith(_SOC)
                or (_THOUGHT + "\n").startswith(stripped) or stripped.startswith(_THOUGHT)
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
            content_so_far = current_text.replace(_EOT, "").strip()
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
        # Strip trailing <turn|>
        while text.endswith(_EOT):
            text = text[:-len(_EOT)]

        # Full marker form
        if _SOC + _THOUGHT in text:
            idx = text.find(_SOC + _THOUGHT)
            after_soc = text[idx + len(_SOC + _THOUGHT):]
            if after_soc.startswith("\n"):
                after_soc = after_soc[1:]
        # Degraded form — only `thought\n` prefix (detokenizer ate <|channel>)
        elif text.lstrip().startswith(_THOUGHT + "\n"):
            stripped = text.lstrip()
            after_soc = stripped[len(_THOUGHT) + 1:]
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
            content = after_soc[eoc_idx + len(_EOC):]
            # Strip <turn|> from content
            while content.endswith(_EOT):
                content = content[:-len(_EOT)]
            content = content.strip()
            return reasoning or None, content or None
        else:
            # Still in thought channel — partial reasoning, no content yet
            # Don't strip trailing partial marker
            reasoning = self._strip_partial_eoc(after_soc).rstrip()
            return reasoning or None, None

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
