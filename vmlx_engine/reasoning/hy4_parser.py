# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash reasoning parser (<think:opensource>...</think:opensource> and <think>...</think>)."""

from __future__ import annotations

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class Hy4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Hy4-flash ``<think:opensource>...</think:opensource>`` blocks.

    The official ``chat_template.jinja`` prefills ``<think:opensource>`` on ``reasoning_effort:high``
    (end-only closing tag in generated text) and ``<think:opensource></think:opensource>`` on
    ``reasoning_effort:no_think``. Also tolerates canonicalized ``<think>...</think>`` tags.
    """

    supports_end_only_reasoning = True

    @property
    def start_token(self) -> str:
        return "<think:opensource>"

    @property
    def end_token(self) -> str:
        return "</think:opensource>"

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if "<think:opensource>" not in model_output and "</think:opensource>" not in model_output:
            if "<think>" in model_output or "</think>" in model_output:
                normalized = model_output.replace("<think>", self.start_token).replace("</think>", self.end_token)
                return super().extract_reasoning(normalized)
        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        *,
        finished: bool = False,
    ) -> DeltaMessage | None:
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
        )

    def reasoning_tag_token_seqs(self, tokenizer) -> dict:
        if tokenizer is None:
            return {"start": [], "end": []}
        try:
            encode = getattr(tokenizer, "encode", None)
            if encode is None and hasattr(tokenizer, "tokenizer"):
                encode = getattr(tokenizer.tokenizer, "encode", None)
            if encode is None:
                return {"start": [], "end": []}
            starts, ends = [], []
            for st in (self.start_token, "<think>"):
                ids = list(encode(st, add_special_tokens=False))
                if ids and ids not in starts:
                    starts.append(ids)
            for et in (self.end_token, "</think>"):
                ids = list(encode(et, add_special_tokens=False))
                if ids and ids not in ends:
                    ends.append(ids)
            return {"start": starts, "end": ends}
        except Exception:
            return {"start": [], "end": []}
