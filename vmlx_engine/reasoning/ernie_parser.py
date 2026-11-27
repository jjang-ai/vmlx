# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 reasoning parser (<ernie:think>...</ernie:think>)."""

from __future__ import annotations

from .base import DeltaMessage
from .think_parser import delta_safe_reasoning_marker_view
from .think_xml_parser import ThinkXmlReasoningParser


class ErnieReasoningParser(ThinkXmlReasoningParser):
    """Reasoning parser for ERNIE 5.5 ``<ernie:think>...</ernie:think>`` blocks."""

    supports_end_only_reasoning = True

    @property
    def start_token(self) -> str:
        return "<ernie:think>"

    @property
    def end_token(self) -> str:
        return "</ernie:think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        *,
        finished: bool = False,
    ) -> DeltaMessage | None:
        markers = (self.start_token, self.end_token)
        previous_text, current_text, delta_text = delta_safe_reasoning_marker_view(
            previous_text,
            current_text,
            markers,
        )
        if not delta_text:
            return None
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
            start_ids = list(encode(self.start_token, add_special_tokens=False))
            end_ids = list(encode(self.end_token, add_special_tokens=False))
            return {
                "start": [start_ids] if start_ids else [],
                "end": [end_ids] if end_ids else [],
            }
        except Exception:
            return {"start": [], "end": []}
