# SPDX-License-Identifier: Apache-2.0
"""
Generic XML <think>...</think> reasoning parser.

MiMo-V2.5 uses plain XML think blocks, but it is not a Qwen-family model. Keep
the parser id separate so capabilities and CLI auto-configuration cannot drift
back to qwen3 while still sharing the common XML tag extraction behavior.
"""

from .think_parser import BaseThinkingReasoningParser


class ThinkXmlReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for generic XML <think>...</think> model output."""

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract generic XML reasoning without hiding untagged content.

        MiMo's thinking markers are plain XML text, not Qwen-style special
        tokens that disappear during detokenization. The shared base parser's
        no-tags + think_in_prompt branch is correct for Qwen/MiniMax special
        token paths, but it is too aggressive for generic XML: a MiMo response
        that says ``FINAL=OK`` without literal tags must remain visible content,
        not become hidden ``reasoning_content``.
        """
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ):
        """Stream generic XML reasoning without classifying no-tag deltas hidden."""
        if self.start_token not in current_text and self.end_token not in current_text:
            return self._content_delta(delta_text)
        return super().extract_reasoning_streaming(previous_text, current_text, delta_text)

    @staticmethod
    def _content_delta(delta_text: str):
        from .base import DeltaMessage

        return DeltaMessage(content=delta_text)
