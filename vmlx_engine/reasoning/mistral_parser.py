# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jinho Jang (eric@jangq.ai) — vMLX / mlxstudio
# Original implementation for Mistral 4 MLA+MoE reasoning with [THINK]/[/THINK]
# token handling, reasoning_effort auto-mapping, and streaming extraction.
# If you're reading this because you're adapting it: please credit the author.
"""
Reasoning parser for Mistral 4 models.

Mistral 4 uses [THINK]...[/THINK] tokens for reasoning content, controlled
via the reasoning_effort field in [MODEL_SETTINGS] within the chat template.

Token IDs: [THINK] = 34, [/THINK] = 35 (extra_special_tokens in tokenizer).
When decoded, these produce the literal text strings "[THINK]" and "[/THINK]".
"""

from .think_parser import BaseThinkingReasoningParser


class MistralReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Mistral 4 models.

    Mistral 4 uses [THINK]...[/THINK] tokens to denote reasoning text.
    The model outputs these when reasoning_effort is set to "high" in
    the [MODEL_SETTINGS] block of the chat template.

    Supports three scenarios:
    1. Both tags in output: [THINK]reasoning[/THINK]content
    2. Only closing tag (think in prompt): reasoning[/THINK]content
    3. No tags: pure content (reasoning_effort="none")

    Example (normal):
        Input: "[THINK]Let me work through this...[/THINK]The answer is 345."
        Output: reasoning="Let me work through this...", content="The answer is 345."
    """

    @property
    def start_token(self) -> str:
        return "[THINK]"

    @property
    def end_token(self) -> str:
        return "[/THINK]"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Mistral 4 output.

        Handles both explicit [THINK]...[/THINK] tags and truncated reasoning
        where the model hit max_tokens during the thinking phase.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        if self.end_token not in model_output:
            if self.start_token in model_output:
                # [THINK] present but no [/THINK] — reasoning was truncated
                return super().extract_reasoning(model_output)
            # No think tags at all — pure content
            return None, model_output

        return super().extract_reasoning(model_output)
