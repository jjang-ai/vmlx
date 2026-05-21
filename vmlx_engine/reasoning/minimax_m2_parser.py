# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for MiniMax M2-family chat templates.

MiniMax M2 generation can start inside a reasoning block. Some tokenizer paths
do not echo the opening ``<think>`` token in decoded text, so the parser must
treat text before the first ``</think>`` as reasoning when the rendered
assistant prefix starts reasoning.
"""

from .think_parser import BaseThinkingReasoningParser


class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    """MiniMax M2 uses standard think tags with implicit prompt-open support."""

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"
