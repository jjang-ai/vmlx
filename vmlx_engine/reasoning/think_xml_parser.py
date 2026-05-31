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
