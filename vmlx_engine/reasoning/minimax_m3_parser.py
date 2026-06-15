# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 reasoning parser: ``<mm:think> ... </mm:think>`` plain-XML think blocks.

M3 marks reasoning with ``<mm:think>`` / ``</mm:think>`` (not ``<think>``). These are
plain XML text markers (like MiMo's), not special tokens that vanish during
detokenization, so we reuse ThinkXmlReasoningParser's no-tags-stay-visible behavior
and only swap the tag tokens. Keep a distinct parser id so capabilities/CLI
auto-config cannot drift back to qwen3 / minimax_m2 / think_xml.

Created by Jinho Jang (eric@jangq.ai).
"""

from .think_xml_parser import ThinkXmlReasoningParser


class MiniMaxM3ReasoningParser(ThinkXmlReasoningParser):
    """Reasoning parser for MiniMax-M3 ``<mm:think>...</mm:think>`` output."""

    @property
    def start_token(self) -> str:
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        return "</mm:think>"
