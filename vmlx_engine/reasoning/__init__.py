# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser module for vmlx-engine.

This module provides parsers for extracting reasoning/thinking content from
model outputs. Supports models like Qwen3, DeepSeek-R1, etc. that use special
tokens (e.g., <think>...</think>) to separate reasoning from final responses.

Usage:
    from vmlx_engine.reasoning import get_parser, list_parsers

    # Get a parser by name
    parser = get_parser("qwen3")()

    # Extract reasoning from complete output
    reasoning, content = parser.extract_reasoning(model_output)

    # For streaming
    parser.reset_state()
    for delta in stream:
        msg = parser.extract_reasoning_streaming(prev, curr, delta)
        if msg:
            # msg.reasoning and/or msg.content will be populated
            ...
"""

from .base import DeltaMessage, ReasoningParser
from .think_parser import BaseThinkingReasoningParser

# Parser registry
_REASONING_PARSERS: dict[str, type[ReasoningParser]] = {}


def register_parser(name: str, parser_class: type[ReasoningParser]) -> None:
    """
    Register a reasoning parser.

    Args:
        name: Name to register the parser under (e.g., "qwen3").
        parser_class: The parser class to register.
    """
    _REASONING_PARSERS[name] = parser_class


def get_parser(name: str) -> type[ReasoningParser]:
    """
    Get a reasoning parser class by name.

    Args:
        name: Name of the parser (e.g., "qwen3", "deepseek_r1").

    Returns:
        The parser class (not an instance).

    Raises:
        KeyError: If parser name is not found.
    """
    if name not in _REASONING_PARSERS:
        available = list(_REASONING_PARSERS.keys())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {available}"
        )
    return _REASONING_PARSERS[name]


def list_parsers() -> list[str]:
    """
    List available parser names.

    Returns:
        List of registered parser names.
    """
    return list(_REASONING_PARSERS.keys())


def _register_builtin_parsers():
    """Register built-in parsers."""
    from .deepseek_r1_parser import DeepSeekR1ReasoningParser
    from .gemma4_parser import Gemma4ReasoningParser
    from .gptoss_parser import GptOssReasoningParser
    from .minimax_m2_parser import MiniMaxM2ReasoningParser
    from .mistral_parser import MistralReasoningParser
    from .qwen3_parser import Qwen3ReasoningParser
    from .think_xml_parser import ThinkXmlReasoningParser
    from .minimax_m3_parser import MiniMaxM3ReasoningParser
    from .muse_glimmer_parser import MuseGlimmerReasoningParser
    from .inkling_parser import InklingReasoningParser
    from .ernie_parser import ErnieReasoningParser
    from .hy4_parser import Hy4ReasoningParser

    register_parser("qwen3", Qwen3ReasoningParser)
    register_parser("deepseek_r1", DeepSeekR1ReasoningParser)
    # Poolside/Laguna v1 uses the same <think>...</think> rail contract as
    # DeepSeek-R1 in vMLX: the template may open the rail in the assistant
    # prefix, so output before </think> is reasoning_content and output after
    # the close marker is visible content. Register the vendor name too so
    # generation_config.json values such as "poolside_v1" are accepted instead
    # of silently depending on a JANG-side canonical alias.
    register_parser("poolside_v1", DeepSeekR1ReasoningParser)
    # GLM-5.3-Flash (glm5_next) uses the same <think>...</think> rail with the
    # template force-opening the block in the assistant prefix
    # ('<|assistant|><think>'). Bundles stamp reasoning_parser=glm_think_block;
    # register that vendor id on the R1-rail parser so the stamp resolves
    # instead of silently nulling reasoning extraction (which drops the whole
    # thinking rail on served responses).
    register_parser("glm_think_block", DeepSeekR1ReasoningParser)
    register_parser("minimax_m2", MiniMaxM2ReasoningParser)
    register_parser("think_xml", ThinkXmlReasoningParser)
    # GPT-OSS / GLM-4.7-Flash / Harmony protocol (<|channel|>analysis/final)
    register_parser("openai_gptoss", GptOssReasoningParser)
    # Inkling is CHANNEL-SWITCHED, not tag-delimited: <|content_thinking|> and
    # <|content_text|> are openers that change which channel later text belongs
    # to, with no closer for thinking. Registering it on any think-tag parser
    # would mis-split every response.
    register_parser("inkling", InklingReasoningParser)
    # Mistral 4 uses [THINK]...[/THINK] tokens for reasoning
    register_parser("mistral", MistralReasoningParser)
    # Gemma 4 uses <|channel>thought...<channel|> for reasoning
    register_parser("gemma4", Gemma4ReasoningParser)
    # MiniMax-M3 uses <mm:think>...</mm:think> plain-XML think blocks
    register_parser("minimax_m3", MiniMaxM3ReasoningParser)
    # ERNIE 5.5 uses <ernie:think>...</ernie:think> blocks
    register_parser("ernie", ErnieReasoningParser)
    register_parser("ernie5_5", ErnieReasoningParser)
    # Hy4-flash uses <think:opensource>...</think:opensource> blocks (prefilled on high effort)
    register_parser("hy4", Hy4ReasoningParser)
    register_parser("hy_v4_flash", Hy4ReasoningParser)
    # dots3_note: plain <think>...</think>, thinking ON by default (the model
    # emits <think> itself off a bare assistant header; token 151721).
    # Thinking OFF is a template-side prefilled CLOSED block plus a
    # <no_think> user marker, so off-mode output carries no think tags at
    # all — the qwen3 think-tag contract matches both modes exactly.
    register_parser("dots3", Qwen3ReasoningParser)
    # Muse Glimmer routes by RECIPIENT, not by an inline think pair or a
    # named channel: "<|start|>assistant to=self<|message|>" is the reasoning
    # rail and "to=user" (or no recipient) is the visible answer.
    register_parser("muse_glimmer", MuseGlimmerReasoningParser)
    register_parser("muse", MuseGlimmerReasoningParser)


# Register built-in parsers on module load
_register_builtin_parsers()


__all__ = [
    # Base classes
    "ReasoningParser",
    "DeltaMessage",
    "BaseThinkingReasoningParser",
    # Registry functions
    "register_parser",
    "get_parser",
    "list_parsers",
]
