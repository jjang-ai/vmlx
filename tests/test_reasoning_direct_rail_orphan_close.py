"""A late close cannot retrospectively make already-visible text private."""

import pytest

from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
from vmlx_engine.reasoning.think_xml_parser import ThinkXmlReasoningParser
from vmlx_engine.reasoning.minimax_m2_parser import MiniMaxM2ReasoningParser
from vmlx_engine.reasoning.mistral_parser import MistralReasoningParser
from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser


@pytest.mark.parametrize("parser_class", [
    Qwen3ReasoningParser, ThinkXmlReasoningParser,
    MiniMaxM2ReasoningParser, MistralReasoningParser,
])
@pytest.mark.parametrize("width", [1, 2, 8, 31, 1000])
def test_explicit_direct_rail_preserves_text_around_orphan_close(parser_class, width):
    # Include a repeated answer deliberately: parsers must not deduplicate data.
    parser = parser_class()
    raw = "Answer: 9 units.\n" + parser.end_token + "\n\nAnswer: 9 units."
    expected = raw.replace(parser.end_token, "")
    parser.reset_state(think_in_prompt=False)
    reasoning, content = parser.extract_reasoning(raw)
    assert reasoning is None
    assert content == expected
    previous = ""
    emitted = []
    private = []
    for offset in range(0, len(raw), width):
        current = raw[:offset + width]
        delta = parser.extract_reasoning_streaming(previous, current, current[len(previous):])
        if delta:
            emitted.append(delta.content or "")
            private.append(delta.reasoning or "")
        previous = current
    assert "".join(emitted) == expected
    assert "".join(private) == ""


def test_unseeded_qwen_parser_retains_implicit_close_compatibility():
    assert Qwen3ReasoningParser().extract_reasoning("plan</think>Answer") == ("plan", "Answer")


def test_seeded_open_qwen_parser_keeps_private_rail():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=True)
    assert parser.extract_reasoning("plan</think>Answer") == ("plan", "Answer")


def test_direct_rail_still_routes_an_explicit_opening_to_reasoning():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)
    assert parser.extract_reasoning("<think>plan</think>Answer") == ("plan", "Answer")


def test_direct_rail_preserves_quoted_close_marker():
    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)
    text = "The marker is `</think>`."
    assert parser.extract_reasoning(text) == (None, text)


@pytest.mark.parametrize("end", ["</mm:think>", "</think>"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_m3_native_implicit_close_contract_is_not_reclassified(end, adaptive):
    parser = MiniMaxM3ReasoningParser()
    parser.reset_state(think_in_prompt=False, adaptive_mode=adaptive)
    assert parser.extract_reasoning("plan" + end + "Answer") == ("plan", "Answer")
