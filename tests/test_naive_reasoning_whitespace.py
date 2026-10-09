"""Naive's native reasoning whitespace must survive non-stream post-cleaning."""
import pytest
from vmlx_engine.api.utils import clean_output_text
from vmlx_engine.reasoning.think_xml_parser import ThinkXmlReasoningParser

@pytest.mark.parametrize("raw", ["<think>\nReason.\n\n</think>Answer", "<think>\nStill reasoning.\n"])
def test_naive_stream_and_complete_reasoning_match_after_cleaning(raw):
    parser = ThinkXmlReasoningParser()
    parser.preserve_native_whitespace = True
    reasoning, _ = parser.extract_reasoning(raw)
    complete = clean_output_text(reasoning, preserve_whitespace=True)
    parts = []
    for i in range(1, len(raw) + 1):
        delta = parser.extract_reasoning_streaming(raw[:i-1], raw[:i], raw[i-1:i])
        if delta and delta.reasoning:
            parts.append(delta.reasoning)
    assert "".join(parts) == complete

def test_other_families_keep_existing_cleaner_contract():
    assert clean_output_text("\nAnswer.\n<|im_end|>") == "Answer."
    assert clean_output_text("\nAnswer.\n<|im_end|>", preserve_whitespace=True) == "\nAnswer.\n"
