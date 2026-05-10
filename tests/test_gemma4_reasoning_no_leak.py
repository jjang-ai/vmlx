# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 reasoning parser no-leak tests.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding R2,
the Gemma 4 reasoning parser had no explicit no-leak test. Gemma 4 uses a
DIFFERENT token shape than other thinking-rail parsers — it uses CHANNEL
markers, not `<think>` tags:
  - `<|channel>` (start-of-channel, U+007C-bracketed)
  - `<channel|>` (end-of-channel)
  - `<turn|>` (end-of-turn / EOS)
  - thought channel name: `thought`

A regression that left these tokens visible to the user would show:
  `<|channel>thought` or `<channel|>` or `<turn|>` in the chat.

The parser also handles a "degraded form" where the detokenizer eats `<|channel>`
and only `thought\n...<channel|>...` reaches the parser.

Behavior: extract_reasoning() must split (reasoning, content) cleanly with no
channel/turn marker substring in either field whenever populated.

Fix: this test file pins the contract — no source patch needed.
"""

import pytest

from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser


# Channel marker tokens (mirror parser source constants for clarity).
_SOC = "<|channel>"
_EOC = "<channel|>"
_EOT = "<turn|>"


@pytest.fixture
def parser():
    return Gemma4ReasoningParser()


class TestGemma4ReasoningParserNoLeak:
    def test_no_thought_channel_pure_content(self, parser):
        """Plain content without channel markers passes through clean."""
        text = "The answer is 42." + _EOT
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "The answer is 42."
        assert _SOC not in (content or "")
        assert _EOC not in (content or "")
        assert _EOT not in (content or "")

    def test_full_marker_form_splits_clean(self, parser):
        """Raw wire form with both <|channel> and <channel|> markers."""
        text = f"{_SOC}thought\nLet me think step by step.{_EOC}The answer is 42.{_EOT}"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning == "Let me think step by step."
        assert content == "The answer is 42."
        # Strict no-leak: no channel/turn markers in either output.
        for field in (reasoning, content):
            assert _SOC not in field
            assert _EOC not in field
            assert _EOT not in field
            assert "thought" not in field or field == "thought"  # exact match guard

    def test_degraded_form_soc_eaten_by_detokenizer(self, parser):
        """Common case: `<|channel>` is a special token the detokenizer drops,
        so only `thought\\n...<channel|>...` reaches the parser."""
        text = f"thought\nReasoning prose here.{_EOC}Visible answer.{_EOT}"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning == "Reasoning prose here."
        assert content == "Visible answer."
        for field in (reasoning, content):
            assert _SOC not in field
            assert _EOC not in field
            assert _EOT not in field

    def test_degraded_form_truncated_mid_thought(self, parser):
        """Degraded form WITHOUT `<channel|>` — output truncated mid-reasoning."""
        text = "thought\nReasoning that never closes."
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning == "Reasoning that never closes."
        assert content is None
        assert _SOC not in reasoning

    def test_full_form_truncated_no_eoc(self, parser):
        """Full marker form WITHOUT `<channel|>` — truncated mid-reasoning."""
        text = f"{_SOC}thought\nUnclosed reasoning."
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning == "Unclosed reasoning."
        assert content is None
        assert _SOC not in reasoning

    def test_orphan_eoc_only_strips_to_content(self, parser):
        """Defensive: prompt placed empty thought block, only `<channel|>` leaks
        into generated text. Parser strips it rather than showing it."""
        text = f"{_EOC}Visible answer.{_EOT}"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "Visible answer."
        assert _EOC not in content
        assert _EOT not in content

    def test_multiple_turn_markers_stripped(self, parser):
        """`<turn|>` may appear multiple times at end (EOS); strip them all."""
        text = f"Visible answer.{_EOT}{_EOT}{_EOT}"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "Visible answer."
        assert _EOT not in content

    def test_tools_called_implies_no_channel_marker_in_content(self, parser):
        """Cross-cutting: every form must NOT leak channel/turn markers
        into either reasoning or content fields."""
        for text in [
            f"{_SOC}thought\nA{_EOC}B{_EOT}",
            f"thought\nA{_EOC}B{_EOT}",
            f"thought\nA{_EOC}B",
            f"{_EOC}B",
            f"A{_EOT}",
            f"{_SOC}thought\nA",
        ]:
            reasoning, content = parser.extract_reasoning(text)
            for field in (reasoning, content):
                if field is not None:
                    assert _SOC not in field, (
                        f"{_SOC} leaked for input: {text!r}\nfield: {field!r}"
                    )
                    assert _EOC not in field, (
                        f"{_EOC} leaked for input: {text!r}\nfield: {field!r}"
                    )
                    assert _EOT not in field, (
                        f"{_EOT} leaked for input: {text!r}\nfield: {field!r}"
                    )

    def test_registry_resolves_gemma4(self):
        """Parser must be registered under `gemma4` alias."""
        from vmlx_engine.reasoning import get_parser
        cls = get_parser("gemma4")
        assert cls is Gemma4ReasoningParser

    def test_used_by_gemma4_families(self):
        """Sanity: gemma4 + gemma4_text registry entries bind to gemma4 parser."""
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        registry = ModelConfigRegistry()
        register_all(registry)

        for family_name in ("gemma4", "gemma4_text"):
            entries = [c for c in registry._configs if c.family_name == family_name]
            assert entries, f"family {family_name!r} not found"
            for entry in entries:
                assert entry.reasoning_parser == "gemma4", (
                    f"{family_name} reasoning_parser changed away from gemma4: "
                    f"{entry.reasoning_parser!r}"
                )
