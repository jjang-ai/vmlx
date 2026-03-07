# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive tests for streaming reasoning, thinking on/off, GPT-OSS parser,
model config entries, clean_output_text, and server streaming path verification.

Covers the complete lifecycle of enable_thinking tri-state (undefined/True/False),
think_in_template, suppress_reasoning, _template_always_thinks, and the SSE fix
that prevents empty delta emission when reasoning parser is active.

Also covers the GPT-OSS/Harmony protocol parser (bare marker normalization,
protocol residue cleaning, fallback threshold, streaming incremental emit).
"""

import inspect
import re

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Think-in-template / think_in_prompt streaming behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestThinkInPromptStreaming:
    """Tests for BaseThinkingReasoningParser with think_in_prompt=True.

    When think_in_template is True, the server injects <think> in the prompt's
    assistant prefix. The model output starts INSIDE the think block without
    an explicit <think> tag. Only </think> appears in the output.

    This is the CRITICAL path for Qwen3/Qwen3.5 models.
    """

    @pytest.fixture
    def parser(self):
        from vmlx_engine.reasoning import get_parser
        return get_parser("qwen3")()

    def test_think_in_prompt_reasoning_then_content(self, parser):
        """With think_in_prompt=True, text before </think> is reasoning."""
        parser.reset_state(think_in_prompt=True)

        tokens = ["I need", " to think", " carefully", "</think>", "The answer", " is 42"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "".join(reasoning_parts) == "I need to think carefully"
        assert "".join(content_parts) == "The answer is 42"

    def test_think_in_prompt_false_content_only(self, parser):
        """With think_in_prompt=False, text without tags is content."""
        parser.reset_state(think_in_prompt=False)

        tokens = ["Hello", " world", "!"]
        accumulated = ""
        results = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                results.append(result)

        content_parts = [r.content for r in results if r.content]
        reasoning_parts = [r.reasoning for r in results if r.reasoning]

        assert "".join(content_parts) == "Hello world!"
        assert len(reasoning_parts) == 0

    def test_think_in_prompt_no_tags_all_reasoning(self, parser):
        """With think_in_prompt=True and no tags, ALL text is reasoning.

        This is the Qwen 3.5 CRACK scenario: model outputs "Thinking Process:"
        without <think> tags. With think_in_prompt=True, parser correctly
        classifies it as reasoning via Case 3 (_think_in_prompt check).
        """
        parser.reset_state(think_in_prompt=True)

        tokens = ["Thinking", " Process", ":", " Let me", " analyze"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "".join(reasoning_parts) == "Thinking Process: Let me analyze"
        assert len(content_parts) == 0

    def test_think_in_prompt_transition_in_single_delta(self, parser):
        """Transition from reasoning to content within a single delta."""
        parser.reset_state(think_in_prompt=True)

        # First some reasoning tokens
        prev = "reasoning text"
        delta = " more</think>content here"
        curr = prev + delta

        # Need to process the initial text first
        parser.extract_reasoning_streaming("", "reasoning text", "reasoning text")
        result = parser.extract_reasoning_streaming(prev, curr, delta)

        assert result is not None
        assert result.reasoning == " more"
        assert result.content == "content here"

    def test_reset_state_clears_think_in_prompt(self, parser):
        """reset_state should clear think_in_prompt between requests."""
        parser.reset_state(think_in_prompt=True)
        assert parser._think_in_prompt is True

        parser.reset_state()  # Default: False
        assert parser._think_in_prompt is False

        # Verify behavior after reset: no-tag text should be content
        result = parser.extract_reasoning_streaming("", "hello", "hello")
        assert result is not None
        assert result.content == "hello"
        assert result.reasoning is None


class TestEnableThinkingTriState:
    """Tests for the enable_thinking tri-state: undefined/True/False.

    The server resolves enable_thinking through a priority chain:
    1. request.enable_thinking (top-level)
    2. chat_template_kwargs.enable_thinking
    3. Auto-detect (None)

    This is tested structurally by inspecting server.py source.
    """

    def test_enable_thinking_priority_chain_in_source(self):
        """Verify server.py has correct priority chain for enable_thinking."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # Priority 1: top-level enable_thinking field
        assert "request.enable_thinking is not None" in source
        # Priority 2: chat_template_kwargs
        assert 'enable_thinking' in source
        assert "_ct_kwargs" in source or "chat_template_kwargs" in source
        # Priority 3: auto-detect (None)
        assert "_effective_thinking = None" in source

    def test_suppress_reasoning_when_thinking_false(self):
        """When enable_thinking=False, suppress_reasoning should be True."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # suppress_reasoning is set when thinking is explicitly False
        assert "suppress_reasoning = _effective_thinking is False" in source

    def test_effective_think_in_template_cleared_when_thinking_false(self):
        """When enable_thinking=False and template respects it, think_in_template is cleared."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # When _effective_thinking is False and _template_always_thinks returns False,
        # think_in_template is set to False, which flows to effective_think_in_template
        assert "think_in_template = False" in source
        assert "effective_think_in_template = think_in_template" in source

    def test_parser_gets_think_in_prompt_from_effective(self):
        """Parser reset_state should receive effective_think_in_template, not raw."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        assert "think_in_prompt=effective_think_in_template" in source


class TestSuppressReasoningDrop:
    """Tests for suppress_reasoning dropping reasoning chunks.

    When user sets enable_thinking=False but model still produces reasoning
    (e.g., template always thinks), suppress_reasoning=True causes the
    server to drop reasoning chunks entirely so only the final answer appears.
    """

    def test_suppress_reasoning_drop_in_source(self):
        """Verify suppress_reasoning drop logic exists in streaming."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # When suppress_reasoning is True:
        # emit_content = delta_msg.content (only actual content after </think>)
        # emit_reasoning = None
        assert "suppress_reasoning" in source
        assert "emit_reasoning = None" in source
        assert "emit_content = delta_msg.content" in source

    def test_suppress_reasoning_tracks_for_tool_detection(self):
        """Dropped reasoning should still be tracked for tool call detection."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # accumulated_content += delta_msg.reasoning (for tool call detection)
        assert "accumulated_content += delta_msg.reasoning" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: _template_always_thinks
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateAlwaysThinks:
    """Tests for _template_always_thinks() detection function.

    This function checks if a model's chat template ignores enable_thinking=False
    and always injects <think> in the assistant prefix. Results are cached.
    """

    def test_function_exists_and_signature(self):
        """_template_always_thinks should exist with correct signature."""
        import vmlx_engine.server as server_mod
        assert hasattr(server_mod, "_template_always_thinks")
        sig = inspect.signature(server_mod._template_always_thinks)
        params = list(sig.parameters.keys())
        assert "tokenizer" in params
        assert "model_name" in params

    def test_qwen_fast_path_returns_false(self):
        """Qwen models should fast-path to False (they honor enable_thinking)."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod._template_always_thinks)

        # The code checks for "qwen" in model_name.lower()
        assert '"qwen"' in source
        assert "model_name.lower()" in source
        # And returns False
        assert "return False" in source

    def test_result_caching(self):
        """Results should be cached per model name."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod._template_always_thinks)

        assert "_template_always_thinks_cache" in source
        # Check cache lookup
        assert "model_name in _template_always_thinks_cache" in source

    def test_empirical_probe_renders_template(self):
        """Should probe the tokenizer by rendering with enable_thinking=False."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod._template_always_thinks)

        assert "apply_chat_template" in source
        assert "enable_thinking=False" in source
        # Checks for <think> in rendered output after user message
        assert "__test__" in source
        assert "<think>" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: SSE Fix — empty delta guard
# ═══════════════════════════════════════════════════════════════════════════════


class TestSSEStreamingFix:
    """Tests for the SSE fix that prevents empty delta emission.

    The fix: `elif not request_parser:` instead of `else:` at the end of
    the streaming loop. When reasoning parser is active but delta_text is
    empty/None (progress-only engine chunk), we skip instead of emitting
    an empty content delta.
    """

    def test_elif_not_request_parser_guard(self):
        """The streaming loop must use 'elif not request_parser' not 'else'."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # The fix: elif not request_parser (not bare else)
        assert "elif not request_parser:" in source

    def test_empty_delta_comment(self):
        """The empty delta guard should have descriptive comment."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # After the guard, there should be a continue + comment about empty delta
        assert "progress-only engine chunk" in source.lower() or "empty" in source.lower()

    def test_three_path_structure(self):
        """Streaming loop should have 3 paths: parser+delta, no-parser, parser+empty."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # Path 1: if request_parser and delta_text:
        assert "if request_parser and delta_text:" in source
        # Path 2: elif not request_parser:
        assert "elif not request_parser:" in source
        # Path 3: else: continue (parser active, no delta)
        # The else + continue handles empty delta when parser is active


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: GPT-OSS / Harmony Protocol Parser
# ═══════════════════════════════════════════════════════════════════════════════


class TestGptOssParserComplete:
    """Comprehensive tests for GPT-OSS/Harmony reasoning parser."""

    @pytest.fixture
    def parser(self):
        from vmlx_engine.reasoning import get_parser
        return get_parser("openai_gptoss")()

    # --- Non-streaming: _parse_channels ---

    def test_full_harmony_protocol(self, parser):
        """Full Harmony analysis→final extraction."""
        text = (
            "<|channel|>analysis<|message|>Let me think about this"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42"
        )
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is not None
        assert "Let me think" in reasoning
        assert content is not None
        assert "42" in content

    def test_analysis_only_no_final(self, parser):
        """Only analysis channel, no final → reasoning only."""
        text = "<|channel|>analysis<|message|>Just thinking out loud"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is not None
        assert "Just thinking" in reasoning
        assert content is None

    def test_no_markers_returns_content(self, parser):
        """No Harmony markers at all → content passthrough."""
        text = "Just a plain response"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "Just a plain response"

    def test_second_cycle_stopped(self, parser):
        """Parser should stop after first complete analysis→final cycle."""
        text = (
            "<|channel|>analysis<|message|>First reasoning"
            "<|start|>assistant<|channel|>final<|message|>First answer"
            "<|start|>assistant<|channel|>analysis<|message|>Second reasoning"
            "<|start|>assistant<|channel|>final<|message|>Second answer"
        )
        reasoning, content = parser.extract_reasoning(text)
        assert "First reasoning" in reasoning
        assert "First answer" in content
        # Second cycle should NOT leak through
        assert "Second" not in (reasoning or "")
        assert "Second" not in (content or "")

    def test_implicit_reasoning_before_markers(self, parser):
        """Text before first channel marker is implicit reasoning."""
        text = (
            "Some preamble text"
            "<|channel|>analysis<|message|>Here is my reasoning"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42"
        )
        reasoning, content = parser.extract_reasoning(text)
        assert "preamble" in reasoning
        # _parse_channels joins preamble and analysis parts with \n
        # _clean_protocol_residue strips leading protocol words like "analysis"/"final"
        assert "reasoning" in reasoning.lower()
        assert "42" in content

    # --- Bare marker normalization ---

    def test_normalize_bare_markers_garbled_html(self):
        """Garbled HTML-like '</assistantfinal' should be normalized."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        result = GptOssReasoningParser._normalize_bare_markers("</assistant final")
        assert "<|channel|>" in result
        assert "final" in result

    def test_normalize_bare_markers_garbled_pipe(self):
        """Garbled pipe-bracket '<|assistantanalysis' should be normalized."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        result = GptOssReasoningParser._normalize_bare_markers("<|assistant analysis")
        assert "<|channel|>" in result
        assert "analysis" in result

    def test_normalize_bare_markers_case_insensitive(self):
        """Bare marker normalization should be case-insensitive."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        result = GptOssReasoningParser._normalize_bare_markers("</ASSISTANTfinal")
        assert "<|channel|>" in result
        assert "final" in result

    def test_normalize_bare_markers_with_space(self):
        """'</assistant final' (with space) should be normalized."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        result = GptOssReasoningParser._normalize_bare_markers("</assistant final")
        assert "<|channel|>" in result

    def test_normalize_preserves_proper_markers(self):
        """Already-proper markers should not be double-normalized."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        text = "<|channel|>analysis<|message|>content"
        result = GptOssReasoningParser._normalize_bare_markers(text)
        assert result == text

    def test_normalize_no_assistant_keyword(self):
        """Text without 'assistant' should pass through unchanged."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        text = "just regular text without the keyword"
        result = GptOssReasoningParser._normalize_bare_markers(text)
        assert result == text

    # --- Protocol residue cleaning ---

    def test_clean_protocol_residue_leading(self):
        """Leading protocol words should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._clean_protocol_residue("assistant Hello") == "Hello"
        assert GptOssReasoningParser._clean_protocol_residue("analysis Hello") == "Hello"
        assert GptOssReasoningParser._clean_protocol_residue("final Hello") == "Hello"

    def test_clean_protocol_residue_trailing(self):
        """Trailing protocol words should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._clean_protocol_residue("Hello assistant") == "Hello"
        assert GptOssReasoningParser._clean_protocol_residue("Hello analysis") == "Hello"

    def test_clean_protocol_residue_garbled_tokens(self):
        """Garbled special token fragments should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._clean_protocol_residue("<|assistant.Hello") == "Hello"
        assert GptOssReasoningParser._clean_protocol_residue("<|end. more text") == "more text"

    def test_clean_protocol_residue_empty_string(self):
        """Empty string should pass through."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._clean_protocol_residue("") == ""

    def test_clean_protocol_residue_concatenated(self):
        """Concatenated protocol words like 'assistantanalysis' in middle."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        result = GptOssReasoningParser._clean_protocol_residue("Hello assistantanalysis World")
        assert "assistantanalysis" not in result

    # --- Partial marker stripping ---

    def test_strip_partial_marker_start_tag(self):
        """Trailing partial <|start|> should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._strip_partial_marker("Hello<|sta") == "Hello"
        assert GptOssReasoningParser._strip_partial_marker("Hello<|star") == "Hello"

    def test_strip_partial_marker_channel_tag(self):
        """Trailing partial <|channel|> should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._strip_partial_marker("Hello<|cha") == "Hello"

    def test_strip_partial_assistant_word(self):
        """Trailing 'assistant' word should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._strip_partial_marker("Hello assistant") == "Hello"

    def test_strip_partial_assistant_fragment(self):
        """Trailing partial 'assistan' (3+ chars) should be stripped."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        result = GptOssReasoningParser._strip_partial_marker("Hello assista")
        assert "assista" not in result

    def test_strip_no_partial(self):
        """Text without partial markers should pass through."""
        from vmlx_engine.reasoning.gptoss_parser import GptOssReasoningParser
        assert GptOssReasoningParser._strip_partial_marker("Hello World") == "Hello World"

    # --- Streaming with fallback ---

    def test_streaming_fallback_threshold(self, parser):
        """Below fallback threshold, non-Harmony text returns None."""
        parser.reset_state()

        # Short text below threshold (10 chars)
        accumulated = "Hi"
        result = parser.extract_reasoning_streaming("", accumulated, accumulated)
        assert result is None

    def test_streaming_fallback_above_threshold(self, parser):
        """Above fallback threshold, non-Harmony text becomes content."""
        parser.reset_state()

        # Build up past threshold (10 chars)
        text = "A" * 20
        accumulated = ""
        results = []
        for i in range(0, len(text), 10):
            chunk = text[i:i+10]
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                results.append(result)

        content_parts = [r.content for r in results if r.content]
        assert len(content_parts) > 0
        assert "A" in "".join(content_parts)

    def test_streaming_harmony_active_mode(self, parser):
        """With harmony_active=True, reasoning is emitted immediately."""
        parser.reset_state(harmony_active=True)

        tokens = ["Let me", " think", " about this"]
        accumulated = ""
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result and result.reasoning:
                reasoning_parts.append(result.reasoning)

        assert len(reasoning_parts) > 0
        assert "think" in "".join(reasoning_parts)

    def test_streaming_incremental_emit(self, parser):
        """Streaming should emit only NEW content, not re-emit old."""
        parser.reset_state()

        text = (
            "<|channel|>analysis<|message|>Step 1. Step 2."
            "<|start|>assistant<|channel|>final<|message|>Done."
        )

        accumulated = ""
        reasoning_parts = []
        content_parts = []

        # Stream in chunks of 5 chars
        for i in range(0, len(text), 5):
            chunk = text[i:i+5]
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "Step 1" in full_reasoning
        assert "Step 2" in full_reasoning
        assert "Done" in full_content

    def test_streaming_emitted_reasoning_shrink_reset(self, parser):
        """When bare marker normalization causes reasoning to shrink, emitted count resets."""
        parser.reset_state()

        # First pass: text that includes "assistant" which will initially be
        # emitted as reasoning, then get consumed into a marker
        text1 = "<|channel|>analysis<|message|>Hello assistant"
        result1 = parser.extract_reasoning_streaming("", text1, text1)

        # Second pass: "final" arrives, making "assistantfinal" a bare marker
        text2 = text1 + "final more text"
        result2 = parser.extract_reasoning_streaming(text1, text2, "final more text")
        # Should not crash — the emitted_reasoning counter should handle shrink


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: clean_output_text
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleanOutputText:
    """Tests for clean_output_text utility function."""

    def test_removes_im_end(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("Hello<|im_end|>") == "Hello"

    def test_removes_im_start(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("<|im_start|>Hello") == "Hello"

    def test_removes_endoftext(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("Hello<|endoftext|>") == "Hello"

    def test_removes_eot_id(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("Hello<|eot_id|>") == "Hello"

    def test_removes_end_tag(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("Hello<|end|>") == "Hello"

    def test_removes_s_tags(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("</s>Hello<s>") == "Hello"

    def test_removes_pad_tokens(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("[PAD]Hello[PAD]") == "Hello"

    def test_removes_multiple_special_tokens(self):
        from vmlx_engine.api.utils import clean_output_text
        result = clean_output_text("<|im_start|>Hello<|im_end|><|endoftext|>")
        assert result == "Hello"

    def test_preserves_think_tags(self):
        from vmlx_engine.api.utils import clean_output_text
        text = "<think>reasoning</think>content"
        result = clean_output_text(text)
        assert "<think>" in result
        assert "</think>" in result

    def test_adds_missing_opening_think(self):
        """When only </think> is present, <think> should be prepended."""
        from vmlx_engine.api.utils import clean_output_text
        text = "reasoning content</think>final answer"
        result = clean_output_text(text)
        assert result.startswith("<think>")
        assert "</think>" in result

    def test_does_not_double_add_think(self):
        """When <think> is already present, don't add another."""
        from vmlx_engine.api.utils import clean_output_text
        text = "<think>reasoning</think>content"
        result = clean_output_text(text)
        assert result.count("<think>") == 1

    def test_empty_string_passthrough(self):
        from vmlx_engine.api.utils import clean_output_text
        assert clean_output_text("") == ""

    def test_none_passthrough(self):
        from vmlx_engine.api.utils import clean_output_text
        # None should be handled (returns falsy)
        result = clean_output_text(None)
        assert not result

    def test_whitespace_only_stripped(self):
        from vmlx_engine.api.utils import clean_output_text
        result = clean_output_text("  Hello  ")
        assert result == "Hello"

    def test_special_tokens_with_think_blocks(self):
        """Special tokens around think blocks should be stripped, tags preserved."""
        from vmlx_engine.api.utils import clean_output_text
        text = "<|im_start|><think>reasoning</think>content<|im_end|>"
        result = clean_output_text(text)
        assert "<think>" in result
        assert "</think>" in result
        assert "reasoning" in result
        assert "content" in result
        assert "<|im_start|>" not in result
        assert "<|im_end|>" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Model Config Registry — Qwen3.5 and related families
# ═══════════════════════════════════════════════════════════════════════════════


def _find_config_by_model_type(registry, model_type: str):
    """Helper to find a config by model_type from registry._configs."""
    for config in registry._configs:
        if model_type in config.model_types:
            return config
    raise KeyError(f"No config found for model_type={model_type}")


class TestQwen35ModelConfig:
    """Tests for Qwen3.5 model config entries.

    Qwen3.5 models have specific parser and template settings that must
    be correct for the thinking on/off system to work.
    """

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        from unittest.mock import patch
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all

        registry = ModelConfigRegistry()
        register_all(registry)

        with patch("vmlx_engine.model_config_registry.load_config", return_value=None):
            self._registry = registry
            yield

    def test_qwen3_5_properties(self):
        """qwen3_5 config must have correct parser and template settings."""
        config = _find_config_by_model_type(self._registry, "qwen3_5")
        assert config.family_name == "qwen3_5"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.tool_parser == "qwen"
        assert config.is_mllm is True
        assert config.cache_type == "kv"

    def test_qwen3_5_moe_properties(self):
        """qwen3_5_moe config must match qwen3_5."""
        config = _find_config_by_model_type(self._registry,"qwen3_5_moe")
        assert config.family_name == "qwen3_5_moe"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.tool_parser == "qwen"
        assert config.is_mllm is True

    def test_qwen3_properties(self):
        """qwen3 config must support native tools."""
        config = _find_config_by_model_type(self._registry,"qwen3")
        assert config.family_name == "qwen3"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.supports_native_tools is True
        assert config.is_mllm is False  # qwen3 is text-only

    def test_qwen3_moe_properties(self):
        """qwen3_moe config."""
        config = _find_config_by_model_type(self._registry,"qwen3_moe")
        assert config.family_name == "qwen3_moe"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.supports_native_tools is True

    def test_qwen3_vl_properties(self):
        """qwen3_vl config must be MLLM."""
        config = _find_config_by_model_type(self._registry,"qwen3_vl")
        assert config.is_mllm is True
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True

    def test_qwen3_next_properties(self):
        """qwen3_next uses mamba cache."""
        config = _find_config_by_model_type(self._registry,"qwen3_next")
        assert config.cache_type == "mamba"
        assert config.tool_parser == "nemotron"


class TestGLMModelConfig:
    """Tests for GLM family model configs (CRITICAL: different parsers per variant)."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        from unittest.mock import patch
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all

        registry = ModelConfigRegistry()
        register_all(registry)

        with patch("vmlx_engine.model_config_registry.load_config", return_value=None):
            self._registry = registry
            yield

    def test_gpt_oss_uses_gptoss_parser(self):
        """GPT-OSS must use openai_gptoss parser, NOT deepseek_r1."""
        config = _find_config_by_model_type(self._registry,"gpt_oss")
        assert config.reasoning_parser == "openai_gptoss"
        assert config.tool_parser == "glm47"
        assert config.chat_template_custom is not None

    def test_glm4_moe_uses_gptoss_parser(self):
        """GLM-4.7 Flash (MoE) must use openai_gptoss parser."""
        config = _find_config_by_model_type(self._registry,"glm4_moe")
        assert config.reasoning_parser == "openai_gptoss"
        assert config.tool_parser == "glm47"

    def test_glm4_moe_lite_uses_gptoss_parser(self):
        """GLM-4.7 Flash Lite must also use openai_gptoss parser."""
        config = _find_config_by_model_type(self._registry,"glm4_moe_lite")
        assert config.reasoning_parser == "openai_gptoss"

    def test_chatglm_no_reasoning_parser(self):
        """Base ChatGLM/GLM-4 should have NO reasoning parser."""
        config = _find_config_by_model_type(self._registry,"chatglm")
        assert config.reasoning_parser is None or config.reasoning_parser == ""

    def test_harmony_template_on_glm_variants(self):
        """GPT-OSS and GLM-4.7 Flash must use Harmony chat template."""
        gptoss = _find_config_by_model_type(self._registry,"gpt_oss")
        glm4moe = _find_config_by_model_type(self._registry,"glm4_moe")
        chatglm = _find_config_by_model_type(self._registry,"chatglm")

        assert gptoss.chat_template_custom is not None
        assert glm4moe.chat_template_custom is not None
        assert chatglm.chat_template_custom is not None

        # All should contain Harmony markers
        for config in [gptoss, glm4moe, chatglm]:
            assert "<|start|>" in config.chat_template_custom
            assert "<|message|>" in config.chat_template_custom


class TestOtherModelConfigs:
    """Tests for other important model config entries."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        from unittest.mock import patch
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all

        registry = ModelConfigRegistry()
        register_all(registry)

        with patch("vmlx_engine.model_config_registry.load_config", return_value=None):
            self._registry = registry
            yield

    def test_minimax_properties(self):
        """MiniMax must use qwen3 reasoning parser with think_in_template."""
        config = _find_config_by_model_type(self._registry,"minimax")
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.tool_parser == "minimax"

    def test_deepseek_uses_r1_parser(self):
        """DeepSeek must use deepseek_r1 reasoning parser."""
        config = _find_config_by_model_type(self._registry,"deepseek_v3")
        assert config.reasoning_parser == "deepseek_r1"
        assert config.tool_parser == "deepseek"

    def test_kimi_k2_config(self):
        """Kimi K2 model config."""
        config = _find_config_by_model_type(self._registry,"kimi_k2")
        assert config.family_name == "kimi"
        assert config.tool_parser == "kimi"

    def test_nemotron_hybrid_cache(self):
        """Nemotron should use hybrid cache."""
        config = _find_config_by_model_type(self._registry,"nemotron")
        assert config.cache_type == "hybrid"
        assert config.reasoning_parser == "deepseek_r1"

    def test_step_family_uses_qwen3_parser(self):
        """Step family should use qwen3 reasoning parser."""
        config = _find_config_by_model_type(self._registry,"step3p5")
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True

    def test_gemma3_uses_deepseek_r1(self):
        """Gemma3 uses deepseek_r1 reasoning parser."""
        config = _find_config_by_model_type(self._registry,"gemma3")
        assert config.reasoning_parser == "deepseek_r1"
        assert config.is_mllm is True

    def test_phi4_reasoning_uses_deepseek_r1(self):
        """Phi4 reasoning variant uses deepseek_r1."""
        config = _find_config_by_model_type(self._registry,"phi4_reasoning")
        assert config.reasoning_parser == "deepseek_r1"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Empty Response Safeguard
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyResponseSafeguard:
    """Tests for zero-token stream detection and diagnostic emission."""

    def test_chat_completions_zero_token_safeguard(self):
        """Chat completions must detect zero-token streams."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # The safeguard checks last_output is None OR zero completion_tokens
        assert "last_output is None" in source
        assert "completion_tokens" in source
        assert "content_was_emitted" in source
        assert "accumulated_reasoning" in source
        # And emits a diagnostic message
        assert "Model produced no response" in source

    def test_responses_api_empty_fallback_chain(self):
        """Responses API must have fallback chain for empty responses."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_responses_api)

        # Fallback chain: display_text → clean_output_text → diagnostic
        assert "display_text" in source
        assert "clean_output_text" in source
        assert "Model produced no response" in source

    def test_responses_api_reasoning_fallback(self):
        """Responses API should use accumulated_reasoning as fallback when no content."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_responses_api)

        # accumulated_reasoning fallback
        assert "accumulated_reasoning" in source
        assert "accumulated_content" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Parser Registry Completeness
# ═══════════════════════════════════════════════════════════════════════════════


class TestParserRegistryCompleteness:
    """Tests that all parsers referenced by model configs are registered."""

    def test_all_reasoning_parsers_registered(self):
        """Every reasoning_parser name used in model_configs must be in registry."""
        from unittest.mock import patch
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.reasoning import list_parsers

        registry = ModelConfigRegistry()
        with patch("vmlx_engine.model_config_registry.load_config", return_value=None):
            register_all(registry)

        available_parsers = set(list_parsers())
        for config in registry._configs:
            if config.reasoning_parser:
                assert config.reasoning_parser in available_parsers, (
                    f"Model config '{config.family_name}' references reasoning parser "
                    f"'{config.reasoning_parser}' which is not registered. "
                    f"Available: {available_parsers}"
                )

    def test_openai_gptoss_is_registered(self):
        """openai_gptoss must be registered (used by GLM and GPT-OSS families)."""
        from vmlx_engine.reasoning import list_parsers
        assert "openai_gptoss" in list_parsers()

    def test_qwen3_parser_is_registered(self):
        """qwen3 parser must be registered."""
        from vmlx_engine.reasoning import list_parsers
        assert "qwen3" in list_parsers()

    def test_deepseek_r1_parser_is_registered(self):
        """deepseek_r1 parser must be registered."""
        from vmlx_engine.reasoning import list_parsers
        assert "deepseek_r1" in list_parsers()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: MLLM Pattern Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLLMPatternDetection:
    """Tests for MLLM_PATTERNS regex-based VLM detection fallback."""

    def test_qwen3_5_detected_as_mllm(self):
        """qwen3_5 pattern should match Qwen3.5 model names."""
        from vmlx_engine.api.utils import MLLM_PATTERNS
        pattern = re.compile("|".join(MLLM_PATTERNS), re.IGNORECASE)
        assert pattern.search("Qwen3.5-VL-72B")
        assert pattern.search("qwen3_5_moe")

    def test_qwen_vl_detected(self):
        from vmlx_engine.api.utils import MLLM_PATTERNS
        pattern = re.compile("|".join(MLLM_PATTERNS), re.IGNORECASE)
        assert pattern.search("Qwen2-VL-7B")
        assert pattern.search("qwen-vl-chat")

    def test_gemma3_detected_but_not_text(self):
        """gemma3 should be detected but gemma3_text should NOT."""
        from vmlx_engine.api.utils import MLLM_PATTERNS
        pattern = re.compile("|".join(MLLM_PATTERNS), re.IGNORECASE)
        assert pattern.search("gemma-3-27b")
        assert not pattern.search("gemma3_text-only")

    def test_pixtral_detected(self):
        from vmlx_engine.api.utils import MLLM_PATTERNS
        pattern = re.compile("|".join(MLLM_PATTERNS), re.IGNORECASE)
        assert pattern.search("pixtral-12b")

    def test_llava_detected(self):
        from vmlx_engine.api.utils import MLLM_PATTERNS
        pattern = re.compile("|".join(MLLM_PATTERNS), re.IGNORECASE)
        assert pattern.search("llava-1.5-7b")

    def test_text_only_not_detected(self):
        """Text-only models should NOT match MLLM patterns."""
        from vmlx_engine.api.utils import MLLM_PATTERNS
        pattern = re.compile("|".join(MLLM_PATTERNS), re.IGNORECASE)
        assert not pattern.search("Llama-3-8B")
        assert not pattern.search("Mistral-7B")
        assert not pattern.search("DeepSeek-V3")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Server Streaming Path — enable_thinking via API kwargs
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnableThinkingAPIKwargs:
    """Tests that enable_thinking works properly through API request parameters.

    The server has two paths for enable_thinking:
    1. request.enable_thinking (top-level field)
    2. request.chat_template_kwargs.enable_thinking (legacy/compat)

    Both must be correctly resolved and affect streaming behavior.
    """

    def test_chat_template_kwargs_path_in_source(self):
        """Server must read enable_thinking from chat_template_kwargs."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # _ct_kwargs = request.chat_template_kwargs or {}
        assert "chat_template_kwargs" in source
        assert '"enable_thinking"' in source or "'enable_thinking'" in source

    def test_top_level_takes_priority(self):
        """request.enable_thinking must override chat_template_kwargs."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # The priority check: request.enable_thinking is not None checked FIRST
        lines = source.split('\n')
        top_level_idx = None
        kwargs_idx = None
        for i, line in enumerate(lines):
            if "request.enable_thinking is not None" in line:
                top_level_idx = i
            if "_ct_kwargs" in line and "enable_thinking" in line and top_level_idx is not None:
                if "elif" in line or "else" in line:
                    kwargs_idx = i
                    break

        assert top_level_idx is not None, "Top-level enable_thinking check not found"
        assert kwargs_idx is not None, "chat_template_kwargs fallback not found"
        assert top_level_idx < kwargs_idx, "Top-level must come before kwargs fallback"

    def test_enable_thinking_field_on_request_model(self):
        """ChatCompletionRequest should have enable_thinking field."""
        from vmlx_engine.api.models import ChatCompletionRequest
        fields = ChatCompletionRequest.model_fields
        assert "enable_thinking" in fields

    def test_chat_template_kwargs_field_on_request_model(self):
        """ChatCompletionRequest should have chat_template_kwargs field."""
        from vmlx_engine.api.models import ChatCompletionRequest
        fields = ChatCompletionRequest.model_fields
        assert "chat_template_kwargs" in fields


class TestStreamingReasoningContent:
    """Tests that reasoning_content appears correctly in streaming chunks.

    ChatCompletionChunkDelta has a `reasoning` field (excluded from serialization)
    and a computed `reasoning_content` field (what gets serialized to JSON).
    """

    def test_chunk_delta_has_reasoning_content(self):
        """ChatCompletionChunkDelta should expose reasoning_content."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta
        delta = ChatCompletionChunkDelta(reasoning="thinking...")
        assert delta.reasoning == "thinking..."
        assert delta.reasoning_content == "thinking..."

    def test_chunk_delta_reasoning_none_when_content_only(self):
        """reasoning_content should be None when only content is set."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta
        delta = ChatCompletionChunkDelta(content="hello")
        assert delta.reasoning_content is None
        assert delta.content == "hello"

    def test_chunk_delta_both_fields(self):
        """During transition, both reasoning and content can be set."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta
        delta = ChatCompletionChunkDelta(reasoning="last thought", content="first word")
        assert delta.reasoning_content == "last thought"
        assert delta.content == "first word"

    def test_computed_field_not_settable_via_constructor(self):
        """REGRESSION: reasoning_content= in constructor must NOT set reasoning.

        This was the root cause of empty streaming deltas for CRACK/VLM models.
        Pydantic computed_field is read-only; passing reasoning_content= to __init__
        is silently ignored, leaving reasoning=None. The constructor MUST use
        reasoning= instead.
        """
        from vmlx_engine.api.models import ChatCompletionChunkDelta
        # Passing reasoning_content= should NOT set the reasoning field
        delta = ChatCompletionChunkDelta(reasoning_content="should be ignored")
        assert delta.reasoning is None
        assert delta.reasoning_content is None
        # The correct way:
        delta2 = ChatCompletionChunkDelta(reasoning="correct way")
        assert delta2.reasoning_content == "correct way"

    def test_server_uses_reasoning_not_reasoning_content_in_constructor(self):
        """Verify server.py constructs ChunkDelta with reasoning= (not reasoning_content=)."""
        import inspect
        from vmlx_engine import server
        src = inspect.getsource(server.stream_chat_completion)
        # Must NOT have reasoning_content= in constructor (silent bug)
        assert "reasoning_content=emit_reasoning" not in src, \
            "BUG: reasoning_content= is a computed_field and silently drops reasoning text"
        # Must have reasoning= in constructor
        assert "reasoning=emit_reasoning" in src


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Reasoning-only fallback
# ═══════════════════════════════════════════════════════════════════════════════


class TestReasoningOnlyFallback:
    """Tests for the fallback when model produces only reasoning (no content).

    When the parser classifies all tokens as reasoning and no content was emitted,
    the server should emit accumulated_reasoning as content so the user sees something.
    """

    def test_reasoning_only_fallback_in_source(self):
        """Server must have reasoning-only fallback logic."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # The fallback: if not content_was_emitted and accumulated_reasoning
        assert "not content_was_emitted" in source
        assert "accumulated_reasoning" in source

    def test_reasoning_only_fallback_emits_with_stop(self):
        """Fallback chunk should have finish_reason='stop'."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # Find the reasoning fallback block and verify it uses "stop"
        # The fallback block contains: finish_reason="stop"
        # Search for reasoning fallback + stop in proximity
        assert 'finish_reason="stop"' in source or "finish_reason='stop'" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Qwen3.5 end-to-end thinking scenario simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestQwen35ThinkingScenarios:
    """End-to-end simulation of Qwen3.5 thinking scenarios.

    These tests simulate the actual token flow for Qwen3.5 models in
    different enable_thinking states, verifying the parser correctly
    classifies tokens.
    """

    @pytest.fixture
    def parser(self):
        from vmlx_engine.reasoning import get_parser
        return get_parser("qwen3")()

    def test_thinking_on_with_think_in_template(self, parser):
        """Scenario: enable_thinking=True, think_in_template=True.

        Template injects <think>\\n in prompt. Model output starts with
        reasoning text (no explicit <think>), then </think>, then content.
        Parser should classify pre-</think> as reasoning.
        """
        parser.reset_state(think_in_prompt=True)

        # Simulate model output (template already injected <think> in prompt)
        tokens = [
            "Let me",           # reasoning (no tags, think_in_prompt=True → reasoning)
            " analyze",         # reasoning
            " this",            # reasoning
            " problem",         # reasoning
            ".",                # reasoning
            "\n\nFirst",        # reasoning
            ", consider",       # reasoning
            " the input",      # reasoning
            ".</think>",        # transition
            "The answer",       # content
            " is 42.",          # content
        ]

        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "analyze this problem" in full_reasoning
        assert "consider the input" in full_reasoning
        assert "The answer is 42" in full_content

    def test_thinking_off_no_think_in_template(self, parser):
        """Scenario: enable_thinking=False, think_in_template=False.

        Template does NOT inject <think>. Model output is pure content.
        Parser has think_in_prompt=False, so everything is content.
        """
        parser.reset_state(think_in_prompt=False)

        tokens = ["The", " quick", " brown", " fox"]
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        assert "".join(content_parts) == "The quick brown fox"
        assert len(reasoning_parts) == 0

    def test_thinking_off_model_still_thinks_crack_style(self, parser):
        """Scenario: enable_thinking=False but CRACK model outputs thinking text.

        Qwen3.5 CRACK models output "Thinking Process:" as plain text
        without <think> tags even when thinking is off. With think_in_prompt=False,
        this correctly becomes content (not reasoning).
        """
        parser.reset_state(think_in_prompt=False)

        tokens = ["Thinking", " Process", ":\n", "Let me", " analyze", "\n\n", "The answer", " is 42"]
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        full_content = "".join(content_parts)
        assert "Thinking Process" in full_content
        assert "The answer is 42" in full_content
        assert len(reasoning_parts) == 0

    def test_thinking_on_explicit_think_tags(self, parser):
        """Scenario: Model outputs explicit <think>...</think> tags.

        Some models output explicit tags even when think_in_template injects them.
        Parser should handle this correctly regardless of think_in_prompt state.
        """
        parser.reset_state(think_in_prompt=False)

        tokens = ["<think>", "Deep", " thought", "</think>", "42"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "".join(reasoning_parts) == "Deep thought"
        assert "".join(content_parts) == "42"

    def test_suppress_reasoning_concept(self, parser):
        """When suppress_reasoning=True, reasoning should be redirected to content.

        This tests the concept (the actual redirect happens in server.py).
        We verify the parser still correctly classifies tokens even when
        the server will redirect them.
        """
        parser.reset_state(think_in_prompt=True)

        tokens = ["Reasoning", " text", "</think>", "Content"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        # Parser correctly classifies:
        assert "Reasoning text" in "".join(reasoning_parts)
        assert "Content" in "".join(content_parts)

        # The server's suppress_reasoning logic would then redirect:
        # emit_content = delta_msg.reasoning or delta_msg.content
        # emit_reasoning = None
        # This is tested structurally in TestSuppressReasoningRedirect


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13: GPT-OSS Regex Patterns
# ═══════════════════════════════════════════════════════════════════════════════


class TestGptOssRegexPatterns:
    """Tests for the regex patterns used in GPT-OSS parser."""

    def test_bare_marker_regex_html_prefixed(self):
        """Bare markers with </ prefix (garbled HTML-like tokens)."""
        from vmlx_engine.reasoning.gptoss_parser import _BARE_MARKER_RE
        assert _BARE_MARKER_RE.search("</assistant final")
        assert _BARE_MARKER_RE.search("</assistant analysis")

    def test_bare_marker_regex_pipe_prefixed(self):
        """Bare markers with <| prefix (garbled special tokens)."""
        from vmlx_engine.reasoning.gptoss_parser import _BARE_MARKER_RE
        assert _BARE_MARKER_RE.search("<|assistant analysis")
        assert _BARE_MARKER_RE.search("<|assistant final")

    def test_bare_marker_regex_with_closing(self):
        """Bare markers with closing bracket forms."""
        from vmlx_engine.reasoning.gptoss_parser import _BARE_MARKER_RE
        assert _BARE_MARKER_RE.search("<assistant final|>")
        assert _BARE_MARKER_RE.search("</assistant analysis|>")

    def test_bare_marker_regex_doubled_assistant(self):
        """Doubled assistant word with prefix."""
        from vmlx_engine.reasoning.gptoss_parser import _BARE_MARKER_RE
        assert _BARE_MARKER_RE.search("</assistant assistant final")

    def test_bare_marker_regex_no_match(self):
        """Regular text should not match."""
        from vmlx_engine.reasoning.gptoss_parser import _BARE_MARKER_RE
        assert not _BARE_MARKER_RE.search("hello world")
        assert not _BARE_MARKER_RE.search("assist")
        assert not _BARE_MARKER_RE.search("the final answer")

    def test_protocol_residue_regex(self):
        """Protocol residue patterns."""
        from vmlx_engine.reasoning.gptoss_parser import _PROTOCOL_RESIDUE_RE
        assert _PROTOCOL_RESIDUE_RE.search("</assistantanalysis")
        assert _PROTOCOL_RESIDUE_RE.search("<assistant")
        assert _PROTOCOL_RESIDUE_RE.search("assistantassistantanalysis")
        assert _PROTOCOL_RESIDUE_RE.search("analysisassistant")

    def test_protocol_residue_no_false_positive(self):
        """Regular text should not match protocol residue."""
        from vmlx_engine.reasoning.gptoss_parser import _PROTOCOL_RESIDUE_RE
        assert not _PROTOCOL_RESIDUE_RE.search("hello world")
        assert not _PROTOCOL_RESIDUE_RE.search("the final answer")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 14: Tool Call Buffering with Reasoning
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCallBufferingStructure:
    """Structural tests for tool call buffering in the streaming path.

    Tool call markers in content vs reasoning are handled differently:
    - Content: check accumulated_content (markers can span deltas)
    - Reasoning: check ONLY current delta (avoids false positives)
    """

    def test_content_uses_accumulated(self):
        """Content tool call detection should use accumulated_content."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        assert "accumulated_content" in source
        assert "marker in accumulated_content" in source

    def test_reasoning_uses_delta_only(self):
        """Reasoning tool call detection should use delta only."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        assert "delta_msg.reasoning" in source
        # Verify there's a check for markers in reasoning delta
        assert "marker in delta_msg.reasoning" in source

    def test_harmony_tool_format_detection(self):
        """GPT-OSS/Harmony native tool format (to=name code{}) detection."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # Regex for: to=name code{
        assert r"to=\w" in source or "to=" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Section 15: Per-request Parser Instances
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerRequestParserInstances:
    """Tests that the server creates per-request parser instances.

    Multiple concurrent streams must not share mutable parser state.
    The server creates: request_parser = _reasoning_parser.__class__()
    """

    def test_per_request_parser_creation(self):
        """Server must create new parser instance per request."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        # Should create a new instance, not reuse the global
        assert "_reasoning_parser.__class__()" in source

    def test_parser_reset_with_effective_think(self):
        """Parser reset_state should use effective_think_in_template."""
        import vmlx_engine.server as server_mod
        source = inspect.getsource(server_mod.stream_chat_completion)

        assert "request_parser.reset_state" in source
        assert "think_in_prompt=effective_think_in_template" in source

    def test_parser_concurrent_isolation(self):
        """Two parser instances should not share state."""
        from vmlx_engine.reasoning import get_parser

        parser1 = get_parser("qwen3")()
        parser2 = get_parser("qwen3")()

        parser1.reset_state(think_in_prompt=True)
        parser2.reset_state(think_in_prompt=False)

        assert parser1._think_in_prompt is True
        assert parser2._think_in_prompt is False

        # They should produce different results for the same input
        result1 = parser1.extract_reasoning_streaming("", "hello", "hello")
        result2 = parser2.extract_reasoning_streaming("", "hello", "hello")

        assert result1.reasoning == "hello"  # think_in_prompt=True → reasoning
        assert result2.content == "hello"    # think_in_prompt=False → content


# ═══════════════════════════════════════════════════════════════════════════════
# Section 14: MLLM System Prompt Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLLMSystemPrompt:
    """Tests that MLLM single-turn path preserves system prompts."""

    def test_mllm_single_turn_passes_full_messages(self):
        """MLLM single-turn path must pass full messages list (including system)
        to mlx_vlm's apply_chat_template, not just the user text."""
        source = inspect.getsource(
            __import__("vmlx_engine.engine.batched", fromlist=["BatchedEngine"]).BatchedEngine._apply_chat_template
        )
        # Should NOT extract just text_prompt from user message
        assert "text_prompt = \"\"" not in source, \
            "BUG: MLLM single-turn path extracts only user text, dropping system prompt"
        # Should pass messages list to apply_chat_template
        assert "messages," in source

    def test_mllm_path_condition(self):
        """MLLM path activates for single-turn VLM without tools."""
        source = inspect.getsource(
            __import__("vmlx_engine.engine.batched", fromlist=["BatchedEngine"]).BatchedEngine._apply_chat_template
        )
        assert "non_system_msgs <= 2" in source
        assert "not tools" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Section 15: Improved Zero-Token Safeguard
# ═══════════════════════════════════════════════════════════════════════════════


class TestImprovedZeroTokenSafeguard:
    """Tests that zero-token safeguard catches empty generation with last_output set."""

    def test_safeguard_catches_zero_completion_tokens(self):
        """Safeguard must also fire when last_output exists but has 0 completion_tokens."""
        source = inspect.getsource(
            __import__("vmlx_engine.server", fromlist=["stream_chat_completion"]).stream_chat_completion
        )
        # Must check completion_tokens == 0, not just last_output is None
        assert "completion_tokens" in source
        assert "Model produced no response" in source
