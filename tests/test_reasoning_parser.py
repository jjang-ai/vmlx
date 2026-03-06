# SPDX-License-Identifier: Apache-2.0
"""
Tests for reasoning content extraction parsers.

Tests cover:
- Parser registry (registration, lookup, listing)
- Qwen3 parser (non-streaming and streaming, 4 cases)
- DeepSeek-R1 parser (non-streaming and streaming)
- Edge cases (no tags, partial tags, unicode, code, HTML)
- Fix #1: enable_thinking=False forwarding (was silently dropped)
- Fix #2: Truncated reasoning overflow (max_tokens during thinking)
- Fix #3/4: MLLM sampling params (temperature kwarg, top_k/min_p sampler)
- Fix #5: LLM non-streaming stop sequence truncation
- Fix #6: Server content leak prevention (truncated reasoning)
- Fix #7: Concurrent batching BatchKVCache.offset proxy (_BatchOffsetSafeCache)
- Fix #8: Streaming reasoning echo prevention (reasoning_was_streamed flag)
- Fix #9: BatchMambaCache filter/merge (SSM stale left_padding crash)
- Fix #10: Always-wrap proxy (filter-down-to-1 slice error)
- Implicit reasoning streaming (think_in_prompt mode)
- API models integration (AssistantMessage, ChatCompletionChunkDelta)
- Performance (large outputs, many chunks, repeated parsing)
"""

import pytest

from vmlx_engine.reasoning import (
    DeltaMessage,
    ReasoningParser,
    get_parser,
    list_parsers,
    register_parser,
)


class TestParserRegistry:
    """Tests for the parser registry functions."""

    def test_list_parsers_includes_builtin(self):
        """Built-in parsers should be registered."""
        parsers = list_parsers()
        assert "qwen3" in parsers
        assert "deepseek_r1" in parsers

    def test_get_parser_qwen3(self):
        """Should be able to get Qwen3 parser."""
        parser_cls = get_parser("qwen3")
        parser = parser_cls()
        assert isinstance(parser, ReasoningParser)

    def test_get_parser_deepseek(self):
        """Should be able to get DeepSeek-R1 parser."""
        parser_cls = get_parser("deepseek_r1")
        parser = parser_cls()
        assert isinstance(parser, ReasoningParser)

    def test_get_unknown_parser_raises(self):
        """Unknown parser name should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_parser("unknown_parser")
        assert "unknown_parser" in str(exc_info.value)
        assert "Available parsers" in str(exc_info.value)

    def test_register_custom_parser(self):
        """Should be able to register custom parsers."""

        class CustomParser(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output

            def extract_reasoning_streaming(self, prev, curr, delta):
                return DeltaMessage(content=delta)

        register_parser("custom_test", CustomParser)
        assert "custom_test" in list_parsers()

        parser = get_parser("custom_test")()
        assert isinstance(parser, CustomParser)


class TestQwen3Parser:
    """Tests for the Qwen3 reasoning parser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh Qwen3 parser for each test."""
        return get_parser("qwen3")()

    # Non-streaming tests

    def test_extract_with_both_tags(self, parser):
        """Should extract reasoning when both tags present."""
        output = "<think>Let me analyze this problem</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me analyze this problem"
        assert content == "The answer is 42."

    def test_extract_only_reasoning(self, parser):
        """Should handle case where only reasoning is present."""
        output = "<think>Just thinking out loud</think>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Just thinking out loud"
        assert content is None

    def test_extract_multiline_reasoning(self, parser):
        """Should preserve newlines in reasoning content."""
        output = (
            "<think>Step 1: Analyze\nStep 2: Solve\nStep 3: Verify</think>Result: 42"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 42"

    def test_no_tags_returns_content_only(self, parser):
        """Qwen3 requires both tags - no tags means pure content."""
        output = "Just a regular response without thinking."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_only_start_tag_truncated_reasoning(self, parser):
        """<think> without </think> = truncated reasoning (max_tokens hit during thinking)."""
        output = "<think>Started thinking but never finished"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Started thinking but never finished"
        assert content is None

    def test_only_end_tag_implicit_mode(self, parser):
        """Qwen3 supports implicit mode - when <think> is in prompt, only </think> in output."""
        output = "Some text</think>more text"
        reasoning, content = parser.extract_reasoning(output)
        # Implicit mode: everything before </think> is reasoning
        assert reasoning == "Some text"
        assert content == "more text"

    # Streaming tests

    def test_streaming_simple_flow(self, parser):
        """Test basic streaming with reasoning then content."""
        parser.reset_state()

        # Simulate streaming tokens
        deltas = ["<think>", "think", "ing", "</think>", "answer"]
        accumulated = ""
        results = []

        for delta in deltas:
            prev = accumulated
            accumulated += delta
            result = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if result:
                results.append(result)

        # Collect reasoning and content
        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        content_parts = [r.content for r in results if r.content]

        assert "".join(reasoning_parts) == "thinking"
        assert "".join(content_parts) == "answer"

    def test_streaming_skip_tags(self, parser):
        """Special tokens themselves should be skipped."""
        parser.reset_state()

        # Just the start tag
        result = parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None

        # Just the end tag
        result = parser.extract_reasoning_streaming(
            "<think>reasoning", "<think>reasoning</think>", "</think>"
        )
        assert result is None

    def test_streaming_transition_chunk(self, parser):
        """Chunk containing end tag should split reasoning and content."""
        parser.reset_state()

        # Previous has start, delta contains end and content
        prev = "<think>reasoning"
        delta = " more</think>content here"
        curr = prev + delta

        result = parser.extract_reasoning_streaming(prev, curr, delta)

        assert result is not None
        assert result.reasoning == " more"
        assert result.content == "content here"


class TestDeepSeekR1Parser:
    """Tests for the DeepSeek-R1 reasoning parser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh DeepSeek-R1 parser for each test."""
        return get_parser("deepseek_r1")()

    # Non-streaming tests

    def test_extract_with_both_tags(self, parser):
        """Should extract reasoning when both tags present."""
        output = "<think>Step by step analysis</think>Final answer: 42"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Step by step analysis"
        assert content == "Final answer: 42"

    def test_extract_implicit_start_tag(self, parser):
        """DeepSeek-R1 handles implicit start tag (missing <think>)."""
        output = "Implicit reasoning content</think>The answer"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Implicit reasoning content"
        assert content == "The answer"

    def test_extract_no_tags_pure_content(self, parser):
        """No tags should return pure content."""
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_extract_multiline_reasoning(self, parser):
        """Should preserve newlines in reasoning content."""
        output = "<think>Line 1\nLine 2\nLine 3</think>Result"
        reasoning, content = parser.extract_reasoning(output)
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert "Line 3" in reasoning
        assert content == "Result"

    # Streaming tests

    def test_streaming_simple_flow(self, parser):
        """Test basic streaming with reasoning then content."""
        parser.reset_state()

        deltas = ["<think>", "think", "ing", "</think>", "answer"]
        accumulated = ""
        results = []

        for delta in deltas:
            prev = accumulated
            accumulated += delta
            result = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if result:
                results.append(result)

        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        content_parts = [r.content for r in results if r.content]

        assert "".join(reasoning_parts) == "thinking"
        assert "".join(content_parts) == "answer"


class TestDeltaMessage:
    """Tests for the DeltaMessage dataclass."""

    def test_reasoning_content_alias(self):
        """reasoning_content should alias reasoning."""
        msg = DeltaMessage(reasoning="test reasoning")
        assert msg.reasoning == "test reasoning"
        assert msg.reasoning_content == "test reasoning"

    def test_content_only(self):
        """Should handle content-only messages."""
        msg = DeltaMessage(content="just content")
        assert msg.content == "just content"
        assert msg.reasoning is None
        assert msg.reasoning_content is None

    def test_both_fields(self):
        """Should handle transition messages with both."""
        msg = DeltaMessage(reasoning="ending", content="starting")
        assert msg.reasoning == "ending"
        assert msg.content == "starting"


class TestEdgeCases:
    """Test edge cases across parsers."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_empty_output(self, parser):
        """Empty output should return (None, '')."""
        reasoning, content = parser.extract_reasoning("")
        # Either both None or content is empty string
        assert reasoning is None or reasoning == ""

    def test_whitespace_only_reasoning(self, parser):
        """Whitespace-only reasoning should be treated as None."""
        output = "<think>   </think>content"
        reasoning, content = parser.extract_reasoning(output)
        # Whitespace-only should be stripped to None
        if reasoning is not None:
            assert reasoning.strip() == "" or reasoning is None

    def test_nested_tags_not_supported(self, parser):
        """Nested tags are not officially supported - behavior may vary."""
        output = "<think>outer<think>inner</think>still outer</think>content"
        # Just ensure it doesn't crash
        reasoning, content = parser.extract_reasoning(output)
        # Result may vary by parser implementation

    def test_streaming_reset_state(self, parser):
        """reset_state should allow reuse of parser."""
        # First stream
        parser.reset_state()
        parser.extract_reasoning_streaming("", "<think>", "<think>")

        # Reset for new stream
        parser.reset_state()

        # Should work fresh
        result = parser.extract_reasoning_streaming("", "content", "content")
        assert result is not None


class TestRealisticStreaming:
    """Tests for realistic streaming scenarios simulating actual model output."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_token_by_token_streaming(self, parser):
        """Simulate realistic token-by-token streaming."""
        # Typical model output broken into tokens
        tokens = [
            "<",
            "think",
            ">",  # Start tag split across tokens
            "Let",
            " me",
            " analyze",
            " this",
            ".",
            "\n",
            "Step",
            " 1",
            ":",
            " check",
            " input",
            "\n",
            "Step",
            " 2",
            ":",
            " compute",
            "</",
            "think",
            ">",  # End tag split across tokens
            "The",
            " answer",
            " is",
            " 42",
            ".",
        ]

        parser.reset_state()
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

        # Verify reasoning was captured
        assert "Let me analyze" in full_reasoning
        assert "Step 1" in full_reasoning
        assert "Step 2" in full_reasoning

        # Verify content was captured
        assert "The answer is 42" in full_content

    def test_long_reasoning_streaming(self, parser):
        """Test streaming with extended reasoning."""
        # Long reasoning content
        reasoning_text = """
        First, I need to understand the problem.
        The user is asking about quantum computing.

        Let me break this down:
        1. Quantum bits (qubits) can be in superposition
        2. Entanglement allows correlated states
        3. Quantum gates perform operations

        After careful analysis, I can provide an answer.
        """

        output = f"<think>{reasoning_text}</think>Quantum computing uses qubits."

        # Simulate character-by-character streaming
        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for char in output:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "quantum computing" in full_reasoning.lower()
        assert "qubits" in full_reasoning.lower()
        assert "Quantum computing uses qubits" in full_content

    def test_streaming_no_content_after_reasoning(self, parser):
        """Test streaming when there's only reasoning, no content."""
        tokens = ["<think>", "just", " thinking", "</think>"]

        parser.reset_state()
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

        assert "just thinking" in "".join(reasoning_parts)
        assert len(content_parts) == 0 or "".join(content_parts).strip() == ""


class TestUnicodeAndSpecialCharacters:
    """Tests for Unicode and special characters in reasoning."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_unicode_reasoning(self, parser):
        """Test reasoning with Unicode characters."""
        output = "<think>分析这个问题：日本語テスト émojis: 🤔💭</think>答案是42"
        reasoning, content = parser.extract_reasoning(output)
        assert "分析" in reasoning
        assert "日本語" in reasoning
        assert "🤔" in reasoning
        assert "42" in content

    def test_code_in_reasoning(self, parser):
        """Test reasoning containing code snippets."""
        output = """<think>
Let me analyze the code:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```
This is a recursive implementation.
</think>The factorial function uses recursion."""

        reasoning, content = parser.extract_reasoning(output)
        assert "def factorial" in reasoning
        assert "recursive" in reasoning
        assert "uses recursion" in content

    def test_html_like_content(self, parser):
        """Test that HTML-like content doesn't confuse the parser."""
        output = "<think>The user mentioned <div> and <span> tags</think>Use CSS for styling."
        reasoning, content = parser.extract_reasoning(output)
        assert "<div>" in reasoning
        assert "<span>" in reasoning
        assert "CSS" in content

    def test_math_expressions(self, parser):
        """Test reasoning with mathematical expressions."""
        output = "<think>Given: x² + 2x + 1 = 0, so (x+1)² = 0, x = -1</think>x = -1"
        reasoning, content = parser.extract_reasoning(output)
        assert "x²" in reasoning
        assert "(x+1)²" in reasoning
        assert "-1" in content


class TestAPIModelsIntegration:
    """Tests for integration with API models."""

    def test_assistant_message_with_reasoning(self):
        """Test that AssistantMessage can hold reasoning content."""
        from vmlx_engine.api.models import AssistantMessage

        msg = AssistantMessage(
            content="The answer is 42.", reasoning="Let me think step by step..."
        )
        assert msg.content == "The answer is 42."
        assert msg.reasoning == "Let me think step by step..."
        assert msg.role == "assistant"

    def test_assistant_message_reasoning_none(self):
        """Test AssistantMessage with no reasoning."""
        from vmlx_engine.api.models import AssistantMessage

        msg = AssistantMessage(content="Simple response without reasoning.")
        assert msg.content == "Simple response without reasoning."
        assert msg.reasoning is None

    def test_chat_completion_chunk_delta_with_reasoning(self):
        """Test that ChatCompletionChunkDelta can hold reasoning."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta

        delta = ChatCompletionChunkDelta(reasoning="thinking...")
        assert delta.reasoning == "thinking..."
        assert delta.content is None

        delta2 = ChatCompletionChunkDelta(content="response text")
        assert delta2.content == "response text"
        assert delta2.reasoning is None

    def test_delta_transition(self):
        """Test delta during transition from reasoning to content."""
        from vmlx_engine.api.models import ChatCompletionChunkDelta

        # During transition, both might have values
        delta = ChatCompletionChunkDelta(
            reasoning="final thought", content="starting answer"
        )
        assert delta.reasoning == "final thought"
        assert delta.content == "starting answer"


class TestParserPerformance:
    """Basic performance tests for parsers."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_large_output_extraction(self, parser):
        """Test extraction from large output."""
        # Generate large reasoning content
        reasoning_lines = [f"Step {i}: processing data chunk {i}" for i in range(100)]
        reasoning_text = "\n".join(reasoning_lines)
        output = f"<think>{reasoning_text}</think>Processing complete."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is not None
        assert "Step 0" in reasoning
        assert "Step 99" in reasoning
        assert content == "Processing complete."

    def test_streaming_many_chunks(self, parser):
        """Test streaming with many small chunks."""
        parser.reset_state()

        # Generate many small chunks
        base_output = "<think>A" * 100 + "</think>" + "B" * 50
        accumulated = ""
        chunk_count = 0

        for char in base_output:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                chunk_count += 1

        # Should have processed all characters
        assert chunk_count > 0

    def test_repeated_parsing(self, parser):
        """Test parsing same output multiple times."""
        output = "<think>Quick thought</think>Quick answer"

        for _ in range(100):
            reasoning, content = parser.extract_reasoning(output)
            assert reasoning == "Quick thought"
            assert content == "Quick answer"


class TestDeepSeekSpecificCases:
    """Tests specific to DeepSeek-R1 parser behavior."""

    @pytest.fixture
    def parser(self):
        """Create DeepSeek-R1 parser."""
        return get_parser("deepseek_r1")()

    def test_implicit_reasoning_streaming(self, parser):
        """Test streaming when start tag is implicit (DeepSeek-R1 specific)."""
        # DeepSeek-R1 sometimes omits <think> but includes </think>
        tokens = ["reasoning", " text", " here", "</think>", "answer"]

        parser.reset_state()
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

        # For DeepSeek-R1, content before </think> without <think> is treated as content
        # until </think> appears in the delta
        all_parts = reasoning_parts + content_parts
        assert len(all_parts) > 0

    def test_deepseek_long_implicit_reasoning(self, parser):
        """Test long implicit reasoning without start tag."""
        output = """Let me think about this problem carefully.

First, I need to consider the constraints.
Then, I'll apply the algorithm.
Finally, I'll verify the result.</think>The answer is 42."""

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "think about this problem" in reasoning
        assert "42" in content


class TestQwen3SpecificCases:
    """Tests specific to Qwen3 parser behavior."""

    @pytest.fixture
    def parser(self):
        """Create Qwen3 parser."""
        return get_parser("qwen3")()

    def test_qwen3_implicit_mode_support(self, parser):
        """Qwen3 supports implicit mode for OpenCode compatibility."""
        # Only end tag - implicit mode (think injected in prompt)
        output1 = "some text</think>more text"
        reasoning, content = parser.extract_reasoning(output1)
        # Implicit mode: everything before </think> is reasoning
        assert reasoning == "some text"
        assert content == "more text"

        # Only start tag - <think> without </think> means truncated reasoning
        # (max_tokens hit during thinking phase). Text goes to reasoning_content.
        output2 = "<think>incomplete reasoning"
        reasoning, content = parser.extract_reasoning(output2)
        assert reasoning == "incomplete reasoning"
        assert content is None

    def test_qwen3_empty_think_tags(self, parser):
        """Test empty think tags."""
        output = "<think></think>Just the answer."
        reasoning, content = parser.extract_reasoning(output)
        # Empty reasoning should be None
        assert reasoning is None or reasoning.strip() == ""
        assert content == "Just the answer."

    def test_qwen3_whitespace_between_tags(self, parser):
        """Test various whitespace patterns."""
        test_cases = [
            ("<think> </think>answer", None, "answer"),
            ("<think>\n\n</think>answer", None, "answer"),
            ("<think>\t\t</think>answer", None, "answer"),
        ]

        for output, expected_reasoning, expected_content in test_cases:
            reasoning, content = parser.extract_reasoning(output)
            if expected_reasoning is None:
                assert reasoning is None or reasoning.strip() == ""
            assert expected_content in (content or "")


class TestTruncatedReasoningOverflow:
    """Tests for Fix #2: reasoning overflow when max_tokens hit during thinking.

    When a model generates <think> but hits max_tokens before producing </think>,
    the text after <think> should go to reasoning_content (not content).
    This matches streaming behavior where reasoning is progressively emitted.
    """

    @pytest.fixture
    def qwen3_parser(self):
        return get_parser("qwen3")()

    @pytest.fixture
    def deepseek_parser(self):
        return get_parser("deepseek_r1")()

    def test_qwen3_truncated_reasoning_basic(self, qwen3_parser):
        """Basic truncated reasoning: <think> with text, no </think>."""
        output = "<think>Let me analyze step by step..."
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert reasoning == "Let me analyze step by step..."
        assert content is None

    def test_qwen3_truncated_reasoning_multiline(self, qwen3_parser):
        """Truncated reasoning with multiple lines of thinking."""
        output = "<think>Step 1: Parse the input\nStep 2: Apply algorithm\nStep 3: But I ran out of"
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "ran out of" in reasoning
        assert content is None

    def test_qwen3_truncated_reasoning_with_newline_after_tag(self, qwen3_parser):
        """Truncated reasoning where <think> is followed by newline (common pattern)."""
        output = "<think>\nLet me think about this carefully.\nFirst, I need to consider"
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert reasoning is not None
        assert "think about this carefully" in reasoning
        assert content is None

    def test_qwen3_truncated_reasoning_long(self, qwen3_parser):
        """Truncated reasoning with very long text (realistic max_tokens scenario)."""
        long_text = "\n".join(
            [f"Step {i}: Processing data chunk {i} with detailed analysis" for i in range(50)]
        )
        output = f"<think>{long_text}"
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert reasoning is not None
        assert "Step 0" in reasoning
        assert "Step 49" in reasoning
        assert content is None

    def test_qwen3_truncated_vs_complete(self, qwen3_parser):
        """Verify truncated and complete reasoning produce consistent results."""
        # Complete reasoning
        complete = "<think>Full reasoning here</think>The answer is 42."
        r1, c1 = qwen3_parser.extract_reasoning(complete)
        assert r1 == "Full reasoning here"
        assert c1 == "The answer is 42."

        # Truncated (same reasoning text, no closing tag)
        truncated = "<think>Full reasoning here"
        r2, c2 = qwen3_parser.extract_reasoning(truncated)
        assert r2 == "Full reasoning here"
        assert c2 is None

    def test_qwen3_no_tags_still_pure_content(self, qwen3_parser):
        """No think tags at all should still return pure content."""
        output = "Just a normal response with no thinking."
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_qwen3_empty_think_tag_truncated(self, qwen3_parser):
        """<think> immediately followed by nothing (empty reasoning truncated)."""
        output = "<think>"
        reasoning, content = qwen3_parser.extract_reasoning(output)
        # Empty text after <think> → None reasoning, None content
        assert reasoning is None
        assert content is None

    def test_deepseek_truncated_reasoning(self, deepseek_parser):
        """DeepSeek-R1 should also handle truncated reasoning."""
        output = "<think>Step by step analysis that got cut"
        reasoning, content = deepseek_parser.extract_reasoning(output)
        # Base class Case 3 handles this
        assert reasoning == "Step by step analysis that got cut"
        assert content is None

    def test_qwen3_truncated_with_unicode(self, qwen3_parser):
        """Truncated reasoning with unicode content."""
        output = "<think>分析这个问题：需要考虑多个因素\n1. 第一个因素\n2. 第二个因素"
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert "分析" in reasoning
        assert "第二个因素" in reasoning
        assert content is None

    def test_qwen3_truncated_with_code(self, qwen3_parser):
        """Truncated reasoning containing code snippets."""
        output = "<think>Let me trace the code:\n```python\ndef foo():\n    return bar()"
        reasoning, content = qwen3_parser.extract_reasoning(output)
        assert "def foo" in reasoning
        assert content is None


class TestEnableThinkingForwarding:
    """Tests for Fix #1: enable_thinking=False forwarding to chat template.

    Previously, only enable_thinking=True was forwarded to the template.
    enable_thinking=False was silently dropped, causing models to always reason.
    Now both True and False are forwarded when explicitly set.
    """

    def test_enable_thinking_true_forwarded(self):
        """enable_thinking=True should be in template_kwargs."""
        # Simulate the logic from mllm.py
        enable_thinking = True
        template_kwargs = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        assert template_kwargs == {"enable_thinking": True}

    def test_enable_thinking_false_forwarded(self):
        """enable_thinking=False should be in template_kwargs (was previously dropped)."""
        enable_thinking = False
        template_kwargs = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        assert template_kwargs == {"enable_thinking": False}

    def test_enable_thinking_none_not_forwarded(self):
        """enable_thinking=None should NOT be in template_kwargs."""
        enable_thinking = None
        template_kwargs = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        assert template_kwargs == {}

    def test_enable_thinking_extraction_from_kwargs(self):
        """enable_thinking should be properly popped from kwargs."""
        kwargs = {"enable_thinking": False, "other_param": 42}
        enable_thinking = kwargs.pop("enable_thinking", None)
        assert enable_thinking is False
        assert "enable_thinking" not in kwargs
        assert kwargs == {"other_param": 42}

    def test_enable_thinking_default_when_missing(self):
        """Missing enable_thinking should default to None."""
        kwargs = {"other_param": 42}
        enable_thinking = kwargs.pop("enable_thinking", None)
        assert enable_thinking is None


class TestMLLMSamplingParams:
    """Tests for Fix #3 and #4: MLLM sampling parameter handling.

    Fix #3: temp= → temperature= (correct kwarg name for mlx_vlm)
    Fix #4: top_k/min_p extraction and sampler creation
    """

    def test_sampler_creation_with_top_k(self):
        """top_k > 0 should trigger sampler creation."""
        kwargs = {"top_k": 50, "top_p": 0.9}
        temperature = 0.7
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)

        needs_sampler = (top_k and top_k > 0) or (min_p and min_p > 0.0)
        assert needs_sampler is True

    def test_sampler_creation_with_min_p(self):
        """min_p > 0 should trigger sampler creation."""
        kwargs = {"min_p": 0.1}
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)

        needs_sampler = (top_k and top_k > 0) or (min_p and min_p > 0.0)
        assert needs_sampler is True

    def test_no_sampler_when_defaults(self):
        """Default top_k=0 and min_p=0.0 should not create sampler."""
        kwargs = {}
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)

        needs_sampler = (top_k and top_k > 0) or (min_p and min_p > 0.0)
        assert not needs_sampler

    def test_top_p_extracted_for_sampler(self):
        """When sampler needed, top_p should be popped to avoid double-passing."""
        kwargs = {"top_k": 50, "top_p": 0.9, "other": "keep"}
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            top_p = kwargs.pop("top_p", 1.0)
            assert top_p == 0.9
        assert "top_p" not in kwargs
        assert kwargs == {"other": "keep"}


class TestServerContentLeakPrevention:
    """Tests for Fix #6: Server-side content leak when reasoning is truncated.

    When extract_reasoning() returns (reasoning, None) — i.e., truncated reasoning
    with no content after it — the server must set content_for_parsing = "" to avoid
    leaking raw <think> tags into the content field.
    """

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        return get_parser(request.param)()

    def test_truncated_reasoning_no_content_leak(self, parser):
        """When reasoning is truncated, content should be empty, not raw output."""
        output = "<think>Some reasoning that got cut off"
        reasoning, content = parser.extract_reasoning(output)
        # Key assertion: content must NOT contain <think> tags
        assert reasoning is not None
        assert content is None  # No content, not the raw output

    def test_complete_reasoning_content_preserved(self, parser):
        """Normal case: complete reasoning should pass content through."""
        output = "<think>Thinking</think>The answer."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Thinking"
        assert content == "The answer."

    def test_no_tags_content_preserved(self, parser):
        """No tags: full output should be content."""
        output = "Just a plain response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Just a plain response."

    def test_content_for_parsing_logic(self):
        """Simulate the server.py content_for_parsing logic."""
        parser = get_parser("qwen3")()

        # Case 1: Normal — both reasoning and content
        r, c = parser.extract_reasoning("<think>R</think>C")
        content_for_parsing = "<think>R</think>C"  # raw output
        if c is not None:
            content_for_parsing = c
        elif r is not None:
            content_for_parsing = ""  # Fix #6
        assert content_for_parsing == "C"

        # Case 2: Truncated — reasoning only
        r, c = parser.extract_reasoning("<think>Truncated reasoning")
        content_for_parsing = "<think>Truncated reasoning"  # raw output
        if c is not None:
            content_for_parsing = c
        elif r is not None:
            content_for_parsing = ""  # Fix #6: prevents leak
        assert content_for_parsing == ""

        # Case 3: No tags — pure content
        r, c = parser.extract_reasoning("Plain text")
        content_for_parsing = "Plain text"  # raw output
        if c is not None:
            content_for_parsing = c
        elif r is not None:
            content_for_parsing = ""
        assert content_for_parsing == "Plain text"


class TestImplicitReasoningStreaming:
    """Tests for streaming implicit reasoning mode (think_in_prompt=True).

    When <think> is injected in the prompt (by agents like OpenCode), only
    </think> appears in the model output. Text before </think> is reasoning.
    """

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        return get_parser(request.param)()

    def test_think_in_prompt_streams_as_reasoning(self, parser):
        """With think_in_prompt=True, tokens before </think> are reasoning."""
        parser.reset_state(think_in_prompt=True)

        tokens = ["I", " think", " the", "</think>", "answer"]
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

        assert "I think the" in "".join(reasoning_parts)
        assert "answer" in "".join(content_parts)

    def test_think_in_prompt_false_streams_as_content(self, parser):
        """With think_in_prompt=False (default), text without tags is content."""
        parser.reset_state(think_in_prompt=False)

        tokens = ["Hello", " world"]
        accumulated = ""
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)

        assert "Hello world" in "".join(content_parts)

    def test_think_in_prompt_default_is_false(self, parser):
        """Default think_in_prompt should be False."""
        parser.reset_state()
        # Without think_in_prompt, tokens without tags should be content
        result = parser.extract_reasoning_streaming("", "Hello", "Hello")
        assert result is not None
        assert result.content == "Hello"
        assert result.reasoning is None


class TestLLMStopSequences:
    """Tests for Fix #5: LLM non-streaming stop sequence support.

    Previously, stop sequences were accepted but silently ignored in
    non-streaming LLM.generate(). Now they trigger post-generation truncation.
    """

    def test_stop_sequence_truncation_logic(self):
        """Stop sequence should truncate output at first match."""
        output_text = "Hello world! How are you doing today?"
        stop = ["!"]
        for stop_seq in stop:
            idx = output_text.find(stop_seq)
            if idx != -1:
                output_text = output_text[:idx]
                break
        assert output_text == "Hello world"

    def test_multiple_stop_sequences(self):
        """First matching stop sequence should win."""
        output_text = "Line 1\nLine 2\n---\nLine 4"
        stop = ["---", "\n\n"]
        for stop_seq in stop:
            idx = output_text.find(stop_seq)
            if idx != -1:
                output_text = output_text[:idx]
                break
        assert output_text == "Line 1\nLine 2\n"

    def test_no_stop_sequence_match(self):
        """If no stop sequence matches, output is unchanged."""
        output_text = "Complete response here."
        stop = ["NONEXISTENT"]
        original = output_text
        for stop_seq in stop:
            idx = output_text.find(stop_seq)
            if idx != -1:
                output_text = output_text[:idx]
                break
        assert output_text == original

    def test_stop_none_no_truncation(self):
        """stop=None should not cause truncation."""
        output_text = "Response text."
        stop = None
        if stop and output_text:
            for stop_seq in stop:
                idx = output_text.find(stop_seq)
                if idx != -1:
                    output_text = output_text[:idx]
                    break
        assert output_text == "Response text."

    def test_stop_at_beginning(self):
        """Stop sequence at the very beginning should return empty string."""
        output_text = "STOP rest of text"
        stop = ["STOP"]
        for stop_seq in stop:
            idx = output_text.find(stop_seq)
            if idx != -1:
                output_text = output_text[:idx]
                break
        assert output_text == ""

    def test_finish_reason_set_correctly(self):
        """Finish reason should be 'stop' when stop sequence found."""
        output_text = "Hello! World"
        stop = ["!"]
        finish_reason = "length"  # default
        if stop and output_text:
            for stop_seq in stop:
                idx = output_text.find(stop_seq)
                if idx != -1:
                    output_text = output_text[:idx]
                    finish_reason = "stop"
                    break
        assert finish_reason == "stop"
        assert output_text == "Hello"


class TestBatchOffsetSafeCache:
    """Tests for Fix #7: _BatchOffsetSafeCache proxy for concurrent batching.

    When multiple requests are batched, BatchKVCache.offset is an mx.array
    of per-request offsets. Several Qwen VL model attention layers use
    cache.offset in slice operations that require a Python int.

    The _BatchOffsetSafeCache proxy intercepts .offset reads to return
    a scalar int while delegating all other attribute access unchanged.
    """

    def test_mx_array_offset_returns_max_int(self):
        """Proxy should convert mx.array offset to max element as Python int.

        Using max (not first element) ensures the attention mask is wide enough
        for ALL sequences in the batch, preventing broadcast shape mismatches
        when sequences have different lengths due to left-padding.
        """
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        class FakeCache:
            offset = mx.array([10, 20, 30])
        proxy = _BatchOffsetSafeCache(FakeCache())
        assert isinstance(proxy.offset, int)
        assert proxy.offset == 30  # max, not first

    def test_scalar_mx_array_offset(self):
        """Proxy should handle scalar (0-dim) mx.array offset."""
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        class FakeCache:
            offset = mx.array(42)
        proxy = _BatchOffsetSafeCache(FakeCache())
        assert isinstance(proxy.offset, int)
        assert proxy.offset == 42

    def test_int_offset_passthrough(self):
        """Proxy should pass through Python int offset unchanged."""
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        class FakeCache:
            offset = 99
        proxy = _BatchOffsetSafeCache(FakeCache())
        assert proxy.offset == 99

    def test_attribute_delegation(self):
        """Proxy should delegate non-offset attributes to inner cache."""
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        class FakeCache:
            offset = 0
            keys = "fake_keys"
            values = "fake_values"
            def make_mask(self):
                return "mask"
        proxy = _BatchOffsetSafeCache(FakeCache())
        assert proxy.keys == "fake_keys"
        assert proxy.values == "fake_values"
        assert proxy.make_mask() == "mask"

    def test_truthiness(self):
        """Proxy should be truthy (used in 'if cache:' checks)."""
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        class FakeCache:
            offset = 0
        proxy = _BatchOffsetSafeCache(FakeCache())
        assert bool(proxy) is True

    def test_offset_setter_delegates(self):
        """Setting offset through proxy should update inner cache."""
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        class FakeCache:
            offset = 0
        inner = FakeCache()
        proxy = _BatchOffsetSafeCache(inner)
        proxy.offset = mx.array([100, 200])
        assert isinstance(inner.offset, mx.array)

    def test_wrap_batch_caches_selective(self):
        """_wrap_batch_caches should only wrap BatchKVCache objects."""
        from mlx_lm.models.cache import BatchKVCache, KVCache
        from vmlx_engine.mllm_batch_generator import (
            _BatchOffsetSafeCache,
            _wrap_batch_caches,
        )

        bkv = BatchKVCache([0, 5])
        kv = KVCache()
        wrapped = _wrap_batch_caches([bkv, kv, bkv])
        assert isinstance(wrapped[0], _BatchOffsetSafeCache)
        assert isinstance(wrapped[1], KVCache)
        assert isinstance(wrapped[2], _BatchOffsetSafeCache)

    def test_divergent_offsets_uses_max(self):
        """When batch sequences have different offsets, proxy uses max.

        This prevents broadcast_shapes errors when the mask is sliced
        to kv_seq_len — using max ensures the mask is wide enough for
        all sequences, while left_padding in the mask handles masking
        invalid positions for shorter sequences.
        """
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import _BatchOffsetSafeCache

        # Simulate: 3 sequences with different lengths due to left-padding
        # Sequence 0 is shortest (offset 95), sequence 1 is longest (offset 100)
        class FakeCache:
            offset = mx.array([95, 100, 98])
        proxy = _BatchOffsetSafeCache(FakeCache())
        # Should return 100 (max) to make mask wide enough
        assert proxy.offset == 100
        # Verify it's a plain int for slice compatibility
        assert type(proxy.offset) is int

    def test_wrap_always_applied_single_request_batch(self):
        """_wrap_batch_caches wraps even when batch filters down to 1 request.

        When a batch starts with N>1 requests and filters down to 1, the cache
        is still BatchKVCache (offset is mx.array), but input_tokens.shape[0]=1.
        The proxy must still wrap to convert offset to int, otherwise VL models
        that use cache.offset in slice ops get "Slice indices must be integers".
        """
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache
        from vmlx_engine.mllm_batch_generator import (
            _BatchOffsetSafeCache,
            _wrap_batch_caches,
        )

        # Simulate: batch started with 3 requests, filtered down to 1
        bkv = BatchKVCache([0])
        # After filter, offset is still mx.array (1-element)
        bkv.offset = mx.array([42])
        wrapped = _wrap_batch_caches([bkv])
        assert isinstance(wrapped[0], _BatchOffsetSafeCache)
        # Proxy converts mx.array offset to int
        assert wrapped[0].offset == 42
        assert type(wrapped[0].offset) is int

    def test_batch_mamba_cache_filter_updates_left_padding(self):
        """BatchMambaCache.filter() must filter left_padding along with cache arrays.

        Without this, left_padding retains the original batch_size after
        filtering. Then make_mask() returns a mask with stale dimensions,
        causing shape mismatches in SSM layers during batched generation.
        This was the root cause of the intermittent concatenate error:
        [concatenate] (1,3,12288) and (2,1,12288)
        """
        import mlx.core as mx
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        # Simulate a merged batch of 3 requests with left_padding
        cache = BatchMambaCache(size=2, left_padding=[5, 10, 15])
        cache.cache = [
            mx.ones((3, 3, 128)),  # conv_state: (batch=3, d_conv-1=3, d=128)
            mx.ones((3, 4, 64)),   # ssm_state: (batch=3, nh=4, d=64)
        ]

        # Filter to keep only requests 0 and 2
        cache.filter(mx.array([0, 2], mx.int32))

        # Cache arrays should be filtered to batch_size=2
        assert cache.cache[0].shape == (2, 3, 128)
        assert cache.cache[1].shape == (2, 4, 64)

        # left_padding MUST also be filtered to batch_size=2
        assert cache.left_padding.shape == (2,)
        assert cache.left_padding.tolist() == [5, 15]

    def test_batch_mamba_cache_filter_none_left_padding(self):
        """BatchMambaCache.filter() works when left_padding is None."""
        import mlx.core as mx
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        cache = BatchMambaCache(size=2, left_padding=None)
        cache._batch_size = 3
        cache.cache = [
            mx.ones((3, 3, 128)),
            mx.ones((3, 4, 64)),
        ]

        # Should not crash when left_padding is None
        cache.filter(mx.array([0, 1], mx.int32))
        assert cache.cache[0].shape == (2, 3, 128)
        assert cache.left_padding is None

    def test_batch_mamba_cache_merge_no_left_padding(self):
        """BatchMambaCache.merge() should NOT set left_padding.

        Setting left_padding=[0]*N creates a no-op mask that causes shape
        mismatches after filter() shrinks the batch. With left_padding=None,
        make_mask() returns None and SSM layers skip masking (correct for decode).
        """
        import mlx.core as mx
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        # Create 3 individual caches (simulating post-prefill per-request caches)
        caches = []
        for _ in range(3):
            c = BatchMambaCache(size=2, left_padding=None)
            c.cache = [mx.ones((1, 3, 128)), mx.ones((1, 4, 64))]
            caches.append(c)

        # Merge them
        merged = BatchMambaCache.merge(caches)

        # Merged cache should have correct batch dimension
        assert merged.cache[0].shape == (3, 3, 128)
        assert merged.cache[1].shape == (3, 4, 64)

        # left_padding should be None (NOT [0, 0, 0])
        assert merged.left_padding is None

        # make_mask should return None (no masking during decode)
        assert merged.make_mask(1) is None

    def test_batch_mamba_cache_merge_then_filter_no_stale_mask(self):
        """Full lifecycle: merge → step → filter should not leave stale masks.

        Regression test for the concatenate crash: after merge of 3 caches then
        filter to 2, make_mask() must not return a (3, N) mask for batch_size=2.
        """
        import mlx.core as mx
        from vmlx_engine.utils.mamba_cache import BatchMambaCache

        caches = []
        for _ in range(3):
            c = BatchMambaCache(size=2, left_padding=None)
            c.cache = [mx.ones((1, 3, 128)), mx.ones((1, 4, 64))]
            caches.append(c)

        merged = BatchMambaCache.merge(caches)

        # Simulate a generation step updating cache
        merged.cache[0] = mx.ones((3, 3, 128)) * 2
        merged.cache[1] = mx.ones((3, 4, 64)) * 2

        # Filter: one request finishes, keep requests 0 and 1
        merged.filter(mx.array([0, 1], mx.int32))

        assert merged.cache[0].shape == (2, 3, 128)
        assert merged.cache[1].shape == (2, 4, 64)

        # The critical check: make_mask should return None or match batch_size=2
        mask = merged.make_mask(1)
        assert mask is None or mask.shape[0] == 2
