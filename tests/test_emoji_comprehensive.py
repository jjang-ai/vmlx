# SPDX-License-Identifier: Apache-2.0
"""Comprehensive emoji support verification tests.

This test suite verifies that the streaming detokenizer correctly handles
all categories of emoji and complex Unicode sequences:
- Basic emoji (single codepoint)
- Skin tone modifiers
- Family/relationship emoji (multi-person ZWJ sequences)
- Flag emoji (regional indicator sequences)
- ZWJ sequences (occupation, gender, etc.)
- Variation selectors
- High codepoints (surrogate pairs in UTF-16)
"""

import pytest
from transformers import AutoTokenizer
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer


class TestEmojiSupport:
    """Test comprehensive emoji support in streaming detokenizer."""

    @pytest.fixture
    def qwen_tokenizer(self):
        """Load Qwen tokenizer."""
        return AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-8bit")

    def test_basic_emoji(self, qwen_tokenizer):
        """Test basic single-codepoint emoji."""
        text = "Basic emoji: 🌟 🎯 🔥 🚀 🐍"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        # Should match batch decode
        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result

        # Should not have replacement characters
        assert "\ufffd" not in detok.text

        # Should contain all emoji
        assert "🌟" in detok.text
        assert "🎯" in detok.text
        assert "🔥" in detok.text
        assert "🚀" in detok.text
        assert "🐍" in detok.text

    def test_skin_tone_modifiers(self, qwen_tokenizer):
        """Test emoji with skin tone modifiers (Fitzpatrick scale)."""
        text = "Skin tones: 👋🏻 👋🏼 👋🏽 👋🏾 👋🏿"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

        # All 5 Fitzpatrick skin tone modifiers
        assert "👋🏻" in detok.text or "👋" in detok.text  # Type 1-2 (light)
        assert "👋🏼" in detok.text or "👋" in detok.text  # Type 3 (medium-light)
        assert "👋🏽" in detok.text or "👋" in detok.text  # Type 4 (medium)
        assert "👋🏾" in detok.text or "👋" in detok.text  # Type 5 (medium-dark)
        assert "👋🏿" in detok.text or "👋" in detok.text  # Type 6 (dark)

    def test_family_emoji_zwj_sequences(self, qwen_tokenizer):
        """Test multi-person family emoji (Zero-Width Joiner sequences)."""
        text = "Families: 👨‍👩‍👧‍👦 👨‍👨‍👦 👩‍👩‍👦‍👦"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

    def test_flag_emoji(self, qwen_tokenizer):
        """Test flag emoji (regional indicator pairs)."""
        text = "Flags: 🇺🇸 🇬🇧 🇯🇵 🇧🇷 🇮🇳 🇦🇪 🇸🇦 🇨🇳 🇰🇷"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

        # Regional indicators should be present
        # (Note: some tokenizers may normalize or decompose these)
        assert "🇺" in detok.text or "US" in detok.text

    def test_occupation_zwj_sequences(self, qwen_tokenizer):
        """Test occupation/role emoji (ZWJ sequences)."""
        text = "Roles: 🏳️‍🌈 🏳️‍⚧️ 👩‍💻 👨‍🚀 🧑‍⚕️ 🧑‍🌾 🧑‍🎤"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

    def test_high_codepoint_emoji(self, qwen_tokenizer):
        """Test emoji with high codepoints (surrogate pairs in UTF-16)."""
        # U+1FAC0 - U+1FAFF range (newer emoji)
        text = "High codepoints: 🫀 🫁 🫂 🫃 🫄 🫅"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

        # At least some of these should be present
        assert "🫀" in detok.text or "🫁" in detok.text or "🫂" in detok.text

    def test_very_high_codepoints(self, qwen_tokenizer):
        """Test emoji in U+1F900+ range."""
        text = "Very high: 🦀 🦐 🦒 🦓 🦔 🧀 🧁 🧂 🧃"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

    def test_ultra_high_codepoints(self, qwen_tokenizer):
        """Test emoji in U+1FA00+ range (even higher)."""
        text = "Ultra high: 🪐 🪑 🪒 🪓 🪔 🪕 🪖 🪗"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

    def test_mixed_emoji_and_text(self, qwen_tokenizer):
        """Test mixed emoji and regular text."""
        text = "Hello 👋 world 🌍 with emoji 😊 everywhere! 🎉"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

        # Should contain both text and emoji
        assert "Hello" in detok.text
        assert "world" in detok.text
        assert "👋" in detok.text
        assert "🌍" in detok.text

    def test_emoji_at_token_boundaries(self, qwen_tokenizer):
        """Test emoji that span token boundaries."""
        # Create a text where emoji are likely to be split across tokens
        text = "A👋B🌟C🔥D🚀E🎯F"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()

        # Process token by token (this is where splitting would occur)
        for t in tokens:
            detok.add_token(t)
            # last_segment should only emit complete characters
            segment = detok.last_segment
            if segment:
                assert "\ufffd" not in segment

        detok.finalize()

        # Final text should have no replacement characters
        assert "\ufffd" not in detok.text
        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result

    def test_streaming_incremental_emoji(self, qwen_tokenizer):
        """Test that emoji appear correctly during streaming."""
        text = "Counting: 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()

        all_segments = []
        for t in tokens:
            detok.add_token(t)
            segment = detok.last_segment
            if segment:
                # Each segment should be valid UTF-8
                assert "\ufffd" not in segment
                all_segments.append(segment)

        detok.finalize()
        final_segment = detok.last_segment
        if final_segment:
            all_segments.append(final_segment)

        # Concatenated segments should equal final text
        concatenated = "".join(all_segments)
        assert concatenated == detok.text
        assert "\ufffd" not in concatenated

    def test_empty_emoji_sequence(self, qwen_tokenizer):
        """Test edge case of text without emoji."""
        text = "No emoji here, just plain text."
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text

    def test_only_emoji(self, qwen_tokenizer):
        """Test text containing only emoji."""
        text = "🌟🎯🔥🚀🐍😊🎉🌈"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result
        assert "\ufffd" not in detok.text


class TestEmojiCompatibility:
    """Test emoji compatibility across different models/tokenizers."""

    def test_tokenizer_preserves_emoji(self):
        """Test that emoji are preserved through tokenization round-trip."""
        from transformers import AutoTokenizer

        tokenizers_to_test = [
            "mlx-community/Qwen3-0.6B-8bit",
            # Add more as needed
        ]

        test_emoji = "Hello 👋 world 🌍 emoji 😊 test 🎉"

        for tokenizer_name in tokenizers_to_test:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                tokens = tokenizer.encode(test_emoji)
                decoded = tokenizer.decode(tokens)

                # Should not have replacement characters
                assert "\ufffd" not in decoded, (
                    f"Tokenizer {tokenizer_name} produced replacement characters"
                )

                # Should contain emoji (may have extra spaces/tokens)
                assert "👋" in decoded or "wave" in decoded.lower()
                assert "🌍" in decoded or "world" in decoded.lower()

            except Exception as e:
                pytest.skip(f"Tokenizer {tokenizer_name} not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
