# SPDX-License-Identifier: Apache-2.0
"""
Tests for API utility functions.

Tests clean_output_text, is_mllm_model, and extract_multimodal_content
from vmlx_engine/api/utils.py. No MLX dependency.
"""

from vmlx_engine.api.utils import (
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,
    is_vlm_model,
    SPECIAL_TOKENS_PATTERN,
)
from vmlx_engine.api.models import Message, ContentPart, ImageUrl


class TestCleanOutputText:
    """Tests for clean_output_text function."""

    def test_empty_string(self):
        assert clean_output_text("") == ""

    def test_none_returns_none(self):
        assert clean_output_text(None) is None

    def test_plain_text_unchanged(self):
        assert clean_output_text("Hello world") == "Hello world"

    def test_removes_im_end(self):
        assert clean_output_text("Hello<|im_end|>") == "Hello"

    def test_removes_im_start(self):
        assert clean_output_text("<|im_start|>Hello") == "Hello"

    def test_removes_endoftext(self):
        assert clean_output_text("Hello<|endoftext|>") == "Hello"

    def test_removes_eot_id(self):
        assert clean_output_text("Hello<|eot_id|>") == "Hello"

    def test_removes_end_token(self):
        assert clean_output_text("Hello<|end|>") == "Hello"

    def test_removes_start_header_id(self):
        result = clean_output_text("<|start_header_id|>assistant<|end_header_id|>Hello")
        assert "<|start_header_id|>" not in result
        assert "<|end_header_id|>" not in result

    def test_removes_s_tags(self):
        assert clean_output_text("<s>Hello</s>") == "Hello"

    def test_removes_pad_tokens(self):
        assert clean_output_text("[PAD]Hello[PAD]") == "Hello"

    def test_removes_sep_cls(self):
        assert clean_output_text("[CLS]Hello[SEP]") == "Hello"

    def test_removes_multiple_special_tokens(self):
        text = "<|im_start|>assistant\nHello world<|im_end|><|endoftext|>"
        result = clean_output_text(text)
        assert result == "assistant\nHello world"

    def test_preserves_think_tags(self):
        text = "<think>Let me think about this.</think>The answer is 42."
        result = clean_output_text(text)
        assert "<think>" in result
        assert "</think>" in result
        assert "The answer is 42." in result

    def test_adds_missing_opening_think_tag(self):
        text = "Some thinking content.</think>The answer is 42."
        result = clean_output_text(text)
        assert result.startswith("<think>")
        assert "</think>" in result

    def test_no_extra_think_tag_when_already_present(self):
        text = "<think>Thinking.</think>Answer."
        result = clean_output_text(text)
        assert result.count("<think>") == 1

    def test_strips_whitespace(self):
        assert clean_output_text("  Hello  ") == "Hello"

    def test_combined_special_tokens_and_think(self):
        text = "<|im_start|><think>I need to think.</think>42<|im_end|>"
        result = clean_output_text(text)
        assert "<think>" in result
        assert "</think>" in result
        assert "42" in result
        assert "<|im_start|>" not in result


class TestSpecialTokensPattern:
    """Tests for the special tokens regex pattern."""

    def test_matches_all_expected_tokens(self):
        tokens = [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|end|>",
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "</s>",
            "<s>",
            "<pad>",
            "[PAD]",
            "[SEP]",
            "[CLS]",
        ]
        for token in tokens:
            assert (
                SPECIAL_TOKENS_PATTERN.search(token) is not None
            ), f"Pattern should match {token}"

    def test_does_not_match_think_tags(self):
        assert SPECIAL_TOKENS_PATTERN.search("<think>") is None
        assert SPECIAL_TOKENS_PATTERN.search("</think>") is None

    def test_does_not_match_normal_text(self):
        assert SPECIAL_TOKENS_PATTERN.search("Hello world") is None


class TestIsMllmModel:
    """Tests for is_mllm_model function.

    VLM detection uses config.json vision_config (for local models) and
    model config registry (for known model_types). No regex fallback —
    users can force VLM mode via session settings / --is-mllm flag.
    """

    def test_force_mllm_always_true(self):
        assert is_mllm_model("any-model-name", force_mllm=True) is True
        assert is_mllm_model("nonexistent/path", force_mllm=True) is True

    def test_local_model_with_vision_config(self, tmp_path):
        import json
        config = {"model_type": "qwen3_5", "vision_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_mllm_model(str(tmp_path)) is True

    def test_local_model_without_vision_config(self, tmp_path):
        import json
        config = {"model_type": "llama", "hidden_size": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_mllm_model(str(tmp_path)) is False

    def test_local_qwen3_5_text_model(self, tmp_path):
        """qwen3_5 model_type WITHOUT vision_config should NOT be VLM."""
        import json
        config = {"model_type": "qwen3_5", "hidden_size": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_mllm_model(str(tmp_path)) is False

    def test_local_qwen3_5_vl_model(self, tmp_path):
        """qwen3_5 model_type WITH vision_config SHOULD be VLM."""
        import json
        config = {"model_type": "qwen3_5", "vision_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_mllm_model(str(tmp_path)) is True

    def test_remote_model_name_returns_false(self):
        """Remote HF names without local config.json default to non-VLM.
        Users must force VLM mode via session settings."""
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit") is False
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit") is False

    def test_non_mllm_models(self):
        assert is_mllm_model("mlx-community/Llama-3.2-3B-Instruct-4bit") is False
        assert is_mllm_model("nonexistent-model") is False

    def test_backwards_compatibility_alias(self):
        assert is_vlm_model is is_mllm_model

    def test_malformed_config_json(self, tmp_path):
        """Malformed config.json should fall through gracefully."""
        (tmp_path / "config.json").write_text("not valid json")
        assert is_mllm_model(str(tmp_path)) is False


class TestExtractMultimodalContent:
    """Tests for extract_multimodal_content function."""

    def test_simple_text_messages(self):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0] == {"role": "system", "content": "You are helpful."}
        assert processed[1] == {"role": "user", "content": "Hello"}
        assert images == []
        assert videos == []

    def test_none_content(self):
        messages = [Message(role="assistant", content=None)]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0] == {"role": "assistant", "content": ""}

    def test_multimodal_with_image_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="What is this?"),
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="https://example.com/img.png"),
                    ),
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What is this?"
        assert images == ["https://example.com/img.png"]
        assert videos == []

    def test_multimodal_with_dict_image_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert images == ["data:image/png;base64,abc"]

    def test_multimodal_with_string_image_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look"},
                    {"type": "image_url", "image_url": "https://example.com/img.png"},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert images == ["https://example.com/img.png"]

    def test_multimodal_with_video(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert videos == ["/path/to/video.mp4"]

    def test_multimodal_with_video_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/v.mp4"},
                    },
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert videos == ["https://example.com/v.mp4"]

    def test_multimodal_with_string_video_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look"},
                    {"type": "video_url", "video_url": "https://example.com/v.mp4"},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert videos == ["https://example.com/v.mp4"]

    def test_multiple_images(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": "img1.png"}},
                    {"type": "image_url", "image_url": {"url": "img2.png"}},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert len(images) == 2

    def test_tool_response_message(self):
        messages = [
            Message(role="tool", content="72F and sunny", tool_call_id="call_1")
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["role"] == "user"
        assert "Tool Result" in processed[0]["content"]
        assert "call_1" in processed[0]["content"]

    def test_tool_response_preserve_native(self):
        messages = [
            Message(role="tool", content="72F and sunny", tool_call_id="call_1")
        ]
        processed, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["role"] == "tool"
        assert processed[0]["tool_call_id"] == "call_1"
        assert processed[0]["content"] == "72F and sunny"

    def test_assistant_with_tool_calls(self):
        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["role"] == "assistant"
        assert "get_weather" in processed[0]["content"]

    def test_assistant_with_tool_calls_preserve_native(self):
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["role"] == "assistant"
        assert processed[0]["content"] == "Let me check."
        assert "tool_calls" in processed[0]
        assert len(processed[0]["tool_calls"]) == 1

    def test_dict_messages(self):
        messages = [
            Message(role="user", content="Hello"),
        ]
        # Also test with raw dicts (the function handles both)
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["content"] == "Hello"

    def test_image_type_content_with_raw_dicts(self):
        # type="image" path handles raw dict content (not Pydantic ContentPart)
        # Pass a raw dict message to avoid Pydantic stripping unknown fields
        raw_messages = [
            type(
                "Msg",
                (),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "image": "https://example.com/img.png"},
                    ],
                    "tool_calls": None,
                    "tool_call_id": None,
                },
            )()
        ]
        processed, images, videos = extract_multimodal_content(raw_messages)
        assert images == ["https://example.com/img.png"]

    def test_empty_messages(self):
        processed, images, videos = extract_multimodal_content([])
        assert processed == []
        assert images == []
        assert videos == []

    def test_tool_response_none_content(self):
        messages = [Message(role="tool", content=None, tool_call_id="call_1")]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["role"] == "user"
        assert "call_1" in processed[0]["content"]

    def test_multiple_text_parts_combined(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert "First part." in processed[0]["content"]
        assert "Second part." in processed[0]["content"]
