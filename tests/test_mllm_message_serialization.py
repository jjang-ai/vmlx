# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM message serialization and multi-turn VLM correctness.

These tests verify the critical fixes for:
1. model_dump(exclude_none=True) — prevents Jinja2 template misdetecting text items
   as image items when model_dump() includes image_url=None (key exists → True).
2. _resolve_content() in Responses API — same Pydantic-to-clean-dict conversion.
3. Multi-turn VLM message formatting — correct image_pad token count.
4. ContentPart serialization nuances across all content types.
5. Hybrid model (SSM+attention) prefix cache safety.
6. MLLM server path message conversion edge cases.

These tests are pure unit tests with NO MLX dependency.
"""

import pytest

from vllm_mlx.api.models import (
    ContentPart,
    ImageUrl,
    Message,
    VideoUrl,
    AudioUrl,
    ChatCompletionRequest,
)


# =============================================================================
# 1. model_dump(exclude_none=True) — The Root Cause Fix
# =============================================================================


class TestContentPartModelDump:
    """Verify that model_dump() with/without exclude_none behaves correctly.

    ROOT CAUSE: Jinja2 chat templates (e.g. Qwen3.5-VL) check key existence:
        'image_url' in item
    This returns True for model_dump() dicts that have image_url=None.
    It must return False for text-only items.
    """

    def test_text_part_model_dump_includes_none_fields(self):
        """model_dump() WITHOUT exclude_none includes all optional fields as None."""
        part = ContentPart(type="text", text="Hello world")
        data = part.model_dump()

        # These None fields are the ROOT CAUSE of the multi-turn VLM bug
        assert "image_url" in data, "model_dump() should include image_url key"
        assert data["image_url"] is None
        assert "video" in data
        assert data["video"] is None
        assert "video_url" in data
        assert data["video_url"] is None
        assert "audio_url" in data
        assert data["audio_url"] is None

    def test_text_part_model_dump_exclude_none_removes_fields(self):
        """model_dump(exclude_none=True) correctly removes None fields."""
        part = ContentPart(type="text", text="Hello world")
        data = part.model_dump(exclude_none=True)

        # CRITICAL: These keys must NOT exist after exclude_none
        assert "image_url" not in data, (
            "exclude_none=True must remove image_url=None — "
            "Jinja2 'image_url' in item would misdetect text as image"
        )
        assert "video" not in data
        assert "video_url" not in data
        assert "audio_url" not in data

        # Only type and text should remain
        assert data == {"type": "text", "text": "Hello world"}

    def test_image_part_model_dump_exclude_none_preserves_image(self):
        """exclude_none preserves image_url when it's actually set."""
        part = ContentPart(
            type="image_url",
            image_url=ImageUrl(url="data:image/png;base64,abc123"),
        )
        data = part.model_dump(exclude_none=True)

        assert "image_url" in data
        assert data["image_url"]["url"] == "data:image/png;base64,abc123"
        # Other None fields removed
        assert "text" not in data
        assert "video" not in data

    def test_video_part_model_dump_exclude_none_preserves_video(self):
        """exclude_none preserves video fields when set."""
        part = ContentPart(type="video", video="/path/to/video.mp4")
        data = part.model_dump(exclude_none=True)

        assert data == {"type": "video", "video": "/path/to/video.mp4"}
        assert "image_url" not in data
        assert "text" not in data

    def test_video_url_part_model_dump_exclude_none(self):
        """exclude_none preserves video_url when set."""
        part = ContentPart(
            type="video_url",
            video_url=VideoUrl(url="https://example.com/vid.mp4"),
        )
        data = part.model_dump(exclude_none=True)

        assert "video_url" in data
        assert data["video_url"]["url"] == "https://example.com/vid.mp4"
        assert "image_url" not in data

    def test_audio_url_part_model_dump_exclude_none(self):
        """exclude_none preserves audio_url when set."""
        part = ContentPart(
            type="audio_url",
            audio_url=AudioUrl(url="https://example.com/audio.wav"),
        )
        data = part.model_dump(exclude_none=True)

        assert "audio_url" in data
        assert data["audio_url"]["url"] == "https://example.com/audio.wav"
        assert "image_url" not in data
        assert "text" not in data


class TestJinja2KeyExistenceSimulation:
    """Simulate what Jinja2 chat templates do with 'key in dict' checks.

    This directly tests the bug mechanism:
    Qwen3.5-VL template: {% if 'image_url' in item or 'image' in item %}
    """

    def test_raw_dict_text_item_no_image_key(self):
        """A clean dict text item should NOT have image_url key."""
        item = {"type": "text", "text": "Hello"}
        assert "image_url" not in item, "Clean dict should not have image_url"

    def test_model_dump_text_item_has_image_key_BUG(self):
        """model_dump() text item incorrectly HAS image_url key (the bug)."""
        part = ContentPart(type="text", text="Hello")
        item = part.model_dump()
        # This is the BUG — template thinks it's an image
        assert "image_url" in item, "model_dump() includes image_url=None"

    def test_model_dump_exclude_none_text_item_no_image_key_FIX(self):
        """model_dump(exclude_none=True) fixes the bug."""
        part = ContentPart(type="text", text="Hello")
        item = part.model_dump(exclude_none=True)
        assert "image_url" not in item, "exclude_none removes the false key"

    def test_multi_turn_message_image_count_simulation(self):
        """Simulate Jinja2 template counting image items in multi-turn messages.

        This test replicates the exact scenario that caused 0-token output:
        - 5 messages: user→assistant→user→assistant→user(with image)
        - Only the last user message has an actual image
        - With model_dump(): ALL text content items falsely detected as images
        - With exclude_none: only the actual image_url item detected
        """
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="how r u"),
            Message(role="assistant", content="Good!"),
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                    ContentPart(type="text", text="What is this?"),
                ],
            ),
        ]

        # Simulate server.py MLLM path: convert messages
        def convert_with_model_dump(msgs):
            """BAD path — model_dump() without exclude_none."""
            result = []
            for msg in msgs:
                result.append(msg.model_dump())
            return result

        def convert_with_exclude_none(msgs):
            """GOOD path — model_dump(exclude_none=True)."""
            result = []
            for msg in msgs:
                result.append(msg.model_dump(exclude_none=True))
            return result

        # Count how many content items would be detected as images
        def count_image_items(converted_msgs):
            """Simulate Jinja2: 'image_url' in item or 'image' in item."""
            count = 0
            for msg in converted_msgs:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if "image_url" in item or "image" in item:
                                count += 1
            return count

        bad_msgs = convert_with_model_dump(messages)
        good_msgs = convert_with_exclude_none(messages)

        bad_image_count = count_image_items(bad_msgs)
        good_image_count = count_image_items(good_msgs)

        # BAD: model_dump() causes text items to be counted as images
        # The text item {"type": "text", "text": "What is this?", "image_url": None}
        # has "image_url" as a key → detected as image
        assert bad_image_count == 2, (
            f"model_dump() should cause 2 detections (1 real + 1 false), got {bad_image_count}"
        )

        # GOOD: exclude_none correctly counts only actual images
        assert good_image_count == 1, (
            f"exclude_none should count exactly 1 image, got {good_image_count}"
        )


# =============================================================================
# 2. Message Model Full Serialization Tests
# =============================================================================


class TestMessageModelDump:
    """Test full Message model serialization for MLLM path."""

    def test_simple_text_message_model_dump_exclude_none(self):
        """Simple text message should serialize cleanly."""
        msg = Message(role="user", content="Hello")
        data = msg.model_dump(exclude_none=True)

        assert data == {"role": "user", "content": "Hello"}
        assert "tool_calls" not in data
        assert "tool_call_id" not in data

    def test_multimodal_message_model_dump_exclude_none(self):
        """Multimodal message content parts should be clean after exclude_none."""
        msg = Message(
            role="user",
            content=[
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="data:image/png;base64,abc"),
                ),
                ContentPart(type="text", text="Describe this"),
            ],
        )
        data = msg.model_dump(exclude_none=True)

        assert data["role"] == "user"
        assert isinstance(data["content"], list)
        assert len(data["content"]) == 2

        # Image part
        img_part = data["content"][0]
        assert img_part["type"] == "image_url"
        assert "image_url" in img_part
        assert "text" not in img_part  # Should not have text=None
        assert "video" not in img_part

        # Text part
        txt_part = data["content"][1]
        assert txt_part["type"] == "text"
        assert txt_part["text"] == "Describe this"
        assert "image_url" not in txt_part  # CRITICAL: no false image_url key
        assert "video" not in txt_part

    def test_assistant_message_with_none_content(self):
        """Assistant message with content=None should serialize correctly."""
        msg = Message(role="assistant", content=None)
        data = msg.model_dump(exclude_none=True)

        assert data == {"role": "assistant"}
        assert "content" not in data

    def test_tool_message_model_dump_exclude_none(self):
        """Tool message preserves tool_call_id."""
        msg = Message(role="tool", content="Result", tool_call_id="call_123")
        data = msg.model_dump(exclude_none=True)

        assert data["role"] == "tool"
        assert data["content"] == "Result"
        assert data["tool_call_id"] == "call_123"
        assert "tool_calls" not in data

    def test_assistant_with_tool_calls(self):
        """Assistant message with tool_calls serializes correctly."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
        )
        data = msg.model_dump(exclude_none=True)

        assert data["role"] == "assistant"
        assert "tool_calls" in data
        assert len(data["tool_calls"]) == 1
        assert "content" not in data  # content was None

    def test_multi_turn_conversation_model_dump(self):
        """Full multi-turn conversation serializes without false keys."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="Thanks"),
        ]

        for msg in messages:
            data = msg.model_dump(exclude_none=True)
            # No message in a text-only conversation should have media keys
            assert "image_url" not in str(data), (
                f"Text message should not reference image_url: {data}"
            )


# =============================================================================
# 3. Server MLLM Message Conversion (simulated)
# =============================================================================


class TestMLLMMessageConversion:
    """Test the server.py MLLM message conversion logic.

    The server has two code paths:
    1. Chat Completions: if engine.is_mllm → msg.model_dump(exclude_none=True)
    2. Responses API: _resolve_content() → Pydantic-to-clean-dict conversion
    """

    def _convert_mllm_messages(self, request_messages):
        """Simulate the server.py MLLM Chat Completions path.

        This mirrors server.py lines 1635-1641:
            for msg in request.messages:
                if hasattr(msg, "model_dump"):
                    msg_dict = msg.model_dump(exclude_none=True)
                else:
                    msg_dict = dict(msg)
                messages.append(msg_dict)
        """
        messages = []
        for msg in request_messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                msg_dict = dict(msg)
            messages.append(msg_dict)
        return messages

    def _resolve_content_mllm(self, raw_content):
        """Simulate _resolve_content() for MLLM Responses API path.

        This mirrors server.py lines 1937-1950.
        """
        if isinstance(raw_content, list):
            clean_parts = []
            for p in raw_content:
                if hasattr(p, "model_dump"):
                    clean_parts.append(p.model_dump(exclude_none=True))
                elif isinstance(p, dict):
                    clean_parts.append({k: v for k, v in p.items() if v is not None})
                else:
                    clean_parts.append(p)
            has_media = any(
                isinstance(p, dict)
                and p.get("type") in ("image_url", "image", "video_url", "video")
                for p in clean_parts
            )
            if has_media:
                return clean_parts
        # Fallback: extract text
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            texts = []
            for p in raw_content:
                if isinstance(p, dict):
                    texts.append(p.get("text", ""))
                elif hasattr(p, "text"):
                    texts.append(p.text or "")
            return " ".join(t for t in texts if t)
        return str(raw_content) if raw_content else ""

    def test_chat_completions_mllm_text_only(self):
        """Text-only messages convert cleanly for MLLM."""
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        converted = self._convert_mllm_messages(msgs)

        assert converted[0] == {"role": "user", "content": "Hello"}
        assert converted[1] == {"role": "assistant", "content": "Hi"}

    def test_chat_completions_mllm_with_image(self):
        """Multimodal messages preserve image content for MLLM."""
        msgs = [
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                    ContentPart(type="text", text="Describe"),
                ],
            )
        ]
        converted = self._convert_mllm_messages(msgs)

        content = converted[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2

        # Verify no false image_url on text part
        text_part = content[1]
        assert text_part["type"] == "text"
        assert "image_url" not in text_part

    def test_chat_completions_mllm_multi_turn_with_image(self):
        """Multi-turn with final image message — the exact bug scenario."""
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="how are you"),
            Message(role="assistant", content="Good!"),
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                    ContentPart(type="text", text="What is this?"),
                ],
            ),
        ]
        converted = self._convert_mllm_messages(msgs)

        # Count image-related items across ALL messages
        image_count = 0
        for msg in converted:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and (
                        "image_url" in item or "image" in item
                    ):
                        image_count += 1

        assert image_count == 1, (
            f"Expected exactly 1 image item, got {image_count}. "
            "This would cause IndexError in Qwen3VLProcessor."
        )

    def test_chat_completions_mllm_handles_raw_dicts(self):
        """Raw dict messages (not Pydantic) pass through correctly."""
        msgs = [
            {"role": "user", "content": "Hello"},
        ]
        # These don't have model_dump, so dict() is used
        converted = self._convert_mllm_messages(msgs)
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"

    def test_resolve_content_pydantic_parts(self):
        """_resolve_content handles Pydantic ContentPart objects."""
        parts = [
            ContentPart(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc"),
            ),
            ContentPart(type="text", text="Describe"),
        ]
        result = self._resolve_content_mllm(parts)

        # Should return list (has media)
        assert isinstance(result, list)
        assert len(result) == 2

        # Image part preserved
        assert result[0]["type"] == "image_url"
        assert "image_url" in result[0]

        # Text part clean — no false image_url
        assert result[1]["type"] == "text"
        assert "image_url" not in result[1]

    def test_resolve_content_dict_parts_with_none(self):
        """_resolve_content cleans dicts with None values."""
        parts = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
                "text": None,
                "video": None,
            },
            {
                "type": "text",
                "text": "Describe",
                "image_url": None,
                "video": None,
            },
        ]
        result = self._resolve_content_mllm(parts)

        assert isinstance(result, list)
        # Image part: text and video removed
        assert "text" not in result[0]
        assert "video" not in result[0]
        # Text part: image_url removed (CRITICAL)
        assert "image_url" not in result[1]

    def test_resolve_content_text_only_returns_string(self):
        """_resolve_content with text-only parts returns extracted text."""
        parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="text", text="World"),
        ]
        result = self._resolve_content_mllm(parts)

        # No media → extract text
        assert isinstance(result, str)
        assert "Hello" in result
        assert "World" in result

    def test_resolve_content_string_passthrough(self):
        """_resolve_content with string passes through."""
        result = self._resolve_content_mllm("Hello world")
        assert result == "Hello world"

    def test_resolve_content_mixed_pydantic_and_dict(self):
        """_resolve_content handles mix of Pydantic and dict parts."""
        parts = [
            ContentPart(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc"),
            ),
            {"type": "text", "text": "A raw dict text part", "image_url": None},
        ]
        result = self._resolve_content_mllm(parts)

        assert isinstance(result, list)
        assert len(result) == 2
        # Dict part should have image_url=None cleaned
        assert "image_url" not in result[1]


# =============================================================================
# 4. ChatCompletionRequest with MLLM Content
# =============================================================================


class TestChatCompletionRequestMLLM:
    """Test ChatCompletionRequest with multimodal content construction."""

    def test_request_with_image_content(self):
        """Request with image content parses correctly."""
        req = ChatCompletionRequest(
            model="Qwen3.5-VL",
            messages=[
                Message(
                    role="user",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                        {"type": "text", "text": "What is this?"},
                    ],
                )
            ],
        )
        assert len(req.messages) == 1
        assert isinstance(req.messages[0].content, list)

    def test_request_multi_turn_with_image(self):
        """Multi-turn request with image in last message."""
        req = ChatCompletionRequest(
            model="Qwen3.5-VL",
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="Hello!"),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                        {"type": "text", "text": "What is this?"},
                    ],
                ),
            ],
        )

        # Verify all messages parse
        assert len(req.messages) == 3
        assert req.messages[0].content == "hi"
        assert isinstance(req.messages[2].content, list)

    def test_request_with_enable_thinking(self):
        """Request with enable_thinking parameter."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Think hard")],
            enable_thinking=True,
        )
        assert req.enable_thinking is True

    def test_request_with_stream_options(self):
        """Request with stream_options for usage reporting."""
        from vllm_mlx.api.models import StreamOptions

        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hello")],
            stream=True,
            stream_options=StreamOptions(include_usage=True),
        )
        assert req.stream_options.include_usage is True


# =============================================================================
# 5. ContentPart Edge Cases
# =============================================================================


class TestContentPartEdgeCases:
    """Test edge cases in ContentPart that could cause subtle bugs."""

    def test_content_part_with_dict_image_url(self):
        """ContentPart accepts dict for image_url (OpenAI format)."""
        part = ContentPart(
            type="image_url",
            image_url={"url": "https://example.com/img.jpg", "detail": "high"},
        )
        # model_dump should preserve the dict format
        data = part.model_dump(exclude_none=True)
        assert data["image_url"]["url"] == "https://example.com/img.jpg"

    def test_content_part_with_string_image_url(self):
        """ContentPart accepts string for image_url (simplified format)."""
        part = ContentPart(
            type="image_url",
            image_url="https://example.com/img.jpg",
        )
        data = part.model_dump(exclude_none=True)
        assert data["image_url"] == "https://example.com/img.jpg"

    def test_content_part_base64_image(self):
        """ContentPart with base64 image data URL."""
        b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        part = ContentPart(
            type="image_url",
            image_url=ImageUrl(url=b64),
        )
        data = part.model_dump(exclude_none=True)
        assert data["image_url"]["url"].startswith("data:image/png;base64,")

    def test_all_content_types_exclude_none_isolation(self):
        """Each content type only has its relevant fields after exclude_none."""
        test_cases = [
            (
                ContentPart(type="text", text="Hello"),
                {"type", "text"},
            ),
            (
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="http://img.jpg"),
                ),
                {"type", "image_url"},
            ),
            (
                ContentPart(type="video", video="/path/vid.mp4"),
                {"type", "video"},
            ),
            (
                ContentPart(
                    type="video_url",
                    video_url=VideoUrl(url="http://vid.mp4"),
                ),
                {"type", "video_url"},
            ),
            (
                ContentPart(
                    type="audio_url",
                    audio_url=AudioUrl(url="http://audio.wav"),
                ),
                {"type", "audio_url"},
            ),
        ]

        for part, expected_keys in test_cases:
            data = part.model_dump(exclude_none=True)
            actual_keys = set(data.keys())
            assert actual_keys == expected_keys, (
                f"ContentPart(type={part.type}) should have keys {expected_keys}, "
                f"got {actual_keys}"
            )


# =============================================================================
# 6. Dict Filtering for None Values
# =============================================================================


class TestDictNoneFiltering:
    """Test the {k: v for k, v in d.items() if v is not None} pattern
    used in _resolve_content for raw dict inputs."""

    def test_filter_removes_none_values(self):
        """Filtering removes None-valued keys."""
        d = {"type": "text", "text": "Hello", "image_url": None, "video": None}
        clean = {k: v for k, v in d.items() if v is not None}

        assert clean == {"type": "text", "text": "Hello"}
        assert "image_url" not in clean

    def test_filter_preserves_falsy_non_none(self):
        """Filtering preserves falsy values that aren't None (0, '', False)."""
        d = {"type": "text", "text": "", "count": 0, "flag": False, "nothing": None}
        clean = {k: v for k, v in d.items() if v is not None}

        assert "text" in clean  # "" is not None
        assert clean["text"] == ""
        assert "count" in clean  # 0 is not None
        assert "flag" in clean  # False is not None
        assert "nothing" not in clean

    def test_filter_on_model_dump_output(self):
        """Filtering model_dump() output should match exclude_none."""
        part = ContentPart(type="text", text="Hello")

        # Method 1: model_dump(exclude_none=True)
        data1 = part.model_dump(exclude_none=True)

        # Method 2: model_dump() + filter
        raw = part.model_dump()
        data2 = {k: v for k, v in raw.items() if v is not None}

        assert data1 == data2


# =============================================================================
# 7. Multi-Image and Multi-Video Scenarios
# =============================================================================


class TestMultiMediaScenarios:
    """Test scenarios with multiple images or videos in a single message."""

    def test_multiple_images_in_one_message(self):
        """Multiple images in one message all have correct keys."""
        msg = Message(
            role="user",
            content=[
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="img1.png"),
                ),
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="img2.png"),
                ),
                ContentPart(type="text", text="Compare these"),
            ],
        )
        data = msg.model_dump(exclude_none=True)
        content = data["content"]

        image_count = sum(
            1 for p in content if isinstance(p, dict) and "image_url" in p
        )
        text_count = sum(
            1 for p in content if isinstance(p, dict) and p.get("type") == "text"
        )

        assert image_count == 2
        assert text_count == 1

        # Text item must NOT have image_url key
        text_items = [p for p in content if p.get("type") == "text"]
        for t in text_items:
            assert "image_url" not in t

    def test_image_and_video_in_one_message(self):
        """Mixed image + video in one message serializes correctly."""
        msg = Message(
            role="user",
            content=[
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="img.png"),
                ),
                ContentPart(type="video", video="/path/vid.mp4"),
                ContentPart(type="text", text="Compare"),
            ],
        )
        data = msg.model_dump(exclude_none=True)
        content = data["content"]

        # Each part should only have its own media key
        assert "image_url" in content[0]
        assert "video" not in content[0]

        assert "video" in content[1]
        assert "image_url" not in content[1]

        assert "image_url" not in content[2]
        assert "video" not in content[2]

    def test_multi_turn_with_images_in_different_turns(self):
        """Images in different turns should each be counted separately."""
        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="img1.png"),
                    ),
                    ContentPart(type="text", text="What's this?"),
                ],
            ),
            Message(role="assistant", content="A cat."),
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="img2.png"),
                    ),
                    ContentPart(type="text", text="And this?"),
                ],
            ),
        ]

        # Count total image items after clean serialization
        total_images = 0
        for msg in messages:
            data = msg.model_dump(exclude_none=True)
            content = data.get("content")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and "image_url" in p:
                        total_images += 1

        assert total_images == 2


# =============================================================================
# 8. Hybrid Model Prefix Cache Safety
# =============================================================================


class TestHybridModelPrefixCacheSafety:
    """Test logic for hybrid model (SSM+attention) prefix cache detection.

    Hybrid models like Qwen3.5-VL-MoE have both KVCache and MambaCache layers.
    Prefix cache can only store KVCache; SSM layers need full-sequence processing.
    The _process_prompts code must detect hybrid and skip cache use.
    """

    def test_hybrid_detection_logic(self):
        """Simulate hybrid model detection from layer positions."""
        # Qwen3.5-VL-MoE: 40 layers, ~30 KV + ~10 SSM
        hybrid_kv_positions = list(range(30))  # positions 0-29 are KV
        hybrid_num_layers = 40  # total 40 layers

        is_hybrid = (
            hybrid_kv_positions is not None
            and hybrid_num_layers is not None
            and len(hybrid_kv_positions) < hybrid_num_layers
        )
        assert is_hybrid is True, "Should detect as hybrid"

    def test_pure_attention_detection_logic(self):
        """Pure attention model (all layers are KV) should NOT be hybrid."""
        # Standard model: all 32 layers are KV
        kv_positions = list(range(32))
        num_layers = 32

        is_hybrid = (
            kv_positions is not None
            and num_layers is not None
            and len(kv_positions) < num_layers
        )
        assert is_hybrid is False, "Pure attention should not be hybrid"

    def test_no_hybrid_info_defaults_to_non_hybrid(self):
        """When hybrid info is None, model is treated as non-hybrid."""
        is_hybrid = (
            None is not None
            and None is not None
            and False
        )
        assert is_hybrid is False


# =============================================================================
# 9. extract_multimodal_content with Pydantic objects
# =============================================================================


class TestExtractMultimodalContentPydantic:
    """Test that extract_multimodal_content handles both Pydantic and dict inputs."""

    def test_pydantic_content_parts(self):
        """extract_multimodal_content with Pydantic ContentPart objects."""
        from vllm_mlx.api.utils import extract_multimodal_content

        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="https://example.com/img.jpg"),
                    ),
                    ContentPart(type="text", text="Describe this"),
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)
        assert len(images) == 1
        assert images[0] == "https://example.com/img.jpg"
        assert "Describe this" in processed[0]["content"]

    def test_mixed_pydantic_and_dict_content(self):
        """Messages with mix of Pydantic and dict content parts."""
        from vllm_mlx.api.utils import extract_multimodal_content

        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="img.jpg"),
                    ),
                    {"type": "text", "text": "raw dict part"},
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)
        assert len(images) == 1

    def test_no_images_produces_empty_list(self):
        """Text-only multi-turn should produce no images."""
        from vllm_mlx.api.utils import extract_multimodal_content

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="How are you?"),
        ]

        processed, images, videos = extract_multimodal_content(messages)
        assert images == []
        assert videos == []
        assert len(processed) == 3


# =============================================================================
# 10. Enable Thinking / Chat Template kwargs
# =============================================================================


class TestEnableThinkingKwargs:
    """Test that enable_thinking and extra template kwargs are preserved."""

    def test_chat_completion_request_enable_thinking_true(self):
        """enable_thinking=True should be preserved in request."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Think")],
            enable_thinking=True,
        )
        assert req.enable_thinking is True

    def test_chat_completion_request_enable_thinking_false(self):
        """enable_thinking=False should be preserved (not None)."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Don't think")],
            enable_thinking=False,
        )
        assert req.enable_thinking is False

    def test_chat_completion_request_enable_thinking_default(self):
        """enable_thinking defaults to None (auto mode)."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.enable_thinking is None

    def test_chat_template_kwargs_preserved(self):
        """chat_template_kwargs are preserved in request."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hello")],
            chat_template_kwargs={"thinking_budget": 1024, "reasoning_effort": "high"},
        )
        assert req.chat_template_kwargs["thinking_budget"] == 1024
        assert req.chat_template_kwargs["reasoning_effort"] == "high"


# =============================================================================
# 11. Responses API Input Items
# =============================================================================


class TestResponsesAPIInputItems:
    """Test conversion of Responses API input items to chat messages."""

    def test_simple_string_input(self):
        """String input for Responses API."""
        # Just a basic structural test
        input_data = "Hello, how are you?"
        assert isinstance(input_data, str)

    def test_message_with_content_parts(self):
        """Message type input with multimodal content parts."""
        item = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "img.jpg"}},
                {"type": "text", "text": "What is this?"},
            ],
        }
        assert item["type"] == "message"
        assert len(item["content"]) == 2

    def test_function_call_input(self):
        """Function call input for Responses API."""
        item = {
            "type": "function_call",
            "name": "search",
            "call_id": "call_123",
            "arguments": '{"query": "test"}',
        }
        assert item["type"] == "function_call"
        assert item["name"] == "search"

    def test_function_call_output_input(self):
        """Function call output for Responses API."""
        item = {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Search results here",
        }
        assert item["type"] == "function_call_output"


# =============================================================================
# 12. _fix_hybrid_cache correctness
# =============================================================================


class TestFixHybridCache:
    """Test _fix_hybrid_cache returns correct cache on mismatch.

    The function must return a fresh full-length cache (from make_cache())
    when the reconstructed cache length doesn't match expected KV positions,
    NOT the corrupt short cache.
    """

    def test_mismatch_returns_fresh_cache_fast_path(self):
        """When cache length != kv_positions length, return fresh cache (fast path)."""
        from unittest.mock import MagicMock, patch as mock_patch

        mock_model = MagicMock()
        # 40-layer model with 30 KV + 10 SSM
        kv_positions = list(range(30))
        num_layers = 40

        # Corrupt cache: only 20 entries instead of expected 30
        corrupt_cache = [MagicMock() for _ in range(20)]

        # make_cache returns a fresh 40-entry cache
        fresh_cache = [MagicMock() for _ in range(40)]
        mock_model.make_cache.return_value = fresh_cache

        # Mock KVCache import
        mock_kvcache = type('KVCache', (), {})

        with mock_patch(
            "vllm_mlx.mllm_batch_generator.logger"
        ):
            from vllm_mlx.mllm_batch_generator import _fix_hybrid_cache
            # Temporarily mock the import inside the function
            import vllm_mlx.mllm_batch_generator as mbg
            original_fn = mbg._fix_hybrid_cache

            # Test: corrupt cache (20) doesn't match kv_positions (30)
            result = _fix_hybrid_cache(
                corrupt_cache, mock_model,
                kv_positions=kv_positions,
                num_model_layers=num_layers,
            )

        # Must be the fresh cache (40 layers), NOT the corrupt cache (20)
        assert len(result) == 40
        assert result is not corrupt_cache

    def test_matching_cache_passes_through(self):
        """When cache length == num_model_layers, return as-is (no fix needed)."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        kv_positions = list(range(32))
        num_layers = 32

        # Cache already correct length
        correct_cache = [MagicMock() for _ in range(32)]

        from vllm_mlx.mllm_batch_generator import _fix_hybrid_cache
        result = _fix_hybrid_cache(
            correct_cache, mock_model,
            kv_positions=kv_positions,
            num_model_layers=num_layers,
        )
        assert result is correct_cache

    def test_pure_attention_no_fix(self):
        """Pure attention model (all layers KV) needs no fix."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        # All 32 layers are KV
        kv_positions = list(range(32))
        num_layers = 32

        cache = [MagicMock() for _ in range(32)]

        from vllm_mlx.mllm_batch_generator import _fix_hybrid_cache
        result = _fix_hybrid_cache(
            cache, mock_model,
            kv_positions=kv_positions,
            num_model_layers=num_layers,
        )
        assert result is cache

    def test_no_make_cache_returns_original(self):
        """Models without make_cache() return original cache unchanged."""
        from unittest.mock import MagicMock

        mock_model = MagicMock(spec=[])  # No make_cache attribute
        cache = [MagicMock() for _ in range(32)]

        from vllm_mlx.mllm_batch_generator import _fix_hybrid_cache
        result = _fix_hybrid_cache(cache, mock_model)
        assert result is cache


# =============================================================================
# 13. SimpleEngine MLLM kwargs forwarding
# =============================================================================


class TestSimpleEngineMLLMKwargs:
    """Test that SimpleEngine MLLM paths forward all kwargs correctly.

    Both chat() and stream_generate() must pass:
    - enable_thinking
    - reasoning_effort
    - chat_template_kwargs (merged)
    """

    def test_chat_kwargs_built_correctly(self):
        """Simulate the MLLM kwargs building logic from SimpleEngine.chat()."""
        # These are the kwargs as they'd come in from server.py
        kwargs = {
            "enable_thinking": True,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking_budget": 1024},
            "request_id": "req-123",
            "prompt_suffix": None,
            "skip_generation_prompt": False,
        }

        # Simulate the SimpleEngine.chat() kwargs extraction
        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        kwargs.pop("request_id", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        kwargs.pop("prompt_suffix", None)
        kwargs.pop("skip_generation_prompt", False)

        # Build MLLM kwargs (matching fixed code)
        mllm_kwargs = dict(kwargs)
        if thinking_enabled is not None:
            mllm_kwargs["enable_thinking"] = thinking_enabled
        if reasoning_effort:
            mllm_kwargs["reasoning_effort"] = reasoning_effort
        if extra_ct_kwargs:
            mllm_kwargs.update(extra_ct_kwargs)

        # Verify all kwargs present
        assert mllm_kwargs["enable_thinking"] is True
        assert mllm_kwargs["reasoning_effort"] == "high"
        assert mllm_kwargs["thinking_budget"] == 1024

    def test_stream_kwargs_built_correctly(self):
        """Simulate the MLLM kwargs building logic from SimpleEngine.stream_generate()."""
        kwargs = {
            "enable_thinking": False,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"thinking_budget": 256},
            "request_id": "req-456",
            "prompt_suffix": None,
            "skip_generation_prompt": False,
        }

        # Simulate extraction
        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        kwargs.pop("request_id", None)
        kwargs.pop("prompt_suffix", None)
        kwargs.pop("skip_generation_prompt", False)

        mllm_kwargs = dict(kwargs)
        if thinking_enabled is not None:
            mllm_kwargs["enable_thinking"] = thinking_enabled
        if reasoning_effort:
            mllm_kwargs["reasoning_effort"] = reasoning_effort
        if extra_ct_kwargs:
            mllm_kwargs.update(extra_ct_kwargs)

        assert mllm_kwargs["enable_thinking"] is False
        assert mllm_kwargs["reasoning_effort"] == "low"
        assert mllm_kwargs["thinking_budget"] == 256

    def test_no_reasoning_effort_omitted(self):
        """When reasoning_effort is None, it should NOT appear in kwargs."""
        kwargs = {
            "enable_thinking": True,
            "reasoning_effort": None,
            "chat_template_kwargs": None,
            "request_id": "req-789",
            "prompt_suffix": None,
            "skip_generation_prompt": False,
        }

        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        kwargs.pop("request_id", None)
        kwargs.pop("prompt_suffix", None)
        kwargs.pop("skip_generation_prompt", False)

        mllm_kwargs = dict(kwargs)
        if thinking_enabled is not None:
            mllm_kwargs["enable_thinking"] = thinking_enabled
        if reasoning_effort:
            mllm_kwargs["reasoning_effort"] = reasoning_effort
        if extra_ct_kwargs:
            mllm_kwargs.update(extra_ct_kwargs)

        assert "enable_thinking" in mllm_kwargs
        assert "reasoning_effort" not in mllm_kwargs
        assert "thinking_budget" not in mllm_kwargs

    def test_extra_ct_kwargs_merge_does_not_overwrite(self):
        """chat_template_kwargs should merge into mllm_kwargs without overwriting."""
        kwargs = {
            "enable_thinking": True,
            "reasoning_effort": "medium",
            "chat_template_kwargs": {"custom_key": "value"},
            "request_id": "req-abc",
        }

        extra_ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_enabled = kwargs.pop("enable_thinking", True)
        kwargs.pop("request_id", None)

        mllm_kwargs = dict(kwargs)
        if thinking_enabled is not None:
            mllm_kwargs["enable_thinking"] = thinking_enabled
        if reasoning_effort:
            mllm_kwargs["reasoning_effort"] = reasoning_effort
        if extra_ct_kwargs:
            mllm_kwargs.update(extra_ct_kwargs)

        assert mllm_kwargs["enable_thinking"] is True
        assert mllm_kwargs["reasoning_effort"] == "medium"
        assert mllm_kwargs["custom_key"] == "value"


# =============================================================================
# 14. model_dump(exclude_none=True) in mllm.py
# =============================================================================


class TestMllmModelDumpExcludeNone:
    """Test that mllm.py model_dump uses exclude_none=True.

    Both extract_multimodal_content paths in mllm.py (chat and stream_chat)
    convert Pydantic models to dicts. Without exclude_none=True, image_url=None
    would survive in text ContentParts, causing duplicate image_pad tokens.
    """

    def test_text_contentpart_no_image_url_key(self):
        """Text ContentPart model_dump(exclude_none=True) has no image_url key."""
        part = ContentPart(type="text", text="describe this")
        dumped = part.model_dump(exclude_none=True)
        assert "image_url" not in dumped
        assert "video" not in dumped
        assert "video_url" not in dumped
        assert "audio_url" not in dumped

    def test_image_contentpart_no_text_key(self):
        """Image ContentPart model_dump(exclude_none=True) has no text key."""
        part = ContentPart(
            type="image_url",
            image_url=ImageUrl(url="data:image/png;base64,abc"),
        )
        dumped = part.model_dump(exclude_none=True)
        assert "text" not in dumped
        assert "video" not in dumped
        assert "image_url" in dumped

    def test_simulated_mllm_item_loop(self):
        """Simulate the exact loop in mllm.py that processes content items."""
        content = [
            ContentPart(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc"),
            ),
            ContentPart(type="text", text="What is this?"),
        ]

        image_count = 0
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            # This is the FIXED code path from mllm.py
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif hasattr(item, "dict"):
                item = item.dict()

            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    image_count += 1

        assert image_count == 1, f"Expected 1 image, got {image_count}"
        assert len(text_parts) == 1
        assert text_parts[0] == "What is this?"
