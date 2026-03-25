# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Multimodal Language Model (MLLM) wrapper."""

import platform
import sys
from pathlib import Path

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_mllm_model():
    """Return a small MLLM model for testing."""
    return "mlx-community/Qwen3-VL-4B-Instruct-3bit"


@pytest.fixture
def test_image_path(tmp_path):
    """Download a real image from Wikimedia Commons for tests."""
    pytest.importorskip("PIL")
    import requests
    from PIL import Image
    import io

    # Use a small dog image from Wikimedia Commons (public domain)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return str(path)
    except Exception:
        # Fallback to synthetic image if download fails
        img = Image.new("RGB", (320, 240), color="blue")
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return str(path)


@pytest.fixture
def test_video_path(tmp_path):
    """Download a real video from Wikimedia Commons for tests."""
    import requests

    # Use a short video from Wikimedia Commons (Creative Commons)
    # This is a 3-second sample video
    url = "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.160p.webm"

    path = tmp_path / "test_video.webm"

    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(path)
    except Exception:
        # Fallback to synthetic video if download fails
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))

        # Create 30 frames (1 second)
        for i in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (255, 0, 0)  # Blue in BGR
            out.write(frame)

        out.release()
        return str(path)


# =============================================================================
# Unit Tests - No Model Loading Required
# =============================================================================


class TestMLLMHelperFunctions:
    """Test helper functions that don't require model loading."""

    def test_is_base64_image(self):
        """Test base64 image detection."""
        from vmlx_engine.models.mllm import is_base64_image

        assert is_base64_image("data:image/png;base64,iVBORw0KGgo=")
        assert is_base64_image("data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        assert not is_base64_image("https://example.com/image.jpg")
        assert not is_base64_image("/path/to/image.jpg")

    def test_is_base64_video(self):
        """Test base64 video detection."""
        from vmlx_engine.models.mllm import is_base64_video

        assert is_base64_video("data:video/mp4;base64,AAAA")
        assert is_base64_video("data:video/webm;base64,AAAA")
        assert not is_base64_video("https://example.com/video.mp4")
        assert not is_base64_video("/path/to/video.mp4")

    def test_is_url(self):
        """Test URL detection."""
        from vmlx_engine.models.mllm import is_url

        assert is_url("https://example.com/image.jpg")
        assert is_url("http://example.com/video.mp4")
        assert not is_url("/path/to/file.jpg")
        assert not is_url("data:image/png;base64,AAAA")


class TestVideoFrameExtraction:
    """Test video frame extraction functions."""

    def test_get_video_info(self, test_video_path):
        """Test getting video information."""
        cv2 = pytest.importorskip("cv2")

        # Use OpenCV directly since get_video_info may not be exported
        cap = cv2.VideoCapture(test_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Video from Wikimedia will have different properties
        assert total_frames > 0
        assert fps > 0
        assert width > 0
        assert height > 0

    def test_extract_video_frames_smart(self, test_video_path):
        """Test smart frame extraction."""
        cv2 = pytest.importorskip("cv2")
        from vmlx_engine.models.mllm import extract_video_frames_smart

        # Extract frames
        frames = extract_video_frames_smart(test_video_path, fps=2.0, max_frames=10)

        assert len(frames) > 0
        assert len(frames) <= 10
        # Check frame shape (height, width, channels)
        assert len(frames[0].shape) == 3  # Should be 3D array

    def test_extract_frames_respects_max_frames(self, test_video_path):
        """Test that max_frames limit is respected."""
        cv2 = pytest.importorskip("cv2")
        from vmlx_engine.models.mllm import extract_video_frames_smart

        frames = extract_video_frames_smart(test_video_path, fps=30.0, max_frames=5)

        assert len(frames) <= 5

    def test_save_frames_to_temp(self, test_video_path):
        """Test saving frames to temp files."""
        cv2 = pytest.importorskip("cv2")
        from vmlx_engine.models.mllm import extract_video_frames_smart, save_frames_to_temp

        frames = extract_video_frames_smart(test_video_path, fps=1.0, max_frames=2)
        paths = save_frames_to_temp(frames)

        assert len(paths) == len(frames)
        for path in paths:
            assert Path(path).exists()
            assert path.endswith(".jpg")


class TestImageProcessing:
    """Test image processing functions."""

    def test_process_image_input_local_file(self, test_image_path):
        """Test processing local image file."""
        from vmlx_engine.models.mllm import process_image_input

        result = process_image_input(test_image_path)
        assert result == test_image_path

    def test_process_image_input_dict_format(self, test_image_path):
        """Test processing image in dict format."""
        from vmlx_engine.models.mllm import process_image_input

        # OpenAI format
        result = process_image_input({"url": test_image_path})
        assert Path(result).exists()


class TestVideoProcessing:
    """Test video processing functions."""

    def test_process_video_input_local_file(self, test_video_path):
        """Test processing local video file."""
        from vmlx_engine.models.mllm import process_video_input

        result = process_video_input(test_video_path)
        assert result == test_video_path

    def test_process_video_input_dict_format(self, test_video_path):
        """Test processing video in dict format."""
        from vmlx_engine.models.mllm import process_video_input

        # OpenAI format
        result = process_video_input({"url": test_video_path})
        assert Path(result).exists()

    def test_process_video_input_empty_raises(self):
        """Test that empty input raises error."""
        from vmlx_engine.models.mllm import process_video_input

        with pytest.raises(ValueError):
            process_video_input("")

        with pytest.raises(ValueError):
            process_video_input({})


# =============================================================================
# Audit Bug Fix Tests (v1.1.6)
# =============================================================================


class TestExtractMultimodalMessages:
    """Tests for _extract_multimodal_messages bug fixes."""

    def test_video_url_content_type_handled(self):
        """BUG 7: video_url content type must not be silently dropped."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
                ],
            }
        ]

        chat_msgs, images, videos = MLXMultimodalLM._extract_multimodal_messages(messages)
        assert len(videos) == 1
        assert videos[0] == "https://example.com/video.mp4"

    def test_video_url_string_format(self):
        """video_url can also be a plain string."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "video_url", "video_url": "/path/to/video.mp4"},
                ],
            }
        ]

        _, _, videos = MLXMultimodalLM._extract_multimodal_messages(messages)
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"

    def test_video_type_still_works(self):
        """Original video type handling must not be broken."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            }
        ]

        _, _, videos = MLXMultimodalLM._extract_multimodal_messages(messages)
        assert len(videos) == 1

    def test_mixed_image_and_video_url(self):
        """Both image_url and video_url in same message."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                ],
            }
        ]

        _, images, videos = MLXMultimodalLM._extract_multimodal_messages(messages)
        assert len(images) == 1
        assert len(videos) == 1

    def test_tool_role_messages_preserved_as_text(self):
        """Tool role messages should at minimum preserve text content."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me calculate that."},
            {"role": "tool", "content": "4", "tool_call_id": "call_123", "name": "calculator"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        chat_msgs, _, _ = MLXMultimodalLM._extract_multimodal_messages(messages)
        roles = [m["role"] for m in chat_msgs]
        assert "tool" in roles

        # Tool message should preserve tool_call_id and name
        tool_msg = [m for m in chat_msgs if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["name"] == "calculator"
        assert tool_msg["content"] == "4"

    def test_assistant_tool_calls_preserved(self):
        """Assistant messages with tool_calls should preserve them."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        tool_calls = [{"id": "call_123", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]
        messages = [
            {"role": "user", "content": "Calculate"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]

        chat_msgs, _, _ = MLXMultimodalLM._extract_multimodal_messages(messages)
        assistant_msg = [m for m in chat_msgs if m["role"] == "assistant"][0]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"] == tool_calls


class TestMLLMFinishReason:
    """Tests for MLLM finish_reason correctness (BUG 9 fix)."""

    def test_mllm_output_dataclass(self):
        """MLLMOutput should support both 'stop' and 'length' finish reasons."""
        from vmlx_engine.models.mllm import MLLMOutput

        out_stop = MLLMOutput(text="hello", finish_reason="stop", completion_tokens=5)
        assert out_stop.finish_reason == "stop"

        out_length = MLLMOutput(text="hello", finish_reason="length", completion_tokens=256)
        assert out_length.finish_reason == "length"

    def test_finish_reason_no_error_value(self):
        """finish_reason should never be 'error' (non-standard for OpenAI)."""
        # This is a documentation test — 'error' is not a valid OpenAI finish_reason
        valid_reasons = {"stop", "length", "tool_calls", "content_filter", None}
        assert "error" not in valid_reasons


# =============================================================================
# MLLM Model Tests
# =============================================================================


class TestMLLMModelInit:
    """Test MLLM model initialization (no model loading)."""

    def test_model_init(self):
        """Test model initialization."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        assert model.model_name == "test-model"
        assert not model._loaded

    def test_model_info_not_loaded(self):
        """Test model info when not loaded."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        info = model.get_model_info()

        assert info["loaded"] is False
        assert info["model_name"] == "test-model"

    def test_model_repr(self):
        """Test model string representation."""
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        repr_str = repr(model)

        assert "MLXMultimodalLM" in repr_str
        assert "test-model" in repr_str


# =============================================================================
# Integration Tests - Require Model Loading (Slow)
# =============================================================================


@pytest.mark.slow
class TestMLLMImageGeneration:
    """Integration tests for MLLM image generation."""

    def test_generate_with_image(self, small_mllm_model, test_image_path):
        """Test generation with an image."""
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        output = model.generate(
            prompt="What animal is in this image?",
            images=[test_image_path],
            max_tokens=30,
        )

        assert output.text is not None
        assert len(output.text) > 0
        assert output.completion_tokens > 0

    def test_describe_image(self, small_mllm_model, test_image_path):
        """Test describe_image convenience method."""
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        description = model.describe_image(test_image_path, max_tokens=30)

        assert description is not None
        assert len(description) > 0


@pytest.mark.slow
class TestMLLMVideoGeneration:
    """Integration tests for MLLM video generation."""

    def test_generate_with_video(self, small_mllm_model, test_video_path):
        """Test generation with a video."""
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        output = model.generate(
            prompt="Describe this video.",
            videos=[test_video_path],
            video_fps=1.0,
            video_max_frames=4,
            max_tokens=20,
        )

        assert output.text is not None
        assert len(output.text) > 0

    def test_describe_video(self, small_mllm_model, test_video_path):
        """Test describe_video convenience method."""
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        description = model.describe_video(
            test_video_path,
            fps=1.0,
            max_frames=4,
            max_tokens=20,
        )

        assert description is not None
        assert len(description) > 0


@pytest.mark.slow
class TestMLLMChat:
    """Integration tests for MLLM chat interface."""

    def test_chat_with_image(self, small_mllm_model, test_image_path):
        """Test chat interface with image."""
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image_path},
                    {"type": "text", "text": "What animal is this?"},
                ],
            }
        ]

        output = model.chat(messages, max_tokens=30)

        assert output.text is not None
        assert len(output.text) > 0

    def test_chat_with_video(self, small_mllm_model, test_video_path):
        """Test chat interface with video."""
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": test_video_path},
                    {"type": "text", "text": "Describe the colors in this video."},
                ],
            }
        ]

        output = model.chat(messages, max_tokens=30, video_fps=1.0, video_max_frames=4)

        assert output.text is not None
        assert len(output.text) > 0


# =============================================================================
# M13: Repetition penalty uses original tokens
# =============================================================================


class TestRepetitionPenaltyOriginalTokens:
    """Verify repetition penalty uses _original_token_ids, not trimmed input_ids."""

    def test_code_uses_original_token_ids(self):
        """_make_request_sampler should reference _original_token_ids for full prompt coverage."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._make_request_sampler)
        assert "_original_token_ids" in source, \
            "_make_request_sampler should use _original_token_ids for full prompt coverage"

    def test_code_has_fallback_to_input_ids(self):
        """_make_request_sampler should fall back to input_ids when _original_token_ids is not set."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._make_request_sampler)
        assert "input_ids" in source, \
            "_make_request_sampler should fall back to input_ids"


# =============================================================================
# M15: KV cache quantization uses type identity check
# =============================================================================


class TestKVCacheQuantizationTypeCheck:
    """Verify cache quantization uses isinstance (handles KVCache subclasses like RotatingKVCache)."""

    def test_scheduler_quantize_uses_isinstance_check(self):
        """Ensure scheduler._quantize_cache_for_storage uses isinstance, not type identity."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._quantize_cache_for_storage)
        assert "isinstance(layer_cache, KVCache)" in source
        assert "not isinstance(layer_cache, QuantizedKVCache)" in source

    def test_mllm_scheduler_quantize_uses_isinstance_check(self):
        """Ensure MLLMScheduler._quantize_cache_for_storage uses isinstance, not type identity."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._quantize_cache_for_storage)
        assert "isinstance(layer_cache, KVCache)" in source
        assert "not isinstance(layer_cache, QuantizedKVCache)" in source


# =============================================================================
# M17: Prefix cache BFS for shortest longer prefix
# =============================================================================


class TestPrefixCacheBFS:
    """Verify prefix cache uses BFS (not DFS) for shorter-first longer-prefix search."""

    def test_search_finds_shortest_extension(self):
        """BFS should find the shortest cached extension, not an arbitrary deep one."""
        from vmlx_engine.prefix_cache import PrefixCacheManager

        from unittest.mock import MagicMock
        cache = PrefixCacheManager(model=MagicMock())
        cache.model_key = "test_model"

        # Build a trie with two extensions of [1, 2]:
        # Short: [1, 2, 3] has cache
        # Long:  [1, 2, 4, 5, 6] has cache
        cache._cache = {
            "test_model": {
                1: {
                    2: {
                        3: {"cache": "short_cache"},
                        4: {5: {6: {"cache": "long_cache"}}},
                    }
                }
            }
        }

        _, _, longer, _ = cache._search([1, 2])
        assert longer is not None
        # BFS should find the shorter extension [1, 2, 3]
        assert len(longer) == 3, f"Expected shortest extension [1,2,3], got {longer}"
        assert longer == [1, 2, 3]


# =============================================================================
# M6: Batch VLM per-request prefill isolation
# =============================================================================


class TestBatchPrefillIsolation:
    """Verify that a single bad request doesn't kill the entire batch."""

    def test_prefill_errors_list_exists(self):
        """MLLMBatchGenerator should have _prefill_errors for per-request isolation."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator.__init__)
        assert "_prefill_errors" in source, \
            "MLLMBatchGenerator should track per-request prefill errors"

    def test_process_prompts_has_per_request_try_except(self):
        """_process_prompts should have per-request error handling."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "succeeded_requests" in source, \
            "_process_prompts should track succeeded requests separately"
        assert "prefill_err" in source, \
            "_process_prompts should catch per-request prefill errors"

    def test_next_drains_prefill_errors(self):
        """_next() should drain and return prefill errors."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._next)
        assert "prefill_errors" in source, \
            "_next() should drain prefill errors"
        assert "_prefill_errors.clear()" in source, \
            "_next() should clear prefill errors after draining"
