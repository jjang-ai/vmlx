# SPDX-License-Identifier: Apache-2.0
"""
MLX Multimodal Language Model (MLLM) wrapper.

This module provides a wrapper around mlx-vlm for multimodal inference,
supporting vision, audio, and video understanding on Apple Silicon.

Features:
- OpenAI-compatible API format for images and video
- Smart video frame extraction with configurable FPS
- Base64 and URL image support
- Streaming generation
- MLLM KV cache for repeated image/video+prompt combinations
"""

import atexit
import base64
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import numpy as np
import requests

from vmlx_engine.mllm_cache import MLLMPrefixCacheManager

logger = logging.getLogger(__name__)


class TempFileManager:
    """Thread-safe manager for tracking and cleaning up temporary files."""

    def __init__(self):
        self._files: set[str] = set()
        self._lock = threading.Lock()
        atexit.register(self.cleanup_all)

    def register(self, path: str) -> str:
        """Register a temp file for tracking. Returns the path for convenience."""
        with self._lock:
            self._files.add(path)
        return path

    def cleanup(self, path: str) -> bool:
        """Clean up a specific temp file. Returns True if successful."""
        try:
            if os.path.exists(path):
                os.unlink(path)
                with self._lock:
                    self._files.discard(path)
                logger.debug(f"Cleaned up temp file: {path}")
                return True
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")
        return False

    def cleanup_all(self) -> int:
        """Clean up all tracked temp files. Returns count of cleaned files."""
        with self._lock:
            files_to_clean = list(self._files)
            self._files.clear()

        cleaned = 0
        for path in files_to_clean:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    cleaned += 1
            except OSError:
                pass

        if cleaned:
            logger.info(f"Cleaned up {cleaned} temp files")
        return cleaned


# Global temp file manager
_temp_manager = TempFileManager()



# Video processing constants
FRAME_FACTOR = 2  # Frames must be divisible by this
DEFAULT_FPS = 2.0  # Default frames per second for video
MIN_FRAMES = 4
MAX_FRAMES = 128  # Practical limit for most MLLMs
IMAGE_FACTOR = 28  # For smart resize

# Security: File size limits (in bytes)
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB max for images
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB max for videos
MAX_BASE64_IMAGE_LENGTH = 30 * 1024 * 1024  # 30 MB base64 string (~22 MB decoded)
MAX_BASE64_VIDEO_LENGTH = 700 * 1024 * 1024  # 700 MB base64 string (~500 MB decoded)


class FileSizeExceededError(Exception):
    """Raised when a downloaded file exceeds the size limit."""

    pass


@dataclass
class MultimodalInput:
    """Input for multimodal generation."""

    prompt: str
    images: list[str] = field(default_factory=list)  # Paths, URLs, or base64
    videos: list[str] = field(default_factory=list)  # Paths
    audio: list[str] = field(default_factory=list)  # Paths


@dataclass
class MLLMOutput:
    """Output from multimodal language model."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def is_base64_image(s: str) -> bool:
    """Check if string is base64-encoded image data."""
    return s.startswith("data:image/") or (
        len(s) > 100 and not s.startswith(("http://", "https://", "/"))
    )


def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://"))


def is_base64_video(s: str) -> bool:
    """Check if string is base64-encoded video data."""
    return s.startswith("data:video/")


def decode_base64_image(
    base64_string: str, max_length: int = MAX_BASE64_IMAGE_LENGTH
) -> bytes:
    """
    Decode base64 image to bytes.

    Args:
        base64_string: Base64 encoded image (optionally with data URL prefix)
        max_length: Maximum allowed length of base64 string

    Returns:
        Decoded image bytes

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 image data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
    if base64_string.startswith("data:"):
        # Extract the base64 part after the comma
        _, data = base64_string.split(",", 1)
        return base64.b64decode(data)
    return base64.b64decode(base64_string)


def download_image(url: str, timeout: int = 30, max_size: int = MAX_IMAGE_SIZE) -> str:
    """
    Download image from URL and return local path.

    Args:
        url: Image URL
        timeout: Download timeout in seconds
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded image

    Raises:
        FileSizeExceededError: If image exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    # First, make a HEAD request to check Content-Length
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "png" in content_type:
        ext = ".png"
    elif "gif" in content_type:
        ext = ".gif"
    elif "webp" in content_type:
        ext = ".webp"
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = Path(path).suffix or ".jpg"

    # Save to temp file with size checking during download
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Image at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    return _temp_manager.register(temp_file.name)


def download_video(url: str, timeout: int = 120, max_size: int = MAX_VIDEO_SIZE) -> str:
    """
    Download video from URL and return local path.

    Args:
        url: Video URL (http/https)
        timeout: Download timeout in seconds (default 120s for larger videos)
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded video

    Raises:
        FileSizeExceededError: If video exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    logger.info(f"Downloading video from: {url}")

    # First, make a HEAD request to check Content-Length
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Video at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Video at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "mp4" in content_type:
        ext = ".mp4"
    elif "webm" in content_type:
        ext = ".webm"
    elif "avi" in content_type:
        ext = ".avi"
    elif "mov" in content_type or "quicktime" in content_type:
        ext = ".mov"
    elif "mkv" in content_type:
        ext = ".mkv"
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = Path(path).suffix or ".mp4"

    # Save to temp file (stream for larger files) with size checking
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Video at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    file_size = Path(temp_file.name).stat().st_size
    logger.info(
        f"Video downloaded: {temp_file.name} ({file_size / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_video(
    base64_string: str, max_length: int = MAX_BASE64_VIDEO_LENGTH
) -> str:
    """
    Decode base64 video to temp file and return path.

    Supports format: data:video/mp4;base64,AAAA...

    Args:
        base64_string: Base64-encoded video with data URL prefix
        max_length: Maximum allowed length of base64 string

    Returns:
        Local file path to decoded video

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 video data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Extract format and data
    if base64_string.startswith("data:video/"):
        # Format: data:video/mp4;base64,AAAA...
        header, data = base64_string.split(",", 1)
        # Extract extension from header (e.g., "data:video/mp4;base64" -> "mp4")
        format_part = header.split(";")[0]  # "data:video/mp4"
        ext = "." + format_part.split("/")[-1]  # ".mp4"
    else:
        # Assume mp4 if no header
        data = base64_string
        ext = ".mp4"

    # Decode, save, and register for cleanup
    video_bytes = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(video_bytes)
    temp_file.close()

    logger.info(
        f"Base64 video decoded: {temp_file.name} ({len(video_bytes) / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def process_video_input(video: str | dict) -> str:
    """
    Process video input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string (data:video/mp4;base64,...)
    - OpenAI format dict: {"url": "..."} or {"url": "data:video/...;base64,..."}

    Args:
        video: Video input in any supported format

    Returns:
        Local file path to video
    """
    # Handle dict format (OpenAI style)
    if isinstance(video, dict):
        url = video.get("url", video.get("video_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        video = url

    if not video:
        raise ValueError("Empty video input")

    # Check if it's a local file
    if Path(video).exists():
        return video

    # Check if it's a URL
    if is_url(video):
        return download_video(video)

    # Check if it's base64
    if is_base64_video(video):
        return decode_base64_video(video)

    raise ValueError(f"Cannot process video: {video[:50]}...")


# Cache for base64 images to avoid re-saving the same image
# OrderedDict for LRU eviction with bounded size
from collections import OrderedDict

_base64_image_cache: OrderedDict[str, str] = OrderedDict()  # hash -> temp file path
_BASE64_IMAGE_CACHE_MAX_SIZE = 100


def save_base64_image(base64_string: str) -> str:
    """Save base64 image to temp file and return path. Caches identical images."""
    import hashlib

    # Hash the full base64 string to avoid collisions
    image_hash = hashlib.sha256(base64_string.encode()).hexdigest()

    # Return cached path if available and file still exists
    if image_hash in _base64_image_cache:
        cached_path = _base64_image_cache[image_hash]
        if Path(cached_path).exists():
            _base64_image_cache.move_to_end(image_hash)  # LRU: mark as recently used
            return cached_path
        else:
            # File was cleaned up, remove stale entry
            del _base64_image_cache[image_hash]

    image_bytes = decode_base64_image(base64_string)

    # Detect format from magic bytes
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif image_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        ext = ".gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        ext = ".webp"
    else:
        ext = ".jpg"  # Default

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    path = _temp_manager.register(temp_file.name)
    _base64_image_cache[image_hash] = path

    # Evict oldest entries if cache exceeds size limit
    # M5: Also delete the temp file on eviction (not just the cache entry)
    while len(_base64_image_cache) > _BASE64_IMAGE_CACHE_MAX_SIZE:
        _, evicted_path = _base64_image_cache.popitem(last=False)
        _temp_manager.cleanup(evicted_path)

    return path


def process_image_input(image: str | dict) -> str:
    """
    Process image input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string
    - OpenAI format dict: {"url": "..."} or {"url": "data:image/...;base64,..."}
    """
    # Handle dict format (OpenAI style)
    if isinstance(image, dict):
        url = image.get("url", image.get("image_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        image = url

    if not image:
        raise ValueError("Empty image input")

    # Check if it's base64 FIRST (before Path.exists() which fails on long strings)
    if is_base64_image(image):
        return save_base64_image(image)

    # Check if it's a URL
    if is_url(image):
        return download_image(image)

    # Check if it's a local file (only for short strings that could be paths)
    if len(image) < 4096 and Path(image).exists():
        return image

    raise ValueError(f"Cannot process image: {image[:50]}...")


def round_by_factor(x: int, factor: int) -> int:
    """Round to nearest multiple of factor."""
    return round(x / factor) * factor


def ceil_by_factor(x: float, factor: int) -> int:
    """Ceiling to next multiple of factor."""
    return math.ceil(x / factor) * factor


def floor_by_factor(x: float, factor: int) -> int:
    """Floor to previous multiple of factor."""
    return math.floor(x / factor) * factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: float = DEFAULT_FPS,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_FRAMES,
) -> int:
    """
    Calculate optimal number of frames to extract from video.

    Uses smart sampling based on video length and target FPS.
    """
    # Calculate duration-based frame count
    duration = total_frames / video_fps if video_fps > 0 else 0
    nframes = duration * target_fps

    # Clamp to min/max
    nframes = max(min_frames, min(nframes, max_frames, total_frames))

    # Round to factor
    nframes = max(FRAME_FACTOR, floor_by_factor(nframes, FRAME_FACTOR))

    return int(nframes)


def extract_video_frames_smart(
    video_path: str,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """
    Extract frames from video with smart sampling.

    Args:
        video_path: Path to video file
        fps: Target frames per second (default: 2.0)
        max_frames: Maximum frames to extract
        resize: Optional (width, height) to resize frames

    Returns:
        List of frame arrays (RGB format)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculate number of frames to extract
    nframes = smart_nframes(
        total_frames=total_frames,
        video_fps=video_fps,
        target_fps=fps,
        max_frames=max_frames,
    )

    # Calculate frame indices (evenly spaced)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    logger.info(
        f"Video: {total_frames} total frames @ {video_fps:.1f} fps, "
        f"extracting {nframes} frames"
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if specified
        if resize:
            frame = cv2.resize(frame, resize)

        frames.append(frame)

    cap.release()

    return frames


def save_frames_to_temp(frames: list[np.ndarray]) -> list[str]:
    """Save frame arrays to temporary files and return paths."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for frame processing")

    paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file.name, "JPEG", quality=85)
        paths.append(_temp_manager.register(temp_file.name))

    return paths


class MLXMultimodalLM:
    """
    Wrapper around mlx-vlm for multimodal inference.

    This class provides a unified interface for multimodal language models
    using Apple's MLX framework. Supports:
    - Image understanding (single and multi-image)
    - Video understanding (smart frame extraction)
    - Audio understanding (for supported models)
    - OpenAI-compatible API format

    Supported models include:
    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    - LLaVA
    - Idefics3
    - PaliGemma
    - And more via mlx-vlm

    Example:
        >>> model = MLXMultimodalLM("mlx-community/Qwen2-VL-2B-Instruct-4bit")
        >>> model.load()
        >>> output = model.generate(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        enable_cache: bool = True,
        cache_size: int = 50,
    ):
        """
        Initialize the MLX multimodal language model.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable KV cache for repeated image/video+prompt (default: True)
            cache_size: Maximum cache entries (default: 50)
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.enable_cache = enable_cache

        self.model = None
        self.processor = None
        self.config = None
        self._loaded = False

        # Initialize MLLM prefix cache manager (with vision embedding caching)
        self._cache_manager: MLLMPrefixCacheManager | None = None
        if enable_cache:
            self._cache_manager = MLLMPrefixCacheManager(max_entries=cache_size)

    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        # JANG VL models: use JANG loader (handles mixed-precision + mlx-vlm sanitization)
        from ..utils.jang_loader import is_jang_model
        if is_jang_model(self.model_name):
            logger.info(f"Loading JANG VL model: {self.model_name}")
            from ..utils.jang_loader import load_jang_vlm_model
            from mlx_vlm.utils import load_config
            self.model, self.processor = load_jang_vlm_model(self.model_name)
            self.config = load_config(self.model_name)
            self._loaded = True
            logger.info(f"JANG VL model loaded: {self.model_name}")
            return

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            logger.info(f"Loading MLLM: {self.model_name}")

            # Bundled Python doesn't include torch/torchvision (~2GB). Preemptively
            # patch video processor loading and suppress warnings so models with
            # video_processor sub-components (Qwen3.5-VL, InternVL, etc.) load cleanly.
            self._patch_video_processor()
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchvision.*", category=UserWarning)
                self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)

            self._loaded = True
            logger.info(f"MLLM loaded successfully: {self.model_name}")

        except ImportError:
            raise ImportError(
                "mlx-vlm is required for multimodal inference. "
                "Install with: pip install mlx-vlm"
            )
        except Exception as e:
            logger.error(f"Failed to load MLLM: {e}")
            raise

    _video_processor_patched = False

    @classmethod
    def _patch_video_processor(cls):
        """Patch AutoVideoProcessor to skip torchvision requirement.

        Some VLM models (Qwen3.5-VL, InternVL) include a video_processor
        sub-component that requires torchvision. Since we only do image inference
        and torchvision requires torch (~2GB), we patch the processor to load
        without it. Called preemptively before first model load. Idempotent.
        """
        if cls._video_processor_patched:
            return

        try:
            from transformers.models.auto import video_processing_auto

            _orig_from_pretrained = video_processing_auto.AutoVideoProcessor.from_pretrained

            @classmethod  # type: ignore[misc]
            def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                try:
                    return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
                except (ValueError, ImportError) as e:
                    if "torchvision" in str(e):
                        logger.info("Video processor skipped (torchvision not available) — image-only mode")
                        return None
                    raise

            video_processing_auto.AutoVideoProcessor.from_pretrained = _patched_from_pretrained
            cls._video_processor_patched = True
            logger.debug("Patched AutoVideoProcessor for torchvision-free loading")
        except Exception:
            logger.warning("Failed to patch AutoVideoProcessor — torchvision-dependent VLMs may fail to load")
            cls._video_processor_patched = True  # Don't retry on every load, log warning once

    def _prepare_images(self, images: list) -> list[str]:
        """Process image inputs and return local file paths."""
        processed = []
        failed_count = 0
        for img in images:
            try:
                path = process_image_input(img)
                processed.append(path)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to process image: {e}")
        if images and not processed:
            raise ValueError(
                f"All {failed_count} image(s) failed to process. "
                f"Check image URLs/paths and try again."
            )
        return processed

    def _prepare_video(
        self,
        video_input: str | dict,
        fps: float = DEFAULT_FPS,
        max_frames: int = MAX_FRAMES,
    ) -> list[str]:
        """
        Process video input and extract frames.

        Supports:
        - Local file paths
        - URLs (http/https) - will be downloaded
        - Base64 encoded videos (data:video/mp4;base64,...)
        - OpenAI format dicts: {"url": "..."} or {"video_url": {"url": "..."}}

        Args:
            video_input: Video in any supported format
            fps: Frames per second to extract
            max_frames: Maximum frames to extract

        Returns:
            List of paths to extracted frame images
        """
        # Process video input (download if URL, decode if base64)
        video_path = process_video_input(video_input)

        # Extract frames
        frames = extract_video_frames_smart(
            video_path,
            fps=fps,
            max_frames=max_frames,
        )
        return save_frames_to_temp(frames)

    @staticmethod
    def _extract_multimodal_messages(
        messages: list[dict],
    ) -> tuple[list[dict], list[str], list]:
        """
        Parse OpenAI-format messages into chat_messages, image URLs, and videos.

        Extracts text, images, and videos from multimodal message content,
        building properly structured chat messages for Qwen3-VL-MoE and similar
        models that expect image tokens before text in user messages.

        Args:
            messages: List of chat messages in OpenAI format

        Returns:
            Tuple of (chat_messages, all_image_urls, videos) where:
            - chat_messages: Properly structured messages for chat template
            - all_image_urls: Raw image URLs/paths to process
            - videos: Raw video inputs to process
        """
        all_image_urls: list[str] = []
        videos: list = []
        chat_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_text = ""
            msg_image_count = 0

            if isinstance(content, str):
                msg_text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        msg_text += item
                        continue

                    # Convert Pydantic models to dicts.
                    # exclude_none=True prevents Jinja2 false key-existence bugs
                    # (checking 'if image_url' returns True for None values). If a
                    # field should be explicitly None (not absent), it would need
                    # special handling.
                    if hasattr(item, "model_dump"):
                        item = item.model_dump(exclude_none=True)
                    elif hasattr(item, "dict"):
                        item = item.dict()
                        # Remove None values to match exclude_none behavior
                        item = {k: v for k, v in item.items() if v is not None}

                    if isinstance(item, dict):
                        item_type = item.get("type", "")

                        if item_type == "text":
                            msg_text += item.get("text", "")

                        elif item_type == "image_url":
                            img_url = item.get("image_url", {})
                            if isinstance(img_url, str):
                                all_image_urls.append(img_url)
                            else:
                                all_image_urls.append(img_url.get("url", ""))
                            msg_image_count += 1

                        elif item_type == "image":
                            all_image_urls.append(
                                item.get("image", item.get("url", ""))
                            )
                            msg_image_count += 1

                        elif item_type == "video":
                            videos.append(item.get("video", item.get("url", "")))

                        elif item_type == "video_url":
                            vid_url = item.get("video_url", {})
                            if isinstance(vid_url, str):
                                videos.append(vid_url)
                            else:
                                videos.append(vid_url.get("url", ""))

            # Build properly structured message for Qwen3-VL-MoE
            # Format: {"role": "...", "content": [{"type": "image"}, ..., {"type": "text", "text": "..."}]}
            # Preserve tool_calls on assistant messages and tool role fields
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            msg_name = msg.get("name")

            if msg_text or msg_image_count > 0 or tool_calls or role == "tool":
                if role == "user" and msg_image_count > 0:
                    content_list: list[dict] = []
                    for _ in range(msg_image_count):
                        content_list.append({"type": "image"})
                    content_list.append(
                        {"type": "text", "text": msg_text, "content": msg_text}
                    )
                    chat_messages.append({"role": role, "content": content_list})
                elif role == "assistant":
                    out_msg: dict = {"role": role, "content": msg_text}
                    if tool_calls:
                        out_msg["tool_calls"] = tool_calls
                    chat_messages.append(out_msg)
                elif role == "tool":
                    out_msg = {"role": role, "content": msg_text}
                    if tool_call_id:
                        out_msg["tool_call_id"] = tool_call_id
                    if msg_name:
                        out_msg["name"] = msg_name
                    chat_messages.append(out_msg)
                else:
                    chat_messages.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "text", "text": msg_text, "content": msg_text}
                            ],
                        }
                    )

        return chat_messages, all_image_urls, videos

    def _apply_chat_template(
        self,
        chat_messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> str:
        """
        Apply chat template to structured messages with enable_thinking support.

        Handles TypeError fallback (for processors that don't support
        enable_thinking), general exception fallback to last user message,
        and stripping of forced reasoning loops for abliterated models.

        Args:
            chat_messages: Structured chat messages from _extract_multimodal_messages()
            enable_thinking: Whether to enable thinking mode (None = don't pass)

        Returns:
            Formatted prompt string
        """
        from mlx_vlm.prompt_utils import get_chat_template

        template_kwargs = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        formatted_prompt = None
        try:
            formatted_prompt = get_chat_template(
                self.processor,
                chat_messages,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError:
            # Processor doesn't support enable_thinking kwarg -- retry without it
            try:
                formatted_prompt = get_chat_template(
                    self.processor,
                    chat_messages,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}, using last user message"
                )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, using last user message"
            )

        # Ensure stripping of forced reasoning loops for abliterated models
        if enable_thinking is False and formatted_prompt:
            if formatted_prompt.endswith("<think>\n"):
                formatted_prompt = formatted_prompt[:-8]
            elif formatted_prompt.endswith("<think>"):
                formatted_prompt = formatted_prompt[:-7]

        if formatted_prompt is None:
            # Fallback to last user message if template fails
            last_user_msg = ""
            for m in reversed(chat_messages):
                if m["role"] == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                last_user_msg = item.get("text", "")
                                break
                    else:
                        last_user_msg = content
                    break
            formatted_prompt = last_user_msg

        return formatted_prompt

    def generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list | None = None,
        audio: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        use_cache: bool = True,
        **kwargs,
    ) -> MLLMOutput:
        """
        Generate text from multimodal input.

        Args:
            prompt: Text prompt/question
            images: List of image paths, URLs, or base64 strings
            videos: List of video inputs (paths, URLs, base64, or OpenAI format dicts)
            audio: List of audio file paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            video_fps: FPS for video frame extraction (default: 2.0)
            video_max_frames: Max frames to extract from video
            use_cache: Whether to use KV cache (default: True)
            **kwargs: Additional generation parameters

        Returns:
            MLLMOutput with generated text

        Example:
            # With local video
            output = model.generate("Describe this video", videos=["video.mp4"])

            # With video URL
            output = model.generate("What happens?", videos=["https://example.com/video.mp4"])

            # With base64 video
            output = model.generate("Describe", videos=["data:video/mp4;base64,AAAA..."])
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate
        from mlx_vlm.models import cache as vlm_cache
        from mlx_vlm.prompt_utils import apply_chat_template

        images = images or []
        videos = videos or []
        audio = audio or []

        # Process all images (including frames from videos)
        all_images = []
        all_sources = []  # Track original sources for cache key

        # Process image inputs
        if images:
            all_images.extend(self._prepare_images(images))
            all_sources.extend(images)

        # Extract frames from videos
        for video_path in videos:
            frames = self._prepare_video(
                video_path,
                fps=video_fps,
                max_frames=video_max_frames,
            )
            all_images.extend(frames)
            # Include video params in cache key
            video_str = video_path if isinstance(video_path, str) else str(video_path)
            all_sources.append(
                f"video:{video_str}:fps{video_fps}:max{video_max_frames}"
            )
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template if needed
        if all_images and hasattr(self.processor, "apply_chat_template"):
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        # Check cache for existing KV state
        prompt_cache = None
        cache_hit = False

        if use_cache and self._cache_manager is not None and all_sources:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = tokenizer.encode(formatted_prompt)
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_sources, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    prompt_cache = cache_entry.kv_cache
                    cache_hit = True
                    logger.info(
                        f"MLLM cache hit for {len(all_sources)} source(s)"
                        + (f", {prefix_match_len} prefix tokens match" if prefix_match_len > 0 else "")
                    )
            except Exception as e:
                logger.debug(f"Generate cache fetch failed: {e}")

        # Create new cache if needed
        if prompt_cache is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Generate with cache
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
            prompt_cache=prompt_cache,
            **kwargs,
        )

        # Store cache for future reuse (only on miss)
        if use_cache and self._cache_manager and all_sources and not cache_hit:
            if prompt_cache is not None:
                try:
                    num_tokens = getattr(result, "prompt_tokens", 0)
                    self._cache_manager.store_cache(
                        all_sources, formatted_prompt, prompt_cache, num_tokens
                    )
                    logger.info(f"MLLM cache stored for {len(all_sources)} source(s)")
                except Exception as e:
                    logger.debug(f"Failed to store MLLM cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        finish_reason = "length" if generation_tokens >= max_tokens else "stop"

        return MLLMOutput(
            text=output_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
        )

    def stream_generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream text generation for multimodal input.

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_fps: FPS for video frame extraction
            video_max_frames: Maximum video frames to extract
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError:
            # Fallback to non-streaming
            output = self.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                video_fps=video_fps,
                **kwargs,
            )
            yield output.text
            return

        images = images or []
        videos = videos or []

        # Process images
        all_images = []
        if images:
            all_images.extend(self._prepare_images(images))
        for video_path in videos:
            frames = self._prepare_video(video_path, fps=video_fps, max_frames=video_max_frames)
            all_images.extend(frames)

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        if all_images:
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        for chunk in stream_generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> MLLMOutput:
        """
        Chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            MLLMOutput with assistant's response
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate

        # Extract text, images, and videos from messages
        chat_messages, all_image_urls, videos = self._extract_multimodal_messages(messages)

        logger.info(f"MLLM.chat() called with {len(messages)} messages")

        # Process images
        all_images = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))

        # Process videos
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        for video_path in videos:
            frames = self._prepare_video(
                video_path, fps=video_fps, max_frames=video_max_frames
            )
            all_images.extend(frames)
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        logger.info(
            f"Applying chat template with {len(chat_messages)} messages, {len(all_images)} images"
        )
        for i, cm in enumerate(chat_messages):
            content_preview = str(cm.get("content", ""))[:80]
            logger.info(
                f"  Chat msg {i}: role={cm['role']}, content={content_preview}..."
            )
        enable_thinking = kwargs.pop("enable_thinking", None)
        formatted_prompt = self._apply_chat_template(chat_messages, enable_thinking)

        # Prefix caching with vision embedding support
        # Following LMCache approach: cache vision embeddings to skip encoder on hit
        from mlx_vlm.models import cache as vlm_cache
        import time

        use_cache = kwargs.pop("use_cache", True)
        cache_entry = None
        prefix_match_len = 0
        cache_hit = False

        # Tokenize prompt for cache lookup
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        token_ids = tokenizer.encode(formatted_prompt)

        # Check prefix cache
        if use_cache and self._cache_manager is not None and all_images:
            try:
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry:
                    cache_hit = True
                    # NOTE: cache_entry.vision_embeddings is not used — mlx-vlm's
                    # generate() does not accept pre-computed vision embeddings.
                    # The KV cache hit path below handles the actual speedup.
                    if prefix_match_len > 0:
                        logger.info(
                            f"[PREFIX CACHE] {prefix_match_len} prefix tokens match"
                        )
            except Exception as e:
                logger.warning(f"Cache fetch failed: {e}")

        # Generate - use KV cache if available from previous identical request
        start_time = time.time()

        # Create or reuse prompt cache for prefix caching speedup
        prompt_cache = None
        skip_prompt_processing = False

        if cache_hit and cache_entry and cache_entry.kv_cache:
            # NOTE: mlx-vlm's generate_step() has its own multimodal KV cache with prefix matching
            # (MULTIMODAL_KV_CACHE_ENABLED in mlx_vlm/utils.py). Let it handle caching.
            # We only use vmlx-engine's cache for text-only requests (no images).
            if all_images:
                # Let mlx-vlm's multimodal cache handle this - don't interfere
                logger.info(
                    "[PREFIX CACHE] Images present - delegating to mlx-vlm multimodal cache"
                )
                prompt_cache = None  # Fresh cache, mlx-vlm will handle prefix matching
                skip_prompt_processing = False
            else:
                # Text-only: can use skip_prompt_processing for maximum speedup
                logger.info(
                    "[PREFIX CACHE] Text-only cache hit - using skip_prompt_processing speedup"
                )
                cached_prompt_cache = cache_entry.kv_cache
                try:
                    import copy

                    prompt_cache = []
                    for layer_cache in cached_prompt_cache:
                        new_cache = copy.copy(layer_cache)
                        if hasattr(layer_cache, "state"):
                            state = layer_cache.state
                            if state is not None:
                                import mlx.core as mx

                                if len(state) >= 2 and state[0] is not None:
                                    new_cache.keys = mx.array(state[0])
                                    new_cache.values = mx.array(state[1])
                                    if len(state) >= 3:
                                        new_cache.offset = state[2]
                                    elif hasattr(layer_cache, "offset"):
                                        new_cache.offset = layer_cache.offset
                        prompt_cache.append(new_cache)
                    skip_prompt_processing = True
                    logger.info(
                        f"[PREFIX CACHE] Skipping {prefix_match_len} token forward pass"
                    )
                except Exception as e:
                    logger.warning(f"[PREFIX CACHE] Failed to copy cache: {e}")
                    prompt_cache = None
                    skip_prompt_processing = False

        if prompt_cache is None and self.model is not None:
            # Create fresh cache
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Extract sampling params not natively supported by mlx_vlm.generate()
        # (top_k, min_p require a custom sampler; they're silently ignored otherwise)
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            from mlx_lm.sample_utils import make_sampler
            top_p = kwargs.pop("top_p", 1.0)
            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
            prompt_cache=prompt_cache,
            skip_prompt_processing=skip_prompt_processing,
            **kwargs,
        )

        # Store KV cache for future reuse (on cache miss)
        # IMPORTANT: We need to store only the prompt portion, not generated tokens
        if (
            use_cache
            and self._cache_manager is not None
            and all_images
            and not cache_hit
            and prompt_cache
        ):
            try:
                import copy
                import mlx.core as mx

                # Get prompt token count (before generation)
                prompt_tokens_count = getattr(result, "prompt_tokens", 0)

                # Deep copy the cache and trim to prompt tokens only
                cache_to_store = []
                for layer_cache in prompt_cache:
                    new_cache = copy.copy(layer_cache)
                    if hasattr(layer_cache, "state"):
                        state = layer_cache.state
                        if (
                            state is not None
                            and len(state) >= 2
                            and state[0] is not None
                        ):
                            # Copy arrays
                            keys = mx.array(state[0])
                            values = mx.array(state[1])
                            # Trim to prompt tokens only (not generated tokens)
                            if (
                                hasattr(layer_cache, "offset")
                                and layer_cache.offset > prompt_tokens_count
                            ):
                                # For caches with offset tracking, slice to prompt length
                                new_cache.keys = keys[:, :, :prompt_tokens_count, :]
                                new_cache.values = values[:, :, :prompt_tokens_count, :]
                                new_cache.offset = prompt_tokens_count
                            else:
                                new_cache.keys = keys
                                new_cache.values = values
                                if len(state) >= 3:
                                    new_cache.offset = state[2]
                                elif hasattr(layer_cache, "offset"):
                                    new_cache.offset = min(
                                        layer_cache.offset, prompt_tokens_count
                                    )
                    cache_to_store.append(new_cache)

                # Estimate num_image_tokens from the model config or token IDs.
                # Different models use different counts (e.g., Gemma3=256, Qwen2-VL varies).
                num_img_tokens = 0
                if all_images and self.config:
                    # Try to get image_token_index from config and count occurrences in token_ids
                    img_token_id = getattr(self.config, "image_token_index", None)
                    if img_token_id is not None and token_ids:
                        num_img_tokens = token_ids.count(img_token_id)
                    if num_img_tokens == 0:
                        # Fallback: use prompt_tokens_count minus a rough text estimate
                        # or just leave as 0 (cache stats only, not critical for correctness)
                        num_img_tokens = 0

                self._cache_manager.store(
                    images=all_images,
                    prompt=formatted_prompt,
                    vision_embeddings=None,  # mlx-vlm doesn't support pre-computed embeddings
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=num_img_tokens,
                    model_name=self.model_name,
                )
                logger.info(
                    f"[PREFIX CACHE] Stored KV cache for {len(all_images)} image(s) ({prompt_tokens_count} prompt tokens)"
                )
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        finish_reason = "length" if generation_tokens >= max_tokens else "stop"

        return MLLMOutput(
            text=output_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
        )

    def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[MLLMOutput]:
        """
        Stream chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            MLLMOutput with incremental text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
        except ImportError:
            # Fallback to non-streaming if stream_generate not available
            output = self.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            yield output
            return

        # Extract text, images, and videos from messages
        chat_messages, all_image_urls, videos = self._extract_multimodal_messages(messages)

        # Process images
        all_images = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))

        # Process videos
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        for video_path in videos:
            frames = self._prepare_video(
                video_path, fps=video_fps, max_frames=video_max_frames
            )
            all_images.extend(frames)

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        enable_thinking = kwargs.pop("enable_thinking", None)
        formatted_prompt = self._apply_chat_template(chat_messages, enable_thinking)

        # Check cache for existing KV state (uses images as cache key)
        from mlx_vlm.models import cache as vlm_cache

        prompt_cache = None
        cache_hit = False
        use_cache = kwargs.pop("use_cache", True)

        if use_cache and self._cache_manager is not None and all_images:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = tokenizer.encode(formatted_prompt)
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    prompt_cache = cache_entry.kv_cache
                    cache_hit = True
                    if prefix_match_len > 0:
                        logger.debug(
                            f"Stream chat prefix cache hit: {prefix_match_len} tokens, "
                            f"{len(all_images)} image(s)"
                        )
                    else:
                        logger.debug(f"Stream chat cache hit for {len(all_images)} image(s)")
            except Exception as e:
                logger.debug(f"Stream chat cache fetch failed: {e}")

        # Create new cache if needed
        if prompt_cache is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Stream generate tokens with cache
        accumulated_text = ""
        token_count = 0
        last_prompt_tokens = 0

        # Ensure processor uses NaiveStreamingDetokenizer for all MLLM models.
        # mlx-vlm's default streaming detokenizer can buffer infinitely for some
        # architectures. NaiveStreamingDetokenizer yields tokens immediately.
        if not hasattr(self.processor, "_patched_detok"):
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

            # Monkey-patch NaiveStreamingDetokenizer to accept kwargs from mlx_vlm
            if not hasattr(NaiveStreamingDetokenizer, "_vml_patched"):
                original_add = NaiveStreamingDetokenizer.add_token
                def _patched_add(self, token, **kwargs):
                    return original_add(self, token)
                NaiveStreamingDetokenizer.add_token = _patched_add
                NaiveStreamingDetokenizer._vml_patched = True

            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            self.processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
            self.processor._patched_detok = True

        # Extract sampling params not natively supported by mlx_vlm.stream_generate()
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            from mlx_lm.sample_utils import make_sampler
            top_p = kwargs.pop("top_p", 1.0)
            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )

        try:
            for chunk in stream_generate(
                self.model,
                self.processor,
                formatted_prompt,
                all_images if all_images else None,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_cache=prompt_cache,
                **kwargs,
            ):
                token_count += 1
                # chunk is a GenerationResult with .text attribute containing the new token
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                accumulated_text += new_text

                chunk_prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                if chunk_prompt_tokens:
                    last_prompt_tokens = chunk_prompt_tokens

                yield MLLMOutput(
                    text=new_text,  # Just the new token for streaming
                    finish_reason=None,
                    prompt_tokens=last_prompt_tokens,
                    completion_tokens=token_count,
                )
        except Exception as e:
            # Catch OOM and other generation errors to prevent server crash.
            # Try to free Metal memory before reporting the error.
            try:
                import mlx.core as mx
                mx.clear_memory_cache()
            except Exception:
                pass
            logger.error(f"VLM stream_generate error: {type(e).__name__}: {e}")
            yield MLLMOutput(
                text=f"\n\n[Generation error: {type(e).__name__}: {e}]",
                finish_reason="stop",
                prompt_tokens=0,
                completion_tokens=token_count,
            )
            return

        # Final yield with finish_reason
        finish_reason = "length" if token_count >= max_tokens else "stop"
        yield MLLMOutput(
            text="",
            finish_reason=finish_reason,
            prompt_tokens=last_prompt_tokens,
            completion_tokens=token_count,
        )

    def describe_image(
        self,
        image: str,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Convenience method to describe an image.

        Args:
            image: Image path, URL, or base64 string
            prompt: Description prompt
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Image description text
        """
        output = self.generate(
            prompt=prompt,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def answer_about_image(
        self,
        image: str,
        question: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: Image path, URL, or base64 string
            question: Question about the image
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Answer text
        """
        output = self.generate(
            prompt=question,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def describe_video(
        self,
        video: str | dict,
        prompt: str = "Describe what happens in this video.",
        fps: float = 2.0,
        max_frames: int = 32,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Describe a video using frame extraction.

        Args:
            video: Video file path, URL, base64, or OpenAI format dict
            prompt: Description prompt
            fps: Frames per second to extract
            max_frames: Maximum frames to extract
            max_tokens: Maximum tokens to generate

        Returns:
            Video description text

        Example:
            # Local file
            model.describe_video("video.mp4")

            # URL
            model.describe_video("https://example.com/video.mp4")

            # OpenAI format
            model.describe_video({"url": "https://example.com/video.mp4"})
        """
        output = self.generate(
            prompt=prompt,
            videos=[video],
            video_fps=fps,
            video_max_frames=max_frames,
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def get_cache_stats(self) -> dict:
        """
        Get MLLM cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit_rate, tokens_saved, etc.)
        """
        if self._cache_manager is None:
            return {"enabled": False}

        stats = self._cache_manager.get_stats()
        stats["enabled"] = True
        stats["cache_entries"] = len(self._cache_manager)
        stats["max_entries"] = self._cache_manager.max_size
        return stats

    def clear_cache(self) -> None:
        """Clear the MLLM KV cache."""
        if self._cache_manager is not None:
            self._cache_manager.clear()
            logger.info("MLLM cache cleared")

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "type": "multimodal-language-model",
            "supports_video": True,
            "supports_streaming": True,
            "cache_enabled": self.enable_cache,
        }

        if self.config:
            info["model_type"] = getattr(self.config, "model_type", "unknown")

        if self._cache_manager is not None:
            info["cache_stats"] = self._cache_manager.get_stats()

        return info

    @staticmethod
    def list_supported_model_families() -> dict[str, str]:
        """
        List supported model families and their patterns.

        Any model on HuggingFace containing these patterns in the name
        is likely compatible with mlx-vlm.
        """
        return {
            "Qwen-VL": "Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)",
            "LLaVA": "LLaVA vision-language models",
            "Idefics": "Idefics vision-language models",
            "PaliGemma": "PaliGemma multimodal models",
            "Pixtral": "Mistral's Pixtral vision models",
            "Molmo": "Allen AI's Molmo models",
            "Phi-3-Vision": "Microsoft's Phi-3 Vision models",
            "CogVLM": "Tsinghua's CogVLM models",
            "InternVL": "InternVL models",
            "MiniCPM-V": "OpenBMB's MiniCPM-V models",
            "Florence": "Microsoft Florence vision models",
            "DeepSeek-VL": "DeepSeek's vision-language models (DeepSeek-VL, DeepSeek-VL2)",
        }

    @staticmethod
    def is_mllm_model(model_name: str) -> bool:
        """Check if a model name indicates an MLLM model.

        Delegates to the canonical implementation in api.utils which checks
        config.json, model registry, and regex patterns.
        """
        from ..api.utils import is_mllm_model as _canonical
        return _canonical(model_name)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXMultimodalLM model={self.model_name} status={status}>"


# Backwards compatibility aliases
MLXVisionLanguageModel = MLXMultimodalLM
VLMOutput = MLLMOutput
is_vlm_model = MLXMultimodalLM.is_mllm_model
