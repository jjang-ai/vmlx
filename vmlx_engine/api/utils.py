# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import functools
import json
import os
import re

from .models import Message

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]"
)


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens.

    Keeps <think>...</think> blocks intact for reasoning models.
    Adds opening <think> tag if missing (happens when thinking is enabled
    in the prompt template but the tag is part of the prompt, not output).

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens removed
    """
    if not text:
        return text
    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    text = text.strip()

    # Add opening <think> tag if response has closing but not opening
    # This happens when enable_thinking=True in the chat template
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return text


# =============================================================================
# Model Detection
# =============================================================================


@functools.lru_cache(maxsize=32)
def resolve_to_local_path(model_name: str) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory.

    Returns the original ``model_name`` unchanged if it already points to an
    existing directory.  For HuggingFace repo IDs (e.g.
    ``"JANGQ-AI/Qwen3.5-122B-A10B-JANG_3L"``), scans the local HF cache and
    returns the snapshot path of the most recently modified revision.
    Returns ``model_name`` as-is if resolution fails (callers will fall
    through gracefully).

    Results are cached for the lifetime of the process via ``@lru_cache``.
    """
    from pathlib import Path

    # Already a local directory?
    if Path(model_name).is_dir():
        return model_name

    # Try HuggingFace cache (no network, no download)
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                for revision in sorted(
                    repo.revisions,
                    key=lambda r: (r.last_modified or 0),
                    reverse=True,
                ):
                    snapshot = str(revision.snapshot_path)
                    if Path(snapshot).is_dir() and (
                        Path(snapshot) / "config.json"
                    ).is_file():
                        return snapshot
    except Exception:
        pass

    return model_name


def is_mllm_model(model_name: str, force_mllm: bool = False) -> bool:
    """
    Check if model is a multimodal language model.

    Primary check: force_mllm flag (highest priority, from --is-mllm / user setting)
    Secondary check: reads the model's config.json for vision_config presence.
    Tertiary: uses the model config registry.

    No regex fallback — users can force VLM mode via session settings if
    auto-detection fails for custom/renamed models.

    Args:
        model_name: HuggingFace model name or local path
        force_mllm: If True, bypass detection and return True immediately

    Returns:
        True if model is detected as MLLM/VLM
    """
    if force_mllm:
        return True

    # Resolve HF repo IDs (e.g. "Org/Model") to local cache path so that
    # file-based checks (jang_config.json, config.json) actually find the files.
    local_path = resolve_to_local_path(model_name)

    # JANG models: check jang config has_vision field, then fall through
    # to config.json check. Don't early-return False — some JANG models
    # have has_vision=false in jang_config but DO have vision_config in
    # config.json (e.g., Mistral 4 JANG VLM).
    from ..utils.jang_loader import is_jang_model, _find_config_path
    from pathlib import Path
    if is_jang_model(local_path):
        try:
            cfg_path = _find_config_path(Path(local_path))
            if cfg_path is not None:
                jang_cfg = json.loads(cfg_path.read_text())
                if jang_cfg.get("architecture", {}).get("has_vision", False):
                    return True
        except Exception:
            pass
        # Fall through to config.json check instead of returning False

    # Primary: check config.json for vision_config (authoritative for local models)
    config_path = os.path.join(local_path, "config.json")
    if os.path.isfile(config_path):
        try:
            model_config = json.loads(open(config_path).read())
            if "vision_config" in model_config:
                return True
        except Exception:
            pass

    # Secondary: use model config registry (reads model_type from config.json).
    # Pass local_path so registry can read config.json even for HF repo IDs,
    # but also try the original model_name for name-based disambiguation
    # (e.g. GLM-Z1, MedGemma regex patterns in registry.lookup).
    try:
        from ..model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        # Try resolved path first (can read config.json from disk)
        reg_config = registry.lookup(local_path)
        if reg_config.family_name == "unknown" and local_path != model_name:
            # Retry with original name for name-based regex disambiguation
            reg_config = registry.lookup(model_name)
        if reg_config.family_name != "unknown":
            return reg_config.is_mllm
    except Exception:
        pass

    return False


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# =============================================================================
# Multimodal Content Extraction
# =============================================================================


def _flatten_content_list(content: list) -> str:
    """Flatten an OpenAI content array to a single text string.

    Extracts text parts from content arrays like:
        [{"type": "text", "text": "hello"}, {"type": "image_url", ...}]
    Returns joined text parts. Used when assistant messages have content
    as an array alongside tool_calls (OpenAI spec says assistant content
    is string|null, but some clients send arrays).
    """
    parts = []
    for item in content:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif hasattr(item, "dict"):
            item = item.dict()
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts) if parts else ""


def extract_multimodal_content(
    messages: list[Message],
    preserve_native_format: bool = False,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        preserve_native_format: If True, preserve native tool message format
            (role="tool", tool_calls field) instead of converting to text.
            Required for models with native tool support in chat templates
            (e.g., Mistral, Llama 3+, DeepSeek V3).

    Returns:
        Tuple of (processed_messages, images, videos)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        # Handle both dict and Pydantic model messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content

        # Map "developer" role to "system" (OpenAI API compatibility)
        if role == "developer":
            role = "system"

        # Handle tool response messages (role="tool")
        if role == "tool":
            if isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id", "") or ""
            else:
                tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_content = content if content else ""

            if preserve_native_format:
                # Preserve native tool format for models that support it
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                # Convert to user role for models without native support
                processed_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                    }
                )
            continue

        # Handle assistant messages with tool_calls
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "assistant" and tool_calls:
            if preserve_native_format:
                # Preserve native tool_calls format
                tool_calls_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tool_calls_list.append(tc)
                    elif hasattr(tc, "model_dump"):
                        tool_calls_list.append(tc.model_dump())
                    elif hasattr(tc, "dict"):
                        tool_calls_list.append(tc.dict())

                # Flatten list content to string for tool_calls messages
                # (assistant content is always string/null in OpenAI spec)
                if isinstance(content, list):
                    content = _flatten_content_list(content)
                msg_dict = {"role": role, "content": content if content else ""}
                if tool_calls_list:
                    msg_dict["tool_calls"] = tool_calls_list
                processed_messages.append(msg_dict)
            else:
                # Convert tool calls to text for models without native support
                tool_calls_text = []
                for tc in tool_calls:
                    if hasattr(tc, "model_dump"):
                        tc = tc.model_dump()
                    elif hasattr(tc, "dict"):
                        tc = tc.dict()
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")

                # Flatten list content to string (fixes list + "\n" crash)
                if isinstance(content, list):
                    text = _flatten_content_list(content)
                else:
                    text = content if content else ""
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)

                processed_messages.append({"role": role, "content": text})
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                # Handle both Pydantic models and dicts
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif hasattr(item, "dict"):
                    item = item.dict()

                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    if isinstance(img_url, str):
                        images.append(img_url)
                    elif isinstance(img_url, dict):
                        images.append(img_url.get("url", ""))

                elif item_type == "image":
                    images.append(item.get("image", item.get("url", "")))

                elif item_type == "video":
                    videos.append(item.get("video", item.get("url", "")))

                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        videos.append(vid_url)
                    elif isinstance(vid_url, dict):
                        videos.append(vid_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos
