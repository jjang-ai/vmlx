# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import functools
import json
import logging
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
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]|"
    # Gemma 4 channel/turn tokens — strip channel headers like
    # "<|channel>default\n<channel|>" that leak into output when reasoning is off
    r"<\|channel>[^\n]*\n<channel\|>|<\|channel>|<channel\|>|<turn\|>|<\|turn>"
)


def strip_marker_tokens_delta(text: str) -> str:
    """Streaming-delta-safe marker stripper.

    Removes Gemma 4 channel / turn tokens, think tags, im_start/im_end etc.
    from an INCREMENTAL streaming delta WITHOUT touching surrounding
    whitespace. Unlike `clean_output_text` (which ends in `.strip()` and
    is designed for a complete output blob), this is safe to apply per
    delta — preserving leading/trailing spaces that tokenizers emit
    between words.

    Use this in streaming paths where multiple deltas are aggregated on
    the client side. Using `clean_output_text` per-delta caused visible
    concatenation (`"Itlooksliketypo"` instead of `"It looks like typo"`)
    because every delta's leading space was stripped.
    """
    if not text:
        return text
    # Strip the full special-token regex (channel markers, im_start/end,
    # think tags only in certain shapes per SPECIAL_TOKENS_PATTERN).
    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    # Delta-safe: DO NOT `.strip()` — that eats single-space deltas like
    # " the" which destroys the streaming word stream.
    return text


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

    # Gemma 4 degraded-form handling: when the tokenizer strips the
    # `<|channel>` special token but leaves the `thought` word, we see
    # a literal `thought\n...<channel|>...` (or `thought\n...` if the
    # endmarker was stripped too). Must run BEFORE SPECIAL_TOKENS_PATTERN
    # so we can still see the `<channel|>` endmarker — if we strip
    # channel markers first, the degraded block has no delimiter and
    # we'd collapse reasoning into content.
    text = re.sub(r"^\s*thought\n.*?<channel\|>", "", text, flags=re.DOTALL).lstrip()

    # Now strip special tokens (including `<|channel>` SOC that may still
    # be there with the degraded form's `thought\n` visible right after).
    text = SPECIAL_TOKENS_PATTERN.sub("", text)

    # After SOC stripping, a bare `thought\n` prefix may now be at the
    # start of the string (e.g. raw `<|channel>thought\n...` with no
    # `<channel|>` endmarker loses only the SOC above; the `thought\n`
    # prefix survives until this step). Strip it here so display output
    # never shows the degraded-form lead. Run AFTER SPECIAL_TOKENS_PATTERN
    # so `<|channel>thought\n...` flows correctly through both stages.
    if text.startswith("thought\n"):
        text = text[len("thought\n"):]

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
    ``"JANGQ-AI/Qwen3.5-122B-A10B-JANG_3L"``), scans:

    1. ``~/.mlxstudio/models/<repo_id>/`` — vMLX desktop app's model cache
    2. HuggingFace cache snapshots (most recent revision with a config.json)

    Returns ``model_name`` as-is if resolution fails (callers will fall
    through gracefully).

    Results are cached for the lifetime of the process via ``@lru_cache``.
    """
    from pathlib import Path

    # Already a local directory?
    if Path(model_name).is_dir():
        return model_name

    # vMLX app cache: ~/.mlxstudio/models/<org>/<name>/
    # (The desktop app downloads here, separate from HF cache.)
    try:
        mlxstudio_path = Path.home() / ".mlxstudio" / "models" / model_name
        if mlxstudio_path.is_dir() and (mlxstudio_path / "config.json").is_file():
            return str(mlxstudio_path)
    except Exception:
        pass

    # HuggingFace cache (no network, no download)
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


# Q2 (audit-2026-04-07): result cache for is_mllm_model() to stop repeated
# INFO log emissions on hot paths (is_mllm_model is called from multiple
# request entry points per turn). Keyed by (model_name, local_path,
# config_mtime) — mtime invalidates the cache if the user edits config.json
# or jang_config.json between loads.
_IS_MLLM_CACHE: dict[tuple, bool] = {}


def _is_mllm_cache_key(model_name: str, local_path: str) -> tuple:
    """Build a cache key that invalidates on config.json / jang_config.json edits."""
    try:
        cfg_mtime = 0.0
        jang_mtime = 0.0
        cfg_path = os.path.join(local_path, "config.json")
        jang_path = os.path.join(local_path, "jang_config.json")
        if os.path.isfile(cfg_path):
            cfg_mtime = os.path.getmtime(cfg_path)
        if os.path.isfile(jang_path):
            jang_mtime = os.path.getmtime(jang_path)
        return (model_name, local_path, cfg_mtime, jang_mtime)
    except Exception:
        return (model_name, local_path, 0.0, 0.0)


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
    # Audit-2026-04-07 risk §6.8: log which detection tier matched at INFO so
    # parser-detection debugging is possible. Tiers are checked in order; the
    # first one to return wins.
    _logger = logging.getLogger("vmlx_engine")

    # Smelt mutual exclusion: when smelt is active, the vision tower is NOT
    # wired through the partial-expert loader. Allowing a VLM to load under
    # smelt would silently produce garbage logits on image input (vision
    # features never reach the language model embeddings). Force text-only
    # mode unconditionally — overriding force_mllm too, since the user has
    # no way to have a working VLM under smelt.
    try:
        from .. import server as _server_module  # local import to avoid cycles
        if getattr(_server_module, '_smelt_enabled', False):
            if force_mllm:
                _logger.warning(
                    "is_mllm_model(%s): smelt mode overrides force_mllm — "
                    "VLM would produce garbage on image input under smelt, "
                    "forcing text-only",
                    model_name,
                )
            else:
                _logger.info(
                    "is_mllm_model(%s): tier=smelt_forces_text_only result=False",
                    model_name,
                )
            return False
    except Exception:
        # Defensive: if server module isn't loaded yet (rare race on startup),
        # fall through to normal detection. The CLI already zeroed args.is_mllm
        # so force_mllm will be False in the common path.
        pass

    # Resolve HF repo IDs (e.g. "Org/Model") to local cache path so that
    # file-based checks (jang_config.json, config.json) actually find the files.
    local_path = resolve_to_local_path(model_name)

    # Nemotron-3-Nano-Omni: bundle has vision_config (would route through MLLM
    # path → mlx_vlm.get_model_and_args → ValueError "nemotron_h not supported"),
    # but the omni dispatcher loads its own encoders + LM at request time. The
    # standard engine load only needs the text-only LLM path (mlx_lm has
    # nemotron_h). Force LLM path; omni multimodal is handled in server.py's
    # /v1/chat/completions, /v1/messages, /v1/responses dispatch hooks.
    try:
        from ..omni_multimodal import is_omni_multimodal_bundle
        if is_omni_multimodal_bundle(local_path):
            _logger.info(
                "is_mllm_model(%s): tier=omni_bundle_routes_via_dispatcher result=False",
                model_name,
            )
            return False
    except Exception:
        pass

    # Mistral-Medium-3.5: outer mistral3 wrapper has vision_config but the
    # current `jang_tools.mistral3.Mistral3ForConditionalGeneration` keeps
    # vision_tower / multi_modal_projector as None stubs (text-only). The
    # MLLM path routes to mlx_vlm which expects a fully-fleshed VLM and
    # raises "Received 438 parameters not in model" on the vision keys.
    # Force LLM path so jang_tools.mistral3.runtime.load handles the
    # vision-strip + text decode. When the vision tower port lands, drop
    # this branch.
    try:
        import json as _json_m3
        from pathlib import Path as _Path_m3
        cfg_path_m3 = _Path_m3(local_path) / "config.json"
        if cfg_path_m3.exists():
            cfg_m3 = _json_m3.loads(cfg_path_m3.read_text())
            if (cfg_m3.get("text_config") or {}).get("model_type") == "ministral3" \
               or cfg_m3.get("model_type") == "ministral3":
                _logger.info(
                    "is_mllm_model(%s): tier=ministral3_routes_via_jang_tools result=False",
                    model_name,
                )
                return False
    except Exception:
        pass

    if force_mllm:
        # Not cached — force_mllm is cheap + callers may toggle at runtime.
        _logger.info("is_mllm_model(%s): tier=force_mllm result=True", model_name)
        return True

    # Q2 result cache check — return without re-logging if we've already
    # resolved this (model_name, local_path, config_mtime, jang_mtime).
    _cache_key = _is_mllm_cache_key(model_name, local_path)
    if _cache_key in _IS_MLLM_CACHE:
        return _IS_MLLM_CACHE[_cache_key]

    def _resolve() -> bool:
        # JANG models: jang_config.has_vision is authoritative.
        # When explicitly set, it overrides config.json vision_config
        # (e.g., Mistral 4 text-only JANG has vision_config in config.json
        # because mistral3 is a VLM wrapper arch, but jang_config says false).
        from ..utils.jang_loader import is_jang_model, _find_config_path
        from pathlib import Path
        if is_jang_model(local_path):
            try:
                cfg_path = _find_config_path(Path(local_path))
                if cfg_path is not None:
                    jang_cfg = json.loads(cfg_path.read_text())
                    arch = jang_cfg.get("architecture", {}) or {}
                    if "has_vision" in arch:
                        has_vision = arch.get("has_vision")
                        if has_vision is True:
                            # Mistral 4 has_vision=true exception: mlx_vlm has
                            # mistral3 (standard attention) and mistral4 (text
                            # only; no VLM class). A Mistral 4 VLM config
                            # routed through the VLM engine stuffs MLA weights
                            # into mistral3's standard-attention skeleton →
                            # garbage tokens. Force text-only until mlx_vlm
                            # ships a mistral4 VLM class; paired with the
                            # loader's text-only fallback as defense in depth.
                            try:
                                hf_cfg_path = os.path.join(local_path, "config.json")
                                if os.path.isfile(hf_cfg_path):
                                    hf_cfg = json.loads(open(hf_cfg_path).read())
                                    tc = hf_cfg.get("text_config") or {}
                                    if (hf_cfg.get("model_type") == "mistral3"
                                            and tc.get("model_type") == "mistral4"):
                                        _logger.warning(
                                            "is_mllm_model(%s): Mistral 4 VLM "
                                            "wrapper unsupported by mlx_vlm — "
                                            "forcing text-only to avoid garbage output",
                                            model_name,
                                        )
                                        return False
                            except Exception:
                                pass
                            _logger.info(
                                "is_mllm_model(%s): tier=jang_config_explicit_true result=True",
                                model_name,
                            )
                            return True
                        if has_vision is False:
                            _logger.info(
                                "is_mllm_model(%s): tier=jang_config_explicit_false result=False",
                                model_name,
                            )
                            return False
            except Exception:
                pass

        # Primary: check config.json for vision_config (authoritative for local models)
        config_path = os.path.join(local_path, "config.json")
        if os.path.isfile(config_path):
            try:
                model_config = json.loads(open(config_path).read())
                if "vision_config" in model_config:
                    _logger.info(
                        "is_mllm_model(%s): tier=config_json_vision_config result=True",
                        model_name,
                    )
                    return True
            except Exception:
                pass

        # Secondary: use model config registry (reads model_type from config.json).
        try:
            from ..model_config_registry import get_model_config_registry

            registry = get_model_config_registry()
            reg_config = registry.lookup(local_path)
            if reg_config.family_name == "unknown" and local_path != model_name:
                reg_config = registry.lookup(model_name)
            if reg_config.family_name != "unknown":
                _logger.info(
                    "is_mllm_model(%s): tier=registry_family_%s result=%s",
                    model_name,
                    reg_config.family_name,
                    reg_config.is_mllm,
                )
                return reg_config.is_mllm
        except Exception:
            pass

        _logger.info("is_mllm_model(%s): tier=fallthrough result=False", model_name)
        return False

    _result = _resolve()
    _IS_MLLM_CACHE[_cache_key] = _result
    return _result


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
