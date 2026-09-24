# SPDX-License-Identifier: Apache-2.0
"""Nemotron-3-Nano-Omni multimodal dispatch for vMLX HTTP server.

Bridges the OpenAI-compatible chat-completions endpoint to
``jang_tools.nemotron_omni_session.OmniSession`` so requests with
``image_url`` / ``input_audio`` / ``video_url`` content parts on
nemotron_h-Omni bundles (MXFP4 / JANGTQ4 / JANGTQ2, paired with the
in-bundle multimodal addon) flow through the Stage-1 PyTorch-encoder +
MLX-LLM bridge described in
``research/NEMOTRON-OMNI-MULTIMODAL-2026-04-28.md``.

Design:
  - Lazy-loads a single ``OmniSession`` per server lifetime — Stage-1 load is
    expensive (~10 min on CPU, ~10s on MPS) so reuse the session across
    requests. The session also carries the persistent KV+SSM cache for
    multi-turn coherence on the same conversation.
  - Per-request reset: when a NEW conversation arrives (detected by the text
    and media identity of its message history not matching the running
    history-prefix), reset the session.
  - Extracts base64-encoded media into ``$TMPDIR/vmlx-omni-XXXX.{jpg,wav,mp4}``
    files because OmniChat's encoders expect file paths or PIL images.
  - Returns plain text for non-streaming requests and forwards real decode-time
    token segments for streaming requests. The HTTP layer keeps reasoning and
    visible content on separate OpenAI-compatible delta rails.

Public surface:
  - ``is_omni_multimodal_bundle(model_path)`` — True iff bundle has
    ``config_omni.json`` and the model_type is nemotron_h.
  - ``OmniMultimodalDispatcher`` — per-process singleton, wraps OmniSession.
  - ``request_has_multimodal(messages)`` — True if any message content part
    is image/audio/video.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_configs import NEMOTRON_H_MODEL_TYPES

logger = logging.getLogger(__name__)

# Supported content-part types per OpenAI chat schema (+ vMLX extension for video).
_IMAGE_TYPES = {"image_url", "image"}
# audio_url is the missing symmetric partner of image_url/video_url. The
# Ollama and Anthropic adapters both emit audio as {"type": "audio_url"},
# and the request schema accepts it — but it used to be recognized nowhere,
# so those requests skipped omni dispatch and were answered text-only with
# no error. Recognize it exactly like image_url/video_url.
_AUDIO_TYPES = {"input_audio", "audio", "audio_url"}
_VIDEO_TYPES = {"video_url", "video"}
_CRADIO_CACHE_REPO = "C_hyphen_RADIOv2_hyphen_H"
_CRADIO_CACHE_REVISION = "0d8f4c18c877166eda07ddae1386bcad256b7a6a"
_OMNI_SESSION_L2_SCHEMA = "nemotron_omni_session_v2"


def _ensure_vendored_cradio_dynamic_module(
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Install the vendored C-RADIO Transformers module into HF's cache.

    Nemotron-Omni's local wrapper delegates RADIO construction to
    ``AutoModel.from_config(..., trust_remote_code=True)``. The model weights
    are local, but Transformers still resolves the ``nvidia/C-RADIOv2-H``
    dynamic Python module on first media request. Shipping this small Apache-2
    module with vMLX keeps notarized app media cold-start offline and
    deterministic.
    """
    src_root = (
        Path(__file__).resolve().parent
        / "vendor"
        / "transformers_modules"
        / "nvidia"
        / _CRADIO_CACHE_REPO
    )
    src_revision = src_root / _CRADIO_CACHE_REVISION
    if not src_revision.is_dir():
        return {
            "installed": False,
            "reason": "vendored_cradio_missing",
            "source": str(src_revision),
        }
    if cache_root is None:
        try:
            from transformers.utils.hub import HF_MODULES_CACHE

            cache_root = HF_MODULES_CACHE
        except Exception:
            cache_root = Path.home() / ".cache" / "huggingface" / "modules"
    cache_base = Path(cache_root) / "transformers_modules" / "nvidia" / _CRADIO_CACHE_REPO
    dst_revision = cache_base / _CRADIO_CACHE_REVISION
    cache_base.mkdir(parents=True, exist_ok=True)
    init_src = src_root / "__init__.py"
    if init_src.is_file():
        shutil.copy2(init_src, cache_base / "__init__.py")
    shutil.copytree(
        src_revision,
        dst_revision,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    pycache = dst_revision / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    for bytecode in dst_revision.rglob("*.py[co]"):
        bytecode.unlink(missing_ok=True)
    return {
        "installed": True,
        "source": str(src_revision),
        "destination": str(dst_revision),
    }


def _patch_omni_encoder_view_for_vendored_cradio() -> bool:
    """Make jang_tools' temporary Omni HF view self-contained for C-RADIO.

    ``jang_tools`` builds a temporary directory where ``config.json`` is the
    Omni wrapper config, then Transformers dynamic-loads ``modeling.py`` from
    that temp view. The nested ``vision_config.auto_map`` in shipped bundles
    still references ``nvidia/C-RADIOv2-H--hf_model.RADIOModel``. In offline
    packaged apps, that makes ``AutoModel.from_config(config.vision_config)``
    call Hugging Face hub even though the C-RADIO code is vendored.

    Patch the temp-view builder once so each view contains the vendored RADIO
    module files and points the nested auto-map at those local files.
    """
    try:
        omni_chat = importlib.import_module("jang_tools.nemotron_omni_chat")
    except Exception:
        return False
    if getattr(omni_chat, "_vmlx_cradio_view_patch", False):
        return True
    original = getattr(omni_chat, "_populate_omni_encoder_view", None)
    if original is None:
        return False

    src_revision = (
        Path(__file__).resolve().parent
        / "vendor"
        / "transformers_modules"
        / "nvidia"
        / _CRADIO_CACHE_REPO
        / _CRADIO_CACHE_REVISION
    )
    if not src_revision.is_dir():
        return False

    def _populate_with_local_cradio(bundle_path: Path, view_dir: Path) -> None:
        original(bundle_path, view_dir)
        for src in src_revision.glob("*.py"):
            shutil.copy2(src, view_dir / src.name)
        try:
            from transformers.utils.hub import HF_MODULES_CACHE

            cache_view = Path(HF_MODULES_CACHE) / "transformers_modules" / view_dir.name
            cache_view.mkdir(parents=True, exist_ok=True)
            for src in src_revision.glob("*.py"):
                shutil.copy2(src, cache_view / src.name)
        except Exception as exc:
            logger.warning(
                "OmniMultimodalDispatcher: failed to pre-seed C-RADIO "
                "dynamic-module cache for %s: %s",
                view_dir,
                exc,
            )
        modeling_path = view_dir / "modeling.py"
        try:
            modeling = modeling_path.read_text()
            needle = (
                "self.vision_model = AutoModel.from_config("
                "config.vision_config, trust_remote_code=True)"
            )
            replacement = (
                "from .hf_model import RADIOModel as _VMLINUX_RADIOModel\n"
                "        self.vision_model = _VMLINUX_RADIOModel(config.vision_config)"
            )
            if needle in modeling:
                modeling_path.write_text(modeling.replace(needle, replacement))
        except Exception as exc:
            logger.warning(
                "OmniMultimodalDispatcher: failed to patch local RADIOModel "
                "constructor in %s: %s",
                modeling_path,
                exc,
            )
        config_path = view_dir / "config.json"
        try:
            cfg = json.loads(config_path.read_text())
            vision_config = cfg.get("vision_config")
            if isinstance(vision_config, dict):
                vision_config["auto_map"] = {
                    "AutoConfig": "configuration_radio.RADIOConfig",
                    "AutoModel": "hf_model.RADIOModel",
                }
                vision_config["_name_or_path"] = str(view_dir)
                config_path.write_text(json.dumps(cfg, indent=2))
        except Exception as exc:
            logger.warning(
                "OmniMultimodalDispatcher: failed to rewrite local C-RADIO "
                "auto_map in %s: %s",
                config_path,
                exc,
            )

    omni_chat._populate_omni_encoder_view = _populate_with_local_cradio
    omni_chat._vmlx_cradio_view_patch = True
    return True


def omni_multimodal_component_status(model_path: str | Path) -> dict[str, Any]:
    """Inspect whether a Nemotron-Omni bundle has its media components.

    Cross-check indexed media tensors in their actual shard headers, including
    encoder presence and both projector chains. This does not replace live
    modality qualification of the complete model.
    """
    p = Path(model_path)
    status: dict[str, Any] = {
        "bundle_compatible": False,
        "config_model_type": None,
        "has_config": False,
        "has_config_omni": False,
        "has_index": False,
        "has_radio_config": False,
        "sound_config_model_type": None,
        "has_radio_weights": False,
        "has_parakeet_weights": False,
        "has_media_projector": False,
        "has_vision_projector": False,
        "has_audio_projector": False,
        "modalities": [],
        "missing": [],
    }
    if not p.is_dir():
        status["missing"].append("directory")
        return status
    cfg = p / "config.json"
    omni_cfg = p / "config_omni.json"
    idx = p / "model.safetensors.index.json"
    radio_cfg = p / "configuration_radio.py"
    video_processor_cfg = p / "video_preprocessor_config.json"
    status["has_config"] = cfg.is_file()
    status["has_config_omni"] = omni_cfg.is_file()
    status["has_index"] = idx.is_file()
    status["has_radio_config"] = radio_cfg.is_file()
    status["has_video_preprocessor_config"] = video_processor_cfg.is_file()
    status["video_bridge_supported"] = False
    for name, present in (
        ("config.json", status["has_config"]),
        ("config_omni.json", status["has_config_omni"]),
        ("model.safetensors.index.json", status["has_index"]),
        ("configuration_radio.py", status["has_radio_config"]),
    ):
        if not present:
            status["missing"].append(name)
    if not (cfg.is_file() and omni_cfg.is_file() and idx.is_file()):
        return status
    try:
        cfg_data = json.loads(cfg.read_text())
        omni_data = json.loads(omni_cfg.read_text())
        status["config_model_type"] = cfg_data.get("model_type")
        sound_config = omni_data.get("sound_config")
        if isinstance(sound_config, dict):
            status["sound_config_model_type"] = sound_config.get("model_type")
        from .omni_media_components import inspect_media_weights
        status.update(inspect_media_weights(
            p, json.loads(idx.read_text()).get("weight_map", {}), cfg_data, omni_data,
        ))
        if status["has_radio_weights"] and status["has_vision_projector"]:
            from .omni_native_video import temporal_video_spec
            status["temporal_video_spec"] = temporal_video_spec(p)
            status["modalities"].append("image")
            # Native temporal projection is independently verified from shards.
            # Older image-only bundles retain the sampled-frame fallback.
            status["video_bridge_supported"] = bool(
                status["temporal_video_spec"] or status["has_video_preprocessor_config"]
            )
            status["video_frame_fallback_supported"] = True
            status["modalities"].append("video")
        if status["has_parakeet_weights"] and status["has_audio_projector"]:
            status["modalities"].append("audio")
        requirements = {
            # Either family spelling — nemotron_h_v2 is the same hybrid
            # architecture, and an Omni v2 bundle carrying the full RADIO/
            # Parakeet sidecar set must not be rejected on the alias alone.
            "model_type=nemotron_h": status["config_model_type"]
            in NEMOTRON_H_MODEL_TYPES,
            "sound_config.model_type=parakeet": status["sound_config_model_type"] == "parakeet",
            "radio weights": status["has_radio_weights"],
            "parakeet weights": status["has_parakeet_weights"],
            "vision projector": status["has_vision_projector"],
            "audio projector": status["has_audio_projector"],
        }
        status["missing"].extend([name for name, ok in requirements.items() if not ok])
        status["modalities"] = ["text"] + sorted(set(status["modalities"]))
        status["bundle_compatible"] = not status["missing"]
        return status
    except Exception as e:  # pragma: no cover
        status["missing"].append(f"inspect_error:{type(e).__name__}:{e}")
        logger.debug(f"omni_multimodal_component_status({p}) check failed: {e}")
        return status


def is_omni_multimodal_bundle(model_path: str | Path) -> bool:
    """Return True iff the bundle has Nemotron-Omni RADIO + Parakeet media."""
    return bool(omni_multimodal_component_status(model_path).get("bundle_compatible"))


def request_has_multimodal(messages: List[Dict[str, Any]]) -> bool:
    """True iff any message's content includes image/audio/video parts."""
    return bool(request_modalities(messages))


def request_modalities(messages: List[Dict[str, Any]]) -> set[str]:
    """Return normalized media modality names used by a chat request."""
    modalities: set[str] = set()
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if ptype in _IMAGE_TYPES:
                    modalities.add("image")
                elif ptype in _AUDIO_TYPES:
                    modalities.add("audio")
                elif ptype in _VIDEO_TYPES:
                    modalities.add("video")
    return modalities


def _decode_data_url(data_url: str) -> Tuple[bytes, str]:
    """Parse ``data:<mime>;base64,<payload>`` → (bytes, suggested_extension).

    Falls back to .bin if mime is unknown.
    """
    if not data_url.startswith("data:"):
        # Not a data URL — assume it's a file path or http URL caller must fetch.
        raise ValueError(
            "OmniMultimodal requires data: URLs or local file paths; got non-data URL"
        )
    head, payload = data_url.split(",", 1)
    raw = base64.b64decode(payload)
    ext_map = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
        "image/webp": ".webp", "image/gif": ".gif",
        "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    }
    mime = head[5:].split(";")[0].strip().lower()
    return raw, ext_map.get(mime, ".bin")


def _materialize_to_temp(data: bytes, suffix: str, scratch_dir: Path) -> Path:
    """Deduplicate media within the caller's request-owned scratch directory."""
    digest = hashlib.sha256(data).hexdigest()[:16]
    out = scratch_dir / f"{digest}{suffix}"
    if not out.exists():
        out.write_bytes(data)
    return out


def _omni_video_policy(video_controls=None, *, temporal_patch_size=None):
    from .video_controls import VideoControls

    controls = video_controls or VideoControls()
    policy = {
        "pipeline": "radio-frame-fallback-v2",
        "fps": controls.fps if controls.fps is not None else float(os.environ.get("VMLINUX_OMNI_VIDEO_FPS", "1")),
        "max_frames": controls.max_frames if controls.max_frames is not None else int(os.environ.get("VMLINUX_OMNI_VIDEO_MAX_FRAMES", "4")),
        "dedup_mad": float(os.environ.get("VMLINUX_OMNI_VIDEO_DEDUP_MAD", "8")),
        "contact_sheet": os.environ.get("VMLINUX_OMNI_VIDEO_CONTACT_SHEET", "1") != "0",
    }
    if temporal_patch_size is not None:
        policy.update(pipeline="radio-temporal-v1", temporal_patch_size=temporal_patch_size)
        policy.pop("dedup_mad")
        policy.pop("contact_sheet")
    return policy


def _extract_omni_video_frames(video_path: Path, scratch_dir: Path, *, video_controls=None) -> List[Path]:
    """Sample an Omni video into image files for the RADIO image encoder."""
    try:
        from PIL import Image
        import numpy as np
        from .models.mllm import extract_video_frames_smart

        policy = _omni_video_policy(video_controls)
        frames = extract_video_frames_smart(
            str(video_path),
            fps=policy["fps"],
            max_frames=policy["max_frames"],
        )
        from .video_controls import subsample_frames_evenly
        sampled_count = len(frames)
        # The shared sampler rounds up to its temporal patch minimum. RADIO
        # consumes independent images, so the requested ceiling still wins.
        frames = subsample_frames_evenly(frames, min(len(frames), policy["max_frames"]))
        if not frames:
            raise ValueError("video contains no readable frames")
        deduped = []
        threshold = policy["dedup_mad"]
        for frame in frames:
            if not deduped:
                deduped.append(frame)
                continue
            delta = np.mean(
                np.abs(frame.astype(np.int16) - deduped[-1].astype(np.int16))
            )
            if delta >= threshold:
                deduped.append(frame)
        out: List[Path] = []
        with video_path.open("rb") as source:
            content_hash = hashlib.file_digest(source, "sha256").hexdigest()
        digest = hashlib.sha256(json.dumps(
            {"content": content_hash, "policy": policy}, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        logger.info("Omni video preprocessing: fps=%s max_frames=%s sampled=%d capped=%d retained=%d contact_sheet=%s",
                    policy["fps"], policy["max_frames"], sampled_count, len(frames), len(deduped), policy["contact_sheet"])
        if (
            len(deduped) > 1
            and policy["contact_sheet"]
        ):
            images = [Image.fromarray(frame).convert("RGB") for frame in deduped]
            width = sum(img.width for img in images)
            height = max(img.height for img in images)
            sheet = Image.new("RGB", (width, height), "white")
            x = 0
            for img in images:
                sheet.paste(img, (x, 0))
                x += img.width
            frame_path = scratch_dir / f"{video_path.stem}-{digest}-contact-sheet.jpg"
            if not frame_path.exists():
                sheet.save(frame_path, "JPEG", quality=92)
            return [frame_path]
        for idx, frame in enumerate(deduped):
            frame_path = scratch_dir / f"{video_path.stem}-{digest}-frame-{idx}.jpg"
            if not frame_path.exists():
                Image.fromarray(frame).save(frame_path, "JPEG", quality=90)
            out.append(frame_path)
        return out
    except Exception as exc:
        # The native video method has fixed sampling defaults. Falling through
        # to it would silently discard the caller's frame policy.
        raise ValueError(f"Omni video frame preprocessing failed: {exc}") from exc


def _extract_parts(
    messages: List[Dict[str, Any]],
    scratch_dir: Path,
    *,
    rehydrate_history_media: bool = False,
    video_controls=None,
    native_video: bool = False,
) -> Tuple[str, List[Path], Optional[Path], Optional[Path]]:
    """Walk all messages, collect text + write media to temp files.

    Returns (concatenated_text, image_paths, audio_path, video_path). Only the
    LAST user turn's text is used as the prompt — earlier turns inform the
    OmniSession via its persistent cache. Media on EARLIER turns are still
    extracted (so they're cached for reuse) but only the last turn's media is
    forwarded to ``session.turn()``.
    """
    last_user_text: str = ""
    cur_images: List[Path] = []
    cur_audio: Optional[Path] = None
    cur_video: Optional[Path] = None
    latest_media_turn: Optional[
        Tuple[List[Path], Optional[Path], Optional[Path]]
    ] = None

    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role != "user":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

        # Reset per-turn collectors so only the LATEST user turn carries media.
        cur_images, cur_audio, cur_video = [], None, None
        last_user_text = ""

        if isinstance(content, str):
            last_user_text = content
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if ptype in _IMAGE_TYPES:
                    src = part.get("image_url") or part.get("image") or {}
                    url = src.get("url") if isinstance(src, dict) else src
                    if isinstance(url, str):
                        if url.startswith("data:"):
                            raw, ext = _decode_data_url(url)
                            cur_images.append(_materialize_to_temp(raw, ext, scratch_dir))
                        elif Path(url).exists():
                            cur_images.append(Path(url))
                elif ptype in _AUDIO_TYPES:
                    src = (
                        part.get("input_audio")
                        or part.get("audio")
                        or part.get("audio_url")
                        or {}
                    )
                    if isinstance(src, dict):
                        b64 = src.get("data")
                        fmt = src.get("format", "wav")
                        if b64:
                            raw = base64.b64decode(b64)
                            cur_audio = _materialize_to_temp(raw, f".{fmt}", scratch_dir)
                        elif src.get("url", "").startswith("data:"):
                            raw, ext = _decode_data_url(src["url"])
                            cur_audio = _materialize_to_temp(raw, ext, scratch_dir)
                        elif isinstance(src.get("url"), str) and Path(src["url"]).exists():
                            cur_audio = Path(src["url"])
                    elif isinstance(src, str):
                        if src.startswith("data:"):
                            raw, ext = _decode_data_url(src)
                            cur_audio = _materialize_to_temp(raw, ext, scratch_dir)
                        elif Path(src).exists():
                            cur_audio = Path(src)
                elif ptype in _VIDEO_TYPES:
                    src = part.get("video_url") or part.get("video") or {}
                    url = src.get("url") if isinstance(src, dict) else src
                    if isinstance(url, str):
                        if url.startswith("data:"):
                            raw, ext = _decode_data_url(url)
                            video_path = _materialize_to_temp(raw, ext, scratch_dir)
                        elif Path(url).exists():
                            video_path = Path(url)
                        else:
                            video_path = None
                        if video_path is not None:
                            if native_video:
                                if cur_video is not None:
                                    raise ValueError("Native incremental video requires one clip per turn")
                                cur_video = video_path
                                continue
                            frame_paths = _extract_omni_video_frames(
                                video_path,
                                scratch_dir,
                                video_controls=video_controls,
                            )
                            if frame_paths:
                                cur_images.extend(frame_paths)
                                cur_video = None
                            else:
                                cur_video = video_path
                elif ptype == "text":
                    last_user_text += (part.get("text") if isinstance(part, dict) else getattr(part, "text", "")) or ""
        if cur_images or cur_audio is not None or cur_video is not None:
            latest_media_turn = (list(cur_images), cur_audio, cur_video)

    if (
        rehydrate_history_media
        and not cur_images
        and cur_audio is None
        and cur_video is None
        and latest_media_turn is not None
    ):
        cur_images, cur_audio, cur_video = latest_media_turn
        logger.info(
            "OmniMultimodalDispatcher: rehydrating latest prior-turn media "
            "after conversation cache reset"
        )
    return last_user_text, cur_images, cur_audio, cur_video


def _build_omni_turn_prompt_with_thinking(
    tokenizer: Any,
    user_text: str,
    n_image_tokens: int = 0,
    n_video_tokens: int = 0,
    n_audio_tokens: int = 0,
    is_first: bool = False,
    enable_thinking: Optional[bool] = None,
    video_prompt: Optional[str] = None,
    template_options=None,
) -> str:
    """Build an OmniSession turn prompt while preserving the API thinking rail.

    `jang_tools.nemotron_omni_session.OmniSession` renders the tokenizer chat
    template without forwarding `enable_thinking`, so Nemotron-Omni media
    requests can ignore explicit thinking-on/off requests. Keep the same
    per-turn/cache semantics, but pass the template variable only when the
    caller explicitly supplied one so the bundle's native default remains
    untouched.
    """
    media = ""
    if n_image_tokens > 0:
        media += "<img>" + ("<image>" * n_image_tokens) + "</img>\n"
    if n_video_tokens > 0:
        # Nemotron-Omni's tokenizer has no real printable <video> token.  The
        # bundle processor reuses image placeholders for video frame embeds.
        if video_prompt is not None:
            if video_prompt.count("<image>") != n_video_tokens:
                raise ValueError("Native video prompt and embedding token counts disagree")
            media += video_prompt
        else:
            media += "<img>" + ("<image>" * n_video_tokens) + "</img>\n"
    if n_audio_tokens > 0:
        media += "<sound>" + ("<so_embedding>" * n_audio_tokens) + "</sound>\n"
    msg_content = media + user_text
    messages = [{"role": "user", "content": msg_content}]
    template_kwargs = {
        **(template_options or {}),
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)

    if is_first:
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    followup_messages = [
        {"role": "user", "content": "__PREV_USER__"},
        {"role": "assistant", "content": "__PREV_ASST__"},
        {"role": "user", "content": msg_content},
    ]
    prev_then_now = tokenizer.apply_chat_template(
        followup_messages,
        **template_kwargs,
    )
    marker = "__PREV_ASST__"
    idx = prev_then_now.find(marker)
    if idx < 0:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    return prev_then_now[idx + len(marker):]


def _media_part_identity(part: Dict[str, Any]) -> Optional[str]:
    """Return a stable, non-secret identity for one media content part.

    Omni's persistent session cache must never treat identical text paired
    with different media as the same conversation prefix.  Hash the wire
    representation instead of retaining base64 payloads in process metadata.
    Conservative false misses (for example, the same file via two paths) are
    safe; a false hit would reuse stale KV/SSM state from different media.
    """
    if not isinstance(part, dict):
        return None
    ptype = str(part.get("type") or "")
    if ptype in _IMAGE_TYPES:
        kind = "image"
        src = part.get("image_url") or part.get("image") or {}
    elif ptype in _AUDIO_TYPES:
        kind = "audio"
        src = part.get("input_audio") or part.get("audio") or part.get("audio_url") or {}
    elif ptype in _VIDEO_TYPES:
        kind = "video"
        src = part.get("video_url") or part.get("video") or {}
    else:
        return None

    if isinstance(src, dict):
        data = src.get("data")
        if isinstance(data, str) and data:
            digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
            fmt = str(src.get("format") or "")
            return f"{kind}:data:{fmt}:{digest}"
        source = src.get("url") or src.get("path")
    else:
        source = src

    if not isinstance(source, str) or not source:
        return f"{kind}:missing"
    if source.startswith("data:"):
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return f"{kind}:data-url:{digest}"

    path = Path(source).expanduser()
    try:
        if path.is_file():
            with path.open("rb") as source_file:
                digest = hashlib.file_digest(source_file, "sha256").hexdigest()
            return f"{kind}:file:{digest}"
    except OSError:
        pass
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"{kind}:source:{digest}"


def _user_turn_signatures(messages: List[Dict[str, Any]]) -> List[str]:
    """Return text+media signatures from each user turn, in order.

    These signatures detect 'same conversation continuing' vs 'fresh
    conversation'. Media identity is mandatory: text-only signatures allowed
    a blue-audio history to reuse an orange-audio KV/SSM session.
    """
    out: List[str] = []
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        if role != "user":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        media: List[str] = []
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: List[str] = []
            for p in content:
                ptype = p.get("type") if isinstance(p, dict) else getattr(p, "type", None)
                if ptype == "text":
                    parts.append((p.get("text") if isinstance(p, dict) else getattr(p, "text", "")) or "")
                if isinstance(p, dict):
                    identity = _media_part_identity(p)
                    if identity:
                        media.append(identity)
            text = " ".join(parts)
        else:
            text = ""
        out.append(
            json.dumps(
                {"text": text, "media": media},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return out


def _hash_user_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # separator so [a,b] != [ab]
    return h.hexdigest()[:16]


def _conversation_signature(messages, enable_thinking, cache_salt=None, *, video_policy=None):
    """Bind retained native state to every supplied role, not user text alone."""
    canonical = []
    for message in messages:
        item = {k: v for k, v in message.items() if v is not None}
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = []
            for part in content:
                identity = _media_part_identity(part)
                item["content"].append({"media_identity": identity} if identity is not None else part)
        canonical.append(item)
    policy = {"video_policy": video_policy or _omni_video_policy()} if "video" in request_modalities(messages) else {}
    payload = json.dumps(
        {"messages": canonical, "enable_thinking": enable_thinking, "cache_salt": cache_salt, **policy},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


from .omni_native_prompt import run_full_history as _run_omni_full_history


class OmniMultimodalDispatcher:
    """Singleton wrapper around OmniSession bound to one model bundle.

    Thread-safe: a single mutex serializes ``chat()`` calls. OmniSession
    holds a persistent KV+SSM cache, so concurrent calls would corrupt the
    cache; we explicitly serialize.
    """

    _instance_lock = threading.Lock()
    _instance: Optional["OmniMultimodalDispatcher"] = None
    _last_signature: Optional[str] = None

    @classmethod
    def get(
        cls,
        bundle_path: str | Path,
        *,
        disk_cache_enabled: Optional[bool] = None,
        disk_cache_policy: Optional[Dict[str, Any]] = None,
    ) -> "OmniMultimodalDispatcher":
        bundle_path = str(Path(bundle_path).resolve())
        with cls._instance_lock:
            if (cls._instance is None or cls._instance.bundle_path != bundle_path
                    or cls._instance._session_l2_fingerprint != cls._bundle_fingerprint(bundle_path)
                    or (disk_cache_policy is not None and
                        cls._instance._session_l2_policy != disk_cache_policy)):
                if cls._instance is not None:
                    logger.info(
                        "OmniMultimodalDispatcher: rebinding from %s to %s",
                        cls._instance.bundle_path, bundle_path,
                    )
                    cls._instance.close()
                cls._instance = cls(
                    bundle_path,
                    disk_cache_enabled=bool(disk_cache_enabled),
                    disk_cache_policy=disk_cache_policy,
                )
            elif disk_cache_enabled is not None:
                cls._instance._disk_cache_enabled = bool(disk_cache_enabled)
            return cls._instance

    def __init__(self, bundle_path: str, *, disk_cache_enabled: bool = False,
                 disk_cache_policy: Optional[Dict[str, Any]] = None):
        self.bundle_path = bundle_path
        self._session = None
        self._lock = threading.Lock()
        # The Stage-1 session owns both PyTorch/MPS encoder state and an MLX
        # language model.  Creating a raw thread per streamed request is not
        # safe: CPython 3.13 can tear down that thread while native MLX/PyTorch
        # objects still run thread-local cleanup, aborting the whole server with
        # `PyThreadState_Get ... GIL is released`.  Keep one long-lived owner
        # thread for session construction, media encoding, and every decode.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vmlx-omni-model",
        )
        self._last_signature: Optional[str] = None
        self._last_snapshot_skip_reason: Optional[str] = None
        self._disk_cache_enabled = bool(disk_cache_enabled)
        self._session_l2_policy = dict(disk_cache_policy or {})
        self._session_l2_store = None
        self._session_l2_fingerprint = self._bundle_fingerprint(bundle_path)
        self._session_l2_path = self._default_session_l2_path(
            self._session_l2_fingerprint, self._session_l2_policy
        )
        self._session_l2_stats: Dict[str, Any] = {
            "schema": _OMNI_SESSION_L2_SCHEMA,
            "enabled": self._disk_cache_enabled,
            "path": str(self._session_l2_path),
            "attention_codec": "native",
            "ssm_codec": "native-arrays",
            "stores": 0,
            "hits": 0,
            "misses": 0,
            "last_store_seconds": None,
            "last_restore_seconds": None,
            "last_error": None,
        }
        self._scratch_dir = Path(tempfile.gettempdir()) / "vmlx-omni-media"
        self._scratch_dir.mkdir(exist_ok=True)
        self._backend = self._pick_backend()
        from .omni_native_video import temporal_video_spec
        self._native_video_spec = temporal_video_spec(bundle_path) if self._backend == "stage1" else None
        self._device = self._pick_device() if self._backend == "stage1" else "metal"
        logger.info(
            "OmniMultimodalDispatcher: bundle=%s, backend=%s, device=%s, scratch=%s",
            bundle_path, self._backend, self._device, self._scratch_dir,
        )

    @staticmethod
    def _bundle_fingerprint(bundle_path: str | Path) -> str:
        """Bind native state to weights, tokenizer, processors and templates."""
        from .omni_bundle_identity import bundle_fingerprint
        return bundle_fingerprint(bundle_path)

    @staticmethod
    def _default_session_l2_path(fingerprint: str, policy=None) -> Path:
        from .utils.omni_session_disk_store import SCHEMA
        root = Path((policy or {}).get("root") or
                    Path.home() / ".cache" / "vmlx-engine" / "block-cache").expanduser()
        namespace = hashlib.sha256(f"{fingerprint}:{SCHEMA}".encode()).hexdigest()[:16]
        return root / namespace / "native_sessions"

    def _native_disk_store(self):
        if self._session_l2_store is None:
            from .utils.omni_session_disk_store import OmniSessionDiskStore
            policy = self._session_l2_policy
            self._session_l2_store = OmniSessionDiskStore(
                root=policy.get("root") or Path.home() / ".cache" / "vmlx-engine" / "block-cache",
                model_key=self._session_l2_fingerprint,
                max_size_bytes=policy.get("max_size_bytes", 10 * 1024**3),
                ttl_minutes=policy.get("ttl_minutes", 0),
            )
        return self._session_l2_store

    @classmethod
    def session_l2_status_for(
        cls,
        bundle_path: str | Path,
        *,
        enabled: bool,
        disk_cache_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved = str(Path(bundle_path).resolve())
        with cls._instance_lock:
            instance = cls._instance
            if instance is not None and instance.bundle_path == resolved:
                instance._disk_cache_enabled = bool(enabled)
                return instance.session_l2_status()
        fingerprint = cls._bundle_fingerprint(resolved)
        path = cls._default_session_l2_path(fingerprint, disk_cache_policy)
        return {
            "schema": _OMNI_SESSION_L2_SCHEMA,
            "enabled": bool(enabled),
            "path": str(path),
            "policy": dict(disk_cache_policy or {}),
            "multiple_exact_snapshots": True,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "attention_codec": "native",
            "ssm_codec": "native-arrays",
            "stores": 0,
            "hits": 0,
            "misses": 0,
            "last_store_seconds": None,
            "last_restore_seconds": None,
            "last_error": None,
        }

    def session_l2_status(self) -> Dict[str, Any]:
        path = self._session_l2_path
        status = dict(self._session_l2_stats)
        status["enabled"] = bool(self._disk_cache_enabled)
        status["path"] = str(path)
        status["policy"] = dict(self._session_l2_policy)
        status["multiple_exact_snapshots"] = True
        status["exists"] = path.is_file()
        status["bytes"] = path.stat().st_size if path.is_file() else 0
        status["pending"] = self._executor._work_queue.qsize()
        status["ram_mirror_policy"] = "disk_only" if self._backend == "stage1" else "native_session"
        status["resident_cache_layers"] = len(getattr(self._session, "_cache", None) or [])
        status["last_snapshot_skip_reason"] = getattr(self, "_last_snapshot_skip_reason", None)
        status["video_representation"] = "native_temporal" if getattr(self, "_native_video_spec", None) else "sampled_images"
        return status

    def _cache_block_types(self) -> List[str]:
        backbone = self._session.mlx_model.backbone
        return [
            layer.block_type
            for layer in backbone.layers
            if layer.block_type in ("M", "*")
        ]

    def _cache_for_persistence(self) -> List[Any]:
        cache = list(self._session._cache or [])
        block_types = self._cache_block_types()
        if len(cache) != len(block_types):
            raise ValueError(
                "Omni cache layer count does not match the model cache topology: "
                f"cache={len(cache)} topology={len(block_types)}"
            )
        # Persist the session in the representation the model ACTUALLY uses.
        #
        # This used to call entry.to_quantized(group_size=64, bits=4) on every
        # attention slot. The Omni backbone's own make_cache returns a plain
        # float KVCache for '*' blocks, so that was not "storing the native
        # form" -- it was imposing a lossy codec the model never asked for.
        # And the restore path assigns the loaded objects straight back into
        # the live session (self._session._cache = cache) with no dequant,
        # because mlx_lm has no dequant API: only to_quantized. So after any
        # restore the session decoded through q4 attention AND appended
        # q4-quantized keys from then on -- permanent, compounding, and warm
        # answers diverging from cold ones. It was reachable on defaults,
        # since the whole path is gated on the block-disk cache being enabled.
        #
        # If a model's cache is natively quantized (DSV4 pool state is), it is
        # already quantized here and gets persisted exactly as-is. That is the
        # rule: store what the live cache holds, add nothing.
        return list(cache)

    def _persist_session_snapshot(self) -> bool:
        if (
            not getattr(self, "_disk_cache_enabled", False)
            or self._backend != "stage1"
            or self._session is None
            or not self._last_signature
            or getattr(self, "_last_snapshot_skip_reason", None)
            or not getattr(self._session, "_cache", None)
        ):
            return False
        started = time.monotonic()
        try:
            from mlx_lm.models.cache import save_prompt_cache

            metadata = {
                "schema": _OMNI_SESSION_L2_SCHEMA,
                "kind": "completed_turn",
                "bundle_fingerprint": self._session_l2_fingerprint,
                "signature": self._last_signature,
                "history_json": json.dumps(
                    getattr(self._session, "_history_text", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            cache = self._cache_for_persistence()
            path = self._native_disk_store().save(
                self._last_signature,
                lambda destination: save_prompt_cache(str(destination), cache, metadata),
            )
            self._session_l2_path = path
            elapsed = time.monotonic() - started
            self._session_l2_stats["stores"] += 1
            self._session_l2_stats["last_store_seconds"] = round(elapsed, 6)
            self._session_l2_stats["last_error"] = None
            self._session_l2_stats["last_snapshot_kind"] = "completed_turn"
            logger.info(
                "OmniMultimodalDispatcher: persisted native-representation session "
                "signature=%s bytes=%d in %.3fs",
                self._last_signature,
                path.stat().st_size,
                elapsed,
            )
            return True
        except Exception as exc:
            self._session_l2_stats["last_error"] = str(exc)
            logger.warning("Omni session L2 persist failed: %s", exc)
            return False

    def finish_request_cache(self) -> None:
        """Finish this request's SSD snapshot on the native model owner thread.

        Decode and persistence belong to one submitted job. Enqueuing a second
        job after decode could allow another queued request to replace the
        session state before its snapshot is written. Token deltas can stream
        while decoding; the terminal event waits for this boundary.
        """
        if self._backend != "stage1":
            return
        skip_reason = getattr(self, "_last_snapshot_skip_reason", None)
        if skip_reason:
            logger.info("Omni post-generation SSD snapshot skipped: %s; earlier valid prompt checkpoints remain usable", skip_reason)
            self.reset()
            return
        if getattr(self, "_disk_cache_enabled", False) and not self._persist_session_snapshot():
            raise RuntimeError(
                "Omni SSD cache write failed: "
                + str(self._session_l2_stats.get("last_error") or "no snapshot produced")
            )
        # Keep model/encoder weights resident, but never retain reusable KV/SSM
        # payloads between requests. A following request restores its matching
        # SSD snapshot or rebuilds the complete supplied history on a miss.
        self.reset()

    def _try_restore_session_snapshot(self, prefix_signature: str) -> bool:
        if (
            not getattr(self, "_disk_cache_enabled", False)
            or self._backend != "stage1"
            or self._session is None
            or not prefix_signature
        ):
            return False
        started = time.monotonic()
        try:
            from mlx_lm.models.cache import load_prompt_cache

            def read_native(path):
                import mlx.core as mx
                cache, metadata = load_prompt_cache(str(path), return_metadata=True)
                # Materialize while eviction is fenced, then release the pool
                # lock before decoding. The representation remains native.
                mx.eval([entry.state for entry in cache])
                return cache, metadata

            store = self._native_disk_store()
            restored = store.load(prefix_signature, read_native)
            if restored is None:
                self._session_l2_stats["misses"] += 1
                return False
            cache, metadata = restored
            self._session_l2_path = store.last_path
            if metadata.get("schema") != _OMNI_SESSION_L2_SCHEMA:
                # Older snapshots may contain reasoning that the next native
                # template removes, or an unconsumed token at a length limit.
                self._session_l2_stats["misses"] += 1
                logger.info("Omni SSD snapshot ignored: older replay-eligibility schema")
                return False
            if metadata.get("bundle_fingerprint") != self._session_l2_fingerprint:
                raise ValueError("Omni session L2 bundle fingerprint mismatch")
            if metadata.get("signature") != prefix_signature:
                self._session_l2_stats["misses"] += 1
                return False
            block_types = self._cache_block_types()
            if len(cache) != len(block_types):
                raise ValueError(
                    "Omni session L2 layer count mismatch: "
                    f"cache={len(cache)} topology={len(block_types)}"
                )
            history = json.loads(metadata.get("history_json") or "[]")
            if not isinstance(history, list):
                raise ValueError("Omni session L2 history is not a list")
            self._session._cache = cache
            self._session._history_text = history
            self._last_signature = prefix_signature
            elapsed = time.monotonic() - started
            self._session_l2_stats["hits"] += 1
            self._session_l2_stats["last_restore_seconds"] = round(elapsed, 6)
            self._session_l2_stats["last_error"] = None
            logger.info(
                "OmniMultimodalDispatcher: restored native-representation session "
                "signature=%s bytes=%d in %.3fs",
                prefix_signature,
                self._session_l2_path.stat().st_size,
                elapsed,
            )
            return True
        except Exception as exc:
            self._session_l2_stats["last_error"] = str(exc)
            logger.warning("Omni session L2 restore rejected: %s", exc)
            return False

    def _clear_native_disk_cache(self):
        # Run behind any queued decode+publication on the native owner thread.
        self.reset()
        return self._native_disk_store().clear()

    @classmethod
    async def clear_disk_cache_for(cls, bundle_path, *, disk_cache_policy):
        import asyncio
        resolved = str(Path(bundle_path).resolve())
        future = None
        with cls._instance_lock:
            instance = cls._instance
            if (instance is not None and instance.bundle_path == resolved
                    and instance._session_l2_policy == disk_cache_policy):
                future = instance.submit(instance._clear_native_disk_cache)
        if future is not None:
            return await asyncio.wrap_future(future)

        # A restarted server may have disk entries without a native session.
        # Clear them without loading encoders or decoder weights.
        fingerprint = cls._bundle_fingerprint(resolved)
        directory = cls._default_session_l2_path(fingerprint, disk_cache_policy)
        if not directory.is_dir():
            return 0

        def clear_unloaded():
            from .utils.omni_session_disk_store import OmniSessionDiskStore
            store = OmniSessionDiskStore(
                root=disk_cache_policy["root"], model_key=fingerprint,
                max_size_bytes=disk_cache_policy["max_size_bytes"],
            )
            try:
                return store.clear()
            finally:
                store.close()
        return await asyncio.to_thread(clear_unloaded)

    def close(self):
        def release():
            self.reset()
            self._session = None
            if self._session_l2_store is not None:
                self._session_l2_store.close()
                self._session_l2_store = None
        try:
            self.submit(release).result()
        finally:
            self._executor.shutdown(wait=True)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        """Submit work to the persistent Omni native-runtime owner thread."""
        return self._executor.submit(fn, *args, **kwargs)

    @staticmethod
    def _pick_backend() -> str:
        """Pick Stage-1 (PyTorch bridge) or Stage-2 (native MLX).

        Per research/NEMOTRON-OMNI-FINAL-2026-04-28.md:
          - **Stage-1** (PyTorch+MPS encoders → MLX LLM): bit-exact vs the
            HuggingFace reference. ~6.8 s/image cold encode on MPS, ~280 s
            cold-load. This is the **production-validated** path.
          - **Stage-2** (native MLX RADIO + Parakeet): ~17× faster RADIO,
            ~15× faster parakeet, but has KNOWN quality gaps still under
            validation:
              · RADIO bilinear pos_embed outliers — up to 22 % patch
                divergence vs PyTorch. Visual semantics drift enough that
                the model sometimes fails to register that an image was
                provided.
              · Parakeet rel-pos is content-bias only (skips Q·R^T term);
                pitches drift by ~2× in our 440 Hz probe.
            Default-OFF until Wave 4 (`bilinear_pos_embed`) and full
            rel-pos parity ship. Opt in for benchmarking only.

        Override: ``VMLX_OMNI_BACKEND={stage1|stage2|pytorch|mlx}`` (legacy
        ``VMLINUX_OMNI_BACKEND`` accepted for back-compat — the CLI used to set
        only the legacy name while this reader read only the canonical one, so
        ``--omni-backend`` was inert).
        Default: ``stage1`` (correct).
        """
        env = (
            os.environ.get("VMLX_OMNI_BACKEND")
            or os.environ.get("VMLINUX_OMNI_BACKEND")
            or ""
        ).strip().lower()
        if env in ("stage1", "pytorch"):
            return "stage1"
        if env in ("stage2", "mlx"):
            return "stage2"
        return "stage1"

    @staticmethod
    def _pick_device() -> str:
        """Stage-1 only: pick Apple MPS for the PyTorch encoder pass (~10×
        faster than CPU). Stage-2 ignores this — MLX runs on Metal."""
        env = os.environ.get("VMLX_OMNI_ENCODER_DEVICE")
        if env in ("cpu", "mps", "cuda"):
            return env
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _ensure_session(self):
        if self._session is not None:
            return
        if self._backend == "stage2":
            self._ensure_session_stage2()
        else:
            self._ensure_session_stage1()
        logger.info("OmniMultimodalDispatcher: %s session ready", self._backend)

    def _ensure_session_stage2(self):
        """Native MLX path: jang_tools.nemotron_omni.model.NemotronHOmni —
        runs RADIO ViT + Parakeet Conformer + projectors in MLX/Metal.
        Same .turn()/.reset() surface as Stage-1 OmniSession.
        """
        from jang_tools.nemotron_omni.model import NemotronHOmni
        import mlx.core as mx
        logger.info(
            "OmniMultimodalDispatcher: loading Stage-2 native MLX NemotronHOmni "
            "(first call only, ~6 s on Metal)..."
        )
        self._session = NemotronHOmni(
            bundle_path=self.bundle_path, dtype=mx.float32,
        )

    def _ensure_session_stage1(self):
        """Reference fallback: jang_tools.nemotron_omni_session.OmniSession —
        PyTorch encoders bridged into MLX LLM. Keeps bit-exact parakeet
        rel-pos for transcription benchmarks but ~10× slower per encode.
        """
        cradio_status = _ensure_vendored_cradio_dynamic_module()
        if cradio_status.get("installed"):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _patch_omni_encoder_view_for_vendored_cradio()
            logger.info(
                "OmniMultimodalDispatcher: vendored C-RADIO module ready at %s",
                cradio_status.get("destination"),
            )
        else:
            logger.warning(
                "OmniMultimodalDispatcher: vendored C-RADIO module unavailable: %s",
                cradio_status,
            )
        from jang_tools.nemotron_omni_session import OmniSession
        native_video_spec = self._native_video_spec

        class _ThinkingAwareOmniSession(OmniSession):
            _vmlx_prefill_checkpoints = True

            def _extract_video_embeddings(self, video_path):
                if not native_video_spec:
                    return super()._extract_video_embeddings(video_path)
                from .omni_native_video import encode_temporal_video
                embeds, self._vmlx_video_prompt = encode_temporal_video(
                    self, video_path,
                    controls=self._vmlx_video_policy,
                    temporal_patch_size=native_video_spec["temporal_patch_size"],
                )
                return embeds

            def _build_turn_prompt(
                self,
                user_text: str,
                n_image_tokens: int = 0,
                n_video_tokens: int = 0,
                n_audio_tokens: int = 0,
                is_first: bool = False,
            ) -> str:
                prompt = _build_omni_turn_prompt_with_thinking(
                    self.tokenizer,
                    user_text,
                    n_image_tokens=n_image_tokens,
                    n_video_tokens=n_video_tokens,
                    n_audio_tokens=n_audio_tokens,
                    is_first=is_first,
                    enable_thinking=getattr(self, "_vmlx_enable_thinking", None),
                    video_prompt=getattr(self, "_vmlx_video_prompt", None),
                    template_options=getattr(self, "_vmlx_template_options", None),
                )
                from .omni_native_controls import record_prompt_rail
                record_prompt_rail(self, prompt)
                return prompt

        logger.info(
            "OmniMultimodalDispatcher: loading Stage-1 PyTorch-bridge OmniSession "
            "on device=%s (first call only, ~10s on MPS / ~10min on CPU)...",
            self._device,
        )
        self._session = _ThinkingAwareOmniSession(
            bundle_path=self.bundle_path, device=self._device
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        force_reset: bool = False,
        cache_salt: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        token_callback: Optional[Callable[[Optional[int], str], None]] = None,
        video_controls=None,
        template_options=None,
        prompt_rail_callback=None,
        tools=None,
        tool_context=False,
    ) -> Dict[str, Any]:
        """Run one OmniSession turn and return an OpenAI-shaped response."""
        # Native encoders synchronously consume these files during this turn.
        # They are request inputs, not a persistent media cache: private child
        # directories prevent cross-server races and clean up on errors and
        # cooperative cancellation without deleting caller-owned local files.
        with self._lock, tempfile.TemporaryDirectory(
            prefix="request-", dir=self._scratch_dir,
        ) as request_scratch:
            scratch_dir = Path(request_scratch)
            self._ensure_session()
            from .omni_native_controls import has_native_thinking_directive
            template_options = dict(template_options or {})
            template_sensitive = bool(template_options) or tool_context or has_native_thinking_directive(messages)
            self._session._vmlx_template_options = template_options
            self._session._vmlx_prompt_rail_callback = prompt_rail_callback
            self._session._vmlx_prompt_thinking_off = enable_thinking is False
            setattr(
                self._session,
                "_vmlx_enable_thinking",
                None if enable_thinking is None else bool(enable_thinking),
            )
            native_video_spec = getattr(self, "_native_video_spec", None)
            video_policy = _omni_video_policy(
                video_controls,
                temporal_patch_size=(native_video_spec or {}).get("temporal_patch_size"),
            )
            self._session._vmlx_video_policy = video_policy
            self._session._vmlx_video_prompt = None
            self._last_snapshot_skip_reason = None
            prefix_hash = _conversation_signature(messages[:-1], enable_thinking, cache_salt, video_policy=video_policy)
            if self._last_signature != prefix_hash and not force_reset and not template_sensitive:
                self._try_restore_session_snapshot(prefix_hash)
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
            last_content = last_user.get("content", "")
            last_parts = last_content if isinstance(last_content, list) else []
            # OmniSession.turn accepts one video. The full transcript assembler
            # preserves multiple clips, including their separate frame labels.
            multi_clip = bool(native_video_spec) and sum(
                part.get("type") in _VIDEO_TYPES for part in last_parts
            ) > 1
            should_reset = force_reset or multi_clip or template_sensitive or prefix_hash != self._last_signature
            checkpoints = None
            if (should_reset and not force_reset and not cache_salt
                    and self._backend == "stage1"
                    and getattr(self, "_disk_cache_enabled", False)
                    and getattr(self._session, "_vmlx_prefill_checkpoints", False)):
                from .omni_native_prefix import NativePrefillCheckpoints
                checkpoints = NativePrefillCheckpoints(
                    self, messages, video_policy, publish=template_sensitive or enable_thinking is not False,
                    tools=tools,
                )
            full_history = getattr(self, "_backend", "stage1") == "stage1" and (
                checkpoints is not None or multi_clip or template_sensitive or (should_reset and len(messages) > 1)
            )
            if should_reset:
                logger.info(
                    "OmniMultimodalDispatcher: cache reset (prefix=%r != last=%r)",
                    prefix_hash, self._last_signature,
                )
                self._session.reset()
            else:
                logger.info(
                    "OmniMultimodalDispatcher: continuing conversation (prefix matches)"
                )

            # The native session reports only this turn's new prefill length.
            # Count the logical attention offset BEFORE decode, not allocated
            # KV capacity, recurrent-state size, or the post-generation offset.
            cached_tokens = 0
            self._session._vmlx_restored_prefix_tokens = 0
            if not should_reset and getattr(self, "_backend", "stage1") == "stage1":
                cache = getattr(self._session, "_cache", None)
                if cache:
                    backbone = self._session.mlx_model.backbone
                    cached_tokens = int(cache[backbone.fa_idx].offset)

            if full_history:
                # The assembler resolves each part exactly once. Do not run the
                # incremental collector over earlier clips or flatten them.
                text = last_content if isinstance(last_content, str) else "".join(
                    part.get("text", "") for part in last_parts if part.get("type") == "text"
                )
                images, audio, video = [], None, None
                n_images = sum(part.get("type") in _IMAGE_TYPES for part in last_parts)
                has_audio = any(part.get("type") in _AUDIO_TYPES for part in last_parts)
                has_video = any(part.get("type") in _VIDEO_TYPES for part in last_parts)
            else:
                text, images, audio, video = _extract_parts(
                    messages if getattr(self, "_backend", "stage1") == "stage2" else [last_user],
                    scratch_dir, rehydrate_history_media=should_reset,
                    video_controls=video_controls,
                    native_video=bool(native_video_spec),
                )
                n_images, has_audio, has_video = len(images), bool(audio), bool(video)
            logger.info(
                "OmniMultimodalDispatcher: turn — text=%dch, images=%d, audio=%s, video=%s",
                len(text or ""), n_images,
                "yes" if has_audio else "no",
                "yes" if has_video else "no",
            )
            turn_kwargs: Dict[str, Any] = {
                "text": text or "",
                "images": images or None,
                "audio": audio,
                "video": video,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "token_callback": token_callback,
            }
            # Stage-1 reads the flag through the prompt-builder override above.
            # Stage-2 owns its prompt directly and must receive the same resolved
            # value explicitly. Auto preserves Stage-2's native thinking-on
            # default.
            if getattr(self, "_backend", "stage1") == "stage2":
                turn_kwargs["enable_thinking"] = (
                    True if enable_thinking is None else bool(enable_thinking)
                )
            if full_history:
                reply = _run_omni_full_history(
                    self._session, messages, scratch_dir=scratch_dir,
                    extract_parts=partial(_extract_parts, video_controls=video_controls,
                                          native_video=bool(native_video_spec)), enable_thinking=enable_thinking,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    token_callback=token_callback,
                    checkpoints=checkpoints,
                    template_options=template_options,
                    tools=tools,
                )
                cached_tokens = self._session._vmlx_restored_prefix_tokens
            else:
                reply = self._session.turn(**turn_kwargs)
            self._session._vmlx_prompt_rail_callback = None
            prompt_thinking_off = self._session._vmlx_prompt_thinking_off
            reasoning, visible = _split_omni_reply(
                reply, explicit_thinking_off=prompt_thinking_off,
            )
            assistant = {"role": "assistant", "content": visible}
            if reasoning:
                assistant["reasoning_content"] = reasoning
            finish_reason = str(getattr(self._session, "_last_finish_reason", "stop") or "stop")
            if self._backend == "stage1" and finish_reason != "stop":
                # The native decoder samples the last token without consuming
                # it into KV/SSM state when the output cap is reached.
                self._last_snapshot_skip_reason = "incomplete_generation"
            elif self._backend == "stage1" and template_sensitive:
                # Budget hints move to the latest user; directives and history
                # controls can revise earlier tokens. Only exact-token prefill
                # checkpoints are eligible, never completed-turn snapshots.
                self._last_snapshot_skip_reason = "native_template_controls_require_exact_prefix"
            elif self._backend == "stage1" and (reasoning or not prompt_thinking_off):
                # This bridge currently uses the native default that truncates
                # earlier thinking. Its post-decode state still contains those
                # tokens, so it is not a prefix of the next rendered transcript.
                self._last_snapshot_skip_reason = "history_template_truncates_reasoning"
            self._last_signature = None if self._last_snapshot_skip_reason else _conversation_signature(
                messages + [assistant], enable_thinking, cache_salt,
                video_policy=video_policy,
            )

        # The OmniChat reasoning parser writes <think>…</think> inline in
        # ``reply``; we hand the raw text back so the standard server-side
        # deepseek_r1 reasoning-content split path handles it.
        return {
            "content": reply,
            "prompt_thinking_off": prompt_thinking_off,
            "n_images": n_images,
            "has_audio": has_audio,
            "has_video": has_video,
            "prompt_tokens": int(
                getattr(self._session, "_last_prompt_tokens", 0) or 0
            ) + cached_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": int(
                getattr(self._session, "_last_completion_tokens", 0) or 0
            ),
            "finish_reason": str(
                getattr(self._session, "_last_finish_reason", "stop") or "stop"
            ),
        }

    def reset(self):
        with self._lock:
            if self._session is not None:
                self._session.reset()
                self._session._vmlx_prompt_rail_callback = None
            self._last_signature = None


# ── HTTP-shape adapter ────────────────────────────────────────────────
#
# Wraps OmniMultimodalDispatcher.chat() into an OpenAI chat-completion
# response (or SSE stream). Lives here (not in server.py) so the FastAPI
# decorator that's directly above the route handler doesn't accidentally
# bind to a helper function.


class _OmniStreamCancelled(RuntimeError):
    """Internal cooperative-cancellation signal for an Omni decode worker."""


def _split_omni_reply(
    raw: str,
    *,
    explicit_thinking_off: bool,
) -> Tuple[Optional[str], str]:
    """Split the completed native Omni reply without leaking think markers."""
    reasoning_content: Optional[str] = None
    content = raw or ""
    if "<think>" in content and "</think>" in content:
        a = content.index("<think>")
        b = content.index("</think>") + len("</think>")
        reasoning_content = content[a + len("<think>") : b - len("</think>")].strip()
        content = (content[:a] + content[b:]).strip()
    elif "</think>" in content:
        b = content.index("</think>")
        reasoning_content = content[:b].strip()
        content = content[b + len("</think>") :].strip()
    elif not explicit_thinking_off and "<think>" in content:
        # Model opened its own rail and was cut before closing it: the prefix
        # is visible prose, everything after the marker is private thinking.
        # Never render the raw marker.
        a = content.index("<think>")
        reasoning_content = content[a + len("<think>") :].strip() or None
        content = content[:a].strip()
    elif not explicit_thinking_off:
        # The Omni template OPENS the thinking rail in the prompt, so a reply
        # that never emits </think> is still entirely INSIDE that rail — it is
        # reasoning, not the answer. This branch used to fall through to
        # "everything is content", so a reply cut by max_tokens mid-thought
        # rendered the model's raw private thinking as the visible answer with
        # reasoning_content empty.
        #
        # MEASURED on the live Omni bundle, identical image request, 243 chars:
        #   STREAM      content_len=0   reasoning_len=243   (correct)
        #   NON-STREAM  content_len=243 reasoning_len=0     (leaked)
        # _OmniIncrementalRailSplitter defaults to mode="reasoning" and flushes
        # an unclosed tail as reasoning; this mirrors it so the two agree.
        reasoning_content = content.strip()
        content = ""
    else:
        content = content.strip()
    if explicit_thinking_off:
        reasoning_content = None
    return reasoning_content, content


class _OmniIncrementalRailSplitter:
    """Incrementally separate Omni think text from visible content."""

    _START = "<think>"
    _END = "</think>"

    def __init__(self, *, explicit_thinking_off: bool):
        self._explicit_off = bool(explicit_thinking_off)
        self._mode = "probe_off" if self._explicit_off else "reasoning"
        self._buffer = ""

    @staticmethod
    def _marker_suffix_length(text: str) -> int:
        keep = 0
        for marker in (
            _OmniIncrementalRailSplitter._START,
            _OmniIncrementalRailSplitter._END,
        ):
            for length in range(1, min(len(text), len(marker) - 1) + 1):
                if text.endswith(marker[:length]):
                    keep = max(keep, length)
        return keep

    def feed(self, text: str, *, final: bool = False) -> List[Tuple[str, str]]:
        self._buffer += text or ""
        events: List[Tuple[str, str]] = []

        while self._buffer:
            if self._mode == "content":
                events.append(("content", self._buffer))
                self._buffer = ""
                break

            if self._mode == "probe_off":
                if self._START.startswith(self._buffer) and not final:
                    break
                if self._buffer.startswith(self._START):
                    self._buffer = self._buffer[len(self._START) :]
                    self._mode = "discard_reasoning"
                    continue
                self._mode = "content"
                continue

            end_index = self._buffer.find(self._END)
            start_index = self._buffer.find(self._START)
            if start_index >= 0 and (end_index < 0 or start_index < end_index):
                prefix = self._buffer[:start_index]
                if prefix and not self._explicit_off:
                    events.append(("reasoning", prefix))
                self._buffer = self._buffer[start_index + len(self._START) :]
                continue
            if end_index >= 0:
                prefix = self._buffer[:end_index]
                if prefix and not self._explicit_off:
                    events.append(("reasoning", prefix))
                self._buffer = self._buffer[end_index + len(self._END) :]
                self._mode = "content"
                continue

            keep = 0 if final else self._marker_suffix_length(self._buffer)
            emit = self._buffer[:-keep] if keep else self._buffer
            if emit and not self._explicit_off:
                events.append(("reasoning", emit))
            self._buffer = self._buffer[len(emit) :]
            break

        if final and self._buffer:
            if self._mode == "content":
                events.append(("content", self._buffer))
            elif not self._explicit_off:
                events.append(("reasoning", self._buffer))
            self._buffer = ""
        return events


def _validate_native_media_sources(messages):
    """Reject sources the native collector cannot consume before cache lookup."""
    from fastapi import HTTPException

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type")
            if kind in _IMAGE_TYPES:
                source = part.get("image_url") or part.get("image")
            elif kind in _VIDEO_TYPES:
                source = part.get("video_url") or part.get("video")
            elif kind in _AUDIO_TYPES:
                source = part.get("input_audio") or part.get("audio") or part.get("audio_url")
            else:
                continue
            payload = source.get("data") if isinstance(source, dict) and kind in _AUDIO_TYPES else None
            url = source.get("url") if isinstance(source, dict) else source
            valid = False
            try:
                if payload:
                    valid = isinstance(payload, str) and bool(base64.b64decode(payload, validate=True))
                elif isinstance(url, str) and url.startswith("data:"):
                    header, encoded = url.split(",", 1)
                    valid = header.endswith(";base64") and bool(base64.b64decode(encoded, validate=True))
                elif isinstance(url, str) and url:
                    if url.lower().startswith(("http://", "https://")):
                        raise HTTPException(status_code=400, detail=(
                            "Native Omni media source HTTP URLs are not supported. "
                            "Send a base64 data URL or a local file path."
                        ))
                    valid = Path(url).is_file()
            except (ValueError, OSError):
                valid = False
            if not valid:
                raise HTTPException(status_code=400, detail=(
                    f"Invalid native Omni media source for {kind}: "
                    "expected nonempty base64 data or an existing local file."
                ))


def _validate_native_media_controls(request, messages):
    """Reject constraints absent from the native media generation path.

    Tool catalogs/history are validated separately against the native tool
    contract. Constrained output and samplers without a native implementation
    stay out of the encoder and cache instead of being silently ignored.
    """
    from fastapi import HTTPException

    unsupported = []
    response_format = getattr(request, "response_format", None)
    if hasattr(response_format, "model_dump"):
        response_format = response_format.model_dump(exclude_none=True)
    if isinstance(response_format, dict) and isinstance(response_format.get("format"), dict):
        # Responses clients send text.format; the bridge must retain that
        # constraint even when the compatibility model adds type="text".
        response_format = response_format["format"]
    if response_format and response_format.get("type") not in (None, "text"):
        unsupported.append("structured output (response_format)")
    for field, neutral in (
        ("top_k", 0), ("min_p", 0), ("repetition_penalty", 1),
        ("frequency_penalty", 0), ("presence_penalty", 0),
        ("logit_bias", {}), ("logprobs", False), ("top_logprobs", 0),
    ):
        value = getattr(request, field, None)
        if value is not None and value != neutral:
            unsupported.append(field)
    if getattr(request, "seed", None) is not None:
        unsupported.append("seed")
    if getattr(request, "stop", None):
        unsupported.append("stop")
    if "image" in request_modalities(messages):
        for field in ("image_token_budget", "image_max_pixels", "image_min_pixels",
                      "image_resized_height", "image_resized_width"):
            if getattr(request, field, None) is not None:
                unsupported.append(field)
    if unsupported:
        raise HTTPException(status_code=400, detail=(
            "Native Omni media does not support these request controls: "
            + ", ".join(unsupported)
            + ". They cannot be silently ignored. Omit them for this media route."
        ))


async def dispatch_omni_chat_completion(
    request,
    bundle_path: str,
    *,
    disk_cache_enabled: bool = False,
    disk_cache_policy: Optional[Dict[str, Any]] = None,
    effective_max_tokens: Optional[int] = None,
    effective_temperature: Optional[float] = None,
    effective_top_p: Optional[float] = None,
):
    """Run a Nemotron Omni multimodal chat turn and return an OpenAI
    chat-completion response or SSE stream.

    Args:
        request: ChatCompletionRequest (Pydantic) — accepts text + image_url +
            input_audio + video_url content parts.
        bundle_path: absolute path of the loaded Omni bundle (used to bind
            the singleton dispatcher).
    """
    import asyncio
    import json as _json
    import time as _time
    import uuid as _uuid

    from fastapi import HTTPException
    from starlette.responses import StreamingResponse

    # This dispatch precedes the standard server thinking-policy resolver.
    # Omni exposes a boolean enable_thinking control, not native mode kwargs.
    # Reject explicit mode requests before constructing/loading a dispatcher.
    request_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    native_mode = request_template_kwargs.get("thinking_mode")
    if native_mode in ("enabled", "disabled", "adaptive"):
        raise HTTPException(
            status_code=400,
            detail=f"Omni does not support native thinking_mode={native_mode!r}; use enable_thinking",
        )

    # This native session has only a total generation limit. The standard
    # server's separate thinking/answer budget policy does not run here, so
    # accepting either budget spelling would silently ignore the constraint.
    if (
        getattr(request, "max_thinking_tokens", None) is not None
        or request_template_kwargs.get("thinking_budget") is not None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Nemotron Omni does not support a separate thinking-token budget. "
                "Omit max_thinking_tokens/reasoning.budget_tokens/"
                "thinking.budget_tokens/chat_template_kwargs.thinking_budget; "
                "use the total output-token limit instead."
            ),
        )

    from .omni_native_controls import native_template_options
    _template_options = native_template_options(request_template_kwargs)

    msgs_dump: list[dict] = []
    for m in (request.messages or []):
        if hasattr(m, "model_dump"):
            msgs_dump.append(m.model_dump(exclude_none=True))
        elif isinstance(m, dict):
            msgs_dump.append(m)
        else:
            msgs_dump.append(dict(m))

    _validate_native_media_controls(request, msgs_dump)
    from .omni_native_tools import prepare_native_tools, NativeToolOutput
    tool_contract = prepare_native_tools(
        getattr(request, "tools", None), getattr(request, "tool_choice", None), msgs_dump,
    )
    msgs_dump = tool_contract.messages
    _validate_native_media_sources(msgs_dump)
    status = omni_multimodal_component_status(bundle_path)
    supported_modalities = set(status.get("modalities") or ["text"])
    requested_modalities = request_modalities(msgs_dump)
    from .video_controls import VideoControls
    video_controls = VideoControls.from_request(request)
    if "video" in requested_modalities and video_controls.has_pixel_controls:
        raise HTTPException(status_code=400, detail=(
            "Native Omni video supports video_fps and video_max_frames. "
            "Pixel, resized-dimension and video-token budgets are not supported "
            "by its RADIO frame processor."
        ))
    unsupported = sorted(requested_modalities - supported_modalities)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                "Omni bundle does not support requested media modality "
                f"{', '.join(unsupported)}. Supported modalities: "
                f"{', '.join(sorted(supported_modalities))}."
            ),
        )

    dispatcher = OmniMultimodalDispatcher.get(
        bundle_path,
        disk_cache_enabled=disk_cache_enabled,
        disk_cache_policy=disk_cache_policy,
    )

    if _template_options and dispatcher._backend != "stage1":
        raise HTTPException(400, "Native Omni template options are supported only by the Stage-1 media runtime")
    if tool_contract.active and dispatcher._backend != "stage1":
        raise HTTPException(400, "Native Omni tools are supported only by the Stage-1 media runtime")

    # Protocol handlers resolve request/session/bundle defaults before this
    # bridge.  Do not replace an omitted request cap with the bridge's old
    # 256-token convenience default: that discarded bundle max_new_tokens and
    # could finalize Auto turns as reasoning-only.
    _request_max_tokens = getattr(request, "max_tokens", None)
    if _request_max_tokens is None:
        _request_max_tokens = getattr(request, "max_completion_tokens", None)
    _max_tokens = (
        effective_max_tokens
        if effective_max_tokens is not None
        else (_request_max_tokens if _request_max_tokens is not None else 256)
    )
    _request_temperature = getattr(request, "temperature", None)
    _temperature = (
        effective_temperature
        if effective_temperature is not None
        else (_request_temperature if _request_temperature is not None else 0.6)
    )
    _request_top_p = getattr(request, "top_p", None)
    _top_p = (
        effective_top_p
        if effective_top_p is not None
        else (_request_top_p if _request_top_p is not None else 0.95)
    )
    _ct_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    _request_enable_thinking = getattr(request, "enable_thinking", None)
    if _request_enable_thinking is not None:
        _enable_thinking = bool(_request_enable_thinking)
    elif isinstance(_ct_kwargs, dict) and "enable_thinking" in _ct_kwargs:
        _enable_thinking = bool(_ct_kwargs["enable_thinking"])
    else:
        _enable_thinking = None

    _explicit_thinking_off = _enable_thinking is False
    completion_id = f"chatcmpl-{_uuid.uuid4().hex[:24]}"
    created = int(_time.time())
    t_start = _time.time()
    # Match the server's public bypass contract: a non-empty salt requests
    # fresh state, not a persistent salted namespace. Neither restore nor
    # publication may run for this request.
    _cache_salt = getattr(request, "cache_salt", None)
    _bypass_cache = getattr(request, "skip_prefix_cache", None) is True or (
        isinstance(_cache_salt, str) and bool(_cache_salt)
    )

    def _run_chat(token_callback=None, prompt_rail_callback=None):
        try:
            result = dispatcher.chat(
                messages=msgs_dump,
                max_tokens=int(_max_tokens),
                temperature=float(_temperature),
                top_p=float(_top_p),
                enable_thinking=_enable_thinking,
                force_reset=_bypass_cache,
                cache_salt=_cache_salt,
                token_callback=token_callback,
                video_controls=video_controls,
                template_options=_template_options,
                prompt_rail_callback=prompt_rail_callback,
                tools=tool_contract.template_tools,
                tool_context=tool_contract.active,
            )
            if tool_contract.active:
                # Use the streaming rail state machine for completed tool
                # responses too. Searching globally for think tags corrupts
                # literal code/string arguments after the reasoning rail.
                rails = _OmniIncrementalRailSplitter(
                    explicit_thinking_off=result.get("prompt_thinking_off", _explicit_thinking_off),
                ).feed(result.get("content") or "", final=True)
                visible = "".join(text for rail, text in rails if rail == "content").strip()
                result["parsed_reasoning"] = "".join(
                    text for rail, text in rails if rail == "reasoning"
                ).strip() or None
                output = NativeToolOutput(tool_contract)
                safe = output.feed(visible)
                tail, calls = output.finish(result.get("finish_reason") or "stop")
                result["visible_content"] = safe + tail
                result["tool_calls"] = calls
                if calls:
                    result["finish_reason"] = "tool_calls"
            if _bypass_cache:
                dispatcher.reset()
            else:
                dispatcher.finish_request_cache()
            return result
        except Exception:
            dispatcher.reset()
            raise

    if not getattr(request, "stream", False):
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wrap_future(
                dispatcher.submit(_run_chat),
                loop=loop,
            )
        except Exception as e:
            logger.error("Omni multimodal dispatch failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Omni multimodal generation failed: {e}"
            )

        elapsed = _time.time() - t_start
        reasoning_content, content = _split_omni_reply(
            result.get("content") or "",
            explicit_thinking_off=result.get("prompt_thinking_off", _explicit_thinking_off),
        )
        content = result.get("visible_content", content)
        reasoning_content = result.get("parsed_reasoning", reasoning_content)
        prompt_tokens = int(result.get("prompt_tokens") or 0)
        completion_tokens = int(result.get("completion_tokens") or 0)
        finish_reason = str(result.get("finish_reason") or "stop")
        logger.info(
            "Omni multimodal chat: %d images, audio=%s, video=%s — "
            "%d-char reply in %.2fs",
            result.get("n_images", 0),
            result.get("has_audio"),
            result.get("has_video"),
            len(content),
            elapsed,
        )
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if result.get("tool_calls"):
            message["tool_calls"] = result["tool_calls"]
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                **({"prompt_tokens_details": {"cached_tokens": result["cached_tokens"]}}
                   if "cached_tokens" in result else {}),
            },
        }

    event_queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    def _enqueue(event: tuple) -> None:
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)
        except RuntimeError:
            # The connection/event loop is already gone.  The next decode
            # callback sees cancel_event and cooperatively stops the turn.
            cancel_event.set()

    def _on_token(token_id: Optional[int], text_delta: str) -> None:
        if cancel_event.is_set():
            raise _OmniStreamCancelled("Omni client disconnected")
        _enqueue(("token", token_id, text_delta))

    def _on_prompt_rail(off: bool) -> None:
        _enqueue(("prompt_rail", off))

    def _on_done(future: Future) -> None:
        try:
            _enqueue(("done", future.result()))
        except _OmniStreamCancelled as exc:
            dispatcher.reset()
            _enqueue(("cancelled", exc))
        except Exception as exc:  # pragma: no cover - exercised by live route
            _enqueue(("error", exc))

    async def _sse_iter():
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        }
        yield f"data: {_json.dumps(first)}\n\n"
        splitter = _OmniIncrementalRailSplitter(
            explicit_thinking_off=_explicit_thinking_off
        )
        future = dispatcher.submit(_run_chat, _on_token, _on_prompt_rail)
        future.add_done_callback(_on_done)
        streamed_reasoning = ""
        streamed_content = ""
        tool_output = NativeToolOutput(tool_contract) if tool_contract.active else None
        result = None
        try:
            while True:
                event = await event_queue.get()
                kind = event[0]
                if kind == "prompt_rail":
                    splitter = _OmniIncrementalRailSplitter(explicit_thinking_off=event[1])
                    continue
                if kind == "token":
                    for rail, delta in splitter.feed(event[2]):
                        if not delta:
                            continue
                        if rail == "reasoning":
                            streamed_reasoning += delta
                            delta_payload = {"reasoning_content": delta}
                        else:
                            if tool_output is not None:
                                delta = tool_output.feed(delta)
                                if not delta:
                                    continue
                            streamed_content += delta
                            delta_payload = {"content": delta}
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": delta_payload,
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {_json.dumps(chunk)}\n\n"
                    continue
                if kind == "done":
                    result = event[1]
                    break
                if kind == "cancelled":
                    return
                if kind == "error":
                    logger.error(
                        "Omni multimodal streaming dispatch failed: %s",
                        event[1],
                        exc_info=(
                            type(event[1]),
                            event[1],
                            event[1].__traceback__,
                        ),
                    )
                    yield "data: " + _json.dumps({
                        "error": {
                            "type": "server_error",
                            "message": (
                                "Omni multimodal generation failed: "
                                f"{event[1]}"
                            ),
                        }
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

            for rail, delta in splitter.feed("", final=True):
                if not delta:
                    continue
                if rail == "reasoning":
                    streamed_reasoning += delta
                    delta_payload = {"reasoning_content": delta}
                else:
                    if tool_output is not None:
                        delta = tool_output.feed(delta)
                        if not delta:
                            continue
                    streamed_content += delta
                    delta_payload = {"content": delta}
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta_payload,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {_json.dumps(chunk)}\n\n"

            if tool_output is not None:
                tail, _ = tool_output.finish(
                    "stop" if (result or {}).get("tool_calls") else (result or {}).get("finish_reason", "stop")
                )
                if tail:
                    streamed_content += tail
                    yield "data: " + _json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                    }) + "\n\n"
                calls = (result or {}).get("tool_calls")
                if calls:
                    # The owner has completed finish_request_cache before its
                    # done event. No client can act on a call ahead of SSD
                    # publication; the next request needs no artificial wait.
                    yield "data: " + _json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": request.model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": index, **call} for index, call in enumerate(calls)
                        ]}, "finish_reason": None}],
                    }) + "\n\n"

            raw = (result or {}).get("content") or ""
            final_reasoning, final_content = _split_omni_reply(
                raw,
                explicit_thinking_off=(result or {}).get("prompt_thinking_off", _explicit_thinking_off),
            )
            final_content = (result or {}).get("visible_content", final_content)
            final_reasoning = (result or {}).get("parsed_reasoning", final_reasoning)
            if final_reasoning and not final_reasoning.startswith(
                streamed_reasoning.strip()
            ):
                logger.warning(
                    "Omni reasoning stream/final reconciliation differs "
                    "(streamed=%d final=%d)",
                    len(streamed_reasoning),
                    len(final_reasoning),
                )
            if final_content and not final_content.startswith(streamed_content.strip()):
                logger.warning(
                    "Omni content stream/final reconciliation differs "
                    "(streamed=%d final=%d)",
                    len(streamed_content),
                    len(final_content),
                )

            prompt_tokens = int((result or {}).get("prompt_tokens") or 0)
            completion_tokens = int((result or {}).get("completion_tokens") or 0)
            finish_reason = str((result or {}).get("finish_reason") or "stop")
            elapsed = _time.time() - t_start
            logger.info(
                "Omni multimodal stream: %d images, audio=%s, video=%s — "
                "%d reasoning chars, %d content chars, %d tokens in %.2fs",
                (result or {}).get("n_images", 0),
                (result or {}).get("has_audio"),
                (result or {}).get("has_video"),
                len(streamed_reasoning),
                len(streamed_content),
                completion_tokens,
                elapsed,
            )
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    **({"prompt_tokens_details": {"cached_tokens": result["cached_tokens"]}}
                       if "cached_tokens" in (result or {}) else {}),
                },
            }
            yield f"data: {_json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            cancel_event.set()

    return StreamingResponse(_sse_iter(), media_type="text/event-stream")
