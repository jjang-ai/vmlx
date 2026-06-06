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
  - Per-request reset: when a NEW conversation arrives (detected by message
    history not matching the running history-prefix), reset the session.
  - Extracts base64-encoded media into ``$TMPDIR/vmlx-omni-XXXX.{jpg,wav,mp4}``
    files because OmniChat's encoders expect file paths or PIL images.
  - Returns plain text — the HTTP layer wraps it in OpenAI chat-completion
    format (and emulates SSE by chunking the final text if streaming was
    requested).

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
import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported content-part types per OpenAI chat schema (+ vMLX extension for video).
_IMAGE_TYPES = {"image_url", "image"}
_AUDIO_TYPES = {"input_audio", "audio"}
_VIDEO_TYPES = {"video_url", "video"}
_CRADIO_CACHE_REPO = "C_hyphen_RADIOv2_hyphen_H"
_CRADIO_CACHE_REVISION = "0d8f4c18c877166eda07ddae1386bcad256b7a6a"


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
        import jang_tools.nemotron_omni_chat as omni_chat
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

    Checks are intentionally header/config-only:
      1. ``config.json`` exists and ``model_type == "nemotron_h"``
      2. ``config_omni.json`` exists alongside (carries the NVLM/parakeet wrapper
         metadata that ``OmniChat`` reads)
      3. ``configuration_radio.py`` exists for the RADIO vision config
      4. The safetensors index includes RADIO/vision keys
      5. The safetensors index includes Parakeet/sound keys
      6. The safetensors index includes the media projector keys
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
        keys = json.loads(idx.read_text()).get("weight_map", {}).keys()
        key_list = [str(k) for k in keys]
        status["has_radio_weights"] = any(
            k.startswith("vision_model.radio_model.") for k in key_list
        )
        status["has_parakeet_weights"] = any(
            k.startswith("sound_encoder.") or k.startswith("parakeet.")
            for k in key_list
        )
        status["has_media_projector"] = any(
            k.startswith("mlp1.")
            or k.startswith("sound_projector.")
            or k.startswith("projector.")
            for k in key_list
        )
        if status["has_radio_weights"]:
            status["modalities"].append("image")
            # The current Stage-1 bridge calls processor.video_processor.
            # Nemotron bundles with only the image processor can still process
            # images, but advertising video turns into a runtime 500.
            status["video_bridge_supported"] = bool(status["has_video_preprocessor_config"])
            status["video_frame_fallback_supported"] = True
            status["modalities"].append("video")
        if status["has_parakeet_weights"]:
            status["modalities"].append("audio")
        requirements = {
            "model_type=nemotron_h": status["config_model_type"] == "nemotron_h",
            "sound_config.model_type=parakeet": status["sound_config_model_type"] == "parakeet",
            "radio weights": status["has_radio_weights"],
            "parakeet weights": status["has_parakeet_weights"],
            "media projector": status["has_media_projector"],
        }
        status["missing"].extend([name for name, ok in requirements.items() if not ok])
        status["modalities"] = ["text"] + sorted(set(status["modalities"]))
        status["bundle_compatible"] = not status["missing"]
        return status
    except Exception as e:  # pragma: no cover
        status["missing"].append(f"inspect_error:{type(e).__name__}")
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
    """Write bytes to a temp file under scratch_dir keyed by content hash.

    Caching by hash means repeated requests with the same media don't blow up
    disk; the OmniChat encoders re-read but our PIL/soundfile decode is fast.
    """
    digest = hashlib.sha256(data).hexdigest()[:16]
    out = scratch_dir / f"{digest}{suffix}"
    if not out.exists():
        out.write_bytes(data)
    return out


def _extract_omni_video_frames(video_path: Path, scratch_dir: Path) -> List[Path]:
    """Sample an Omni video into image files for the RADIO image encoder."""
    try:
        from PIL import Image
        import numpy as np
        from .models.mllm import extract_video_frames_smart

        frames = extract_video_frames_smart(
            str(video_path),
            fps=float(os.environ.get("VMLINUX_OMNI_VIDEO_FPS", "1")),
            max_frames=int(os.environ.get("VMLINUX_OMNI_VIDEO_MAX_FRAMES", "4")),
        )
        deduped = []
        threshold = float(os.environ.get("VMLINUX_OMNI_VIDEO_DEDUP_MAD", "8"))
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
        digest = hashlib.sha256(str(video_path).encode("utf-8")).hexdigest()[:10]
        if (
            len(deduped) > 1
            and os.environ.get("VMLINUX_OMNI_VIDEO_CONTACT_SHEET", "1") != "0"
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
        logger.warning(
            "Omni video frame fallback failed for %s; trying native video path: %s",
            video_path,
            exc,
        )
        return []


def _extract_parts(
    messages: List[Dict[str, Any]], scratch_dir: Path
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
                    src = part.get("input_audio") or part.get("audio") or {}
                    if isinstance(src, dict):
                        b64 = src.get("data")
                        fmt = src.get("format", "wav")
                        if b64:
                            raw = base64.b64decode(b64)
                            cur_audio = _materialize_to_temp(raw, f".{fmt}", scratch_dir)
                        elif src.get("url", "").startswith("data:"):
                            raw, ext = _decode_data_url(src["url"])
                            cur_audio = _materialize_to_temp(raw, ext, scratch_dir)
                    elif isinstance(src, str) and Path(src).exists():
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
                            frame_paths = _extract_omni_video_frames(
                                video_path,
                                scratch_dir,
                            )
                            if frame_paths:
                                cur_images.extend(frame_paths)
                                cur_video = None
                            else:
                                cur_video = video_path
                elif ptype == "text":
                    last_user_text += (part.get("text") if isinstance(part, dict) else getattr(part, "text", "")) or ""
    return last_user_text, cur_images, cur_audio, cur_video


def _build_omni_turn_prompt_with_thinking(
    tokenizer: Any,
    user_text: str,
    n_image_tokens: int = 0,
    n_video_tokens: int = 0,
    n_audio_tokens: int = 0,
    is_first: bool = False,
    enable_thinking: Optional[bool] = None,
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
        media += "<img>" + ("<image>" * n_video_tokens) + "</img>\n"
    if n_audio_tokens > 0:
        media += "<sound>" + ("<so_embedding>" * n_audio_tokens) + "</sound>\n"
    msg_content = media + user_text
    messages = [{"role": "user", "content": msg_content}]
    template_kwargs = {
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


def _user_texts_in_order(messages: List[Dict[str, Any]]) -> List[str]:
    """Return text from each user turn, in order. Used to compute a
    cumulative hash that detects 'same conversation continuing' vs
    'fresh conversation' robustly across single- and multi-turn requests.
    """
    out: List[str] = []
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        if role != "user":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            parts: List[str] = []
            for p in content:
                ptype = p.get("type") if isinstance(p, dict) else getattr(p, "type", None)
                if ptype == "text":
                    parts.append((p.get("text") if isinstance(p, dict) else getattr(p, "text", "")) or "")
            out.append(" ".join(parts))
        else:
            out.append("")
    return out


def _hash_user_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # separator so [a,b] != [ab]
    return h.hexdigest()[:16]


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
    def get(cls, bundle_path: str | Path) -> "OmniMultimodalDispatcher":
        bundle_path = str(Path(bundle_path).resolve())
        with cls._instance_lock:
            if cls._instance is None or cls._instance.bundle_path != bundle_path:
                if cls._instance is not None:
                    logger.info(
                        "OmniMultimodalDispatcher: rebinding from %s to %s",
                        cls._instance.bundle_path, bundle_path,
                    )
                cls._instance = cls(bundle_path)
            return cls._instance

    def __init__(self, bundle_path: str):
        self.bundle_path = bundle_path
        self._session = None
        self._lock = threading.Lock()
        self._last_signature: Optional[str] = None
        self._scratch_dir = Path(tempfile.gettempdir()) / "vmlx-omni-media"
        self._scratch_dir.mkdir(exist_ok=True)
        self._backend = self._pick_backend()
        self._device = self._pick_device() if self._backend == "stage1" else "metal"
        logger.info(
            "OmniMultimodalDispatcher: bundle=%s, backend=%s, device=%s, scratch=%s",
            bundle_path, self._backend, self._device, self._scratch_dir,
        )

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

        Override: ``VMLX_OMNI_BACKEND={stage1|stage2|pytorch|mlx}``.
        Default: ``stage1`` (correct).
        """
        env = os.environ.get("VMLX_OMNI_BACKEND", "").strip().lower()
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

        class _ThinkingAwareOmniSession(OmniSession):
            def _build_turn_prompt(
                self,
                user_text: str,
                n_image_tokens: int = 0,
                n_video_tokens: int = 0,
                n_audio_tokens: int = 0,
                is_first: bool = False,
            ) -> str:
                return _build_omni_turn_prompt_with_thinking(
                    self.tokenizer,
                    user_text,
                    n_image_tokens=n_image_tokens,
                    n_video_tokens=n_video_tokens,
                    n_audio_tokens=n_audio_tokens,
                    is_first=is_first,
                    enable_thinking=getattr(self, "_vmlx_enable_thinking", None),
                )

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
        enable_thinking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run one OmniSession turn and return an OpenAI-shaped response."""
        with self._lock:
            self._ensure_session()
            setattr(
                self._session,
                "_vmlx_enable_thinking",
                None if enable_thinking is None else bool(enable_thinking),
            )
            # Cumulative-prefix signature: hash all USER texts EXCLUDING the
            # current (last) one. If it matches the hash we stored after the
            # previous turn (= hash of all user texts including the one we
            # just answered), this is the next turn of the same conversation.
            user_texts = _user_texts_in_order(messages)
            prefix_hash = _hash_user_texts(user_texts[:-1])
            current_hash = _hash_user_texts(user_texts)

            should_reset = force_reset or prefix_hash != self._last_signature
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
            self._last_signature = current_hash

            text, images, audio, video = _extract_parts(messages, self._scratch_dir)
            logger.info(
                "OmniMultimodalDispatcher: turn — text=%dch, images=%d, audio=%s, video=%s",
                len(text or ""), len(images),
                "yes" if audio else "no",
                "yes" if video else "no",
            )
            reply = self._session.turn(
                text=text or "",
                images=images or None,
                audio=audio,
                video=video,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # The OmniChat reasoning parser writes <think>…</think> inline in
        # ``reply``; we hand the raw text back so the standard server-side
        # deepseek_r1 reasoning-content split path handles it.
        return {
            "content": reply,
            "n_images": len(images),
            "has_audio": bool(audio),
            "has_video": bool(video),
        }

    def reset(self):
        with self._lock:
            if self._session is not None:
                self._session.reset()
            self._last_signature = ""


# ── HTTP-shape adapter ────────────────────────────────────────────────
#
# Wraps OmniMultimodalDispatcher.chat() into an OpenAI chat-completion
# response (or SSE stream). Lives here (not in server.py) so the FastAPI
# decorator that's directly above the route handler doesn't accidentally
# bind to a helper function.

async def dispatch_omni_chat_completion(request, bundle_path: str):
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

    msgs_dump: list[dict] = []
    for m in (request.messages or []):
        if hasattr(m, "model_dump"):
            msgs_dump.append(m.model_dump(exclude_none=True))
        elif isinstance(m, dict):
            msgs_dump.append(m)
        else:
            msgs_dump.append(dict(m))

    status = omni_multimodal_component_status(bundle_path)
    supported_modalities = set(status.get("modalities") or ["text"])
    requested_modalities = request_modalities(msgs_dump)
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

    dispatcher = OmniMultimodalDispatcher.get(bundle_path)

    _max_tokens = (
        getattr(request, "max_tokens", None)
        or getattr(request, "max_completion_tokens", None)
        or 256
    )
    _temperature = getattr(request, "temperature", None)
    if _temperature is None:
        _temperature = 0.6
    _top_p = getattr(request, "top_p", None)
    if _top_p is None:
        _top_p = 0.95
    _ct_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    _request_enable_thinking = getattr(request, "enable_thinking", None)
    if _request_enable_thinking is not None:
        _enable_thinking = bool(_request_enable_thinking)
    elif isinstance(_ct_kwargs, dict) and "enable_thinking" in _ct_kwargs:
        _enable_thinking = bool(_ct_kwargs["enable_thinking"])
    else:
        _enable_thinking = None

    loop = asyncio.get_running_loop()
    t_start = _time.time()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: dispatcher.chat(
                messages=msgs_dump,
                max_tokens=int(_max_tokens),
                temperature=float(_temperature),
                top_p=float(_top_p),
                enable_thinking=_enable_thinking,
            ),
        )
    except Exception as e:
        logger.error("Omni multimodal dispatch failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Omni multimodal generation failed: {e}"
        )

    elapsed = _time.time() - t_start
    raw = result["content"] or ""
    reasoning_content: Optional[str] = None
    content = raw
    _explicit_thinking_off = _enable_thinking is False
    if "<think>" in raw and "</think>" in raw:
        a = raw.index("<think>")
        b = raw.index("</think>") + len("</think>")
        reasoning_content = raw[a + len("<think>"):b - len("</think>")].strip()
        content = (raw[:a] + raw[b:]).strip()
    elif "</think>" in raw:
        # deepseek_r1 templates often emit only the closing tag (the open
        # `<think>` is baked into the assistant prompt prefix). Treat
        # everything before `</think>` as reasoning, after as content.
        b = raw.index("</think>")
        reasoning_content = raw[:b].strip()
        content = raw[b + len("</think>"):].strip()
    if _explicit_thinking_off:
        reasoning_content = None

    completion_id = f"chatcmpl-{_uuid.uuid4().hex[:24]}"
    created = int(_time.time())
    try:
        prompt_tokens = sum(
            len((m.get("content") or "").split()) if isinstance(m.get("content"), str) else 0
            for m in msgs_dump
        )
    except Exception:
        prompt_tokens = 0
    completion_tokens = max(1, len(content.split()))

    logger.info(
        "Omni multimodal chat: %d images, audio=%s, video=%s — %d-char reply in %.2fs",
        result.get("n_images", 0),
        result.get("has_audio"),
        result.get("has_video"),
        len(content),
        elapsed,
    )

    if not getattr(request, "stream", False):
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def _sse_iter():
        first = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {_json.dumps(first)}\n\n"
        if reasoning_content:
            for i in range(0, len(reasoning_content), 80):
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": request.model,
                    "choices": [{"index": 0,
                                 "delta": {"reasoning_content": reasoning_content[i:i + 80]},
                                 "finish_reason": None}],
                }
                yield f"data: {_json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)
        for i in range(0, len(content), 80):
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": request.model,
                "choices": [{"index": 0,
                             "delta": {"content": content[i:i + 80]},
                             "finish_reason": None}],
            }
            yield f"data: {_json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)
        final = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {_json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse_iter(), media_type="text/event-stream")
