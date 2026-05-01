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
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported content-part types per OpenAI chat schema (+ vMLX extension for video).
_IMAGE_TYPES = {"image_url", "image"}
_AUDIO_TYPES = {"input_audio", "audio"}
_VIDEO_TYPES = {"video_url", "video"}


def is_omni_multimodal_bundle(model_path: str | Path) -> bool:
    """Return True iff the bundle is a Nemotron-3-Nano-Omni text+multimodal bundle.

    Checks:
      1. ``config.json`` exists and ``model_type == "nemotron_h"``
      2. ``config_omni.json`` exists alongside (carries the NVLM/parakeet wrapper
         metadata that ``OmniChat`` reads)
      3. The bundle's safetensors index includes ``vision_model.*`` keys
         (proves multimodal weights are merged into the LLM bundle — addon
         already absorbed). All 3 published JANGQ-AI Omni bundles satisfy this.
    """
    p = Path(model_path)
    if not p.is_dir():
        return False
    cfg = p / "config.json"
    omni_cfg = p / "config_omni.json"
    idx = p / "model.safetensors.index.json"
    if not (cfg.is_file() and omni_cfg.is_file() and idx.is_file()):
        return False
    try:
        if json.loads(cfg.read_text()).get("model_type") != "nemotron_h":
            return False
        keys = json.loads(idx.read_text()).get("weight_map", {}).keys()
        return any(k.startswith("vision_model.") for k in keys)
    except Exception as e:  # pragma: no cover
        logger.debug(f"is_omni_multimodal_bundle({p}) check failed: {e}")
        return False


def request_has_multimodal(messages: List[Dict[str, Any]]) -> bool:
    """True iff any message's content includes image/audio/video parts."""
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if ptype in _IMAGE_TYPES or ptype in _AUDIO_TYPES or ptype in _VIDEO_TYPES:
                    return True
    return False


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
                            cur_video = _materialize_to_temp(raw, ext, scratch_dir)
                        elif Path(url).exists():
                            cur_video = Path(url)
                elif ptype == "text":
                    last_user_text += (part.get("text") if isinstance(part, dict) else getattr(part, "text", "")) or ""
    return last_user_text, cur_images, cur_audio, cur_video


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
        from jang_tools.nemotron_omni_session import OmniSession
        logger.info(
            "OmniMultimodalDispatcher: loading Stage-1 PyTorch-bridge OmniSession "
            "on device=%s (first call only, ~10s on MPS / ~10min on CPU)...",
            self._device,
        )
        self._session = OmniSession(
            bundle_path=self.bundle_path, device=self._device
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        force_reset: bool = False,
    ) -> Dict[str, Any]:
        """Run one OmniSession turn and return an OpenAI-shaped response."""
        with self._lock:
            self._ensure_session()
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

    dispatcher = OmniMultimodalDispatcher.get(bundle_path)

    msgs_dump: list[dict] = []
    for m in (request.messages or []):
        if hasattr(m, "model_dump"):
            msgs_dump.append(m.model_dump(exclude_none=True))
        elif isinstance(m, dict):
            msgs_dump.append(m)
        else:
            msgs_dump.append(dict(m))

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
    _ct_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    _explicit_thinking_off = (
        isinstance(_ct_kwargs, dict) and _ct_kwargs.get("enable_thinking") is False
    )
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
