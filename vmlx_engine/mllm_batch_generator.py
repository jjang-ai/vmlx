# SPDX-License-Identifier: Apache-2.0
# Base architecture from waybarrios/vllm-mlx. _BatchOffsetSafeCache, hybrid SSM
# cache merge, MoE CacheList support, and vision encoding pipeline added by
# Jinho Jang (eric@jangq.ai) for vMLX (github.com/jjang-ai/vmlx).
"""
MLLM Batch Generator -- continuous batching engine for multimodal models.

This module is the low-level generation engine that the MLLMScheduler delegates
to. It handles vision preprocessing, prefill, cache management, and batched
token-by-token decode on Apple Metal.

KEY INSIGHT
-----------
VLM models have a ``model.language_model`` which is a standard LLM.
After the initial forward pass with vision encoding, text generation uses
only the language model -- which CAN be batched using the same BatchKVCache
pattern as pure LLM inference.

GENERATION PIPELINE
-------------------
::

    _process_prompts()                        step()
    ==================                        ======
    For each request:                         For all active requests:
    1. _preprocess_request()                  1. language_model(y, cache=cache)
       - Pixel processing + tokenization      2. Sample next token
       - Vision cache lookup                  3. Check stop conditions
    2. Cache fetch (paged/memory/legacy/disk)  4. Return responses
    3. _run_vision_encoding()                  5. Filter finished requests
       - Full VLM forward (vision + LM)
       - Populates KV cache
    4. Capture SSM state (hybrid models)
    5. Merge per-request caches -> BatchKVCache
    6. Return MLLMBatch

CACHE FETCH ORDER (in _process_prompts)
-----------------------------------------
Each request tries caches in this priority::

    1. Paged cache (block_aware_cache.fetch_cache)
       +-- Hybrid model: also check HybridSSMStateCache
    2. Memory-aware or legacy cache (fetch/fetch_cache)
    3. Disk cache L2 fallback (disk_cache.fetch)

On cache HIT for pure attention models:
  - ``req.prompt_cache`` = reconstructed KV cache
  - ``req.input_ids`` trimmed to remaining (uncached) tokens
  - ``req.pixel_values/attention_mask/image_grid_thw`` = None (no re-encoding)

On cache HIT for hybrid models (KV + SSM):
  - With SSM companion HIT: full cache (KV + SSM), skip all prefix tokens
  - Without SSM companion: forced full prefill (SSM state is path-dependent)

HYBRID MODEL HANDLING
---------------------
Hybrid models (e.g., Qwen3.5-VL 122B: 36 SSM + 12 attention layers) require
special treatment because SSM state is cumulative -- you can't skip prefix
computation for SSM layers even if you have the KV cache.

``HybridSSMStateCache``:
  - Companion LRU cache (max 50 entries) storing SSM layer states
  - Keyed by hash(tuple(token_ids[:prompt_len])) for text-only prefixes
  - Media placeholder prefixes are not stored under token-only keys; image,
    video, and audio embeddings are path-dependent and need a media-aware cache
    key before reuse can be safe
  - Stored after prefill (before cache merge destroys per-request state)
  - Deep-copies SSM arrays with mx.contiguous() for safety
  - Enables groundbreaking full prefix skip for hybrid VLMs

``_fix_hybrid_cache()``:
  - Expands KV-only reconstructed cache to full layer count
  - Inserts fresh ArraysCache at SSM positions from model.make_cache() template
  - Pre-computed ``_hybrid_kv_positions`` and ``_hybrid_num_layers`` for speed

METAL OPTIMIZATIONS
-------------------
- ``mx.metal.set_cache_limit()``: 25% of max working set (floor 512MB)
  Bounds the Metal allocator's free-list so prefix cache and OS get memory.
- ``mx.async_eval()``: Used in prefill loop for GPU/CPU overlap.
  Submits sampled token + cache states to GPU without blocking.
- ``mx.contiguous()``: Applied to extracted cache keys/values in
  ``MLLMBatch.extract_cache()`` to release batch tensor references.
- ``mx.new_stream()``: Dedicated Metal stream for generation.
- Old limits restored in ``close()`` for clean teardown.

KEY CLASSES
-----------
- ``HybridSSMStateCache`` -- Companion LRU cache for SSM layer states
- ``MLLMBatchRequest`` -- Per-request data (tokens, pixels, sampling params)
- ``MLLMBatchResponse`` -- Per-request step output (token, logprobs, cache)
- ``MLLMBatch`` -- Active batch state (all requests being generated together)
- ``MLLMBatchStats`` -- Throughput and timing statistics
- ``MLLMBatchGenerator`` -- Main batch generator class

HELPER FUNCTIONS
----------------
- ``_dequantize_cache()`` -- QuantizedKVCache -> KVCache for batch generation
- ``_fix_hybrid_cache()`` -- Expand KV-only cache for hybrid models
- ``_merge_caches()`` -- Merge per-request caches into batch-aware caches
"""

import hashlib
import importlib
import logging
import inspect
import math
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .errors import (
    PromptTooLongError,
    UnsupportedMediaModalityError,
    VLMImagePrefillBudgetError,
)
from .vision_embedding_cache import VisionEmbeddingCache
from .cache_key import CACHE_EXTRA_SCOPES_KEY, scope_cache_extra_key
from .utils.prefix_hit import (
    disk_prefix_hit_tail_and_cached_tokens as _shared_disk_prefix_hit,
    prefix_hit_tail_and_cached_tokens as _shared_prefix_hit,
)
from .utils.memory_limits import (
    get_effective_metal_working_set_bytes,
    get_metal_ws_guard_threshold,
)
from .mlx_memory import clear_mlx_memory_cache
from .native_mtp_cache_telemetry import (
    native_mtp_cache_lifecycle_snapshot,
    native_mtp_cache_snapshot,
)
from .native_mtp_adaptive import (
    NativeMTPAdaptiveValueState,
    adaptive_value_snapshot,
    arm_depth_cycle,
    choose_depth_by_value,
    finish_armed_depth_cycle,
    note_forced_depth_change,
)
from .native_mtp_profile import (
    NativeMTPProfileStore,
    profile_key as native_mtp_profile_key,
)

logger = logging.getLogger(__name__)
_MIMO_AUDIO_TOKENIZER_CACHE: Dict[str, Any] = {}


def _read_config_field(obj: Any, field: str) -> Any:
    """Read config fields from object-style or dict-style configs."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _positive_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _mllm_input_ids_token_count(input_ids: Any) -> int:
    """Return the token count from one VLM processor ``input_ids`` payload."""
    if input_ids is None:
        return 0
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        try:
            if len(shape) == 0:
                return int(getattr(input_ids, "size", 0) or 0)
            return int(shape[-1])
        except Exception:
            pass
    if hasattr(input_ids, "tolist"):
        try:
            input_ids = input_ids.tolist()
        except Exception:
            return int(getattr(input_ids, "size", 0) or 0)
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    return int(getattr(input_ids, "size", 0) or 0)


def _hash_mllm_media_text(hasher: Any, label: str, value: Any) -> None:
    hasher.update(label.encode("utf-8"))
    hasher.update(hashlib.sha256(str(value).encode("utf-8")).digest())


def _hash_mllm_media_source(hasher: Any, label: str, value: Any) -> None:
    """Hash local media by bytes so equivalent temp paths share a key."""
    try:
        path = Path(value)
        if path.is_file():
            content_hash = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    content_hash.update(chunk)
            hasher.update(label.encode("utf-8"))
            hasher.update(b"local-media-content")
            hasher.update(content_hash.digest())
            return
    except (OSError, TypeError, ValueError):
        pass
    _hash_mllm_media_text(hasher, label, value)


def _hash_mllm_media_array(hasher: Any, label: str, value: Any) -> None:
    if value is None:
        return
    hasher.update(label.encode("utf-8"))
    try:
        hasher.update(str(getattr(value, "shape", "")).encode("utf-8"))
        hasher.update(str(getattr(value, "dtype", "")).encode("utf-8"))
        import numpy as np

        hasher.update(np.array(value).tobytes())
    except Exception:
        _hash_mllm_media_text(hasher, label, value)


def _mllm_media_item_digest(
    request: Any,
    modality: str,
    source: Any,
) -> str:
    """Stable identity for one causal media item in a rendered prompt."""
    hasher = hashlib.sha256()
    hasher.update(b"vmlx-mllm-media-item-v1")
    _hash_mllm_media_source(hasher, modality, source)
    if modality == "image":
        _hash_mllm_media_text(
            hasher,
            "image_token_budget",
            getattr(request, "image_token_budget", None),
        )
    elif modality == "video":
        _hash_mllm_media_text(
            hasher, "video_fps", getattr(request, "video_fps", None)
        )
        _hash_mllm_media_text(
            hasher,
            "video_max_frames",
            getattr(request, "video_max_frames", None),
        )
    return hasher.hexdigest()


def _mllm_media_source_items(request: Any) -> List[Tuple[str, Any]]:
    """Return source-backed media grouped in processor argument order."""
    items: List[Tuple[str, Any]] = []
    items.extend(("image", value) for value in getattr(request, "images", None) or [])
    items.extend(("video", value) for value in getattr(request, "videos", None) or [])
    audio = getattr(request, "audio", None) or getattr(request, "audios", None)
    if audio is not None and not isinstance(audio, (list, tuple)):
        audio = [audio]
    items.extend(("audio", value) for value in audio or [])
    return items


_MLLM_MEDIA_PREFIX_CACHE_DEFAULT_FAMILIES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_vl",
        "qwen4_exp",
        "muse_glimmer",
        "gemma4",
        "gemma4_unified",
        "step3p7",
        "dots3_note",
    }
)


def _mllm_media_prefix_cache_family_enabled(model_type: Any) -> bool:
    """One policy source for generator lookup and scheduler publication.

    A duplicated allowlist made it possible for one half of the media SSD path
    to admit a family while the other half silently skipped it. Default-on is
    restricted to families with a source-owned media-conditioned cache path;
    every other family retains the historical explicit double opt-in.
    """
    enabled = os.environ.get("VMLINUX_MLLM_MEDIA_PREFIX_CACHE", "").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return False
    if str(model_type or "").lower() in _MLLM_MEDIA_PREFIX_CACHE_DEFAULT_FAMILIES:
        return True
    if enabled not in ("1", "true", "yes", "on"):
        return False
    unsafe_ack = os.environ.get(
        "VMLINUX_MLLM_MEDIA_PREFIX_CACHE_UNSAFE_ACK", ""
    ).strip().lower()
    return unsafe_ack in ("1", "true", "yes", "on")


def _clear_mllm_request_media_payloads(request: Any) -> None:
    """Drop every processor payload once restored KV covers all media tokens."""
    for attr in (
        "pixel_values",
        "image_grid_thw",
        "video_pixel_values",
        "video_grid_thw",
        "audio_codes",
        "audio_embeds",
        "audio_features",
        "audio_features_mask",
        "audio_chunk_meta",
    ):
        try:
            setattr(request, attr, None)
        except Exception:
            pass
    try:
        request.audio_features_are_raw_input_features = False
    except Exception:
        pass


def _mllm_media_cache_extra_keys(request: Any) -> Optional[Dict[str, str]]:
    """Return a stable media fingerprint for paged VLM prefix-cache keys.

    Token ids are not enough for media prompts: two different images with the
    same grid shape render the same placeholder tokens. The model state after
    prefill, however, depends on pixel values and media grids. This side key is
    mixed into paged/block hashes while token counts stay based on real model
    tokens only.
    """
    if request is None:
        return None
    has_media = bool(
        getattr(request, "images", None)
        or getattr(request, "videos", None)
        or getattr(request, "audio_codes", None) is not None
        or getattr(request, "audio_embeds", None) is not None
        or getattr(request, "audio_features", None) is not None
        or getattr(request, "audio_features_mask", None) is not None
        or getattr(request, "audio", None)
        or getattr(request, "audios", None)
        or getattr(request, "pixel_values", None) is not None
        or getattr(request, "image_grid_thw", None) is not None
        or getattr(request, "pixel_values_videos", None) is not None
        or getattr(request, "video_pixel_values", None) is not None
        or getattr(request, "video_grid_thw", None) is not None
    )
    if not has_media:
        return None

    hasher = hashlib.sha256()
    hasher.update(b"vmlx-mllm-media-cache-v1")
    audio_sources = getattr(request, "audio", None) or getattr(
        request, "audios", None
    )
    if audio_sources is not None and not isinstance(audio_sources, (list, tuple)):
        audio_sources = [audio_sources]
    media_sources = (
        list(getattr(request, "images", None) or [])
        + list(getattr(request, "videos", None) or [])
        + list(audio_sources or [])
    )

    for source in getattr(request, "images", None) or []:
        _hash_mllm_media_source(hasher, "image", source)
    if getattr(request, "images", None):
        _hash_mllm_media_text(
            hasher,
            "image_token_budget",
            getattr(request, "image_token_budget", None),
        )
    for source in getattr(request, "videos", None) or []:
        _hash_mllm_media_source(hasher, "video", source)
    if getattr(request, "videos", None):
        _hash_mllm_media_text(
            hasher, "video_fps", getattr(request, "video_fps", None)
        )
        _hash_mllm_media_text(
            hasher,
            "video_max_frames",
            getattr(request, "video_max_frames", None),
        )
    for source in audio_sources or []:
        _hash_mllm_media_source(hasher, "audio", source)
    _hash_mllm_media_array(
        hasher, "image_grid_thw", getattr(request, "image_grid_thw", None)
    )
    _hash_mllm_media_array(
        hasher, "video_grid_thw", getattr(request, "video_grid_thw", None)
    )
    _hash_mllm_media_array(
        hasher, "audio_codes", getattr(request, "audio_codes", None)
    )
    _hash_mllm_media_array(
        hasher, "audio_embeds", getattr(request, "audio_embeds", None)
    )
    _hash_mllm_media_array(
        hasher, "audio_features", getattr(request, "audio_features", None)
    )
    _hash_mllm_media_array(
        hasher,
        "audio_features_mask",
        getattr(request, "audio_features_mask", None),
    )
    if not media_sources:
        # Fallback for callers that hand us preprocessed pixel tensors without
        # source URLs/paths. Do not hash attention_mask: it changes with text
        # history length and would make the same image miss across turns.
        _hash_mllm_media_array(
            hasher, "pixel_values", getattr(request, "pixel_values", None)
        )
        _hash_mllm_media_array(
            hasher,
            "pixel_values_videos",
            getattr(request, "pixel_values_videos", None)
            if getattr(request, "pixel_values_videos", None) is not None
            else getattr(request, "video_pixel_values", None),
        )

    return {"mllm_media": hasher.hexdigest()}


def _merge_mllm_cache_extra_keys(
    base: Optional[Any],
    addition: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge cache side-key dictionaries without dropping request-owned axes."""
    if not base and not addition:
        return None
    merged: Dict[str, Any] = {}
    if base:
        if isinstance(base, dict):
            merged.update({str(k): v for k, v in base.items()})
        else:
            merged["request"] = repr(base)
    if addition:
        for key, value in addition.items():
            key = str(key)
            if (
                key == CACHE_EXTRA_SCOPES_KEY
                and isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged or None


def _ssm_companion_cache_extra_keys(request: Any) -> Optional[Any]:
    """Return every request-owned discriminator used by companion state.

    Generation-prompt identity is present on text and media requests.  It must
    not be dropped on text-only stores: a later media request can share the
    same pre-placeholder token/KV prefix, and its scoped media key resolves
    away at that boundary, leaving the generation-prompt discriminator as the
    common key.  Media safety is decided by the caller before this helper is
    used; this function only keeps SSM keying aligned with paged/block keying.
    """
    return getattr(request, "_cache_extra_keys", None)


def _uses_ssm_companion_cache(
    kv_positions: Optional[List[int]],
    num_layers: Optional[int],
    *,
    mixed_attention: bool,
) -> bool:
    """Return whether a mixed cache layout needs a separate state companion.

    Gemma/Step mixed-SWA layouts contain only KV-like cache objects
    (``KVCache`` + ``RotatingKVCache``). Their paged blocks already preserve
    rotating-window metadata and must be reconstructed directly. Treating that
    shape as SSM hybrid makes a valid mixed-SWA block hit wait for a second,
    unrelated companion entry and downgrade to full prefill.
    """
    return bool(
        not mixed_attention
        and kv_positions is not None
        and num_layers is not None
        and len(kv_positions) < num_layers
    )


@dataclass(frozen=True)
class VLMImagePrefillBudgetDecision:
    over_budget: bool
    predicted_attention_bytes: int
    active_memory_bytes: int
    max_working_set_bytes: int
    detail: str


@dataclass(frozen=True)
class MiMoTightMemoryTextPrefillDecision:
    # Named for what it does: this one genuinely refuses (PromptTooLongError).
    # It is narrowly scoped to mimo_v2 under an active tight-memory drain and
    # documents a Metal abort that kills the process before Python can catch
    # it. The VLM image budget next door is the opposite: an estimate that
    # only advises.
    should_reject: bool
    prompt_tokens: int
    generation_tokens: int
    max_prompt_tokens: int
    max_total_tokens: int
    active_memory_bytes: int
    max_working_set_bytes: int
    free_memory_bytes: int
    detail: str


def _mimo_tight_memory_text_prefill_budget(
    *,
    model_type: str,
    has_media_payload: bool,
    tight_memory_prefill_drain: bool,
    seq_len: int,
    generation_tokens: int,
    active_memory_bytes: int,
    max_working_set_bytes: int,
    reject_tokens: int,
    max_total_tokens: int,
    min_free_bytes: int,
    guard_enabled: bool,
) -> MiMoTightMemoryTextPrefillDecision:
    """Reject MiMo text-only shapes that hard-abort Metal under no headroom.

    This guard is not a fake success path: the request still fails. Its purpose
    is to keep the server process alive for users and for the release harness
    when a giant MiMo JANG_2L load leaves too little Metal working-set headroom
    for longer text-prefill/tool prompts.
    """
    tokens = max(0, int(seq_len or 0))
    gen_tokens = max(0, int(generation_tokens or 0))
    total_tokens = tokens + gen_tokens
    max_tokens = max(1, int(reject_tokens or 1))
    max_total = max(1, int(max_total_tokens or max_tokens))
    active = max(0, int(active_memory_bytes or 0))
    max_ws = max(0, int(max_working_set_bytes or 0))
    min_free = max(0, int(min_free_bytes or 0))
    free = max(0, max_ws - active) if max_ws > 0 else 0

    def gb(value: int) -> float:
        return value / (1024**3)

    base = dict(
        prompt_tokens=tokens,
        generation_tokens=gen_tokens,
        max_prompt_tokens=max_tokens,
        max_total_tokens=max_total,
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        free_memory_bytes=free,
    )
    if (
        not guard_enabled
        or model_type != "mimo_v2"
        or has_media_payload
        or not tight_memory_prefill_drain
        or max_ws <= 0
        or free >= min_free
        or (tokens <= max_tokens and total_tokens <= max_total)
    ):
        return MiMoTightMemoryTextPrefillDecision(
            should_reject=False,
            detail="MiMo-V2 tight-memory text prefill budget ok or not applicable",
            **base,
        )

    limiting_tokens = max_total if total_tokens > max_total else max_tokens
    return MiMoTightMemoryTextPrefillDecision(
        should_reject=True,
        detail=(
            "MiMo-V2 tight-memory text prefill rejected before Metal forward: "
            f"prompt has {tokens} tokens and generation budget is {gen_tokens} "
            f"tokens while Metal headroom is {gb(free):.1f}GB, below guard "
            f"{gb(min_free):.1f}GB. This request shape can hard-abort "
            "Metal before Python can recover; reduce prompt length, close other "
            "sessions, use a smaller MiMo quant, or set "
            "VMLINUX_MIMO_TEXT_PREFILL_GUARD=0 at OOM risk."
        ),
        max_prompt_tokens=limiting_tokens,
        prompt_tokens=total_tokens if total_tokens > max_total else tokens,
        generation_tokens=gen_tokens,
        max_total_tokens=max_total,
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        free_memory_bytes=free,
    )


def _raise_if_mimo_tight_memory_text_prefill_exceeds_budget(
    *,
    model_type: str,
    has_media_payload: bool,
    tight_memory_prefill_drain: bool,
    seq_len: int,
    generation_tokens: int,
    request_id: str,
) -> None:
    if os.environ.get("VMLINUX_MIMO_TEXT_PREFILL_GUARD", "1") == "0":
        guard_enabled = False
    else:
        guard_enabled = True
    try:
        reject_tokens = int(os.environ.get("VMLINUX_MIMO_TEXT_PREFILL_REJECT_TOKENS", "256"))
    except (TypeError, ValueError):
        reject_tokens = 256
    reject_tokens = max(16, reject_tokens)
    try:
        max_total_tokens = int(
                os.environ.get("VMLINUX_MIMO_TEXT_PREFILL_TOTAL_TOKENS", "384")
        )
    except (TypeError, ValueError):
        max_total_tokens = 192
    max_total_tokens = max(16, max_total_tokens)
    min_free_gb = _parse_positive_float_env("VMLINUX_MIMO_TEXT_PREFILL_MIN_FREE_GB")
    if min_free_gb is None:
        min_free_gb = 2.0
    try:
        active, max_ws = get_effective_metal_working_set_bytes(mx)
    except Exception:
        active = 0
        max_ws = 0
    decision = _mimo_tight_memory_text_prefill_budget(
        model_type=model_type,
        has_media_payload=has_media_payload,
        tight_memory_prefill_drain=tight_memory_prefill_drain,
        seq_len=seq_len,
        generation_tokens=generation_tokens,
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        reject_tokens=reject_tokens,
        max_total_tokens=max_total_tokens,
        min_free_bytes=int(min_free_gb * 1024**3),
        guard_enabled=guard_enabled,
    )
    if decision.should_reject:
        logger.warning(decision.detail)
        raise PromptTooLongError(
            decision.prompt_tokens,
            decision.max_prompt_tokens,
            source="mimo_v2_tight_memory_text_prefill",
            request_id=request_id,
        )


def _vlm_image_request_cache_limit_bytes(
    *,
    active_memory_bytes: int,
    max_working_set_bytes: int,
    max_limit_bytes: int,
    free_fraction: float,
    floor_bytes: int,
) -> int:
    """Return a conservative MLX allocator cache limit for media requests.

    The startup cache limit is intentionally generous for text decode, but
    high-res VLM requests need transient Metal headroom for image preprocessing,
    vision encoding, one-shot language prefill, and the first decode KV writes.
    Keep reusable allocator memory small on media requests so cached/free-list
    blocks do not compete with those transient tensors.
    """
    active = max(0, int(active_memory_bytes or 0))
    max_ws = max(0, int(max_working_set_bytes or 0))
    max_limit = max(0, int(max_limit_bytes or 0))
    floor = max(0, int(floor_bytes or 0))
    fraction = max(0.0, float(free_fraction or 0.0))

    if max_limit <= 0:
        return 0
    if max_ws <= 0:
        return max(floor, max_limit)

    free = max(0, max_ws - active)
    fractional = int(free * fraction)
    if fractional <= 0:
        return floor
    return max(floor, min(max_limit, fractional))


def _apply_vlm_image_request_cache_limit() -> bool:
    """Tighten the Metal reusable cache before VLM media work.

    This is a preflight memory-safety control, not a model behavior change. It
    leaves text-only requests on the normal scheduler cache policy and only
    shrinks MLX's allocator free-list ceiling for media requests that otherwise
    have large transient tensors.

    Returns True when a tightened limit was actually applied, so the caller
    can restore the steady-state scheduler limit once the media prefill spike
    is over instead of leaving a ~1GB allocator ceiling on the whole session.
    (A/B measured this restore as hygiene, not a decode-speed lever: decode
    throughput was unchanged with the limit still tightened.)
    """
    if os.environ.get("VMLX_VLM_IMAGE_CACHE_LIMIT", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False
    try:
        max_limit = int(
            float(os.environ.get("VMLX_VLM_IMAGE_CACHE_LIMIT_GB", "1.0"))
            * 1024**3
        )
    except (TypeError, ValueError):
        max_limit = 1024**3
    try:
        free_fraction = float(
            os.environ.get("VMLX_VLM_IMAGE_CACHE_LIMIT_FREE_FRACTION", "0.10")
        )
    except (TypeError, ValueError):
        free_fraction = 0.10
    try:
        floor = int(
            float(os.environ.get("VMLX_VLM_IMAGE_CACHE_LIMIT_FLOOR_GB", "0.25"))
            * 1024**3
        )
    except (TypeError, ValueError):
        floor = 256 * 1024**2

    if max_limit <= 0:
        return False
    try:
        active, max_ws = get_effective_metal_working_set_bytes(mx)
        limit = _vlm_image_request_cache_limit_bytes(
            active_memory_bytes=active,
            max_working_set_bytes=max_ws,
            max_limit_bytes=max_limit,
            free_fraction=free_fraction,
            floor_bytes=floor,
        )
        if limit <= 0:
            return False
        set_cache = getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit
        set_cache(limit)
        logger.info(
            "VLM image request Metal cache limit tightened to %.2fGB "
            "(active=%.1fGB, max_ws=%.1fGB, free_fraction=%.2f)",
            limit / (1024**3),
            active / (1024**3),
            max_ws / (1024**3) if max_ws else 0.0,
            free_fraction,
        )
        return True
    except Exception as exc:
        logger.debug("VLM image request cache limit not applied: %s", exc)
        return False


def _max_tokens_under_attention_bytes(byte_budget: int, heads: int) -> int:
    """Largest media-expanded prompt whose heads x tokens^2 x 2B fits.

    The guard's cost model is quadratic in prompt length, so "shorten the
    prompt" is unactionable without this inverse (vmlx#256).
    """
    import math

    if byte_budget <= 0 or heads <= 0:
        return 0
    return int(math.isqrt(int(byte_budget) // (2 * int(heads))))


def _vlm_image_prefill_budget(
    *,
    has_images: bool,
    seq_len: int,
    num_attention_heads: int,
    active_memory_bytes: int,
    max_working_set_bytes: int,
    reject_pct: float,
    single_buffer_limit_bytes: int,
    guard_enabled: bool,
    image_token_count: Optional[int] = None,
) -> VLMImagePrefillBudgetDecision:
    """Budget a one-shot image prefill before it reaches Metal.

    Image/video prompts cannot use the text chunking path because the vision
    wrapper needs the full media-expanded prompt. The expensive tensor is the
    language attention score buffer, roughly heads * seq_len^2 * bf16 bytes.
    Rejecting here turns a process-killing Metal command-buffer OOM into a
    normal request error the scheduler can report to the client.
    """
    heads = max(1, int(num_attention_heads or 1))
    tokens = max(0, int(seq_len or 0))
    predicted = int(heads * tokens * tokens * 2)
    active = max(0, int(active_memory_bytes or 0))
    max_ws = max(0, int(max_working_set_bytes or 0))
    reject_pct = float(reject_pct or 98.0)
    single_limit = max(0, int(single_buffer_limit_bytes or 0))

    def gb(value: int) -> float:
        return value / (1024**3)

    if not guard_enabled or not has_images:
        return VLMImagePrefillBudgetDecision(
            over_budget=False,
            predicted_attention_bytes=predicted,
            active_memory_bytes=active,
            max_working_set_bytes=max_ws,
            detail="VLM image prefill guard bypassed for text-only or disabled path",
        )

    exceeds_single_buffer = single_limit > 0 and predicted > single_limit
    exceeds_working_set = False
    projected_pct = 0.0
    if max_ws > 0:
        projected_pct = ((active + predicted) / max_ws) * 100.0
        exceeds_working_set = projected_pct >= reject_pct

    if not (exceeds_single_buffer or exceeds_working_set):
        return VLMImagePrefillBudgetDecision(
            over_budget=False,
            predicted_attention_bytes=predicted,
            active_memory_bytes=active,
            max_working_set_bytes=max_ws,
            detail=(
                "VLM image prefill budget ok: "
                f"seq_len={tokens}, heads={heads}, "
                f"predicted_attention={gb(predicted):.1f}GB"
            ),
        )

    reasons: List[str] = []
    if exceeds_single_buffer:
        reasons.append(
            f"predicted attention buffer {gb(predicted):.1f}GB "
            f"(= {heads} heads x ({tokens:,} tokens)^2 x 2B) exceeds "
            f"single-buffer guard {gb(single_limit):.1f}GB"
        )
    if exceeds_working_set:
        reasons.append(
            f"projected Metal working set {projected_pct:.0f}% "
            f"({gb(active):.1f}GB already resident + {gb(predicted):.1f}GB "
            f"predicted of {gb(max_ws):.1f}GB) exceeds "
            f"threshold {reject_pct:.0f}%"
        )

    # The buffer grows with the SQUARE of the media-expanded prompt, so the
    # actionable number is how many total tokens actually fit — without it a
    # user can only guess-and-retry (vmlx#256).
    fitting_budgets = []
    if exceeds_single_buffer:
        fitting_budgets.append(_max_tokens_under_attention_bytes(single_limit, heads))
    if exceeds_working_set:
        headroom = int((reject_pct / 100.0) * max_ws) - active
        fitting_budgets.append(_max_tokens_under_attention_bytes(headroom, heads))
    fits_tokens = min([b for b in fitting_budgets if b >= 0], default=0)

    budget_line = (
        f" This device fits about {fits_tokens:,} media-expanded tokens under "
        f"the binding limit; this prompt is {tokens:,}"
    )
    if image_token_count is not None and image_token_count >= 0:
        text_tokens = max(0, tokens - int(image_token_count))
        budget_line += (
            f" ({int(image_token_count):,} image + {text_tokens:,} text)"
        )
        if fits_tokens > int(image_token_count):
            budget_line += (
                f" — keeping these images, the text must fit in "
                f"{fits_tokens - int(image_token_count):,} tokens"
            )
        else:
            budget_line += (
                " — the images alone exceed the budget, so a smaller image "
                "or fewer images is required"
            )
    else:
        budget_line += f", i.e. {max(0, tokens - fits_tokens):,} too many"
    budget_line += "."

    return VLMImagePrefillBudgetDecision(
        over_budget=True,
        predicted_attention_bytes=predicted,
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        detail=(
            "VLM image prefill is above the estimated attention budget: "
            + "; ".join(reasons)
            + "."
            + budget_line
            + " This is an ESTIMATE (heads x seq^2 x 2B assumes a fully "
            "materialised score matrix, which a fused attention kernel does "
            "not produce), so it is reported and not enforced. If the "
            "allocation genuinely does not fit, Metal will say so."
        ),
    )


def _parse_positive_float_env(name: str) -> Optional[float]:
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _resolve_vlm_image_prefill_single_buffer_limit(
    max_working_set_bytes: int,
) -> int:
    """Resolve the one-shot VLM image attention-buffer guard.

    The old fixed 8GB default was appropriate for smaller Apple Silicon
    machines, but it incorrectly rejected Gemma 4 12B image prompts on high-RAM
    M-series systems even when the Metal working-set budget had enough headroom.
    Keep explicit env overrides exact, otherwise scale the default guard with
    the effective Metal working set while preserving an 8GB floor.
    """
    gib = 1024**3
    explicit_gb = _parse_positive_float_env("VMLX_VLM_IMAGE_PREFILL_BUFFER_GB")
    if explicit_gb is not None:
        return int(explicit_gb * gib)

    floor = 8 * gib
    max_ws = max(0, int(max_working_set_bytes or 0))
    if max_ws <= 0:
        return floor

    fraction = _parse_positive_float_env("VMLX_VLM_IMAGE_PREFILL_BUFFER_FRACTION")
    if fraction is None:
        fraction = 0.16
    fraction = min(max(fraction, 0.01), 0.50)

    cap_gb = _parse_positive_float_env("VMLX_VLM_IMAGE_PREFILL_BUFFER_MAX_GB")
    cap = int((cap_gb if cap_gb is not None else 24.0) * gib)

    return min(cap, max(floor, int(max_ws * fraction)))


# Media prefill chunking. The floor is high on purpose -- see
# _media_prefill_chunk_tokens: chunking trades peak memory for repeated weight
# streaming, so the aim is the largest chunk that stays off the cliff.
_MEDIA_PREFILL_CHUNK_FLOOR = 4096
# Below this the one-shot forward wins outright: it was never the shape that
# ran out of memory, and it reads the weights once.
_MEDIA_PREFILL_CHUNK_MIN_SEQ = 8192


def _named_params(fn) -> set:
    """Named parameters of a callable, EXCLUDING anything absorbed by **kwargs.

    `**kwargs` makes every introspection question answer "yes". Several
    wrappers in this tree accept `position_ids` only in the sense that it
    disappears into `**kwargs` and is never read, so a capability probe that
    counts `**kwargs` as support will happily build a chunked prefill on a
    model that silently ignores the positions and returns confident garbage.
    Only NAMED parameters count as support.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def _media_embed_kwarg_name(language_model) -> Optional[str]:
    """Return the embeddings kwarg this language model actually names.

    Two spellings exist in this tree: `inputs_embeds` almost everywhere,
    `input_embeddings` on minimax (zaya accepts both). Returns None when the
    model names neither, which is the signal to keep the one-shot path.
    """
    if language_model is None:
        return None
    call = getattr(language_model, "__call__", None)
    if call is None:
        return None
    names = _named_params(call)
    for candidate in ("inputs_embeds", "input_embeddings"):
        if candidate in names:
            return candidate
    return None


def _video_pixel_values_kwarg_name(model: Any) -> str:
    """Return the video tensor kwarg the loaded wrapper actually consumes.

    Processor output is standardized as ``pixel_values_videos``, but older
    vMLX wrappers name the forwarded model argument ``video_pixel_values``.
    Do not count ``**kwargs`` as support: Gemma 4 accepts the legacy spelling
    there and silently ignores it in ``get_input_embeddings``.
    """
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        names = _named_params(get_input_embeddings)
        if "pixel_values_videos" in names:
            return "pixel_values_videos"
        if "video_pixel_values" in names:
            return "video_pixel_values"

    call = getattr(model, "__call__", None)
    if callable(call):
        names = _named_params(call)
        if "pixel_values_videos" in names:
            return "pixel_values_videos"
        if "video_pixel_values" in names:
            return "video_pixel_values"

    # Preserve the established bridge spelling for wrappers whose contract is
    # opaque or intentionally resolves the alias from **kwargs.
    return "video_pixel_values"


def _media_chunk_boundaries(
    seq_len: int,
    chunk: int,
    media_runs: Optional[List[Tuple[int, int]]] = None,
) -> List[int]:
    """Chunk end positions, snapped away from the middle of a media run.

    Once vision embeddings are merged into the embedding sequence, splitting
    inside a run of media placeholders is harmless for every family whose
    masks are built from the cache offset -- which is all of them here today.
    It stops being harmless the moment a family builds a mask from
    whole-sequence image geometry (gemma4's config already declares
    `use_bidirectional_attention: "vision"` even though the MLX language
    model does not implement it yet, and qwen3_vl's deepstack injection is
    keyed to visual rows in the current window).

    Snapping costs nothing -- boundaries are approximate already -- and means
    this does not silently break the day one of those masks lands.
    """
    if chunk <= 0 or seq_len <= 0:
        return [seq_len]
    runs = list(media_runs or [])
    bounds: List[int] = []
    pos = 0
    while pos < seq_len:
        end = min(pos + chunk, seq_len)
        if end < seq_len:
            for run_start, run_end in runs:
                if run_start < end < run_end:
                    # Prefer the run's start; fall back to its end if that
                    # would make no forward progress.
                    end = run_start if run_start > pos else run_end
                    end = min(max(end, pos + 1), seq_len)
                    break
        bounds.append(end)
        pos = end
    return bounds


def _media_placeholder_runs(
    token_ids: Optional[List[int]], media_ids: set
) -> List[Tuple[int, int]]:
    """Half-open [start, end) spans of consecutive media placeholder tokens."""
    if not token_ids or not media_ids:
        return []
    runs: List[Tuple[int, int]] = []
    start = None
    for index, token in enumerate(token_ids):
        if token in media_ids:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(token_ids)))
    return runs


def _step3p7_media_item_runs(
    request: Any,
    source_items: List[Tuple[str, Any]],
    runs: List[Tuple[int, int]],
    *,
    model_type: Optional[str],
) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
    """Collapse Step3.7 crop/full-image runs into one span per source image.

    Step's processor expands one source image to ``num_patches`` adaptive-crop
    ``<im_patch>`` runs plus one final full-image run. Structural
    ``<patch_start>/<patch_end>`` and ``<im_start>/<im_end>`` tokens separate
    those runs, so counting consecutive image-token runs does not count media
    items. The processor preserves the exact per-source ``num_patches`` list in
    ``request.extra_kwargs``; consume it only when it reconciles every source
    and every observed run. Any disagreement returns ``None`` so the caller
    retains its aggregate fail-closed behavior.
    """
    if str(model_type or "").lower() != "step3p7":
        return None
    if not source_items or any(modality != "image" for modality, _ in source_items):
        return None
    extra_kwargs = getattr(request, "extra_kwargs", None)
    if not isinstance(extra_kwargs, dict):
        return None
    num_patches = extra_kwargs.get("num_patches")
    if hasattr(num_patches, "tolist"):
        try:
            num_patches = num_patches.tolist()
        except Exception:
            return None
    if not isinstance(num_patches, (list, tuple)):
        return None
    if len(num_patches) != len(source_items):
        return None

    run_group_sizes: List[int] = []
    for value in num_patches:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        run_group_sizes.append(value + 1)
    if sum(run_group_sizes) != len(runs):
        return None

    grouped: List[Tuple[int, int]] = []
    cursor = 0
    for group_size in run_group_sizes:
        owned_runs = runs[cursor : cursor + group_size]
        if len(owned_runs) != group_size:
            return None
        grouped.append((owned_runs[0][0], owned_runs[-1][1]))
        cursor += group_size
    if cursor != len(runs):
        return None
    return grouped, run_group_sizes


def _mllm_grid_rows(value: Any) -> Optional[List[Tuple[int, int, int]]]:
    """Normalize a processor grid tensor/list without guessing malformed rows."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    if value and not isinstance(value[0], (list, tuple)):
        value = [value]
    rows: List[Tuple[int, int, int]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        try:
            row = tuple(int(component) for component in raw)
        except (TypeError, ValueError):
            return None
        if any(component <= 0 for component in row):
            return None
        rows.append(row)
    return rows


def _muse_glimmer_media_item_runs(
    request: Any,
    source_items: List[Tuple[str, Any]],
    token_ids: List[int],
    runs: List[Tuple[int, int]],
    grouped_ids: Dict[str, set[int]],
    *,
    model_type: Optional[str],
) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
    """Collapse Muse's timestamp-separated video-frame runs per source clip.

    The native processor emits one video placeholder run per temporal group,
    separated by timestamp text and ``<|vid_frame_separator|>``. Cache side
    keys, however, own one source video. Group exactly ``grid_t`` runs for each
    clip so a later video salts only its first causal position and leaves an
    unchanged earlier image chain reusable.
    """
    if str(model_type or "").lower() != "muse_glimmer":
        return None
    expected = {
        modality: sum(
            1
            for item_modality, _source in source_items
            if item_modality == modality
        )
        for modality in ("image", "video")
    }
    if expected["image"] + expected["video"] != len(source_items):
        return None
    video_grids = _mllm_grid_rows(getattr(request, "video_grid_thw", None))
    if video_grids is None or len(video_grids) != expected["video"]:
        return None
    video_group_sizes: Deque[int] = deque(grid[0] for grid in video_grids)

    grouped: List[Tuple[int, int]] = []
    group_sizes: List[int] = []
    seen = {"image": 0, "video": 0}
    cursor = 0
    while cursor < len(runs):
        run_start, _run_end = runs[cursor]
        token_id = int(token_ids[run_start])
        modalities = [
            modality
            for modality in ("image", "video")
            if token_id in grouped_ids.get(modality, set())
        ]
        if len(modalities) != 1:
            return None
        modality = modalities[0]
        if modality == "image":
            size = 1
        else:
            if not video_group_sizes:
                return None
            size = int(video_group_sizes.popleft())
        if size <= 0 or cursor + size > len(runs):
            return None
        owned = runs[cursor : cursor + size]
        modality_ids = grouped_ids.get(modality, set())
        if any(int(token_ids[start]) not in modality_ids for start, _end in owned):
            return None
        grouped.append((owned[0][0], owned[-1][1]))
        group_sizes.append(size)
        seen[modality] += 1
        cursor += size

    if video_group_sizes or seen != expected or len(grouped) != len(source_items):
        return None
    return grouped, group_sizes


def _dots3_media_item_runs(
    request: Any,
    token_ids: List[int],
    grouped_ids: Dict[str, set[int]],
    *,
    model_type: Optional[str],
) -> Optional[
    Tuple[List[Tuple[int, int]], List[int], List[Tuple[str, Any]]]
]:
    """Use processor-owned item intervals for prompt-ordered dots media.

    Image and video retain distinct native token IDs, while their feature rows
    share one prompt-ordered buffer. The processor carries fail-closed item
    metadata with exact expanded intervals and source indices; validate it
    against both the processed tokens and the original request payloads before
    using it.
    """
    if str(model_type or "").lower() != "dots3_note":
        return None
    extra_kwargs = getattr(request, "extra_kwargs", None)
    if not isinstance(extra_kwargs, dict):
        return None
    raw_items = extra_kwargs.get("_vmlx_dots3_media_items")
    if not isinstance(raw_items, (list, tuple)) or not raw_items:
        return None

    sources = {
        "image": list(getattr(request, "images", None) or []),
        "video": list(getattr(request, "videos", None) or []),
        "audio": list(
            getattr(request, "audio", None)
            or getattr(request, "audios", None)
            or []
        ),
    }
    assignments: List[Tuple[str, Any]] = []
    ranges: List[Tuple[int, int]] = []
    seen_indices = {modality: set() for modality in sources}
    prior_end = 0
    for raw in raw_items:
        if not isinstance(raw, dict):
            return None
        modality = str(raw.get("modality") or "").lower()
        if modality not in sources:
            return None
        try:
            source_index = int(raw["source_index"])
            start = int(raw["token_start"])
            end = int(raw["token_end"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            source_index < 0
            or source_index >= len(sources[modality])
            or source_index in seen_indices[modality]
            or start < prior_end
            or end <= start
            or end > len(token_ids)
        ):
            return None
        expected_ids = grouped_ids.get(modality, set())
        if not expected_ids or any(
            int(token) not in expected_ids for token in token_ids[start:end]
        ):
            return None
        seen_indices[modality].add(source_index)
        assignments.append((modality, sources[modality][source_index]))
        ranges.append((start, end))
        prior_end = end

    if any(
        seen_indices[modality] != set(range(len(values)))
        for modality, values in sources.items()
    ):
        return None
    return ranges, [1] * len(ranges), assignments


def _raise_if_image_prefill_exceeds_budget(
    *,
    has_images: bool,
    has_audio_payload: bool = False,
    seq_len: int,
    language_model: Any,
    image_token_count: Optional[int] = None,
) -> None:
    if os.environ.get("VMLX_VLM_IMAGE_PREFILL_GUARD", "1") == "0":
        guard_enabled = False
    else:
        guard_enabled = True

    try:
        reject_pct = get_metal_ws_guard_threshold(
            default=float(
                os.environ.get(
                    "VMLX_VLM_IMAGE_PREFILL_REJECT_PCT",
                    os.environ.get("VMLX_METAL_WS_REJECT_PCT", "98"),
                )
            )
        )
    except (TypeError, ValueError):
        reject_pct = get_metal_ws_guard_threshold(98.0)

    try:
        active, max_ws = get_effective_metal_working_set_bytes(mx)
    except Exception:
        active = 0
        max_ws = 0
    single_buffer_limit = _resolve_vlm_image_prefill_single_buffer_limit(max_ws)

    decision = _vlm_image_prefill_budget(
        has_images=bool(has_images or has_audio_payload),
        seq_len=seq_len,
        num_attention_heads=_infer_attention_heads_for_hybrid_oom_guard(
            language_model
        ),
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        reject_pct=reject_pct,
        single_buffer_limit_bytes=single_buffer_limit,
        guard_enabled=guard_enabled,
        image_token_count=image_token_count,
    )
    if decision.over_budget:
        # ADVISE, NEVER REFUSE. This used to raise, and the arithmetic behind
        # it is a guess with veto power: `heads * seq^2 * 2B` assumes a full
        # head x seq x seq score matrix is materialised, which a fused
        # attention kernel never does, and the "single-buffer limit" it is
        # compared against is not a device limit at all -- it is 16% of the
        # Metal working set. Measured on an M5 Max: a 33,913-token media
        # prompt was refused against a "17.2GB" ceiling with 89.4GB free, and
        # because a chat client re-sends the image every turn, that killed
        # the conversation permanently from the first turn that crossed it.
        #
        # If a request genuinely does not fit, Metal and the allocator fail
        # loudly on their own. Predicting that badly and blocking on the
        # prediction is strictly worse than attempting the operation.
        logger.warning(
            "%s (PROCEEDING ANYWAY: this is an estimate, not a measurement, "
            "and it may not refuse the request)",
            decision.detail,
        )


def _infer_attention_heads_for_hybrid_oom_guard(
    language_model: Any,
    default: int = 32,
) -> int:
    """Infer text-backbone attention heads for hybrid one-shot OOM estimates.

    MLLM wrappers can hide text model config under ``config.text_config`` or
    an inner ``.model`` / ``.language_model`` object. This helper keeps the
    OOM guard and clean-SSM rederive guard on the same traversal path.
    """
    candidates: List[Any] = []
    seen: set[int] = set()

    def add_candidate(obj: Any) -> None:
        if obj is None:
            return
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(obj)

    add_candidate(language_model)
    index = 0
    while index < len(candidates) and index < 8:
        obj = candidates[index]
        index += 1

        for attr in ("args", "config", "text_config"):
            cfg = _read_config_field(obj, attr)
            if cfg is None:
                continue
            heads = _positive_int_or_none(
                _read_config_field(cfg, "num_attention_heads")
            )
            if heads is not None:
                return heads
            text_cfg = _read_config_field(cfg, "text_config")
            heads = _positive_int_or_none(
                _read_config_field(text_cfg, "num_attention_heads")
            )
            if heads is not None:
                return heads

        add_candidate(_read_config_field(obj, "model"))
        add_candidate(_read_config_field(obj, "language_model"))

    return default


def _batch_shares_sampler_params(requests: List[Any]) -> bool:
    """Return True when requests can share one request-scoped sampler call."""
    if not requests:
        return False
    first_req = requests[0]
    # A seeded sampler owns mutable request-local PRNG state.  Sharing the first
    # request's sampler across a batch would make every other request consume
    # that state instead of its own seed.
    if getattr(first_req, "seed", None) is not None:
        return False
    if getattr(first_req, "repetition_penalty", 1.0) not in (None, 1.0):
        return False
    first_key = (
        getattr(first_req, "temperature", None),
        getattr(first_req, "top_p", None),
        getattr(first_req, "top_k", None),
        getattr(first_req, "min_p", None),
    )
    for req in requests[1:]:
        if getattr(req, "seed", None) is not None:
            return False
        if getattr(req, "repetition_penalty", 1.0) not in (None, 1.0):
            return False
        key = (
            getattr(req, "temperature", None),
            getattr(req, "top_p", None),
            getattr(req, "top_k", None),
            getattr(req, "min_p", None),
        )
        if key != first_key:
            return False
    return True


def _prefix_hit_tail_and_cached_tokens(
    *,
    token_list: List[int],
    remaining: List[int],
    gen_prompt_suffix: List[int],
) -> Tuple[List[int], int]:
    """Thin adapter over the shared N-1 prefix-hit arithmetic."""
    return _shared_prefix_hit(
        key_tokens=token_list,
        remaining=remaining,
        gen_prompt_suffix=gen_prompt_suffix,
    )


def _disk_prefix_hit_tail_and_cached_tokens(
    *,
    token_list: List[int],
    matched_tokens: List[int],
    gen_prompt_suffix: List[int],
) -> Tuple[List[int], int]:
    """Thin adapter over the shared disk L2 partial-hit arithmetic.

    This used to be a private copy that reported ``cached=len(matched)`` and
    dropped ``matched[-1]`` from the tail, on the stated premise that the MLLM
    disk store "stores the clean prompt-boundary cache under the same token key
    length it reports". That premise is false on BOTH MLLM store branches: the
    plain VLM path truncates through ``_truncate_hybrid_cache`` (prompt_len-1)
    and the mixed-SWA / ZAYA clean path re-prefills ``token_list[:prompt_len-1]``
    -- and both then write under the FULL N-token key. So the last matched token
    was never in the payload, was never re-fed, and never got KV: a warm
    disk-prefix turn answered differently from the same turn cold. It also ate
    the boundary-swap sentinel, silently serving the previous turn's thinking
    mode. The text lane fixed this in 12ee1c8ee; this copy did not get the fix,
    which is why there is no longer a copy.
    """
    return _shared_disk_prefix_hit(
        fetch_tokens=token_list,
        matched_tokens=matched_tokens,
        gen_prompt_suffix=gen_prompt_suffix,
    )


def _cache_offset_for_position_ids(cache: Optional[List[Any]], language_model: Any) -> int:
    """Return the first usable attention-cache offset for absolute positions."""
    if not cache:
        return 0

    candidate_indices: List[int] = []
    try:
        fa_idx = getattr(getattr(language_model, "model", None), "fa_idx", None)
        if isinstance(fa_idx, int) and 0 <= fa_idx < len(cache):
            candidate_indices.append(fa_idx)
    except Exception:
        pass
    candidate_indices.extend(i for i in range(len(cache)) if i not in candidate_indices)

    for idx in candidate_indices:
        cache_obj = cache[idx]
        offset = getattr(cache_obj, "offset", None)
        if offset is None:
            continue
        try:
            if isinstance(offset, int):
                return max(0, offset)
            if isinstance(offset, mx.array):
                value = offset if offset.ndim == 0 else offset[0]
                return max(0, int(value.item()))
            return max(0, int(offset))
        except Exception:
            continue
    return 0


def _absolute_text_position_ids(
    input_ids: mx.array,
    cache: Optional[List[Any]],
    language_model: Any,
) -> Optional[mx.array]:
    """Build absolute text-only position_ids for cache-hit prompt tails.

    Qwen3.5/3.6 mRoPE language models keep module-level rope state. Passing
    explicit text positions avoids depending on ``_rope_deltas`` for both
    cold text-only prefill and cache-hit tails, and makes partial-prefix reuse
    match full prefill.
    """
    offset = _cache_offset_for_position_ids(cache, language_model)
    if input_ids.ndim == 1:
        batch_size = 1
        seq_len = input_ids.shape[0]
    else:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
    if seq_len <= 0:
        return None
    pos = mx.arange(offset, offset + seq_len, dtype=mx.int32).reshape(1, seq_len)
    pos = mx.broadcast_to(pos, (batch_size, seq_len))
    return mx.broadcast_to(pos[None, ...], (3, batch_size, seq_len))


def _seed_text_rope_delta_for_decode(language_model: Any, input_ids: mx.array) -> None:
    """Seed Qwen-style rope delta state after explicit cache-tail positions.

    When a cache-hit tail is prefed with explicit absolute ``position_ids``,
    mlx-vlm's Qwen language model does not compute/update ``_rope_deltas``.
    The next decode token would then recompute positions from zero. Text-only
    prompts have zero mRoPE delta, so seed that state explicitly and clear the
    stale cached ``_position_ids`` slice.
    """
    if language_model is None or not hasattr(language_model, "_rope_deltas"):
        return
    batch_size = input_ids.shape[0] if getattr(input_ids, "ndim", 0) > 1 else 1
    dtype = getattr(input_ids, "dtype", mx.int32)
    try:
        language_model._rope_deltas = mx.zeros((batch_size, 1), dtype=dtype)
        if hasattr(language_model, "_position_ids"):
            language_model._position_ids = None
    except Exception:
        return


# Dedicated GPU stream for prefill + sample + materialize, matching the
# pattern used by mlx_lm.generate_step, mlx_vlm.generate.generate_step,
# and the reference jang_tools generate_vl helper.
#
# WHY: native TurboQuant Metal kernels (P3/P15/P17/P18 in jang_tools)
# dispatch async work onto an internal stream. Subsequent ops (the
# materialize routine, async_eval, sampled.item()) running on the
# scheduler thread without a stream context raise:
#   RuntimeError: There is no Stream(gpu, 1) in current thread.
# Pinning all prefill+sample+materialize work to a single dedicated
# stream means every op shares the same Stream handle, eliminating the
# cross-thread/cross-stream resolution failure.
#
# Lazy-created on first use so the device handle binds at runtime.
_GENERATION_STREAM: Optional[Any] = None
_GENERATION_STREAM_OWNER: int | None = None


def _gen_stream() -> Any:
    """Lazily create the dedicated GPU stream for prefill/decode work."""
    global _GENERATION_STREAM, _GENERATION_STREAM_OWNER
    owner = threading.get_ident()
    if _GENERATION_STREAM is None or _GENERATION_STREAM_OWNER != owner:
        try:
            _GENERATION_STREAM = mx.new_stream(mx.default_device())
            _GENERATION_STREAM_OWNER = owner
        except Exception:
            try:
                _GENERATION_STREAM = mx.default_stream(mx.default_device())
                _GENERATION_STREAM_OWNER = owner
            except Exception:
                _GENERATION_STREAM = None
                _GENERATION_STREAM_OWNER = owner
    # P0 VL/audio stream bug: mlx_vlm.generate.generation_stream is a module
    # global bound to the IMPORT thread; mlx_vlm internals (wired_limit /
    # generate) call mx.synchronize(generation_stream), which raises "There is
    # no Stream(gpu, 0) in current thread" when MLLM generation (image OR audio)
    # runs on the mllm-worker executor. Rebind it to OUR worker-thread stream so
    # those syncs resolve. Extends the simple-MLLM FIX#7 to MLLMBatchGenerator /
    # the /v1/responses + dev-build-UI VL+audio path. Idempotent + cheap.
    if _GENERATION_STREAM is not None:
        try:
            # ``mlx_vlm`` exports a top-level ``generate`` function.  The
            # dotted import form can therefore bind that function rather than
            # the submodule, leaving the real module-global stream unchanged.
            _mvg = importlib.import_module("mlx_vlm.generate")
            if getattr(_mvg, "generation_stream", None) is not _GENERATION_STREAM:
                _mvg.generation_stream = _GENERATION_STREAM
        except Exception:
            pass
    return _GENERATION_STREAM


def reset_generation_streams() -> None:
    """Drop thread-local MLX stream handles after MLLM engine teardown.

    MLX stream objects are bound to the worker thread that created them. Deep
    sleep and model switch tear down the scheduler/executor and create a new
    worker on wake. Keeping these globals across that lifecycle can make the
    next generation try to use a stream from the old thread.
    """
    global _GENERATION_STREAM, _GENERATION_STREAM_OWNER
    _GENERATION_STREAM = None
    _GENERATION_STREAM_OWNER = None
    cls = globals().get("MLLMBatchGenerator")
    if cls is not None:
        try:
            cls._stream = None
        except Exception:
            pass


class _MaybeStream:
    """Context manager that wraps mx.stream(stream) only if stream exists."""
    __slots__ = ("_cm",)

    def __init__(self):
        s = _gen_stream()
        self._cm = mx.stream(s) if s is not None else None

    def __enter__(self):
        if self._cm is not None:
            self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        if self._cm is not None:
            return self._cm.__exit__(*exc)
        return False


def _as_input_mapping(inputs: Any) -> Dict[str, Any]:
    """Normalize processor outputs to a plain mapping.

    HuggingFace processors may return BatchFeature/BatchEncoding objects,
    dataclass-like outputs, or a regular dict. The batched VLM path only needs
    key lookup, so convert once at the boundary.
    """
    if isinstance(inputs, dict):
        return inputs
    if hasattr(inputs, "data") and isinstance(getattr(inputs, "data"), dict):
        return dict(inputs.data)
    try:
        return dict(inputs)
    except Exception:
        pass
    out: Dict[str, Any] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "images",
        "image_grid_thw",
        "video_grid_thw",
    ):
        if hasattr(inputs, key):
            out[key] = getattr(inputs, key)
    return out


def _shape_images_for_processor_call(
    processor: Any, images: Optional[List[str]]
) -> Optional[List[Any]]:
    """Normalize image argument shape for processors with conversation nesting.

    Mistral3/Pixtral processors expect ``images`` as a list of per-sample image
    lists. The API path stores data URLs as local temp file paths, and passing
    ``["/tmp/image.png"]`` directly makes those processors iterate the path
    string and hand ``"/"`` to Transformers' image loader.
    """
    if not images:
        return images
    proc_type = type(processor)
    proc_key = f"{proc_type.__module__}.{proc_type.__name__}".lower()
    if not any(name in proc_key for name in ("mistral3", "pixtral")):
        return images
    if all(isinstance(image, str) and image.startswith("/") for image in images):
        return [list(images)]
    return images


def _processor_audio_sampling_rate(processor: Any) -> int:
    """Best-effort source sampling rate a VLM audio processor expects (Hz)."""
    for obj in (
        getattr(processor, "feature_extractor", None),
        getattr(processor, "audio_processor", None),
        getattr(processor, "audio_feature_extractor", None),
        processor,
    ):
        sr = getattr(obj, "sampling_rate", None) or getattr(obj, "sample_rate", None)
        if isinstance(sr, (int, float)) and sr > 0:
            return int(sr)
    return 16000


def _load_audio_waveforms_for_processor(processor: Any, audio: List[Any]) -> List[Any]:
    """Load file-path / data-URL audio entries into float32 waveforms.

    process_audio_input() returns a file PATH string (base64 -> temp .wav ->
    path). Strict HF audio processors (e.g. Gemma 4) expect a list of float32
    waveform arrays, not paths, and otherwise try to float() the path string
    ("could not convert string to float: '/.../tmp.wav'"). Load each path with
    librosa at the processor's source sampling rate. Entries that are already
    arrays pass through untouched. (MiMo uses its own mel path and is excluded by
    the caller.)
    """
    import numpy as np
    sr = _processor_audio_sampling_rate(processor)
    out: List[Any] = []
    for a in audio:
        if isinstance(a, str):
            try:
                import librosa
                wf, _ = librosa.load(a, sr=sr, mono=True)
                out.append(np.asarray(wf, dtype=np.float32))
            except Exception as exc:
                logger.warning("Audio waveform load failed for %s: %s", a, exc)
                out.append(a)
        else:
            out.append(a)
    return out


def _normalize_qwen_video_arrays_for_processor(
    processor: Any, videos: List[Any]
) -> List[Any]:
    """Restore the uint8 contract after mlx-vlm's float32 video resize.

    ``mlx_vlm.video_generate.fetch_video`` returns resized TCHW arrays as
    float32 while preserving the original 0..255 range.  The numpy Qwen3-VL
    video processor intentionally rescales only uint8 arrays; feeding that
    float32 payload skips the 1/255 conversion and turns a blue channel near
    255 into a normalized value near 509.  Restrict the repair to that exact
    processor contract and leave already-normalized float video untouched.
    """
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return videos
    processor_type = type(video_processor)
    processor_key = (
        f"{processor_type.__module__}.{processor_type.__name__}".lower()
    )
    if "qwen3_vl" not in processor_key or "videoprocessor" not in processor_key:
        return videos
    if not bool(getattr(video_processor, "do_rescale", False)):
        return videos

    import numpy as np

    normalized: List[Any] = []
    repaired = 0
    for video in videos:
        array = np.asarray(video) if not isinstance(video, np.ndarray) else video
        if (
            np.issubdtype(array.dtype, np.floating)
            and array.size
            and float(array.min()) >= 0.0
            and 1.5 < float(array.max()) <= 255.5
        ):
            array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
            repaired += 1
        normalized.append(array)
    if repaired:
        logger.info(
            "Restored uint8 0..255 contract for %d Qwen video tensor(s) "
            "before processor rescaling",
            repaired,
        )
    return normalized


def _call_processor_direct_unscoped(
    processor: Any,
    *,
    prompts: Any,
    images: Optional[List[str]],
    videos: Optional[List[Any]] = None,
    video_fps: Optional[List[float]] = None,
    video_timestamps: Optional[List[List[float]]] = None,
    audio: Optional[List[Any]] = None,
    add_special_tokens: bool,
) -> Dict[str, Any]:
    """Call a VLM processor without mlx_vlm.process_inputs' bad `.process` trap.

    mlx-vlm's process_inputs() does `getattr(processor, "process", processor)`
    and immediately calls inspect.signature() on it. Some modern processor
    wrappers expose a non-callable `.process` field that points at the tokenizer
    wrapper, which raises `TypeError: TokenizerWrapper object is not callable`
    before the actual processor __call__ can run. Prefer a callable
    `processor.process`; otherwise invoke the processor itself with signature
    filtering.
    """
    process_method = getattr(processor, "process", None)
    if callable(process_method) and not videos and not audio:
        from mlx_vlm.utils import process_inputs

        return _as_input_mapping(
            process_inputs(
                processor,
                prompts=prompts,
                images=images,
                add_special_tokens=add_special_tokens,
            )
        )

    if not callable(processor):
        raise TypeError(
            f"VLM processor {type(processor).__name__} is not callable and "
            "does not expose a callable .process method"
        )

    processor_type = type(processor)
    processor_blob = " ".join(
        str(item)
        for item in (
            getattr(processor_type, "__module__", ""),
            getattr(processor_type, "__name__", ""),
            getattr(processor, "model_type", ""),
            getattr(getattr(processor, "config", None), "model_type", ""),
        )
    ).lower()
    skip_audios_alias = "mimo" in processor_blob and "v2" in processor_blob

    try:
        params = inspect.signature(processor).parameters
    except (TypeError, ValueError):
        params = {}

    kwargs: Dict[str, Any] = {
        "text": prompts,
        "padding": True,
        "return_tensors": "mlx",
        "add_special_tokens": add_special_tokens,
    }
    if images:
        kwargs["images"] = _shape_images_for_processor_call(processor, images)
    if videos:
        # Mirror mlx_vlm.prepare_inputs' video handling EXACTLY: load string
        # paths through load_video and pass the per-video fps alongside the
        # frames. The fps is the temporal metadata Qwen-style processors use
        # to build video_grid_thw; without it the temporal grid degenerates
        # and the model perceives the clip as one static frame repeated.
        # Measured live on Qwen3.6-4D: video-only turns (which route through
        # prepare_inputs) answered "the ball moves right", while mixed
        # image+video turns (which route HERE because the image forces the
        # safe direct path) enumerated 12 identical frames at fixed
        # coordinates. Same video, same engine — the only difference was this
        # call's missing load/fps step.
        try:
            from mlx_vlm.utils import load_video as _load_video

            _loaded, _video_fps = [], []
            for _index, _v in enumerate(videos):
                _fps_hint = (
                    float(video_fps[_index])
                    if video_fps is not None and _index < len(video_fps)
                    else 2.0
                )
                if isinstance(_v, (str, bytes)):
                    _arr, _s_fps = _load_video(str(_v), fps=_fps_hint)
                else:
                    _arr, _s_fps = _v, _fps_hint
                _loaded.append(_arr)
                _video_fps.append(_s_fps)
            _loaded = _normalize_qwen_video_arrays_for_processor(
                processor, _loaded
            )
            kwargs["videos"] = _loaded
            accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            if "fps" in params or accepts_var_kwargs:
                kwargs["fps"] = _video_fps
            if video_timestamps is not None and "video_timestamps" in params:
                kwargs["video_timestamps"] = video_timestamps
        except Exception:
            kwargs["videos"] = videos
    if audio:
        processor_audio = (
            audio
            if skip_audios_alias
            else _load_audio_waveforms_for_processor(processor, audio)
        )
        kwargs["audio"] = processor_audio
        if not skip_audios_alias and "audios" in params:
            kwargs["audios"] = processor_audio
    if params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return _as_input_mapping(processor(**kwargs))


_GEMMA4_IMAGE_TOKEN_BUDGETS = {70, 140, 280, 560, 1120}


@contextmanager
def _temporary_gemma4_image_token_budget(processor: Any, budget: Optional[int]):
    """Apply a request-local Gemma 4 visual budget and restore the processor.

    Gemma4Processor does not forward arbitrary kwargs to its image processor;
    it reads ``image_processor.max_soft_tokens`` synchronously. The MLLM
    scheduler preprocesses requests serially, so a tightly-scoped mutation is
    safe and avoids changing the model bundle's configured default.
    """
    if budget is None:
        yield
        return
    if budget not in _GEMMA4_IMAGE_TOKEN_BUDGETS:
        allowed = ", ".join(str(value) for value in sorted(_GEMMA4_IMAGE_TOKEN_BUDGETS))
        raise ValueError(f"image_token_budget must be one of: {allowed}")

    processor_type = type(processor)
    processor_blob = " ".join(
        str(item)
        for item in (
            getattr(processor_type, "__module__", ""),
            getattr(processor_type, "__name__", ""),
            getattr(processor, "model_type", ""),
            getattr(getattr(processor, "config", None), "model_type", ""),
        )
    ).lower()
    if "gemma4" not in processor_blob and "gemma_4" not in processor_blob:
        yield
        return

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "max_soft_tokens"):
        raise TypeError(
            f"Gemma 4 processor {type(processor).__name__} has no mutable "
            "image_processor.max_soft_tokens"
        )
    previous = image_processor.max_soft_tokens
    image_processor.max_soft_tokens = int(budget)
    try:
        yield
    finally:
        image_processor.max_soft_tokens = previous


def _call_processor_direct(
    processor: Any,
    *,
    prompts: Any,
    images: Optional[List[str]],
    videos: Optional[List[Any]] = None,
    video_fps: Optional[List[float]] = None,
    video_timestamps: Optional[List[List[float]]] = None,
    audio: Optional[List[Any]] = None,
    add_special_tokens: bool,
    image_token_budget: Optional[int] = None,
) -> Dict[str, Any]:
    with _temporary_gemma4_image_token_budget(processor, image_token_budget):
        return _call_processor_direct_unscoped(
            processor,
            prompts=prompts,
            images=images,
            videos=videos,
            video_fps=video_fps,
            video_timestamps=video_timestamps,
            audio=audio,
            add_special_tokens=add_special_tokens,
        )


def _sampled_video_timestamps(
    video_path: Any,
    frame_count: int,
    sample_fps: float,
) -> List[float]:
    """Rebuild the timestamps used by mlx-vlm's uniform frame sampler."""
    if frame_count <= 0:
        return []
    try:
        fallback_rate = float(sample_fps)
    except (TypeError, ValueError):
        fallback_rate = 2.0
    if fallback_rate <= 0:
        fallback_rate = 2.0
    fallback = [index / fallback_rate for index in range(frame_count)]
    if not isinstance(video_path, (str, os.PathLike)):
        return fallback
    path = str(video_path)
    if path.startswith("file://"):
        path = path[7:]
    try:
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return fallback
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        if total_frames <= 0 or source_fps <= 0:
            return fallback
        indices = np.linspace(0, total_frames - 1, frame_count).round().astype(int)
        return [float(index) / source_fps for index in indices]
    except Exception:
        return fallback


def _resolve_mimo_audio_bundle_path(model: Any, processor: Any) -> Optional[Path]:
    """Resolve the loaded MiMo bundle path without hardcoding local model names."""

    candidates: List[Any] = []
    for obj in (
        processor,
        getattr(processor, "tokenizer", None),
        getattr(processor, "image_processor", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ):
        if obj is None:
            continue
        if isinstance(obj, (str, os.PathLike)):
            candidates.append(obj)
            continue
        for attr in (
            "name_or_path",
            "_name_or_path",
            "pretrained_model_name_or_path",
            "model_path",
        ):
            value = getattr(obj, attr, None)
            if value:
                candidates.append(value)
    for candidate in candidates:
        try:
            path = Path(candidate).expanduser()
        except TypeError:
            continue
        if path.is_dir() and (path / "config.json").is_file():
            return path
    return None


def _mimo_audio_processor_config(model: Any) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    if isinstance(cfg, dict):
        raw = cfg.get("processor_config") or {}
    else:
        raw = getattr(cfg, "processor_config", None) or {}
    return raw if isinstance(raw, dict) else {}


def _mimo_audio_log_mel(path: str, processor_config: Dict[str, Any]):
    """Decode one audio file and produce MiMo tokenizer mel frames [T, n_mels]."""

    import numpy as np

    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Audio ingestion requires librosa, which is not installed. "
            "Install the audio extra: pip install 'vmlx[audio]'"
        ) from exc

    target_sr = int(processor_config.get("audio_sampling_rate") or 24000)
    n_fft = int(processor_config.get("audio_nfft") or 960)
    hop_length = int(processor_config.get("audio_hop_length") or 240)
    win_length = int(processor_config.get("audio_window_size") or n_fft)
    n_mels = int(processor_config.get("audio_n_mels") or 128)
    fmin = float(processor_config.get("audio_fmin") or 0.0)
    fmax_raw = processor_config.get("audio_fmax")
    fmax = None if fmax_raw is None else float(fmax_raw)

    waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"MiMo audio input is empty: {path}")
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return mx.array(np.log(np.maximum(mel, 1e-10)).T.astype(np.float32))


def _mimo_audio_token_id(model: Any) -> Optional[int]:
    cfg = getattr(model, "config", None)
    processor_config = _mimo_audio_processor_config(model)
    for source in (processor_config, cfg):
        for key in ("audio_token_id", "audio_token_index"):
            value = _read_config_field(source, key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _mimo_audio_group_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    audio_config = _read_config_field(cfg, "audio_config")
    processor_config = _mimo_audio_processor_config(model)
    for source, key in (
        (processor_config, "audio_group_size"),
        (audio_config, "group_size"),
    ):
        value = _read_config_field(source, key)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return 4


def _input_ids_contains_token(input_ids: Any, token_id: int) -> bool:
    if input_ids is None:
        return False
    try:
        arr = mx.array(input_ids)
        return bool(mx.any(arr == int(token_id)).item())
    except Exception:
        try:
            return int(token_id) in input_ids
        except Exception:
            return False


def _expand_mimo_audio_token_placeholders(
    *,
    input_ids: Any,
    attention_mask: Any,
    token_id: int,
    target_count: int,
) -> tuple[mx.array, Any]:
    """Expand one MiMo audio pad token to the number of audio embeddings."""

    ids = mx.array(input_ids)
    if ids.ndim == 1:
        rows = [ids.tolist()]
        batched = False
    elif ids.ndim == 2 and int(ids.shape[0]) == 1:
        rows = ids.tolist()
        batched = True
    else:
        raise ValueError(
            "MiMo-V2 audio prompt expansion supports one request at a time; "
            f"got input_ids shape {tuple(ids.shape)}"
        )
    row = [int(x) for x in rows[0]]
    current_count = sum(1 for x in row if x == int(token_id))
    if current_count == int(target_count):
        return ids, attention_mask
    if current_count != 1:
        raise ValueError(
            "MiMo-V2 audio token count "
            f"{current_count} does not match target embeddings {target_count}"
        )

    expanded: list[int] = []
    for value in row:
        if value == int(token_id):
            expanded.extend([int(token_id)] * int(target_count))
        else:
            expanded.append(value)
    new_ids = mx.array([expanded] if batched else expanded, dtype=ids.dtype)

    if attention_mask is None:
        return new_ids, attention_mask
    mask = mx.array(attention_mask)
    if mask.ndim == 1:
        mask_row = [int(x) for x in mask.tolist()]
        mask_batched = False
    elif mask.ndim == 2 and int(mask.shape[0]) == 1:
        mask_row = [int(x) for x in mask.tolist()[0]]
        mask_batched = True
    else:
        return new_ids, attention_mask
    if len(mask_row) != len(row):
        return new_ids, attention_mask
    expanded_mask: list[int] = []
    for value, mask_value in zip(row, mask_row):
        if value == int(token_id):
            expanded_mask.extend([mask_value] * int(target_count))
        else:
            expanded_mask.append(mask_value)
    return new_ids, mx.array([expanded_mask] if mask_batched else expanded_mask, dtype=mask.dtype)


def _build_mimo_audio_codes_from_paths(
    *,
    model: Any,
    processor: Any,
    audio_paths: List[str],
    input_ids: Any,
) -> Optional[mx.array]:
    """Convert raw MiMo audio paths to 20-channel audio_codes when possible."""

    if not audio_paths:
        return None
    token_id = _mimo_audio_token_id(model)
    if token_id is None:
        raise UnsupportedMediaModalityError(
            "audio",
            "MiMo-V2 audio request has raw audio but no audio_token_id in model processor_config.",
            family="mimo_v2",
        )
    if not _input_ids_contains_token(input_ids, token_id):
        raise UnsupportedMediaModalityError(
            "audio",
            "MiMo-V2 audio request has raw audio but the tokenized prompt contains no audio token.",
            family="mimo_v2",
        )
    bundle = _resolve_mimo_audio_bundle_path(model, processor)
    if bundle is None:
        raise UnsupportedMediaModalityError(
            "audio",
            "MiMo-V2 audio request cannot resolve the loaded bundle path for audio_tokenizer weights.",
            family="mimo_v2",
        )
    audio_dir = bundle / "audio_tokenizer"
    if not (audio_dir / "config.json").is_file() or not (audio_dir / "model.safetensors").is_file():
        raise UnsupportedMediaModalityError(
            "audio",
            f"MiMo-V2 audio tokenizer sidecar is incomplete under {audio_dir}.",
            family="mimo_v2",
        )
    cache_key = str(bundle.resolve())
    tokenizer = _MIMO_AUDIO_TOKENIZER_CACHE.get(cache_key)
    if tokenizer is None:
        from .models import mllm as local_mllm

        local_mllm._register_mimo_v2_mlx_vlm_runtime()
        import sys

        module = sys.modules.get("mlx_vlm.models.mimo_v2")
        if module is None or not hasattr(module, "load_mimo_audio_tokenizer_from_bundle"):
            raise UnsupportedMediaModalityError(
                "audio",
                "MiMo-V2 audio tokenizer runtime is not registered.",
                family="mimo_v2",
            )
        tokenizer = module.load_mimo_audio_tokenizer_from_bundle(bundle)
        _MIMO_AUDIO_TOKENIZER_CACHE[cache_key] = tokenizer
    processor_config = _mimo_audio_processor_config(model)
    segment_size = int(processor_config.get("audio_segment_size") or 6000)
    channels = int(processor_config.get("audio_channels") or 20)
    mels = [
        _mimo_audio_log_mel(str(path), processor_config)
        for path in audio_paths
    ]
    codes_per_audio = tokenizer.encode_audio_to_codes(
        mels,
        segment_size=segment_size,
        n_q=channels,
    )
    if not codes_per_audio:
        return None
    codes = mx.concatenate([mx.array(c).astype(mx.int32) for c in codes_per_audio], axis=0)
    return codes[:, :channels]

def _should_use_safe_processor_path(
    processor: Any,
    *,
    has_image_literal: bool,
    has_images: bool,
) -> bool:
    """Decide whether to bypass mlx_vlm.prepare_inputs / process_inputs.

    Three failure modes flow into the safe path (``_call_processor_direct``):

    1. Images present but no ``<image>`` literal in the prompt — prepare_inputs'
       BaseImageProcessor branch hardcodes ``split("<image>")`` and silently
       drops the image (Gemma 4 ``<|image|>``, Qwen3.5/3.6 native tokens).
    2. Processor exposes a ``.process`` attribute that is not callable —
       ``mlx_vlm.utils.process_inputs`` runs ``inspect.signature(.process)``
       which raises ``TypeError: <TokenizerWrapper ...> is not a callable
       object`` (vmlx#145).
    3. Processor lacks ``.process`` entirely AND is not itself callable —
       ``getattr(processor, "process", processor)`` then ``inspect.signature``
       on a TokenizerWrapper raises the same ``not a callable object`` error.
       This is the Case D edge that the original guard missed; some JANGTQ4
       VLM bundles expose only the tokenizer wrapper as the processor.

    Returns ``True`` when the safe path should be used. Pure helper for
    testing — no side effects.
    """
    if not has_images:
        return False
    if not has_image_literal:
        return True
    process_attr = getattr(processor, "process", None)
    if process_attr is None:
        # Falls through to processor itself in mlx_vlm; safe iff callable.
        return not callable(processor)
    return not callable(process_attr)


# TurboQuantKVCache class name for isinstance-free detection.
# TQ is a drop-in replacement for KVCache (positional, sliceable, has .state/.keys/.values)
# but does NOT inherit from KVCache. This constant enables KV-like detection without
# importing TQ (which may not be installed for non-JANG users).
_TQ_CLASS_NAME = "TurboQuantKVCache"
_ATTENTION_CACHE_CLASS_NAMES = {
    "KVCache",
    "BatchKVCache",
    "RotatingKVCache",
    "BatchRotatingKVCache",
    "QuantizedKVCache",
    _TQ_CLASS_NAME,
}


def _is_kv_like(c) -> bool:
    """Check if cache is attention-KV compatible.

    Gemma4 mixed-SWA uses RotatingKVCache for sliding-window attention layers
    and KVCache for full-attention layers. Both are attention cache slots; they
    must not be mistaken for SSM/ArraysCache hybrid state.
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    return isinstance(c, (KVCache, RotatingKVCache)) or type(c).__name__ == _TQ_CLASS_NAME


def _is_attention_cache_slot(cache: Any) -> bool:
    """Recognize attention cache slots across mlx-lm and mlx-vlm namespaces.

    ``mlx_vlm.models.cache`` vendors classes named ``KVCache`` and
    ``RotatingKVCache`` that are not instances of the same-named mlx-lm
    classes. Cache-layout detection previously classified every Gemma 4 slot
    as non-KV, then ``_fix_hybrid_cache`` replaced a valid reconstructed
    48-layer prefix with a fresh all-empty template. Keep merge mechanics on
    their existing concrete-type checks; this predicate is only for structural
    attention-vs-SSM layout decisions.
    """
    return _is_kv_like(cache) or type(cache).__name__ in _ATTENTION_CACHE_CLASS_NAMES


# Allow the CLEAN RE-DERIVE to run chunked even for recurrent (SSM) slots.
# Default ON.
#
# The 2026-08-11 note here said this was NOT what capped hybrid prefix reuse,
# because turning it on left Qwen3.6 frozen at its first turn. That was true at
# the time and is no longer: the freeze then was the blanket hybrid store skip,
# which masked this. With hybrid wired into the clean store, chunk-safety became
# the binding constraint for everything past ~12.5k tokens, where the one-shot
# buffer prediction exceeds the Metal single-buffer limit and the store is
# skipped outright. Both were real; they were just in series.
#
# The actual cap is the deliberate store-skip in mllm_scheduler: hybrid + cache
# hit skips the store, because promoting a live extended cache compounds
# reconstruction error until Bonsai/Qwen3.5 collapse into a token loop. Lifting
# THAT is the real unlock, and it needs a long coherence run to justify.
from .utils.memory_limits import get_effective_metal_working_set_bytes
from .utils.prefill_admission import (
    PrefillAdmissionError,
    fit_peak_model,
    hybrid_chunk_valve_check,
    prefill_keep_alloc_enabled,
    prefill_valve_enabled,
    prefill_valve_min_margin_bytes,
    span_admission_check,
    turn_peak_admission_check,
)
from .utils.prefill_admission import (
    max_prefill_chunk_tokens,
    prefill_valve_enabled as _prefill_valve_enabled,
)

# Adaptive chunk sizing from MEASURED per-chunk transient. Default OFF.
#
# It works as designed — it measured an 18.32GB transient for a 2048-token chunk
# and shrank 2048 -> 1731 -> 1137 -> 469 — and the process STILL aborted, because
# by then ACTIVE memory was already ~95GB against a ~107GB limit. No chunk size
# helps once the resident set is that close to the ceiling.
#
# Weights (~15GB) plus the measured cache (~6GB) account for only ~21GB of that
# 95GB, so ~74GB is live memory that is neither the cache list nor the MLX
# allocator cache (cache=0.00GB throughout). That is a LEAK, and sizing cannot
# fix a leak. Shipping this on would add cost without preventing the crash —
# the same mistake as the reverted budget clamp (a3cedb29f).
#
# Kept because the mechanism is right and measurement-driven: turn it on to
# study per-chunk transients, or once the leak is fixed.
_HYBRID_ADAPTIVE_CHUNK = os.environ.get(
    "VMLX_HYBRID_ADAPTIVE_CHUNK", "0"
).strip().lower() in {"1", "true", "yes", "on"}

_HYBRID_MIN_CHUNK = max(1, int(os.environ.get("VMLX_HYBRID_MIN_CHUNK", "64") or 64))

# Ceiling on the PROJECTED tight-memory prefill step.
#
# Bigger is not simply better: the per-chunk admission valve projects the
# next chunk's transient by the context ratio, so a step that merely fits
# the current context is declined a chunk later. Measured on dots3: at
# step 2048 a 2015-token prompt ran at 591.6 pp/s but an 8k prompt was
# 413'd at chunk [2048:4096); step 1024 was independently measured FASTER
# than 2048 at 12k context while halving the transient. So the cap buys
# depth AND speed.
_TIGHT_PROJECTED_STEP_CAP = max(
    64, int(os.environ.get("VMLX_TIGHT_PROJECTED_STEP_CAP", "1024") or 1024)
)

# One-shot attention buffer size above which a hybrid prefill switches to the
# chunked path (and above which the SSM clean re-derive declines) rather than
# ask Metal for a single allocation it will refuse.
#
# This was written out twice as a bare `8 * 1024 ** 3` local, which is the
# duplicated-constant pattern that has bitten this project repeatedly. Worse,
# the chunked path's own comment asks that "correctness should be spot-checked
# by the caller" for families other than qwen3_5 — and with the threshold
# hardcoded there was NO WAY to make a family take the chunked path at a prompt
# size where the one-shot path also works, so the comparison the comment asks
# for could not be performed at all. Lowering this makes chunking reachable on a
# small prompt, which is what turns that caveat into something testable.
_HYBRID_ONE_SHOT_GUARD_BYTES = max(
    1,
    int(
        os.environ.get("VMLX_HYBRID_ONE_SHOT_GUARD_BYTES", str(8 * 1024**3))
        or 8 * 1024**3
    ),
)

# Hybrid text model_types whose CHUNKED prefill is MECHANISM-equivalence
# proven against one-shot (tests/test_hybrid_chunked_prefill_equivalence.py:
# bit-identical final logits, KV, and GDN conv/ssm state across aligned,
# non-divisor, ragged and per-token chunk grids, plus warm-cache suffixes, on
# the fused-SDPA path).
#
# This set does NOT flip the default yet. The flip was tried and RETRACTED
# (2026-08-23): on MLX 0.32.1 a live temperature-0 A/B diverged the reasoning
# trajectory between lanes on Qwen3.8-27B because head_dim=256 used the
# materializing SDPA fallback. MLX 0.32.2 added a fused NAX D=256 path, and the
# vendored Qwen attention now forces that path for eligible non-divisor tails,
# but the replacement live answer-byte gate must pass before the lane default
# changes. See the path decision inside _run_vision_encoding_inner.
#
# The set still labels the auto-chunk OOM escape hatch below as
# "verified" (the chunked MATH is exact; the hatch only fires where one-shot
# would die anyway), and it is the allow-list a future answer-byte-proven
# flip would key on.
_HYBRID_CHUNKED_PROVEN_TEXT_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_vl",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def _qwen_fused_d256_owns_attention_allocation(
    language_model: Any, model_type: str
) -> bool:
    """Whether MLX fused SDPA makes the old materialized-score guard stale."""
    if model_type not in _HYBRID_CHUNKED_PROVEN_TEXT_MODEL_TYPES:
        return False
    enabled = (
        os.environ.get("VMLINUX_QWEN35_FUSED_PREFILL")
        or os.environ.get("VMLX_QWEN35_FUSED_PREFILL")
        or "1"
    )
    if enabled.strip().lower() in {"0", "false", "no", "off"}:
        return False
    args = getattr(language_model, "args", None)
    head_dim = _read_config_field(args, "head_dim")
    try:
        from importlib.metadata import PackageNotFoundError, version

        mlx_version = tuple(
            int(part) for part in version("mlx").split(".")[:3]
        )
    except (ImportError, PackageNotFoundError, ValueError):
        return False
    return mlx_version >= (0, 32, 2) and int(head_dim or 0) == 256

# Drain the generation stream before each per-chunk clear_cache.
#
# DEFAULT OFF — this was tried against the hybrid retention and MEASURED to do
# NOTHING. The theory was that clear_cache reclaims nothing because the dead
# buffers are still "active" until the runtime catches up on a stream that is
# never synchronized. With the drain ON the curve was byte-identical: base
# 87.46GB / peak 104.41GB at chunk 46, exactly as without it. The flag value was
# confirmed True in the serving process first, so this is a real negative and
# not a guard that silently declined.
#
# Kept behind a flag rather than deleted because the reasoning is sound for
# other allocation shapes and it costs one synchronize per chunk to re-test.
# Do NOT turn it on as a fix without new evidence.
# Pre-size KV slots to the whole prefill span so the chunk loop stops
# reallocating every layer's K/V per chunk. ON by default: it removes the
# dominant source of per-chunk garbage. VMLX_KV_PRESIZE_SPAN=0 restores the
# incremental 256-token growth.
_KV_PRESIZE_SPAN = os.environ.get(
    "VMLX_KV_PRESIZE_SPAN", "1"
).strip().lower() not in {"0", "false", "no", "off"}

_HYBRID_PREFILL_DRAIN = os.environ.get(
    "VMLX_HYBRID_PREFILL_DRAIN", "0"
).strip().lower() not in {"0", "false", "no", "off"}

# Deep-span cache relief. The MLX allocator cache retains freed buffers up
# to its limit (25% of the working set — 26.9GB on a 128GB box). Across a
# long multiturn session that cache fills with dead buffers, and a deep
# span's peak (block reconstruction transient + full-span KV presize +
# chunk transients + MTP buffers) then lands ON TOP of it — Metal aborts
# with an UNCATCHABLE command-buffer OOM. MEASURED: a 16-turn incremental
# grow crashed at a ~94k-token turn while the IDENTICAL request (same
# restored 88,918-token chain, same delta prefill) completed cleanly on a
# fresh process whose cache started empty. Clearing the cache before a
# deep span returns that garbage exactly when the room is needed; a span
# this size runs for seconds, so one clear is noise. 0 disables.
try:
    _DEEP_SPAN_CACHE_CLEAR_TOKENS = max(
        0,
        int(
            os.environ.get("VMLX_DEEP_SPAN_CACHE_CLEAR_TOKENS", "32768")
            or "0"
        ),
    )
except (TypeError, ValueError):
    _DEEP_SPAN_CACHE_CLEAR_TOKENS = 32768


# Decode headroom folded into the span presize. With step == span exactly,
# the FIRST decode token's lazy eval grows EVERY attention layer's KV to the
# next step multiple in one graph — a full second copy of the entire span's
# KV while the old buffers are still alive. MEASURED (R5 intra-turn RSS
# poller): a ~23GB spike in <4s at the prefill->decode boundary of a ~96k
# turn, the exact moment of four identical Metal OOM aborts; the per-turn
# peak stride (~5GB/turn) is this spike growing with span. Presizing to
# span + headroom makes decode write IN PLACE instead.
try:
    _DECODE_PRESIZE_HEADROOM = max(
        0,
        int(os.environ.get("VMLX_DECODE_PRESIZE_HEADROOM", "4096") or "0"),
    )
except (TypeError, ValueError):
    _DECODE_PRESIZE_HEADROOM = 4096


def _presize_kv_slots_for_span(cache, span_tokens: int) -> int:
    """Instance-scope KVCache steps to span+headroom; returns count set.

    Must cover BOTH forward lanes (fresh chunked prefill AND the hybrid
    cache-hit delta — the reconstructed cache had no presize at all, so
    every delta chunk and the decode start reallocated the full span).
    """
    if not _KV_PRESIZE_SPAN or span_tokens <= 0:
        return 0
    count = 0
    for _slot in (cache or []):
        if type(_slot).__name__ != "KVCache":
            continue
        if "step" in getattr(_slot, "__dict__", {}):
            continue  # already instance-scoped; leave it alone
        try:
            _slot.step = int(span_tokens + _DECODE_PRESIZE_HEADROOM)
            count += 1
        except Exception:  # noqa: BLE001
            pass
    return count


def _maybe_clear_deep_span_cache(total_span_tokens: int) -> None:
    """Synchronize + clear the MLX allocator cache ahead of a deep span.

    Must be called from EVERY lane that runs a large forward: the fresh
    chunked-prefill lane AND the hybrid cache-hit delta lane — the first
    landing of this fix covered only the fresh lane and the 16-turn grow
    crashed again at ~96k with ZERO engagement lines (the standing
    one-of-two-lanes failure class, proven by the r2 rerun).
    """
    if _DEEP_SPAN_CACHE_CLEAR_TOKENS <= 0:
        return
    span = int(total_span_tokens or 0)
    if span < _DEEP_SPAN_CACHE_CLEAR_TOKENS:
        return
    try:
        mx.synchronize()
    except Exception:  # noqa: BLE001
        pass
    try:
        _clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
        _clear()
        logger.info(
            "Deep-span prefill: cleared MLX allocator cache before a "
            "%d-token span (threshold %d)",
            span,
            _DEEP_SPAN_CACHE_CLEAR_TOKENS,
        )
    except Exception:  # noqa: BLE001
        pass

# Cross-turn peak-walk admission (the third valve; see
# turn_peak_admission_check for why the other two cannot see this failure and
# why the allowance defaults to 0 — the measured boundary cases project to the
# same value, so refusal at the device limit is the only side that never
# aborts the process).
_TURN_PEAK_ADMISSION = os.environ.get(
    "VMLX_TURN_PEAK_ADMISSION", "1"
).strip().lower() not in {"0", "false", "no", "off"}
try:
    _TURN_PEAK_ALLOWANCE_BYTES = int(
        os.environ.get("VMLX_TURN_PEAK_ALLOWANCE_MB", "0") or "0"
    ) * 1024 * 1024
except ValueError:
    # A malformed override must not kill the engine at import; the safe
    # default is the measured one.
    _TURN_PEAK_ALLOWANCE_BYTES = 0
# Recent turns only: residency changes (eviction, an unload, a pool-config
# change) shift the walk's intercept, and stale points from a heavier regime
# would over-project and refuse servable turns.
_TURN_PEAK_WALK_MAXLEN = 8

_HYBRID_PREFILL_MEM_TRACE = os.environ.get(
    "VMLX_HYBRID_PREFILL_MEM_TRACE", ""
).strip().lower() in {"1", "true", "yes", "on"}

_CHUNKED_SSM_REDERIVE = os.environ.get(
    "VMLX_CHUNKED_SSM_REDERIVE", ""
).strip().lower() not in {"0", "false", "no", "off"}


_HYBRID_BASE_SPLICE = os.environ.get(
    "VMLX_HYBRID_BASE_SPLICE", ""
).strip().lower() not in {"0", "false", "no", "off"}


def _hybrid_base_splice_enabled() -> bool:
    """Opt in to completing a paged-rebuilt base with companion SSM state.

    DEFAULT ON, after an A/B that proved both halves. The reuse win already landed: hybrid prefixes
    extend and long documents cache. What has NOT landed is the LATENCY win --
    a base rebuilt from paged blocks carries attention KV only, so on a hybrid
    model the layout check refuses it and the store re-derives the WHOLE prompt
    every turn (measured: 43.7k-token document reused 99.9% yet each follow-up
    still cost ~113s, essentially all of it that re-derive).

    The missing half is already on disk: the SSM companion is stored complete at
    the same absolute token key the base covers, so pairing the two reconstructs
    a correctly typed hybrid cache and only the delta needs forwarding.

    Both halves were then measured on a 43.7k-token document, same probe, same
    conversation, only this gate differing:

    - OFF: three prefills, each `resume_at=0 base_tokens=43733 matches=False` --
      the base is refused every turn and all 43.7k tokens re-derive.
      Q3 **107.3s**, Q4 **114.1s**.
    - ON: only the cold first prefill; later turns need NO prefill at all.
      Q3 **5.2s**, Q4 **8.7s**. About 20x.

    Correctness cleared the same bar the clean store had to: the 8-turn matrix
    (reasoning effort changed mid-conversation, tools appearing and disappearing,
    thinking toggled off) at temperature 0 is **9/9 byte-identical** with the gate
    ON versus OFF.

    Set VMLX_HYBRID_BASE_SPLICE=0 to revert. It remains a cache-correctness change
    on path-dependent state with a silent failure mode, so any future change here
    needs the same two measurements -- and read `resume_at` from the log rather
    than grepping for your own markers, which is how this was misjudged twice.
    """
    return _HYBRID_BASE_SPLICE


def _is_recognised_recurrent_slot(cache: Any) -> bool:
    """Is this a recurrent cache slot whose layout we actually recognise?

    mlx-lm's recurrent caches (ArraysCache and friends) carry their rolling
    tensors on a ``state`` attribute. That is the structural signal used here,
    so a slot we merely failed to recognise as attention does not get treated
    as chunk-safe by default.
    """
    if _is_attention_cache_slot(cache):
        return False
    if type(cache).__name__ in _RECOGNISED_RECURRENT_CACHE_CLASS_NAMES:
        return True
    return hasattr(cache, "state")


_RECOGNISED_RECURRENT_CACHE_CLASS_NAMES = {
    "ArraysCache",
    "MambaCache",
    "ConvCache",
    "RecurrentCache",
}


def _cache_requires_one_shot_rederive(
    cache_slots: Any, *, ignore_chunk_override: bool = False
) -> bool:
    """Does re-deriving this cache need one contiguous forward pass?

    Only recurrent slots (Mamba/ArraysCache-style SSM state) do: their
    offset/mask bookkeeping is populated by the ``BatchKVCache`` wrappers the
    re-derive path does not use, so a second chunk reads uninitialised state.
    Attention slots — including Gemma 4's mixed RotatingKVCache/KVCache stack —
    are chunk-safe, because the live prefill already advances them chunk by
    chunk via ``prefill_step_size``.

    Unknown/unrecognised slots count as recurrent: treating a slot we cannot
    classify as chunk-safe would risk storing silently wrong state, whereas the
    one-shot path merely costs memory and can decline.
    """
    if _CHUNKED_SSM_REDERIVE and not ignore_chunk_override and all(
        _is_attention_cache_slot(slot) or _is_recognised_recurrent_slot(slot)
        for slot in (
            cache_slots if isinstance(cache_slots, (list, tuple)) else [cache_slots]
        )
        if slot is not None
    ):
        # Treat RECOGNISED recurrent slots as chunk-safe too. ON by default.
        #
        # Deliberately not fail-open: a slot we cannot classify still forces the
        # one-shot path, because the evidence below covers known recurrent
        # caches and says nothing about an unrecognised layout.
        #
        # The one-shot rule was written from a failure on the 2nd chunk, blamed
        # on ArraysCache's lengths/left_padding being unpopulated on a fresh
        # make_cache(). Reading mlx-lm, those fields are INERT rather than
        # uninitialised when None: make_mask() returns None (correct for an
        # unpadded batch of 1) and advance() is a no-op.
        #
        # The rule was over-conservative and the cost was total: requiring one
        # contiguous pass means the re-derive declines whenever the predicted
        # attention buffer exceeds the Metal single-buffer limit, which on a
        # ~30-head model is only ~12.5k tokens. Past that the store is skipped
        # entirely, so a long document is re-prefilled from scratch on EVERY
        # follow-up. Measured on Qwen3.8 over a 43.7k-token document: four
        # questions, cached 0 every time, ~122s each.
        #
        # It was gated behind a byte-exactness A/B against the one-shot path
        # plus a long multiturn coherence run. Both were done before this
        # flipped:
        #   - 8-turn matrix varying reasoning effort, tools on/off and thinking
        #     off: 9/9 byte-identical to the one-shot path on Qwen3.8
        #   - same 43.7k document: reuse 0% -> 99.9% on every follow-up, with
        #     both planted facts still retrieved verbatim
        #   - needles at three depths across 14k/44k/90k prompts: 9/9 retrieved
        #   - Nemotron-3.5-Lightning, a different hybrid family, 9 turns with no
        #     token loop and reuse climbing to 96.6%
        #
        # Set VMLX_CHUNKED_SSM_REDERIVE=0 to restore the one-shot rule. Note the
        # store still re-derives the WHOLE prompt each turn on hybrid models,
        # because the base rebuilt from paged blocks has no recurrent slots and
        # is refused by the layout check; the reuse is real but the latency win
        # is not, until that base is completed from the SSM companion.
        return False
    slots = cache_slots if isinstance(cache_slots, (list, tuple)) else [cache_slots]
    for slot in slots:
        if slot is None:
            continue
        nested = getattr(slot, "caches", None)
        if isinstance(nested, (list, tuple)):
            if _cache_requires_one_shot_rederive(
                nested, ignore_chunk_override=ignore_chunk_override
            ):
                return True
            continue
        if not _is_attention_cache_slot(slot):
            return True
    return False


_CACHE_OWNER_WRAPPER_ATTRS = (
    "language_model",
    "model",
    "inner",
    "base_model",
    "text_model",
    "transformer",
)


def _iter_cache_owner_candidates(
    model: Any,
    language_model: Optional[Any] = None,
    *,
    limit: int = 12,
):
    """Yield wrapper/model candidates that may own ``make_cache``.

    JANG-affine VLM loading can leave the callable on the real text backbone
    behind a compatibility wrapper.  Other cache/config detection in this file
    already walks ``language_model`` and inner ``model`` objects; keep cache
    ownership on the same structural path without copying methods onto wrappers.
    """

    queue: List[Tuple[str, Any]] = []
    seen: set[int] = set()

    def add(label: str, value: Any) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        queue.append((label, value))

    add("language_model", language_model)
    add("model", model)
    index = 0
    while index < len(queue) and index < limit:
        label, value = queue[index]
        index += 1
        yield label, value
        for attr in _CACHE_OWNER_WRAPPER_ATTRS:
            try:
                child = getattr(value, attr, None)
            except Exception:
                child = None
            if child is not None and child is not value:
                add(f"{label}.{attr}", child)


def _resolve_make_cache_owner(
    model: Any,
    language_model: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Return the first callable ``make_cache`` owner and its wrapper path."""

    for label, candidate in _iter_cache_owner_candidates(model, language_model):
        try:
            make_cache = getattr(candidate, "make_cache", None)
        except Exception:
            make_cache = None
        if callable(make_cache):
            return candidate, label
    return None, None


def _hybrid_cache_layout(
    model: Any,
    language_model: Optional[Any] = None,
) -> Tuple[
    Optional[Any],
    Optional[str],
    Optional[List[Any]],
    Optional[List[int]],
    Optional[str],
]:
    """Resolve the authoritative cache template and its attention positions."""

    owner, owner_path = _resolve_make_cache_owner(model, language_model)
    if owner is None:
        return None, None, None, None, "no callable make_cache owner"
    try:
        template = list(owner.make_cache() or [])
    except Exception as exc:
        return owner, owner_path, None, None, f"{type(exc).__name__}: {exc}"
    positions = [
        index
        for index, cache in enumerate(template)
        if _is_attention_cache_slot(cache)
    ]
    return owner, owner_path, template, positions, None


def _config_as_hybrid_probe(config: Any, *, depth: int = 0) -> Optional[Dict[str, Any]]:
    """Normalize dict/dataclass configs for the hybrid-family warning gate."""

    if config is None or depth > 3:
        return None
    if isinstance(config, dict):
        return config
    raw = getattr(config, "_raw_config", None)
    if isinstance(raw, dict):
        return raw
    probe: Dict[str, Any] = {}
    for field in (
        "model_type",
        "hybrid_override_pattern",
        "layer_types",
        "layer_type",
        "layers_block_type",
    ):
        value = getattr(config, field, None)
        if value is not None:
            probe[field] = value
    text_config = getattr(config, "text_config", None)
    text_probe = _config_as_hybrid_probe(text_config, depth=depth + 1)
    if text_probe:
        probe["text_config"] = text_probe
    return probe or None


def _declares_hybrid_ssm_model(model: Any, language_model: Optional[Any] = None) -> bool:
    """Return True when wrapper metadata declares SSM/linear-attention layers."""

    from .utils.ssm_companion_cache import is_hybrid_ssm_config

    for _, candidate in _iter_cache_owner_candidates(model, language_model):
        for attr in ("config", "args", "text_config"):
            probe = _config_as_hybrid_probe(getattr(candidate, attr, None))
            if probe is not None and is_hybrid_ssm_config(probe):
                return True
    return False


def _warn_if_hybrid_detection_disabled(
    *,
    model: Any,
    language_model: Optional[Any],
    is_hybrid: bool,
    owner_path: Optional[str],
    template: Optional[List[Any]],
    error: Optional[str],
) -> None:
    """Make a declared hybrid family's disabled companion path impossible to hide."""

    if is_hybrid or not _declares_hybrid_ssm_model(model, language_model):
        return
    template_names = (
        [type(cache).__name__ for cache in template]
        if template is not None
        else None
    )
    logger.warning(
        "Hybrid-family model resolved _is_hybrid=False; SSM companion lookup/store "
        "is disabled and attention-KV prefix hits cannot be consumed safely. "
        "model_type=%s make_cache_owner=%s template=%s error=%s",
        _runtime_model_type(model) or "unknown",
        owner_path or "none",
        template_names,
        error or "none",
    )


def _is_tq_batch_api(c) -> bool:
    """TurboQuant cache with real batch filter/extract/extend semantics."""
    if type(c).__name__ != _TQ_CLASS_NAME:
        return False
    if getattr(c, "_vmlx_batch_api", None) != "turboquant_kv_v1":
        return False
    return all(
        callable(getattr(c, name, None))
        for name in ("extend", "filter", "extract", "prepare", "finalize")
    )


def _paged_hybrid_cache_detail(
    *,
    disk_hit: bool,
    mixed_attention: bool,
    disk_only: bool = False,
    tq_native: bool = False,
) -> str:
    if disk_only:
        base = "block-disk+mixed_swa" if mixed_attention else "block-disk+ssm"
    else:
        base = "paged+mixed_swa" if mixed_attention else "paged+ssm"
        if disk_hit:
            base = f"{base}+disk"
    return f"{base}+tq-native" if tq_native else base


def _paged_attention_cache_detail(
    *,
    disk_hit: bool,
    mixed_attention: bool,
    disk_only: bool = False,
    tq_native: bool = False,
) -> str:
    if disk_only:
        base = "block-disk+mixed_swa" if mixed_attention else "block-disk"
    else:
        base = "paged+mixed_swa" if mixed_attention else "paged"
        if disk_hit:
            base = f"{base}+disk"
    return f"{base}+tq-native" if tq_native else base


def _paged_reconstruct_disk_source(
    *,
    fetch_disk_hit: bool,
    block_aware_cache: Any,
    reconstructed: Any,
) -> tuple[bool, int]:
    """Merge fetch-time and lazy worker-side L2 source accounting.

    Frugal paged entries can keep an in-process block index while releasing the
    tensor payload. ``fetch_cache`` then finds the chain without reading disk;
    the actual L2 read happens later inside ``reconstruct_cache``. Sampling only
    the fetch-time disk counter mislabels that restore as RAM-only.
    """
    if reconstructed is None:
        return bool(fetch_disk_hit), 0
    try:
        disk_blocks = int(
            getattr(block_aware_cache, "_last_reconstruct_disk_blocks", 0) or 0
        )
    except Exception:
        disk_blocks = 0
    return bool(fetch_disk_hit or disk_blocks > 0), max(0, disk_blocks)


def _cache_layer_debug_summary(cache: Optional[List[Any]], limit: int = 8) -> str:
    if not cache:
        return ""
    parts: List[str] = []
    for i, layer in enumerate(cache[:limit]):
        cls_name = type(layer).__name__
        details = [f"L{i}:{cls_name}"]
        for attr in ("offset", "_idx", "max_size", "keep"):
            if hasattr(layer, attr):
                try:
                    value = getattr(layer, attr)
                    if isinstance(value, mx.array):
                        value = value.tolist()
                    details.append(f"{attr}={value}")
                except Exception:
                    pass
        keys = getattr(layer, "keys", None)
        if keys is not None and hasattr(keys, "shape"):
            details.append(f"keys={tuple(keys.shape)}")
        parts.append(":".join(details))
    return ";".join(parts)


def _model_uses_zaya_cache_contract(model: Any) -> bool:
    """Return True when a model has ZAYA CCA typed cache slots."""
    if model is None:
        return False

    def _model_type_is_zaya(obj: Any) -> bool:
        cfgs = []
        for attr in ("config", "args"):
            cfg = getattr(obj, attr, None)
            if cfg is not None:
                cfgs.append(cfg)
                nested = (
                    cfg.get("text_config")
                    if isinstance(cfg, dict)
                    else getattr(cfg, "text_config", None)
                )
                if nested is not None:
                    cfgs.append(nested)
        for cfg in cfgs:
            mt = cfg.get("model_type") if isinstance(cfg, dict) else getattr(cfg, "model_type", None)
            if str(mt or "").lower() in ("zaya", "zaya1_vl"):
                return True
        return False

    if _model_type_is_zaya(model):
        return True
    if not hasattr(model, "make_cache"):
        return False
    try:
        cache = model.make_cache() or []
        names = [type(c).__name__ for c in cache]
        if "ZayaNoStateCache" in names:
            return True
        return any(
            type(c).__name__ == "CacheList"
            and "zaya" in str(type(c).__module__).lower()
            for c in cache
        )
    except Exception:
        return False


def _runtime_model_type(model: Any) -> str:
    """Resolve model_type across VLM wrappers and inner language models.

    Some JANG VLM wrappers expose an empty top-level config model type while
    the real text runtime family lives under ``language_model.config`` or
    ``args``. MiMo-specific thinking-off and XML tool processors depend on
    this value, so the batch generator must inspect the inner runtime too.
    """

    def _cfg_value(cfg: Any, key: str) -> Any:
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)

    def _cfg_candidates(obj: Any) -> list[Any]:
        out: list[Any] = []
        for attr in ("config", "args", "text_config"):
            cfg = getattr(obj, attr, None)
            if cfg is not None:
                out.append(cfg)
        for cfg in list(out):
            raw = _cfg_value(cfg, "_raw_config")
            if isinstance(raw, dict):
                out.append(raw)
            text_cfg = _cfg_value(cfg, "text_config")
            if text_cfg is not None:
                out.append(text_cfg)
        return out

    objects: list[Any] = []
    for obj in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if obj is not None and obj not in objects:
            objects.append(obj)
    for obj in list(objects):
        for nested in (
            getattr(obj, "language_model", None),
            getattr(obj, "model", None),
        ):
            if nested is not None and nested not in objects:
                objects.append(nested)

    for obj in objects:
        for cfg in _cfg_candidates(obj):
            model_type = _cfg_value(cfg, "model_type")
            if model_type:
                return str(model_type).lower()
    return ""


def _mimo_v2_token_trace_enabled() -> bool:
    return os.environ.get("VMLINUX_MIMO_V2_TOKEN_TRACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _trace_mimo_v2_generated_token(
    generator: Any,
    request: Any,
    token_id: Any,
    *,
    phase: str,
    finish_reason: str | None = None,
    logprobs: Any = None,
) -> None:
    """Log MiMo generated token IDs for root-cause diagnostics only.

    This is opt-in instrumentation, not a decode policy. It is used to
    distinguish runtime stop/control-token handling from model/artifact logits
    quality when MiMo emits an empty one-token response.
    """

    if not _mimo_v2_token_trace_enabled():
        return
    if getattr(generator, "_model_type", "") != "mimo_v2":
        return
    try:
        token_int = int(token_id)
    except Exception:
        token_int = token_id
    tokenizer = getattr(generator.processor, "tokenizer", generator.processor)
    decoded = None
    if tokenizer is not None and isinstance(token_int, int):
        try:
            decoded = tokenizer.decode([token_int], skip_special_tokens=False)
        except TypeError:
            try:
                decoded = tokenizer.decode([token_int])
            except Exception:
                decoded = None
        except Exception:
            decoded = None
    top_tokens: list[dict[str, Any]] = []
    try:
        top_k = int(os.environ.get("VMLINUX_MIMO_V2_TOKEN_TRACE_TOPK", "0") or "0")
    except Exception:
        top_k = 0
    if top_k > 0 and logprobs is not None and tokenizer is not None:
        try:
            arr = logprobs
            if hasattr(arr, "ndim") and int(arr.ndim) > 1:
                arr = arr.reshape((-1,))
            values = arr.tolist() if hasattr(arr, "tolist") else list(arr)
            ranked = sorted(
                enumerate(values),
                key=lambda item: float(item[1]),
                reverse=True,
            )[:top_k]
            for cand_id, cand_score in ranked:
                try:
                    cand_decoded = tokenizer.decode(
                        [int(cand_id)],
                        skip_special_tokens=False,
                    )
                except TypeError:
                    cand_decoded = tokenizer.decode([int(cand_id)])
                except Exception:
                    cand_decoded = None
                top_tokens.append(
                    {
                        "id": int(cand_id),
                        "decoded": cand_decoded,
                        "score": float(cand_score),
                    }
                )
        except Exception as exc:
            top_tokens.append({"error": f"{type(exc).__name__}: {exc}"})
    stop_tokens = set(getattr(generator, "stop_tokens", set()) or set())
    token_ids = getattr(generator, "_mimo_v2_thinking_off_token_ids", {}) or {}
    logger.info(
        "MiMo-V2 token trace: request=%s phase=%s token_id=%r decoded=%r "
        "enable_thinking=%r output_len=%d is_stop=%s finish_reason=%r "
        "think_ids=%s eos_ids=%s stop_tokens=%s top_tokens=%s",
        getattr(request, "request_id", None),
        phase,
        token_int,
        decoded,
        getattr(request, "enable_thinking", None),
        len(getattr(request, "output_tokens", []) or []),
        token_int in stop_tokens if isinstance(token_int, int) else False,
        finish_reason,
        sorted(token_ids.get("think_ids", set()) or []),
        sorted(token_ids.get("eos_ids", set()) or []),
        sorted(stop_tokens),
        top_tokens,
    )


def _recompress_to_tq(cache: List[Any], language_model) -> List[Any]:
    """Re-wrap reconstructed KVCache layers into threshold-aware TQ objects.

    After paged cache reconstruction, KV layers are full-precision float16.
    If live encode is enabled, mirror the same thresholded transition used by
    generation. The current jang attention path retains a decoded float view,
    so this function does not claim a resident-memory reduction.
    1. Gets a TQ template from model.make_cache() to extract TQ config
    2. For each KVCache layer, creates a matching TurboQuantKVCache
    3. Copies keys/values and encodes only when the family gate is active

    Returns the cache list with KV layers replaced by compressed TQ layers.
    Non-KV layers (SSM/ArraysCache) pass through unchanged.
    """
    if not hasattr(language_model, 'make_cache'):
        return cache
    # Guard: cache must be a list, not a BatchKVCache or other non-list type.
    if not isinstance(cache, list):
        return cache

    try:
        from jang_tools.turboquant.cache import TurboQuantKVCache
        from mlx_lm.models.cache import KVCache
    except ImportError:
        return cache

    # Get template and check if it contains ANY TurboQuantKVCache objects.
    # This is more robust than checking make_cache.__name__ which can vary.
    try:
        template = language_model.make_cache()
    except Exception:
        return cache
    if not isinstance(template, list) or not any(type(t).__name__ == _TQ_CLASS_NAME for t in template):
        return cache  # No TQ layers in template — not a TQ model

    tq_count = 0
    encoded_count = 0
    resident_before = 0
    resident_after = 0
    # Deep spans: the per-layer encode below is heavy (a 90k-token RQ encode
    # per attention layer). Left lazy, ALL layers' encodes chain into one
    # command buffer together with any pending graph work — measured five
    # identical Metal OOM aborts at a ~96k hybrid-hit turn, ~32s of silent
    # GPU work between the hit acceptance and the abort, before the forward
    # ever logged. Optionally skip live recompression outright above a span
    # cap (0 = no cap): plain KVCache layers pass through and generation is
    # exact — this function claims no resident-memory reduction anyway.
    try:
        _tq_recompress_cap = max(
            0,
            int(
                os.environ.get("VMLX_TQ_RECOMPRESS_MAX_TOKENS", "0") or "0"
            ),
        )
    except (TypeError, ValueError):
        _tq_recompress_cap = 0
    result = list(cache)
    for i, layer in enumerate(result):
        if not isinstance(layer, KVCache):
            continue
        if layer.keys is None or layer.offset == 0:
            continue
        # Find matching TQ template layer
        if i >= len(template) or type(template[i]).__name__ != _TQ_CLASS_NAME:
            continue
        tpl = template[i]
        # Create TQ cache with actual tensor dimensions (not template dims).
        # MLA models have different key/value dims than template due to
        # compression (e.g. template key_dim=128, actual KV dim=256).
        actual_key_dim = layer.keys.shape[-1]
        actual_val_dim = layer.values.shape[-1]
        tq = TurboQuantKVCache(
            key_dim=actual_key_dim,
            value_dim=actual_val_dim,
            key_bits=tpl.key_bits,
            value_bits=tpl.value_bits,
            seed=getattr(tpl, "_seed", 42),
            compress_after=int(getattr(tpl, "compress_after", 0) or 0),
            sink_tokens=tpl.sink_tokens,
        )
        # Copy reconstructed data without changing the model's attention dtype.
        # These attributes also let TQ-native prompt serialization preserve the
        # dtype even when the cache has not crossed its live compression limit.
        tq.keys = layer.keys
        tq.values = layer.values
        tq._vmlx_tq_key_dtype = layer.keys.dtype
        tq._vmlx_tq_value_dtype = layer.values.dtype
        tq.offset = layer.offset
        tq.step = getattr(layer, 'step', layer.keys.shape[2]) if layer.keys.ndim >= 3 else layer.offset
        resident_before += int(layer.keys.nbytes + layer.values.nbytes)
        if (
            tq.compress_after > 0
            and tq.offset > tq.compress_after
            and (
                _tq_recompress_cap <= 0
                or int(tq.offset) <= _tq_recompress_cap
            )
        ):
            tq.compress(tq.compress_after)
            encoded_count += 1
            # Materialize THIS layer's encode before the next layer's graph
            # is built, bounding the transient to one layer instead of the
            # whole stack fused with ambient pending work.
            _pending = [
                _v
                for _v in (
                    getattr(tq, _n, None)
                    for _n in (
                        "keys",
                        "values",
                        "_decoded_k_buffer",
                        "_decoded_v_buffer",
                        "_joined_k",
                        "_joined_v",
                    )
                )
                if _v is not None and hasattr(_v, "dtype")
            ]
            if _pending:
                try:
                    mx.eval(*_pending)
                except Exception:  # noqa: BLE001
                    pass
        for name in (
            "keys", "values", "_decoded_k_buffer", "_decoded_v_buffer",
            "_joined_k", "_joined_v",
        ):
            value = getattr(tq, name, None)
            if value is not None and hasattr(value, "nbytes"):
                resident_after += int(value.nbytes)
        resident_after += int(getattr(tq, "compressed_nbytes", 0) or 0)
        result[i] = tq
        tq_count += 1

    if tq_count > 0:
        logger.info(
            "Re-wrapped %d KV layers as TurboQuant objects: encoded=%d, "
            "resident_before=%d, resident_after=%d, delta=%+d bytes; no "
            "resident-memory reduction claimed",
            tq_count,
            encoded_count,
            resident_before,
            resident_after,
            resident_after - resident_before,
        )
    return result


def _dequantize_cache(cache: List[Any]) -> List[Any]:
    """Dequantize QuantizedKVCache layers to KVCache for batch generation.

    BatchGenerator requires full-precision KVCache objects for merge/extract.
    Returns original cache unmodified if no quantized layers found.
    Recurses into CacheList sub-caches for MoE models.
    """
    try:
        from mlx_lm.models.cache import KVCache, QuantizedKVCache
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None
    except ImportError:
        return cache

    has_quantized = any(isinstance(c, QuantizedKVCache) for c in cache)
    has_cachelist = _CacheList is not None and any(isinstance(c, _CacheList) for c in cache)
    if not has_quantized and not has_cachelist:
        return cache

    result = []
    for layer_cache in cache:
        if _CacheList is not None and isinstance(layer_cache, _CacheList):
            # MoE: recurse into each sub-cache
            dequantized_subs = []
            for sc in layer_cache.caches:
                if isinstance(sc, QuantizedKVCache):
                    if sc.keys is not None:
                        try:
                            kv = KVCache()
                            kv.keys = mx.dequantize(
                                sc.keys[0], sc.keys[1],
                                sc.keys[2], sc.group_size, sc.bits,
                            )
                            kv.values = mx.dequantize(
                                sc.values[0], sc.values[1],
                                sc.values[2], sc.group_size, sc.bits,
                            )
                            kv.offset = sc.offset
                            dequantized_subs.append(kv)
                        except Exception as e:
                            logger.warning(f"KV dequantization failed in CacheList sub-cache: {e}")
                            return None
                    else:
                        dequantized_subs.append(KVCache())
                else:
                    dequantized_subs.append(sc)
            result.append(_CacheList(*dequantized_subs))
        elif isinstance(layer_cache, QuantizedKVCache):
            if layer_cache.keys is not None:
                try:
                    kv = KVCache()
                    kv.keys = mx.dequantize(
                        layer_cache.keys[0], layer_cache.keys[1],
                        layer_cache.keys[2], layer_cache.group_size, layer_cache.bits,
                    )
                    kv.values = mx.dequantize(
                        layer_cache.values[0], layer_cache.values[1],
                        layer_cache.values[2], layer_cache.group_size, layer_cache.bits,
                    )
                    kv.offset = layer_cache.offset
                    result.append(kv)
                except Exception as e:
                    logger.warning(f"KV dequantization failed: {e}, discarding cached prefix")
                    return None  # Caller should do full prefill instead of using broken cache
            else:
                # QuantizedKVCache with keys=None — empty layer, use fresh KVCache
                # (cannot pass QuantizedKVCache to BatchGenerator)
                result.append(KVCache())
        else:
            result.append(layer_cache)
    return result


def _validate_prompt_cache(cache: Any, *, source: str) -> bool:
    """Drop unsafe live prefix-cache objects before model forward."""
    try:
        from .cache_record_validator import reject_live_cache_or_warn
        return reject_live_cache_or_warn(cache, source=source)
    except Exception:
        return cache is not None and (not isinstance(cache, list) or len(cache) > 0)


def _cache_has_materialized_state(cache: Any) -> bool:
    """Return true when a reconstructed prefix owns at least one real tensor.

    A cache list full of freshly constructed KV/rotating objects has the right
    classes and layer count but represents zero prefix tokens. Such a list must
    never be accepted for a non-zero cache hit.
    """
    seen: set[int] = set()

    def _contains_tensor(value: Any) -> bool:
        if value is None:
            return False
        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return all(int(dim) > 0 for dim in shape)
            except Exception:
                return True
        if isinstance(value, dict):
            return any(_contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(_contains_tensor(item) for item in value)
        for attr in ("state", "keys", "values", "cache", "caches"):
            try:
                child = getattr(value, attr, None)
            except Exception:
                child = None
            if child is not None and child is not value and _contains_tensor(child):
                return True
        return False

    return _contains_tensor(cache)


def _block_table_block_count(block_table: Any) -> int:
    """Return the number of blocks represented by a paged-cache table.

    The production ``BlockTable`` field is ``block_ids``. Keep the legacy
    ``blocks`` fallback only for compatibility with older test doubles.
    """
    block_ids = getattr(block_table, "block_ids", None)
    if block_ids is not None:
        return len(block_ids)
    return len(getattr(block_table, "blocks", []) or [])


def _fix_hybrid_cache(
    cache: List[Any],
    language_model: nn.Module,
    kv_positions: Optional[List[int]] = None,
    num_model_layers: Optional[int] = None,
) -> List[Any]:
    """Fix reconstructed cache for hybrid models (SSM + attention layers).

    Prefix cache stores ONLY KVCache (attention) layers — SSM/ArraysCache layers
    are cumulative state and get skipped during extraction. This means the
    reconstructed cache list has fewer entries than total model layers.

    For example, Qwen3.5 9B has 32 layers (8 attention + 24 SSM), but prefix
    cache only stores the 8 attention layers. The reconstructed list of 8 must
    be expanded back to 32 by inserting fresh ArraysCache at SSM positions.

    Args:
        cache: Reconstructed cache list (may be shorter than model layers)
        language_model: The language model (for make_cache() template)
        kv_positions: Pre-computed KVCache layer indices (skips recomputation)
        num_model_layers: Pre-computed total layer count
    """
    if not hasattr(language_model, 'make_cache'):
        return cache
    # Guard: cache must be a list, not BatchKVCache or other non-list type
    if not isinstance(cache, list):
        return cache

    try:
        from mlx_lm.models.cache import KVCache

        # Fast path: use pre-computed positions to check if fix is needed
        if kv_positions is not None and num_model_layers is not None:
            # Not a hybrid model (all layers are KVCache) — no fix needed
            if len(kv_positions) == num_model_layers:
                return cache
            # Cache already correct length — still need type-mismatch check below
            if len(cache) == num_model_layers:
                pass  # Fall through to type-mismatch repair at line 203+
            # Cache length doesn't match expected KV layer count — return fresh cache
            elif len(cache) != len(kv_positions):
                logger.warning(
                    f"Cache length mismatch: {len(cache)} reconstructed vs "
                    f"{len(kv_positions)} KV positions in {num_model_layers}-layer model, "
                    "returning fresh cache"
                )
                return language_model.make_cache()

        # Need make_cache() for fresh SSM objects at non-KV positions
        template = language_model.make_cache()
        n_layers = len(template)

        if len(cache) == n_layers:
            # Same length — check for type mismatches (KVCache at SSM positions)
            fixed = False
            result = list(cache)
            for i, (tmpl, cached) in enumerate(zip(template, cache)):
                if (
                    not _is_attention_cache_slot(tmpl)
                    and _is_attention_cache_slot(cached)
                ):
                    result[i] = tmpl
                    fixed = True
            if fixed:
                logger.debug("Fixed hybrid cache: replaced KVCache at SSM positions")
            return result

        # Cache shorter than model — expand using template
        positions = kv_positions if kv_positions is not None else [
            i for i, t in enumerate(template) if _is_attention_cache_slot(t)
        ]
        if len(cache) != len(positions):
            logger.warning(
                f"Cache length mismatch: {len(cache)} reconstructed vs "
                f"{len(positions)} KV positions in {n_layers}-layer model, "
                "returning fresh cache"
            )
            return template

        result = list(template)
        for cache_idx, model_idx in enumerate(positions):
            result[model_idx] = cache[cache_idx]

        logger.debug(
            f"Expanded hybrid cache: {len(cache)} KV layers -> "
            f"{n_layers} total ({len(positions)} KV + "
            f"{n_layers - len(positions)} SSM)"
        )
        return result
    except Exception as e:
        logger.warning(f"_fix_hybrid_cache failed: {e}, returning fresh cache")
        if hasattr(language_model, 'make_cache'):
            return language_model.make_cache()
        return cache


# SSM companion cache moved to vmlx_engine/utils/ssm_companion_cache.py per
# REQ-A3-001 (option C — Agent 3 owns the file forever, Agent 2 owns the
# call sites). Imported here as `HybridSSMStateCache` (back-compat alias) so
# existing call sites in this file and `scheduler.py` continue to work.
# The new class signature is:
#   store(token_ids, num_tokens, ssm_states, is_complete: bool = True)
#   fetch(token_ids, num_tokens) -> Optional[Tuple[List[Any], bool]]
# Fetch sites in this file have been updated to unpack the new tuple.
# See `agentprogress/3/notes-to-2.md` for the migration guide and
# `agentprogress/2/decisions.md` D-A2-007 for the option-C rationale.
from .utils.ssm_companion_cache import (  # noqa: F401
    DEFAULT_SSM_COMPANION_ENTRIES,
    HybridSSMStateCache,
    SSMCompanionCache,
    make_ssm_prefix_lookup,
    normalize_ssm_telemetry_request_id,
    sanitize_ssm_prefix_lookup,
)


def _ssm_telemetry_attr(value: Any, name: str, default: Any = None) -> Any:
    """Read optional telemetry without allowing descriptors to affect serving."""

    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _ssm_telemetry_store_size(state_cache: Any) -> int:
    try:
        return int(_ssm_telemetry_attr(state_cache, "size", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _attach_request_ssm_prefix_lookup(
    request: Any,
    lookup: Dict[str, Any],
) -> Dict[str, Any]:
    """Overwrite stale request telemetry with one normalized lookup record."""

    request_id = normalize_ssm_telemetry_request_id(
        _ssm_telemetry_attr(request, "request_id")
    )
    current = dict(lookup)
    current["request_id"] = request_id
    execution = dict(_ssm_telemetry_attr(request, "_cache_execution") or {})
    execution["request_id"] = request_id
    execution["ssm_prefix_lookup"] = current
    request._cache_execution = execution
    return current


def _record_request_ssm_exact_lookup(
    request: Any,
    state_cache: Any,
    *,
    max_len: int,
    matched: bool,
    is_complete: bool,
) -> Dict[str, Any]:
    """Attach the actual exact-boundary lookup used by the fast path."""

    lookup = make_ssm_prefix_lookup(
        max_len=max_len,
        candidate_lengths=[max_len] if max_len > 0 and matched else [],
        attempted_candidate_lengths=[max_len] if max_len > 0 else [],
        matched=max_len > 0 and matched,
        checkpoint_tokens=max_len if matched else 0,
        is_complete=is_complete if matched else False,
        source=(
            "exact_boundary_l1_or_l2"
            if max_len > 0 and matched
            else "none"
        ),
        reason=(
            "matched"
            if max_len > 0 and matched
            else "candidate_fetch_miss"
            if max_len > 0
            else "non_positive_max_len"
        ),
        store_size=_ssm_telemetry_store_size(state_cache),
    )
    return _attach_request_ssm_prefix_lookup(request, lookup)


def _fetch_request_ssm_longest_prefix(
    request: Any,
    state_cache: Any,
    *,
    enabled: bool,
    token_ids: List[int],
    max_len: int,
    cache_extra_keys: Any = None,
    exact_boundary_already_missed: bool = False,
) -> Tuple[Optional[Tuple[int, List[Any], bool]], Dict[str, Any]]:
    """Fetch and attach fresh request-owned, path-free SSM lookup telemetry."""

    request_id = normalize_ssm_telemetry_request_id(
        _ssm_telemetry_attr(request, "request_id")
    )
    fetch_property_failed = False
    try:
        fetch_fn = getattr(state_cache, "fetch_longest_prefix", None)
    except Exception:
        fetch_fn = None
        fetch_property_failed = True
    store_size = _ssm_telemetry_store_size(state_cache)
    result: Optional[Tuple[int, List[Any], bool]] = None
    if not enabled:
        lookup = make_ssm_prefix_lookup(
            max_len=max_len,
            reason="ssm_prefix_resume_disabled",
            store_size=store_size,
            request_id=request_id,
        )
    elif fetch_property_failed:
        lookup = make_ssm_prefix_lookup(
            max_len=max_len,
            reason="lookup_exception",
            store_size=store_size,
            request_id=request_id,
        )
    elif not callable(fetch_fn):
        lookup = make_ssm_prefix_lookup(
            max_len=max_len,
            reason="lookup_unavailable",
            store_size=store_size,
            request_id=request_id,
        )
    elif max_len <= 0:
        lookup = make_ssm_prefix_lookup(
            max_len=0,
            reason="non_positive_max_len",
            store_size=store_size,
            request_id=request_id,
        )
    else:
        try:
            fetch_kwargs = {"cache_extra_keys": cache_extra_keys}
            if exact_boundary_already_missed:
                fetch_kwargs["exact_boundary_already_missed"] = True
            result = fetch_fn(token_ids, max_len, **fetch_kwargs)
        except Exception:
            lookup = make_ssm_prefix_lookup(
                max_len=max_len,
                attempted_candidate_lengths=[max_len],
                reason="lookup_exception",
                store_size=store_size,
                request_id=request_id,
            )
        else:
            lookup = sanitize_ssm_prefix_lookup(
                _ssm_telemetry_attr(state_cache, "last_prefix_lookup"),
                request_id=request_id,
                fallback_max_len=max_len,
                fallback_store_size=store_size,
                fallback_attempted_candidate_lengths=[max_len],
            )
            result_matches_lookup = (
                lookup.get("max_len") == max_len
                and (
                    (
                        result is None
                        and lookup.get("matched") is False
                        and lookup.get("checkpoint_tokens") == 0
                    )
                    or (
                        isinstance(result, tuple)
                        and len(result) == 3
                        and lookup.get("matched") is True
                        and type(result[0]) is int
                        and lookup.get("checkpoint_tokens") == result[0]
                        and lookup.get("is_complete") == bool(result[2])
                    )
                )
            )
            if not result_matches_lookup:
                lookup = make_ssm_prefix_lookup(
                    max_len=max_len,
                    attempted_candidate_lengths=[max_len],
                    reason="malformed_lookup",
                    store_size=store_size,
                    request_id=request_id,
                )

    lookup = _attach_request_ssm_prefix_lookup(request, lookup)
    return result, lookup


@dataclass
class MLLMNativeMTPStats:
    cycles: int = 0
    accepts: int = 0
    rejects: int = 0
    init_emits: int = 0
    draft_emits: int = 0
    bonus_emits: int = 0
    verify_emits: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    # Cycles where the confidence gate stopped the draft chain early. Reported
    # so a threshold that never fires -- or fires on every cycle -- is visible
    # rather than something to infer from the throughput.
    margin_truncated_cycles: int = 0
    verify_ms: float = 0.0
    sample_ms: float = 0.0
    draft_ms: float = 0.0
    snapshot_ms: float = 0.0
    restore_ms: float = 0.0
    replay_ms: float = 0.0
    materialize_ms: float = 0.0
    accepted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    drafted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    seed_main_forwards: int = 0
    verify_main_forwards: int = 0
    replay_main_forwards: int = 0
    mtp_forwards: int = 0
    # Calls actually routed through the optional four-row q4 verifier for this
    # request.  A process-wide installed flag is insufficient because q6 and
    # other ineligible artifact layouts correctly fall back to stock MLX.
    verify_qmm_calls: int = 0
    # Calls actually routed through an optional proposal-only head for this
    # request. The loaded model exposes a cumulative counter; draft-chain
    # boundaries below record only the request-local delta.
    draft_head_calls: int = 0
    draft_head: Dict[str, Any] = field(default_factory=dict)
    # MLLM native MTP recreates the head cache after verifier rejection.
    # This deliberately does not claim to count every cache discard/reset.
    mtp_cache_recreated_on_rejects: int = 0
    mtp_cache_retained_on_rejects: int = 0
    mtp_head_cache: Dict[str, Any] = field(default_factory=dict)
    adaptive_depth_value: Dict[str, Any] = field(default_factory=dict)
    profile_seed: str = ""
    profile_key_label: str = ""
    prompt_primed_pairs: int = 0
    prompt_prime_source: str = "unprimed"
    stochastic_distribution_cycles: int = 0
    stochastic_ratio_checks: int = 0
    stochastic_ratio_accepts: int = 0
    stochastic_residual_corrections: int = 0

    def to_dict(
        self,
        *,
        request_id: str,
        finish_reason: str,
        final_depth: int,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        def _rate(accepted: int, drafted: int) -> Optional[float]:
            if drafted <= 0:
                return None
            return accepted / drafted

        timings = {
            "verify": self.verify_ms,
            "sample": self.sample_ms,
            "draft": self.draft_ms,
            "snapshot": self.snapshot_ms,
            "restore": self.restore_ms,
            "replay": self.replay_ms,
            "materialize": self.materialize_ms,
        }
        total_ms = sum(float(value or 0.0) for value in timings.values())
        timings["total"] = total_ms
        timings["avg_cycle"] = total_ms / max(1, int(self.cycles or 0))

        depth_rates = {}
        for index, label in enumerate(("d1", "d2", "d3")):
            accepted = (
                int(self.accepted_by_depth[index])
                if index < len(self.accepted_by_depth)
                else 0
            )
            drafted = (
                int(self.drafted_by_depth[index])
                if index < len(self.drafted_by_depth)
                else 0
            )
            depth_rates[label] = _rate(accepted, drafted)

        draft_head = dict(self.draft_head)
        if draft_head:
            draft_head["calls"] = int(self.draft_head_calls)
            draft_head["active_observed"] = bool(self.draft_head_calls)

        return {
            "request_id": request_id,
            "finish_reason": finish_reason,
            "final_depth": int(final_depth or 1),
            "cycles": int(self.cycles),
            "accepts": int(self.accepts),
            "rejects": int(self.rejects),
            "init_emits": int(self.init_emits),
            "draft_emits": int(self.draft_emits),
            "bonus_emits": int(self.bonus_emits),
            "verify_emits": int(self.verify_emits),
            "drafted_tokens": int(self.drafted_tokens),
            "accepted_tokens": int(self.accepted_tokens),
            "margin_truncated_cycles": int(self.margin_truncated_cycles),
            "acceptance_rate": _rate(
                int(self.accepted_tokens),
                int(self.drafted_tokens),
            ),
            "accepted_by_depth": list(self.accepted_by_depth),
            "drafted_by_depth": list(self.drafted_by_depth),
            "depth_acceptance_rates": depth_rates,
            "forwards": {
                "seed_main": int(self.seed_main_forwards),
                "verify_main": int(self.verify_main_forwards),
                "replay_main": int(self.replay_main_forwards),
                "mtp": int(self.mtp_forwards),
            },
            "verify_qmm": {
                "calls": int(self.verify_qmm_calls),
                "accelerated": bool(self.verify_qmm_calls),
            },
            "draft_head": draft_head,
            "timings_ms": timings,
            "cache_lifecycle": native_mtp_cache_lifecycle_snapshot(
                head_cache=self.mtp_head_cache,
                recreated_on_rejects=self.mtp_cache_recreated_on_rejects,
                retained_on_rejects=self.mtp_cache_retained_on_rejects,
            ),
            "adaptive_depth_value": dict(self.adaptive_depth_value),
            "profile_seed": self.profile_seed,
            "profile_key": self.profile_key_label,
            "prompt_priming": {
                "source": self.prompt_prime_source,
                "folded_pairs": int(self.prompt_primed_pairs),
            },
            "stochastic_verify": {
                "distribution_cycles": int(self.stochastic_distribution_cycles),
                "ratio_checks": int(self.stochastic_ratio_checks),
                "ratio_accepts": int(self.stochastic_ratio_accepts),
                "residual_corrections": int(
                    self.stochastic_residual_corrections
                ),
            },
            "profiled_phase_timing": _native_mtp_trace_enabled(),
            "fallback_reason": fallback_reason,
        }


def _native_mtp_draft_head_status(language_model: Any) -> Dict[str, Any]:
    getter = getattr(language_model, "mtp_draft_head_status", None)
    if not callable(getter):
        return {}
    try:
        status = getter()
    except Exception:  # noqa: BLE001 - telemetry must never break generation
        return {}
    return dict(status) if isinstance(status, dict) else {}


def _record_native_mtp_draft_head_delta(
    stats: MLLMNativeMTPStats,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> None:
    if not after:
        return
    before_calls = int(before.get("calls", 0) or 0)
    after_calls = int(after.get("calls", 0) or 0)
    stats.draft_head_calls += max(0, after_calls - before_calls)
    stats.draft_head = dict(after)


@dataclass
class MLLMNativeMTPState:
    """Private draft/verify state for one native-MTP MLLM request."""

    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)
    mtp_cache: Optional[List[Any]] = None
    next_main: Optional[Any] = None
    drafts: List[Any] = field(default_factory=list)
    draft_lps: List[Any] = field(default_factory=list)
    draft_ids: List[int] = field(default_factory=list)
    depth: int = 1
    stats: MLLMNativeMTPStats = field(default_factory=MLLMNativeMTPStats)
    ar_fallback_pending: bool = False
    ar_fallback_reason: Optional[str] = None
    # Highest depth this request is allowed to probe.  This is a capability
    # ceiling, not a one-way latch: workload phases can make a previously slow
    # depth profitable again, so the rolling wall-value controller may retry it
    # after a cooldown.
    depth_ceiling: int = 3
    adaptive_value: NativeMTPAdaptiveValueState = field(
        default_factory=NativeMTPAdaptiveValueState
    )
    # Aligned head cache: chain pairs (deeper-level drafts) appended to the
    # head cache during the last draft phase.  Trimmed after every verify so an
    # unverified draft can never persist in the head's context.
    head_chain_pairs: int = 0
    # In-flight prefetched verify: {snapshot, logits, hidden, n_inputs} or None.
    pending_verify: Optional[Dict[str, Any]] = None
    # Snapshot and emission ledger for the verify cycle currently being
    # drained. If max_tokens or a stop token lands before all accepted drafts
    # are emitted, the main cache must rewind to the exact visible boundary.
    terminal_snapshot: Optional[List[Optional["_NativeMTPCacheObjectSnapshot"]]] = None
    terminal_n_inputs: int = 0
    terminal_base_token: Optional[Any] = None
    terminal_emitted: List[Tuple[int, str]] = field(default_factory=list)
    # Runtime cost telemetry (wall-clock, trace-free): the seed forward is a
    # true AR step, and the span since the first verify cycle divided by
    # emitted tokens is the real MTP cost per token.
    ar_step_ms: float = 0.0
    cycle_span_start: float = 0.0
    # The request began from a restored prefix. The MTP head cache starts
    # COLD on such requests (backbone hiddens are not stored), so the first
    # gate windows measure a context-starved head: run 3 of a live A/B
    # demoted D3->D1 at cycle 129 on d2=0.574 that recovers to ~0.85 once
    # the head warms, and the lowered ceiling made 17.4 t/s permanent.
    restored_prefix: bool = False
    # Session-profile identity this request seeds from / reports back to.
    profile_key: Optional[Tuple[str, bool, str]] = None


@dataclass
class _NativeMTPCacheObjectSnapshot:
    obj: Any
    attrs: Dict[str, Any]


def _native_mtp_depth() -> int:
    from .native_mtp import native_mtp_effective_depth

    depth, _source = native_mtp_effective_depth()
    return depth


def _native_mtp_depth_for_request(request: Any) -> int:
    """Return the configured starting depth for every request surface.

    Tool metadata is not a decode boundary. The prior blanket D1 cap slowed
    every tool-enabled app turn, including ordinary reasoning and prose before
    a tool call. Exact terminal-boundary rewind now owns the real hazard.
    """

    return _native_mtp_depth()


def _native_mtp_depth_ceiling_for_request(request: Any) -> int:
    """Return the verifier-supported adaptive-depth ceiling."""

    return 3


def _native_mtp_logprobs(logits_2d: mx.array) -> mx.array:
    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _native_mtp_sampler_accepts_logits(sampler: Callable[[mx.array], mx.array]) -> bool:
    return bool(getattr(sampler, "_vmlx_accepts_logits", False))


def _native_mtp_sampler_is_greedy(sampler: Callable[[mx.array], mx.array]) -> bool:
    return bool(getattr(sampler, "_vmlx_is_greedy", False))


def _native_mtp_hidden_tensor(hidden_states: Any) -> Any:
    """Return the final hidden tensor from text or VLM output contracts."""

    if isinstance(hidden_states, (list, tuple)):
        hidden_states = next(
            (item for item in reversed(hidden_states) if item is not None),
            None,
        )
    if hidden_states is None:
        raise RuntimeError("native MTP forward returned no hidden states")
    return hidden_states


def _native_mtp_default_draft_margin_threshold(model_type: Optional[str]) -> float:
    """Use the measured confidence gate only for Qwen4 fixed-D3.

    Fixed D3 remains the configured ceiling, but a low-confidence first
    proposal must not force two more draft-head forwards.  The same 1.0 gap
    gate beat AR on sampled-prose A/Bs for Qwen4 JANG_6S and JANG_4M.
    Adaptive mode has its own value controller and regressed when composed
    with this gate, so its default remains disabled.
    """

    if str(model_type or "").lower() != "qwen4_exp":
        return 0.0

    adaptive = _native_mtp_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH",
        "VMLX_NATIVE_MTP_ADAPTIVE_DEPTH",
    )
    depth = _native_mtp_env_int(
        3,
        "VMLINUX_NATIVE_MTP_DEPTH",
        "VMLX_NATIVE_MTP_DEPTH",
        minimum=1,
    )
    return 1.0 if not adaptive and depth >= 3 else 0.0


def _native_mtp_draft_margin_threshold(model_type: Optional[str] = None) -> float:
    """Logit gap below which the draft chain stops extending. 0 disables.

    Speculative decoding pays for a draft whether or not it is accepted. On
    high-entropy positions -- reasoning and prose, where measured acceptance
    falls to 44-58% against 85-98% on code and counting -- deep chains spend
    head forwards on tokens that are about to be rejected. Measured on
    Qwen3.8-27B at depth 3: code with thinking off runs 51.5 t/s, the same
    prompt with thinking on runs 28.1. The difference is entropy, not the
    engine.

    The head's own top-1-minus-top-2 logit gap is a cheap confidence proxy for
    that, available from logits already computed. It costs one reduction over
    the vocabulary next to a head forward that is a full transformer layer plus
    a 5120x248320 projection.
    """
    raw = os.environ.get("VMLINUX_NATIVE_MTP_DRAFT_MARGIN") or os.environ.get(
        "VMLX_NATIVE_MTP_DRAFT_MARGIN"
    )
    default = _native_mtp_default_draft_margin_threshold(model_type)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid native-MTP draft margin %r; using %.2f",
            raw,
            default,
        )
        return default
    return max(0.0, value)


def _native_mtp_top2_margin(logits_2d: mx.array) -> mx.array:
    """Top-1 minus top-2 logit for the final position, as a 0-d array.

    Deliberately NOT a full sort. The sampler's full-vocabulary sort was
    measured at roughly 30% of decode on wide-vocabulary bundles, so this takes
    a k=2 partial reduction instead.
    """
    row = logits_2d[-1] if logits_2d.ndim > 1 else logits_2d
    top2 = mx.topk(row, 2)
    return mx.abs(top2[..., 0] - top2[..., 1])


def _native_mtp_sample_one(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[mx.array, Optional[mx.array]]:
    if _native_mtp_sampler_accepts_logits(sampler):
        token = _native_mtp_ensure_uint32(sampler(logits_2d))
        if _native_mtp_sampler_is_greedy(sampler):
            return token, None
        return token, _native_mtp_logprobs(logits_2d).squeeze(0)
    logprobs = _native_mtp_logprobs(logits_2d)
    token = _native_mtp_ensure_uint32(sampler(logprobs))
    return token, logprobs.squeeze(0)


def _native_mtp_sample_rows(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
    also_eval: Tuple[Any, ...] = (),
) -> Tuple[List[mx.array], List[Optional[mx.array]], List[int]]:
    """Sample one row per verify position and read the ids back to the host.

    ``also_eval`` is materialized in the SAME ``mx.eval`` as the sampled
    tokens.  A verify cycle otherwise pays two separate blocking round-trips —
    one here and one in ``_native_mtp_materialize_draft_ids`` — and each drains
    the device before the host can continue.  Folding them into one sync halves
    the per-cycle stalls; the caller's later ``mx.eval`` on the same arrays is
    then a no-op.
    """
    if _native_mtp_sampler_accepts_logits(sampler):
        sampled = _native_mtp_ensure_uint32(sampler(logits_2d))
        normalized = (
            None
            if _native_mtp_sampler_is_greedy(sampler)
            else _native_mtp_logprobs(logits_2d)
        )
        if normalized is None:
            mx.eval(sampled, *also_eval)
        else:
            mx.eval(sampled, normalized, *also_eval)
        sampled_ids = [int(value) for value in sampled.tolist()]
        rows = int(sampled.shape[0])
        if normalized is None:
            logprobs = [None] * rows
        else:
            logprobs = [normalized[i] for i in range(rows)]
        return [sampled[i : i + 1] for i in range(rows)], logprobs, sampled_ids
    logprobs = _native_mtp_logprobs(logits_2d)
    sampled = _native_mtp_ensure_uint32(sampler(logprobs))
    mx.eval(sampled, *also_eval)
    sampled_ids = [int(value) for value in sampled.tolist()]
    rows = int(sampled.shape[0])
    return [sampled[i : i + 1] for i in range(rows)], [
        logprobs[i] for i in range(rows)
    ], sampled_ids


def _native_mtp_sample_and_decide(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
    drafts: List[Any],
    depth: int,
):
    """Sample the verify rows and decide acceptance in ONE device round-trip.

    Returns ``(target_tokens, target_lps, target_ids, draft_ids, accepted)``.

    Only the logits-consuming samplers (greedy / compact-top-k, i.e. exactly
    the ones MTP runs under) take this path; stochastic samplers need the full
    log-prob rows on the host anyway, so they keep the original two-step route.
    """
    sampled = _native_mtp_ensure_uint32(sampler(logits_2d))
    bundle, _ = _native_mtp_decision_bundle(sampled, drafts, depth)
    mx.eval(bundle)
    flat = [int(v) for v in bundle.tolist()]
    target_ids = flat[: depth + 1]
    draft_ids = flat[depth + 1 : depth + 1 + depth]
    accepted = flat[-1]
    rows = int(sampled.shape[0])
    target_tokens = [sampled[i : i + 1] for i in range(rows)]
    return target_tokens, [None] * rows, target_ids, draft_ids, accepted


def _native_mtp_decision_bundle(
    sampled: mx.array,
    drafts: List[Any],
    depth: int,
) -> Tuple[mx.array, int]:
    """Build one small device array carrying the whole verify decision.

    Recreates the shape of MTPLX's decode loop, which submits the verify and
    then reads back a single tiny "decision bundle" holding the sampled tokens,
    the drafts, and the accept flags -- the comparison happens ON DEVICE.

    vMLX instead pulled the sampled ids to the host with ``.tolist()``, pulled
    the draft ids back in a SECOND blocking eval, and then compared them in
    Python.  Two device round-trips per cycle, each draining the queue and
    stalling the host while the GPU idles.  At depth 1 that stall amortizes
    over the fewest emitted tokens, which is exactly where it hurts most.

    Layout: ``[sampled(depth+1) | drafts(depth) | accepted_count(1)]``.
    ``accepted_count`` is the LEADING run of matches, computed with a cumulative
    product so a mismatch zeroes everything after it.
    """
    draft_arr = mx.concatenate(
        [_native_mtp_ensure_uint32(tok).reshape(1) for tok in drafts]
    )
    target_head = sampled[:depth].astype(mx.int32)
    matches = (target_head == draft_arr.astype(mx.int32)).astype(mx.int32)
    # cumprod: 1 until the first mismatch, 0 from there on -> sum == leading run
    accepted = mx.cumprod(matches).sum().reshape(1).astype(mx.uint32)
    bundle = mx.concatenate(
        [sampled.astype(mx.uint32), draft_arr.astype(mx.uint32), accepted]
    )
    return bundle, depth


# Speculative rejection sampling for MTP drafts.
#
# Exact-match acceptance is only the correct test under greedy decode.  At
# temperature > 0 the target and the draft head each draw independently, so a
# draft that is a perfectly good sample from the target distribution is thrown
# away merely for disagreeing with the target's own draw.  That collapse is the
# reason MTP bundles have to pin temperature to 0 today.
#
# Standard speculative sampling instead accepts a draft x with probability
# min(1, p_target(x) / p_draft(x)).  Upstream (ml-explore/mlx-lm PR #990)
# reports this recovers greedy-level acceptance at temperature: 84.8% at
# temp 0.6 vs 88.3% at temp 0 on Qwen3.5-27B 4-bit, which is why upstream never
# needs to pin temperature.
#
# Default ON.  Both the text scheduler and this MLLM scheduler now use the
# shared speculative rejection-sampling rule, so a sampled request preserves
# the target sampler distribution instead of silently dropping to AR.  Keep an
# environment kill switch for exact rollback if a model-specific live gate
# exposes a defect.
_NATIVE_MTP_STOCHASTIC_ACCEPT = os.environ.get(
    "VMLX_MTP_STOCHASTIC_ACCEPT", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# Fold the draft-id materialization into the verify sample's eval so a cycle
# pays ONE blocking device round-trip instead of two.  Each blocking eval
# drains the device and stalls the host, and at depth 1 that stall amortizes
# over the fewest emitted tokens.
#
# Default OFF: semantically identical (same arrays, one eval instead of two)
# but NOT yet proven to move the needle, and run-to-run spread on this path is
# wide enough (14.1 vs 24.0 t/s on identical config) that nothing ships here
# without an N>=3 settled A/B.  VMLX_MTP_FUSED_SYNC=1 enables it.
_NATIVE_MTP_FUSED_SYNC = os.environ.get(
    "VMLX_MTP_FUSED_SYNC", "0"
).strip().lower() not in {"0", "false", "no", "off"}


# Skip the replay forward on rejection by rolling the hybrid SSM state back to
# the post-confirmed snapshot the model can capture for us.
#
# Every rejected cycle currently pays a FULL extra backbone forward: the verify
# is rolled back to BEFORE it ran, then the confirmed tokens are re-run to
# rebuild their state.  Measured on Qwen3.8-27B-JANG_4D-CRACK: verify 42.3ms,
# draft 2.47ms, replay ~40ms -- so a reject costs 84.8ms against an accept's
# 44.8ms, and the logs show `replay_main=28` for 64 cycles.
#
# The replay is unnecessary.  Passing ``n_confirmed`` to the verify forward makes
# GatedDeltaNet process the confirmed prefix separately and stash the resulting
# (conv, ssm) pair as ``cache.rollback_state`` (patches/mlx_vlm_mtp/qwen35_vl.py).
# Restoring THAT instead of the pre-verify snapshot lands exactly where the
# replay would have, and the corrected token's hidden state is already sitting in
# the verify output.  A raw harness doing precisely this runs 26.8 t/s where the
# engine runs 22.1.
#
# At depth 1 ``n_confirmed=1`` is exact: accept keeps the cache untouched, reject
# restores the rollback point.  Deeper drafts fall back to the replay path.
#
# VMLX_MTP_SKIP_REPLAY=1 enables.
# Default ON since 2026-08-18: proven live across many runs (replay_main
# 391 -> 0, output byte-correct), and it changes MTP's economics - a rejected
# cycle costs the same as an accepted one, so the profitability floor drops
# from ~0.68 to roughly the draft cost (~6%).
_NATIVE_MTP_SKIP_REPLAY = os.environ.get(
    "VMLX_MTP_SKIP_REPLAY", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def _native_mtp_rollback_to_confirmed(
    cache: List[Any],
    reject_tokens: int,
    accepted_drafts: int = 0,
) -> bool:
    """Roll back a rejected verify without re-running the confirmed tokens.

    SSM layers restore the ``rollback_state`` captured after the confirmed
    prefix (``accepted_drafts == 0``) or recompute the state after the
    accepted drafts through the layer-bound ``rollback_to`` closure — the
    SAME chunk kernel the forward used, seeded with the post-confirmed
    states, so the result is exactly what a replay would rebuild.  Plain KV
    layers simply trim the rejected positions.  Returns False if any layer
    cannot be rolled back this way, so the caller can fall back to the
    snapshot+replay path rather than continue on a corrupt cache.

    This is the same design the DFlash2 lane runs on every cycle; keeping
    it depth-1-only made every deeper rejection pay a full main-model
    replay forward (32% of cold cycles, 61% warm — measured live).
    """
    if reject_tokens <= 0:
        return False
    accepted_drafts = max(0, int(accepted_drafts))
    for layer in cache:
        if layer is None:
            continue
        rollback = getattr(layer, "rollback_state", None)
        rollback_to = getattr(layer, "rollback_to", None)
        rollback_aux = getattr(layer, "rollback_aux_state", None)
        rollback_aux_to = getattr(layer, "rollback_aux_to", None)
        if rollback is not None or rollback_to is not None:
            if accepted_drafts > 0:
                if rollback_to is None:
                    return False
                try:
                    conv_state, ssm_state = rollback_to(accepted_drafts)
                except Exception:
                    return False
            else:
                if rollback is None:
                    return False
                try:
                    conv_state, ssm_state = rollback
                except (TypeError, ValueError):
                    return False
            # Write through __setitem__, exactly as the model does
            # (`cache[0] = conv_f; cache[1] = ssm_f`).  Assigning `.state`
            # replaces the backing list with a TUPLE, and the next forward's
            # indexed write then dies with "'tuple' object does not support
            # item assignment".  ArraysCache carries no offset -- the SSM state
            # IS the position -- so nothing else needs rewinding here.
            try:
                layer[0] = conv_state
                layer[1] = ssm_state
            except (TypeError, IndexError, AttributeError):
                return False
            if rollback_aux is not None or rollback_aux_to is not None:
                try:
                    aux_state = (
                        rollback_aux_to(accepted_drafts)
                        if accepted_drafts > 0
                        else rollback_aux
                    )
                    layer[2], layer[3] = aux_state
                except (TypeError, ValueError, IndexError, AttributeError):
                    return False
            layer.rollback_state = None
            if rollback_to is not None:
                layer.rollback_to = None
            if rollback_aux is not None:
                layer.rollback_aux_state = None
            if rollback_aux_to is not None:
                layer.rollback_aux_to = None
            continue
        if hasattr(layer, "is_trimmable") and layer.is_trimmable():
            layer.trim(reject_tokens)
            continue
        return False
    return True


# Whether the MTP head's KV cache survives ACCEPTED cycles.  Retention sounds
# right but the accumulated context is gappy (bonus tokens never pass through
# the head), and measured live it nearly halves depth-2 acceptance (74.6% fresh
# vs 41.7% retained) while being neutral at depth 1.  Default: fresh per cycle.
_NATIVE_MTP_RETAIN_HEAD_CACHE = os.environ.get(
    "VMLX_MTP_RETAIN_HEAD_CACHE", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# ALIGNED head cache — upstream PR #990's "batched MTP cache commits ...
# maintaining cache alignment between backbone and MTP head", which vMLX never
# implemented.  Every confirmed token's (backbone_hidden_i, token_{i+1}) pair
# is committed through the head in the SAME forward that drafts the next token
# (the level-0 draft samples from the last position, so a multi-token input is
# free).  The head then always drafts with complete, correctly-paired context
# instead of one fused pair (fresh mode) or a history with a hole at every
# bonus token (retain mode).  Chain pairs from deeper draft levels are trimmed
# after each verify so a rejected draft can never poison the cache — the exact
# failure ledger 343 measured (retained d2 16.7% vs recreate 66.7%).
# VMLX_MTP_ALIGNED_HEAD_CACHE=0 reverts to the fresh-per-cycle behaviour.
_NATIVE_MTP_ALIGNED_HEAD_CACHE = os.environ.get(
    "VMLX_MTP_ALIGNED_HEAD_CACHE", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# Verify prefetch: submit the NEXT verify forward (async) the moment a cycle's
# decision refills the emit queue, so the GPU crunches it while Python spends
# ~5ms per token on detokenize/stream/stats for the 2-3 queued tokens.  Same
# forwards, zero extra compute — pure overlap of the measured 10-18ms/cycle
# host gap.  VMLX_MTP_VERIFY_PREFETCH=0 disables.
_NATIVE_MTP_VERIFY_PREFETCH = os.environ.get(
    "VMLX_MTP_VERIFY_PREFETCH", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def _native_mtp_trim_head_chain(state: "MLLMNativeMTPState") -> None:
    """Drop the head-cache entries added by deeper draft levels last cycle.

    Chain pairs are drafted before verification, so one of them may carry a
    rejected token; they are also built from the head's own post-norm hidden
    rather than the backbone's.  The commit that follows re-adds the accepted
    ones with proper backbone hiddens, so trimming is always safe.
    """
    n = int(getattr(state, "head_chain_pairs", 0) or 0)
    if n <= 0 or not state.mtp_cache:
        state.head_chain_pairs = 0
        return
    for layer in state.mtp_cache:
        if layer is None:
            continue
        if hasattr(layer, "is_trimmable") and layer.is_trimmable():
            layer.trim(n)
    state.head_chain_pairs = 0


# Whether a verifier rejection destroys the MTP head's own KV cache.
#
# The MLLM path has always recreated it from scratch, so after every rejected
# draft the head predicts the next token with ZERO history.  The text path does
# the opposite by design.  On predictable text rejections are rare and this
# never shows; on real prose ~37% of cycles reject, so the head is blinded over
# and over and acceptance spirals down -- which is exactly the gap between
# 96.6% acceptance on a counting prompt and 62.5% on prose for the same bundle.
#
# VMLX_MTP_RECREATE_HEAD_CACHE_ON_REJECT=0 retains the cache instead.
_NATIVE_MTP_RECREATE_HEAD_CACHE_ON_REJECT = os.environ.get(
    "VMLX_MTP_RECREATE_HEAD_CACHE_ON_REJECT", "1"
).strip().lower() not in {"0", "false", "no", "off"}

from .native_mtp_acceptance import accepted_count as _shared_accepted_count


def _native_mtp_accepted_count(
    draft_ids: List[int],
    target_ids: List[int],
    draft_lps: List[Optional[mx.array]],
    target_lps: List[Optional[mx.array]],
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    telemetry: Optional[Dict[str, int]] = None,
) -> int:
    """Count leading accepted drafts for one MTP verify cycle.

    Delegates to the shared rule so this path and the text path
    (``patches/mlx_lm_mtp/batch_generator.py``) can never disagree about how a
    draft is accepted — that divergence is what forced ``--is-mllm`` bundles to
    pin temperature to 0 while text bundles ran fine at any temperature.
    """
    return _shared_accepted_count(
        draft_ids,
        target_ids,
        draft_lps,
        target_lps,
        stochastic=_NATIVE_MTP_STOCHASTIC_ACCEPT,
        sampler=sampler,
        telemetry=telemetry,
    )


def _native_mtp_rejection_correction(
    target_token: mx.array,
    target_id: int,
    target_lp: Optional[mx.array],
    draft_lp: Optional[mx.array],
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[mx.array, int]:
    """Return the verifier correction after one rejected proposal.

    Greedy verification emits the verifier token directly.  Stochastic
    verification must instead draw from the positive target-minus-proposal
    residual; otherwise a rejection changes the target distribution.
    """

    if (
        not _NATIVE_MTP_STOCHASTIC_ACCEPT
        or target_lp is None
        or draft_lp is None
    ):
        return target_token, int(target_id)

    from .native_mtp_acceptance import accept_lp_for, residual_sample

    target_accept_lp = accept_lp_for(sampler, target_lp)
    draft_accept_lp = accept_lp_for(sampler, draft_lp)
    correction_id, _ = residual_sample(
        target_accept_lp,
        draft_accept_lp,
        sampler=sampler,
    )
    return mx.array([correction_id], dtype=mx.uint32), correction_id


def _sample_mllm_prefill_logits(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[mx.array, Optional[mx.array]]:
    """Sample the first MLLM decode token without logprobs when possible.

    Generic mlx-lm stochastic samplers consume normalized log-probabilities,
    while vMLX's greedy and compact-top-k wrappers explicitly opt into raw
    logits via ``_vmlx_accepts_logits``.  Decode must use this same helper so
    token zero and every later token share one input-space contract.
    """
    if _native_mtp_sampler_accepts_logits(sampler):
        sampled = _native_mtp_ensure_uint32(sampler(logits_2d))
        return sampled, logits_2d if _mimo_v2_token_trace_enabled() else None
    logprobs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
    sampled = _native_mtp_ensure_uint32(sampler(logprobs))
    return sampled, logprobs


# One device fence per MTP verify cycle, defaulted BY DEPTH AND FAMILY.
#
# depth >= 2 — fence ON.  Measured 2026-08-15 on 35B MXFP8 MTP depth 2: the
# fence recovers the cache-on async-accumulation stall (1.45x -> 1.68x, MTP arm
# 108.5 -> 128.0 t/s, byte-equal) and is free where the stall is absent
# (cache-off 1.932x fenced vs 1.943x unfenced, within run noise, byte-equal).
#
# qwen3.5-family depth == 1 — fence OFF.  Measured 2026-08-18 live in the app on
# Qwen3.8-27B-JANG_4D-CRACK, 64 prompt, fresh chat, IDENTICAL 4139-token
# output: fenced 26.5 t/s vs unfenced 37.9 / 37.5 t/s = +43%.  A depth-1 cycle
# issues one draft forward and one 2-token verify, so there is almost no lazy
# work for the barrier to bound — only its cost lands, and it lands every
# single cycle.  Fencing depth 1 was costing more than MTP itself was winning.
#
# This cannot be a global depth-1 exemption.  Dots3-note later fell from
# 43.5 t/s to 11.6 t/s at identical 89% acceptance after its fence was removed;
# its large MoE path accumulates enough lazy work even at D1 to need the bound.
# Unknown/unmeasured families therefore fail safe to fenced.  Only the measured
# Qwen3.5-family runtimes take the unfenced D1 default.
#
# VMLX_MTP_CYCLE_FENCE=1/0 forces either way and overrides the depth default.
_NATIVE_MTP_CYCLE_FENCE_ENV = os.environ.get("VMLX_MTP_CYCLE_FENCE", "").strip().lower()

_NATIVE_MTP_D1_UNFENCED_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_vl",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def _native_mtp_cycle_fence_enabled(
    depth: int,
    *,
    model_type: Optional[str] = None,
) -> bool:
    """Whether to issue the per-cycle device fence for this runtime lane."""
    if _NATIVE_MTP_CYCLE_FENCE_ENV in {"0", "false", "no", "off"}:
        return False
    if _NATIVE_MTP_CYCLE_FENCE_ENV in {"1", "true", "yes", "on"}:
        return True
    normalized_depth = max(1, int(depth or 1))
    normalized_model_type = str(model_type or "").strip().lower()
    if (
        normalized_depth == 1
        and normalized_model_type in _NATIVE_MTP_D1_UNFENCED_MODEL_TYPES
    ):
        return False
    # Default ON for depth >= 2 and every unknown/unmeasured family.  This
    # preserves the Dots3 lazy-accumulation fix while avoiding the separately
    # measured Qwen3.5 D1 barrier regression above.
    return True


def _native_mtp_trace_enabled() -> bool:
    return os.environ.get("VMLINUX_NATIVE_MTP_TRACE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_mtp_trace_start() -> float:
    if _native_mtp_trace_enabled():
        try:
            mx.synchronize()
        except Exception:
            pass
    return time.perf_counter()


def _native_mtp_trace_stop(
    stats: MLLMNativeMTPStats,
    attr: str,
    start: float,
) -> None:
    if start <= 0.0:
        return
    if _native_mtp_trace_enabled():
        try:
            mx.synchronize()
        except Exception:
            pass
    setattr(stats, attr, getattr(stats, attr) + (time.perf_counter() - start) * 1000.0)


def _native_mtp_trace_eval(*arrays: Any) -> None:
    if not _native_mtp_trace_enabled():
        return
    arrays = tuple(array for array in arrays if array is not None)
    if not arrays:
        return
    try:
        mx.eval(*arrays)
    except Exception:
        pass


def _native_mtp_async_eval(*arrays: Any) -> None:
    arrays = tuple(array for array in arrays if array is not None)
    if not arrays:
        return
    try:
        mx.async_eval(*arrays)
    except Exception:
        mx.eval(*arrays)


def _m3_affine2_decode_needs_sync(cache: Any = None) -> bool:
    """MiniMax-M3 + affine-2 fast path requires SYNCHRONOUS decode-token eval.

    The async_eval decode pipeline races with the custom affine-2 SwitchGLU Metal
    kernel + the lazily-grown MSA cache state and RARELY corrupts long-context
    decode (degenerate / null-byte output). Evidence (2026-06-15): synchronous
    decode ~15/15 coherent across fresh processes vs async ~10-25% corrupt; the
    per-call kernel is numerically exact + deterministic, so this is an eval/
    materialization-ordering hazard, not a kernel or architecture bug. Forcing
    mx.eval each step materializes the cache state in lockstep and removes it.
    Scoped to M3+affine-2 so all other models keep async CPU/GPU overlap.
    """
    if cache is None:
        return False
    try:
        if not any(type(c).__name__ == "MiniMaxM3SparseCache" for c in cache):
            return False
        from .models.minimax_m3.m3_affine2_switch import _disabled as _aff_disabled
        return not _aff_disabled()
    except Exception:
        return False


def _mllm_decode_sync_eval_enabled(cache: Any = None) -> bool:
    if _m3_affine2_decode_needs_sync(cache):
        return True
    return os.environ.get("VMLINUX_MLLM_DECODE_SYNC_EVAL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _submit_decode_token_eval(value: Any, cache: Any = None) -> None:
    if _mllm_decode_sync_eval_enabled(cache):
        mx.eval(value)
    else:
        mx.async_eval(value)


def _mllm_prefill_trace_enabled() -> bool:
    return os.environ.get("VMLINUX_MLLM_PREFILL_TRACE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mllm_prefill_trace_sync() -> None:
    try:
        mx.synchronize()
    except Exception:
        pass


class _MLLMPrefillTrace:
    """Env-gated timing for diagnosing MLLM/VL prefill throughput."""

    def __init__(
        self,
        *,
        request_id: str,
        prompt_tokens: int,
        has_images: bool,
        is_hybrid: bool,
        native_mtp: bool,
        prefix_cache_enabled: bool,
        language_model_class: str = "",
        force_text_rope_1d: bool = False,
        supports_return_logits: bool = False,
    ) -> None:
        self.enabled = _mllm_prefill_trace_enabled()
        self.request_id = request_id
        self.prompt_tokens = int(prompt_tokens or 0)
        self.has_images = bool(has_images)
        self.is_hybrid = bool(is_hybrid)
        self.native_mtp = bool(native_mtp)
        self.prefix_cache_enabled = bool(prefix_cache_enabled)
        self.language_model_class = str(language_model_class or "")
        self.force_text_rope_1d = bool(force_text_rope_1d)
        self.supports_return_logits = bool(supports_return_logits)
        self.cached_tokens = 0
        self.cache_detail = "none"
        self.cache_before_forward = ""
        self.cache_after_forward = ""
        self._segments: Dict[str, float] = {}
        self._open: Dict[str, float] = {}
        self._start = 0.0
        if self.enabled:
            _mllm_prefill_trace_sync()
            self._start = time.perf_counter()

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        _mllm_prefill_trace_sync()
        self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        start = self._open.pop(name, None)
        if start is None:
            return
        _mllm_prefill_trace_sync()
        elapsed = time.perf_counter() - start
        if elapsed > 0.0:
            self._segments[name] = self._segments.get(name, 0.0) + elapsed

    def set(self, **values: Any) -> None:
        if not self.enabled:
            return
        if "prompt_tokens" in values:
            self.prompt_tokens = int(values["prompt_tokens"] or 0)
        if "cached_tokens" in values:
            self.cached_tokens = int(values["cached_tokens"] or 0)
        if "cache_detail" in values:
            self.cache_detail = str(values["cache_detail"] or "none")
        if "has_images" in values:
            self.has_images = bool(values["has_images"])
        if "cache_before_forward" in values:
            self.cache_before_forward = str(values["cache_before_forward"] or "")
        if "cache_after_forward" in values:
            self.cache_after_forward = str(values["cache_after_forward"] or "")

    @staticmethod
    def _ms(value: float) -> float:
        return round(value * 1000.0, 3)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_detail": self.cache_detail,
            "has_images": self.has_images,
            "is_hybrid": self.is_hybrid,
            "native_mtp": self.native_mtp,
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "language_model_class": self.language_model_class,
            "force_text_rope_1d": self.force_text_rope_1d,
            "supports_return_logits": self.supports_return_logits,
        }
        if self.cache_before_forward:
            data["cache_before_forward"] = self.cache_before_forward
        if self.cache_after_forward:
            data["cache_after_forward"] = self.cache_after_forward
        if self.enabled:
            for name, value in self._segments.items():
                data[f"{name}_ms"] = self._ms(value)
            _mllm_prefill_trace_sync()
            data["total_ms"] = self._ms(time.perf_counter() - self._start)
        return data

    def log(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        data = self.to_dict()
        ordered = [
            "request_id",
            "prompt_tokens",
            "cached_tokens",
            "cache_detail",
            "has_images",
            "is_hybrid",
            "native_mtp",
            "prefix_cache_enabled",
            "language_model_class",
            "force_text_rope_1d",
            "supports_return_logits",
            "total_ms",
            "preprocess_ms",
            "cache_lookup_ms",
            "cache_prepare_ms",
            "forward_ms",
            "sample_ms",
            "logits_eval_ms",
            "clear_cache_ms",
            "sample_call_ms",
            "cache_submit_ms",
            "token_item_ms",
            "ssm_capture_ms",
            "cache_merge_ms",
            "cache_before_forward",
            "cache_after_forward",
        ]
        parts = [f"{key}={data[key]}" for key in ordered if key in data]
        logger.info("VMLINUX_MLLM_PREFILL_TRACE %s", " ".join(parts))
        return data


def _native_mtp_ensure_uint32(token: mx.array) -> mx.array:
    return token if token.dtype == mx.uint32 else token.astype(mx.uint32)


def _native_mtp_model_has_head(language_model: Any) -> bool:
    return bool(
        language_model is not None
        and callable(getattr(language_model, "mtp_forward", None))
        and callable(getattr(language_model, "make_mtp_cache", None))
        and getattr(language_model, "mtp", None) is not None
    )


def _lm_supports_return_logits(lm: Any) -> bool:
    try:
        sig = inspect.signature(lm.__call__)
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    ):
        return True
    return "return_logits" in sig.parameters


def _lm_supports_position_ids(lm: Any) -> bool:
    try:
        sig = inspect.signature(lm.__call__)
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    ):
        return True
    return "position_ids" in sig.parameters


def _call_lm_prefix_without_logits(
    lm: Any,
    input_ids: mx.array,
    kwargs: Dict[str, Any],
) -> Any:
    """Run a prefix-only language-model step without projecting full logits when supported."""
    if _lm_supports_return_logits(lm):
        return lm(input_ids, **kwargs, return_logits=False)
    return lm(input_ids, **kwargs)


def _companion_exempt_cache(cache_obj: Any) -> bool:
    """Positional full-latent caches are EXEMPT from SSM companion snapshots.

    The companion lane was designed for fixed-size recurrent state (GDN
    conv/ssm, a few MB per layer). dots3's full-attention Dots3LatentCache
    slots look "SSM-like" (a .cache list, not trimmable KV) but hold O(ctx)
    positional latent streams — cloning them per checkpoint made the
    companion cache grow QUADRATICALLY with context (measured: +13 full-
    length latents per stored boundary, ~8.6GB retained by 36k, the real
    dots3 deep-context wall). They do not need companion state at all: the
    block lane reconstructs them positionally and the adoption lane
    re-types them (`dots3 adopted N restored full-layer cache(s)` fires on
    every block hit today). The discriminator: positionally rewindable
    (trim_to_boundary) with no window = a positional stream, not state.
    """
    return (
        hasattr(cache_obj, "trim_to_boundary")
        and getattr(cache_obj, "window", 1) is None
    )


def _prefill_cache_materialization_items(cache: Optional[List[Any]]) -> List[Any]:
    """Collect KV/SSM cache arrays that should be realized after prefix prefill."""
    items: List[Any] = []

    def _collect(cache_obj: Any) -> None:
        if cache_obj is None:
            return
        keys = getattr(cache_obj, "keys", None)
        values = getattr(cache_obj, "values", None)
        if keys is not None or values is not None:
            for value in (keys, values):
                if isinstance(value, (list, tuple)):
                    items.extend(v for v in value if v is not None)
                elif value is not None:
                    items.append(value)
            return
        nested = getattr(cache_obj, "caches", None)
        if isinstance(nested, (list, tuple)):
            for sub_cache in nested:
                _collect(sub_cache)
            return
        ssm_cache = getattr(cache_obj, "cache", None)
        if isinstance(ssm_cache, list):
            items.extend(arr for arr in ssm_cache if arr is not None)
            return
        state = getattr(cache_obj, "state", None)
        if isinstance(state, (list, tuple)):
            items.extend(arr for arr in state if arr is not None)
        elif state is not None:
            items.append(state)

    for entry in cache or []:
        _collect(entry)
    return items


def _materialize_prefill_cache_state(cache: Optional[List[Any]]) -> None:
    items = _prefill_cache_materialization_items(cache)
    if not items:
        return
    try:
        mx.eval(*items)
    except RuntimeError as eval_err:
        if "Stream" in str(eval_err):
            mx.synchronize()
        else:
            raise


def _native_mtp_clear_rollback(cache: List[Any]) -> None:
    for layer in cache:
        if hasattr(layer, "rollback_state") and layer.rollback_state is not None:
            layer.rollback_state = None
        if getattr(layer, "rollback_to", None) is not None:
            layer.rollback_to = None
        if getattr(layer, "rollback_aux_state", None) is not None:
            layer.rollback_aux_state = None
        if getattr(layer, "rollback_aux_to", None) is not None:
            layer.rollback_aux_to = None


def _native_mtp_snapshot_value(value: Any) -> Any:
    if isinstance(value, mx.array):
        return value
    if isinstance(value, list):
        return [_native_mtp_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_native_mtp_snapshot_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _native_mtp_snapshot_value(item)
            for key, item in value.items()
        }
    if hasattr(value, "__dict__") and (
        hasattr(value, "state")
        or hasattr(value, "is_trimmable")
        or value.__class__.__module__.startswith("mlx_lm.models.cache")
    ):
        return _native_mtp_snapshot_object(value)
    return value


def _native_mtp_restore_value(value: Any) -> Any:
    if isinstance(value, _NativeMTPCacheObjectSnapshot):
        _native_mtp_restore_object(value)
        return value.obj
    if isinstance(value, list):
        return [_native_mtp_restore_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_native_mtp_restore_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _native_mtp_restore_value(item)
            for key, item in value.items()
        }
    return value


def _native_mtp_snapshot_object(obj: Any) -> _NativeMTPCacheObjectSnapshot:
    attrs = {
        key: _native_mtp_snapshot_value(value)
        for key, value in getattr(obj, "__dict__", {}).items()
    }
    return _NativeMTPCacheObjectSnapshot(obj=obj, attrs=attrs)


def _native_mtp_restore_object(snapshot: _NativeMTPCacheObjectSnapshot) -> None:
    attrs = getattr(snapshot.obj, "__dict__", None)
    if attrs is None:
        return
    attrs.clear()
    attrs.update(
        {
            key: _native_mtp_restore_value(value)
            for key, value in snapshot.attrs.items()
        }
    )


def _native_mtp_should_snapshot_layer(layer: Any, advance_len: int = 0) -> bool:
    if layer is None:
        return False
    if hasattr(layer, "rollback_state"):
        return True
    # TurboQuant live-encode crossing: trim() rewinds offset only, so if the
    # one-time compress() fires inside a rejected verify advance, draft KV
    # stays baked into the decoded/compressed buffers. compress() rebinds to
    # new arrays, so a by-ref __dict__ snapshot is a sound rollback here.
    compress_after = getattr(layer, "compress_after", None)
    if compress_after is not None and advance_len > 0:
        try:
            threshold = int(compress_after or 0)
            not_compressed = int(getattr(layer, "_compressed_tokens", 0) or 0) == 0
            offset = int(getattr(layer, "offset", 0) or 0)
            if threshold > 0 and not_compressed and offset + advance_len >= threshold:
                return True
        except (TypeError, ValueError):
            return True
    is_trimmable = getattr(layer, "is_trimmable", None)
    if callable(is_trimmable):
        try:
            return not bool(is_trimmable())
        except Exception:
            return True
    return hasattr(layer, "__dict__")


def _native_mtp_snapshot_replay_cache(
    cache: List[Any],
    advance_len: int = 0,
) -> List[Optional[_NativeMTPCacheObjectSnapshot]]:
    return [
        _native_mtp_snapshot_object(layer)
        if _native_mtp_should_snapshot_layer(layer, advance_len)
        else None
        for layer in cache
    ]


def _native_mtp_restore_replay_cache(
    cache: List[Any],
    snapshot: List[Optional[_NativeMTPCacheObjectSnapshot]],
    token_count: int,
) -> bool:
    """Restore cache to the pre-verify point before replaying confirmed tokens."""
    for layer, layer_snapshot in zip(cache, snapshot):
        if layer_snapshot is not None:
            _native_mtp_restore_object(layer_snapshot)
            continue
        if hasattr(layer, "is_trimmable") and layer.is_trimmable():
            layer.trim(max(1, int(token_count)))
        elif token_count > 0:
            return False
    return True



def _native_mtp_bump_emit(state: MLLMNativeMTPState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


def _native_mtp_log_stats(
    request_id: str,
    stats: MLLMNativeMTPStats,
    reason: str,
    mtp_cache: Any = None,
) -> None:
    # Capture the head-cache shape only at terminal publication. Keeping this
    # out of the per-draft cycle is important: this telemetry exists to
    # diagnose MTP overhead and must not itself add Python work to every token.
    if mtp_cache is not None or not stats.mtp_head_cache:
        stats.mtp_head_cache = native_mtp_cache_snapshot(mtp_cache)
    rate = (stats.accepted_tokens / stats.drafted_tokens * 100.0) if stats.drafted_tokens else 0.0
    logger.info(
        "MLLM MTP[%s] finish=%s cycles=%d accepted=%d/%d (%.1f%%) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] margin_truncated=%d",
        request_id,
        reason,
        stats.cycles,
        stats.accepted_tokens,
        stats.drafted_tokens,
        rate,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.margin_truncated_cycles,
    )
    if any(stats.drafted_by_depth) or any(stats.accepted_by_depth):
        logger.info(
            "MLLM MTP[%s] accept_by_depth[d1=%d/%d,d2=%d/%d,d3=%d/%d] "
            "forwards[seed_main=%d,verify_main=%d,replay_main=%d,mtp=%d]",
            request_id,
            stats.accepted_by_depth[0],
            stats.drafted_by_depth[0],
            stats.accepted_by_depth[1],
            stats.drafted_by_depth[1],
            stats.accepted_by_depth[2],
            stats.drafted_by_depth[2],
            stats.seed_main_forwards,
            stats.verify_main_forwards,
            stats.replay_main_forwards,
            stats.mtp_forwards,
        )
    if stats.stochastic_distribution_cycles:
        logger.info(
            "MLLM MTP[%s] stochastic_verify[cycles=%d,ratio_checks=%d,"
            "ratio_accepts=%d,residual_corrections=%d]",
            request_id,
            stats.stochastic_distribution_cycles,
            stats.stochastic_ratio_checks,
            stats.stochastic_ratio_accepts,
            stats.stochastic_residual_corrections,
        )
    total_ms = (
        stats.verify_ms
        + stats.sample_ms
        + stats.draft_ms
        + stats.snapshot_ms
        + stats.restore_ms
        + stats.replay_ms
        + stats.materialize_ms
    )
    if total_ms > 0.0:
        avg_cycle = total_ms / max(1, stats.cycles)
        logger.info(
            "MLLM MTP[%s] timings_ms[verify=%.2f sample=%.2f draft=%.2f "
            "snapshot=%.2f restore=%.2f replay=%.2f materialize=%.2f "
            "avg_cycle=%.2f]",
            request_id,
            stats.verify_ms,
            stats.sample_ms,
            stats.draft_ms,
            stats.snapshot_ms,
            stats.restore_ms,
            stats.replay_ms,
            stats.materialize_ms,
            avg_cycle,
        )


def _native_mtp_capture_head_cache_before_discard(
    stats: MLLMNativeMTPStats,
    mtp_cache: Any,
) -> None:
    """Preserve one bounded snapshot before adaptive AR frees the head cache."""

    stats.mtp_head_cache = native_mtp_cache_snapshot(mtp_cache)


def _native_mtp_debug_enabled() -> bool:
    return os.environ.get("VMLINUX_NATIVE_MTP_DEBUG_TOKENS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_mtp_burst_enabled() -> bool:
    return os.environ.get("VMLINUX_NATIVE_MTP_BURST", "1") not in {
        "0",
        "false",
        "FALSE",
        "no",
        "NO",
        "off",
        "OFF",
    }


def _native_mtp_env_value(*names: str) -> Optional[str]:
    for name in names:
        if name in os.environ:
            return os.environ.get(name)
    return None


def _native_mtp_env_flag(default: bool, *names: str) -> bool:
    raw = _native_mtp_env_value(*names)
    if raw is None:
        return default
    return str(raw) not in {
        "0",
        "false",
        "FALSE",
        "no",
        "NO",
        "off",
        "OFF",
    }


def _native_mtp_env_int(default: int, *names: str, minimum: int = 0) -> int:
    raw = _native_mtp_env_value(*names)
    if raw is None:
        return max(minimum, int(default))
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _native_mtp_env_float(default: float, *names: str) -> float:
    raw = _native_mtp_env_value(*names)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _native_mtp_value_policy_enabled() -> bool:
    return _native_mtp_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_ADAPTIVE_VALUE",
        "VMLX_NATIVE_MTP_ADAPTIVE_VALUE",
    )


def _native_mtp_value_window() -> int:
    return _native_mtp_env_int(
        16,
        "VMLINUX_NATIVE_MTP_VALUE_WINDOW",
        "VMLX_NATIVE_MTP_VALUE_WINDOW",
        minimum=4,
    )


def _native_mtp_value_min_samples() -> int:
    return _native_mtp_env_int(
        8,
        "VMLINUX_NATIVE_MTP_VALUE_MIN_SAMPLES",
        "VMLX_NATIVE_MTP_VALUE_MIN_SAMPLES",
        minimum=2,
    )


def _native_mtp_finish_value_cycle(
    state: MLLMNativeMTPState,
    *,
    depth: int,
    accepted: int,
    now: float,
) -> None:
    if not _native_mtp_value_policy_enabled():
        return
    finish_armed_depth_cycle(
        state.adaptive_value,
        depth=depth,
        accepted_drafts=accepted,
        cycle=int(state.stats.cycles),
        now=now,
        window=_native_mtp_value_window(),
    )


def _native_mtp_arm_value_cycle(
    state: MLLMNativeMTPState,
    *,
    now: float,
) -> None:
    if not _native_mtp_value_policy_enabled() or state.ar_fallback_pending:
        return
    arm_depth_cycle(
        state.adaptive_value,
        depth=int(state.depth or 1),
        now=now,
    )


def _native_mtp_maybe_choose_value_depth(
    request_id: str,
    state: MLLMNativeMTPState,
    current: int,
) -> bool:
    """Run one bounded adjacent-depth experiment from rolling wall value."""

    if not _native_mtp_value_policy_enabled():
        return False
    # Preserve compatibility with request-state objects created before the
    # rolling wall-value controller was added. Acceptance-based adaptation is
    # still safe for those requests; value-based probing simply has no state to
    # consume and must decline without crashing the request.
    adaptive_value = getattr(state, "adaptive_value", None)
    if adaptive_value is None:
        return False
    minimum_samples = _native_mtp_value_min_samples()
    decision = choose_depth_by_value(
        adaptive_value,
        current_depth=current,
        depth_ceiling=int(getattr(state, "depth_ceiling", 3) or 3),
        cycle=int(state.stats.cycles),
        minimum_samples=minimum_samples,
        cooldown_cycles=_native_mtp_env_int(
            8,
            "VMLINUX_NATIVE_MTP_VALUE_COOLDOWN_CYCLES",
            "VMLX_NATIVE_MTP_VALUE_COOLDOWN_CYCLES",
            minimum=2,
        ),
        probe_interval_cycles=_native_mtp_env_int(
            48,
            "VMLINUX_NATIVE_MTP_VALUE_PROBE_INTERVAL_CYCLES",
            "VMLX_NATIVE_MTP_VALUE_PROBE_INTERVAL_CYCLES",
            minimum=4,
        ),
        hysteresis=_native_mtp_env_float(
            0.05,
            "VMLINUX_NATIVE_MTP_VALUE_HYSTERESIS",
            "VMLX_NATIVE_MTP_VALUE_HYSTERESIS",
        ),
        raise_min_acceptance=_native_mtp_env_float(
            0.88,
            "VMLINUX_NATIVE_MTP_VALUE_RAISE_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_VALUE_RAISE_MIN_ACCEPT",
        ),
        initial_probe_cycles=_native_mtp_env_int(
            48,
            "VMLINUX_NATIVE_MTP_VALUE_INITIAL_PROBE_CYCLES",
            "VMLX_NATIVE_MTP_VALUE_INITIAL_PROBE_CYCLES",
            minimum=2,
        ),
    )
    state.stats.adaptive_depth_value = adaptive_value_snapshot(
        adaptive_value,
        minimum_samples=minimum_samples,
    )
    if decision is None:
        return False
    target = max(
        1,
        min(
            int(getattr(state, "depth_ceiling", 3) or 3),
            int(decision.target_depth),
        ),
    )
    if target != current:
        state.depth = target
    logger.info(
        "MLLM MTP[%s] adaptive value %s D%d -> D%d after cycles=%d: %s",
        request_id,
        decision.event,
        current,
        target,
        state.stats.cycles,
        decision.reason,
    )
    return True


def _native_mtp_depth_rate(stats: MLLMNativeMTPStats, depth: int) -> Optional[float]:
    index = int(depth) - 1
    if index < 0 or index >= len(stats.drafted_by_depth):
        return None
    drafted = int(stats.drafted_by_depth[index])
    if drafted <= 0:
        return None
    accepted = int(stats.accepted_by_depth[index])
    return accepted / drafted


def _native_mtp_timing_total_ms(stats: MLLMNativeMTPStats) -> float:
    return (
        float(stats.verify_ms)
        + float(stats.sample_ms)
        + float(stats.draft_ms)
        + float(stats.snapshot_ms)
        + float(stats.restore_ms)
        + float(stats.replay_ms)
        + float(stats.materialize_ms)
    )


def _native_mtp_confirmed_tokens_from_cycles(stats: MLLMNativeMTPStats) -> int:
    # Every verify cycle emits one verifier token plus the accepted draft prefix.
    return max(0, int(stats.cycles)) + max(0, int(stats.accepted_tokens))


def _native_mtp_cost_ratio(
    stats: MLLMNativeMTPStats,
    ar_step_ms: float,
) -> Optional[Tuple[float, float]]:
    if ar_step_ms <= 0.0:
        return None
    mtp_ms = _native_mtp_timing_total_ms(stats)
    if mtp_ms <= 0.0:
        return None
    confirmed = _native_mtp_confirmed_tokens_from_cycles(stats)
    if confirmed <= 0:
        return None
    mtp_ms_per_token = mtp_ms / confirmed
    return mtp_ms_per_token / ar_step_ms, mtp_ms_per_token


def _native_mtp_maybe_cost_fallback(
    request_id: str,
    state: MLLMNativeMTPState,
    current_depth: int,
) -> bool:
    if not _native_mtp_env_flag(
        False,
        "VMLINUX_NATIVE_MTP_COST_FALLBACK",
        "VMLX_NATIVE_MTP_COST_FALLBACK",
    ):
        return False
    ar_step_ms = _native_mtp_env_float(
        0.0,
        "VMLINUX_NATIVE_MTP_AR_STEP_MS",
        "VMLX_NATIVE_MTP_AR_STEP_MS",
        "VMLINUX_NATIVE_MTP_COST_AR_STEP_MS",
        "VMLX_NATIVE_MTP_COST_AR_STEP_MS",
    )
    ratio_and_cost = _native_mtp_cost_ratio(state.stats, ar_step_ms)
    if ratio_and_cost is None:
        return False
    ratio, mtp_ms_per_token = ratio_and_cost
    threshold = _native_mtp_env_float(
        1.0,
        "VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD",
        "VMLX_NATIVE_MTP_COST_RATIO_THRESHOLD",
    )
    if ratio < threshold:
        return False

    state.depth = 1
    state.ar_fallback_pending = True
    state.ar_fallback_reason = (
        f"cost_ratio={ratio:.3f}>=threshold={threshold:.3f} "
        f"mtp_ms_per_token={mtp_ms_per_token:.2f} ar_step_ms={ar_step_ms:.2f}"
    )
    logger.info(
        "MLLM MTP[%s] adaptive depth D%d -> AR after cycles=%d "
        "cost_ratio=%.3f threshold=%.3f mtp_ms_per_token=%.2f "
        "ar_step_ms=%.2f",
        request_id,
        current_depth,
        state.stats.cycles,
        ratio,
        threshold,
        mtp_ms_per_token,
        ar_step_ms,
    )
    return True


def _native_mtp_maybe_adapt_depth(request_id: str, state: MLLMNativeMTPState) -> None:
    """Lower future recursive draft depth when measured acceptance is poor.

    The current verify cycle always finishes at the depth it started with.
    This only changes the next draft suffix, so it cannot invalidate the
    current verifier cache state.
    """
    if not _native_mtp_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH",
        "VMLX_NATIVE_MTP_ADAPTIVE_DEPTH",
    ):
        return
    current = max(1, int(state.depth or 1))

    warmup = _native_mtp_env_int(
        12,
        "VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        minimum=1,
    )
    # Restored-prefix requests start with a COLD head cache, so the early
    # gate windows measure a context-starved head, not the bundle. Stretch
    # every demotion sample window so warm cycles dilute the cold ones
    # before any gate may fire.
    _restore_scale = 4 if getattr(state, "restored_prefix", False) else 1
    warmup = warmup * _restore_scale
    if int(state.stats.cycles) < warmup:
        return

    target = current
    accelerated_d3 = False
    if target >= 3:
        drafted_d3 = int(state.stats.drafted_by_depth[2]) if len(state.stats.drafted_by_depth) > 2 else 0
        rate_d3 = _native_mtp_depth_rate(state.stats, 3)
        # Use only request-local proof that an eligible projection actually
        # took the optional verifier.  The dispatcher may be installed while
        # this artifact (notably a q6 MTP head) falls back on every call.
        accelerated_d3 = int(getattr(state.stats, "verify_qmm_calls", 0)) > 0
        min_d3 = _native_mtp_env_float(
            0.65 if accelerated_d3 else 0.85,
            "VMLINUX_NATIVE_MTP_D3_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_D3_MIN_ACCEPT",
        )
        d3_min_sample = _restore_scale * _native_mtp_env_int(
            128 if accelerated_d3 else 48,
            "VMLINUX_NATIVE_MTP_DEPTH_GATE_MIN_SAMPLE",
            "VMLX_NATIVE_MTP_DEPTH_GATE_MIN_SAMPLE",
            minimum=1,
        )
        # accepted_by_depth[2] is a JOINT rate: d1, d2 and d3 all accepted.
        # The configured floor is conditional on reaching d3, so scale it by
        # the measured joint d2 rate exactly as the d2 gate is scaled by d1.
        # Comparing the joint d3 rate directly with 0.85 permanently demoted
        # healthy chains (Qwen3.8 measured 0.842 joint d2 / 0.759 joint d3,
        # i.e. 90.1% conditional d3 acceptance). The accelerated verifier gets
        # a longer sample and a 65% conditional floor: its 48-cycle cold window
        # measured 45.8% joint, its 128-cycle window measured 73.6% conditional,
        # and the same completed 767-cycle response settled at 90.1%. The lower
        # floor is profitable only with the guarded four-row verifier; the stock
        # verifier retains its conservative 85% floor.
        rate_d2_for_d3_gate = _native_mtp_depth_rate(state.stats, 2)
        joint_floor_d3 = min_d3 * (
            rate_d2_for_d3_gate if rate_d2_for_d3_gate else 1.0
        )
        if (
            drafted_d3 >= max(warmup, d3_min_sample)
            and rate_d3 is not None
            and rate_d3 < joint_floor_d3
        ):
            target = 2
    if target >= 2:
        drafted_d2 = int(state.stats.drafted_by_depth[1]) if len(state.stats.drafted_by_depth) > 1 else 0
        rate_d2 = _native_mtp_depth_rate(state.stats, 2)
        # Floor 0.70 and a REAL sample, not the 12-cycle warmup.  Measured
        # 2026-08-18 (controlled chain test, 150 cycles, code workload,
        # Qwen3.8-27B-JANG_4D-CRACK): true d2 acceptance is 88/118 = 74.6% —
        # healthy decay from d1's 78.7% — yet the engine demoted D2 after
        # reading 3/12 = 25% on the cold dozen cycles right after climbing.
        # With the depth ceiling recording demotions, that snap judgment
        # permanently locked the request out of depth 2.  Same cold-window
        # failure as the AR fallback (fixed at 64); the depth gates kept the
        # old 12.  0.75 also sat above the bundle's real 74.6%, so even a fair
        # sample would have flapped.
        min_d2 = _native_mtp_env_float(
            0.70,
            "VMLINUX_NATIVE_MTP_D2_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_D2_MIN_ACCEPT",
        )
        depth_min_sample = _restore_scale * _native_mtp_env_int(
            128 if (accelerated_d3 and current >= 3) else 48,
            "VMLINUX_NATIVE_MTP_DEPTH_GATE_MIN_SAMPLE",
            "VMLX_NATIVE_MTP_DEPTH_GATE_MIN_SAMPLE",
            minimum=1,
        )
        # rate_d2 is a JOINT rate (cycles where BOTH d1 and d2 accepted /
        # cycles drafted at depth 2), but the floor is calibrated as a
        # CONDITIONAL rate.  Comparing them raw demands conditional ~90% at
        # d1=0.78 — depth 2 could mathematically never stick.  Scale the floor
        # by the observed d1 rate so the comparison is joint-vs-joint.
        rate_d1_for_gate = _native_mtp_depth_rate(state.stats, 1)
        joint_floor_d2 = min_d2 * (rate_d1_for_gate if rate_d1_for_gate else 1.0)
        if (
            drafted_d2 >= max(warmup, depth_min_sample)
            and rate_d2 is not None
            and rate_d2 < joint_floor_d2
        ):
            target = 1

    if _native_mtp_maybe_cost_fallback(request_id, state, current):
        return

    # Runtime cost gate (default ON, wall-clock, no trace needed).  Acceptance
    # gates cannot catch a request whose CYCLE COST exploded while acceptance
    # stayed healthy — measured live on dots3: a prefix restored from
    # block-disk (mixed-SWA lane) keeps ~60-90% acceptance but MTP decodes at
    # 11.7-12.2 t/s while plain AR on the SAME restored cache does 35.1.  The
    # seed forward is a true AR step; if MTP's measured ms-per-emitted-token
    # exceeds it by the margin over a real sample, AR is simply faster and the
    # request falls back.  Runtime measurement choosing the faster path.
    if _native_mtp_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_RUNTIME_COST_GATE",
        "VMLX_NATIVE_MTP_RUNTIME_COST_GATE",
    ):
        ar_ms = float(getattr(state, "ar_step_ms", 0.0) or 0.0)
        span_start = float(getattr(state, "cycle_span_start", 0.0) or 0.0)
        cycles_done = int(state.stats.cycles)
        cost_sample = (
            4 if getattr(state, "restored_prefix", False) else 1
        ) * _native_mtp_env_int(
            48,
            "VMLINUX_NATIVE_MTP_RUNTIME_COST_MIN_CYCLES",
            "VMLX_NATIVE_MTP_RUNTIME_COST_MIN_CYCLES",
            minimum=8,
        )
        margin = _native_mtp_env_float(
            1.25,
            "VMLINUX_NATIVE_MTP_RUNTIME_COST_MARGIN",
            "VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN",
        )
        if ar_ms > 0.0 and span_start > 0.0 and cycles_done >= cost_sample:
            emitted = cycles_done + int(state.stats.accepted_tokens)
            span_ms = (time.perf_counter() - span_start) * 1000.0
            if emitted > 0 and span_ms > 0.0:
                mtp_ms_per_tok = span_ms / emitted
                if mtp_ms_per_tok > ar_ms * margin:
                    state.depth = 1
                    state.ar_fallback_pending = True
                    state.ar_fallback_reason = (
                        f"runtime_cost mtp_ms_per_tok={mtp_ms_per_tok:.1f}"
                        f">ar_step_ms={ar_ms:.1f}x{margin:.2f}"
                    )
                    logger.info(
                        "MLLM MTP[%s] adaptive depth D%d -> AR after cycles=%d "
                        "runtime cost %.1fms/token vs AR %.1fms (margin %.2f)",
                        request_id,
                        current,
                        cycles_done,
                        mtp_ms_per_tok,
                        ar_ms,
                        margin,
                    )
                    return

    # Default ON since 2026-08-15: Qwen3.8-27B-JANG_4D served at temp 0 ran
    # its 58.6% d1-acceptance head for entire requests (483 cycles, 200
    # replay forwards for 768 tokens) because this gate was opt-in — a
    # sub-breakeven head must fall back to AR (measured breakeven ~0.68 on
    # the MLLM path where a rejected cycle pays verify + replay; healthy
    # heads run 0.93-0.98, so the 0.65 floor cleanly separates).
    # VMLX_NATIVE_MTP_AR_FALLBACK=0 reverts.
    if target <= 1 and _native_mtp_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_AR_FALLBACK",
        "VMLX_NATIVE_MTP_AR_FALLBACK",
    ):
        drafted_d1 = int(state.stats.drafted_by_depth[0]) if state.stats.drafted_by_depth else 0
        rate_d1 = _native_mtp_depth_rate(state.stats, 1)
        # With the replay skipped (default), a rejected depth-1 cycle costs
        # the same backbone forward an accepted one does, so MTP is profitable
        # down to roughly the draft-head cost (~6% acceptance).  0.35 keeps a
        # wide margin over that.  With the replay active a reject pays a full
        # extra forward and the old ~0.68 break-even applies, so 0.65 stays.
        # Live case this decides: the reasoning workload measured d1=0.649 and
        # was demoted by the old floor at 25.2 t/s vs 22.1 AR - a real win
        # discarded by one thousandth.
        min_d1 = _native_mtp_env_float(
            0.35 if (_NATIVE_MTP_SKIP_REPLAY and current == 1) else 0.65,
            "VMLINUX_NATIVE_MTP_D1_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_D1_MIN_ACCEPT",
        )
        # Demoting to AR is irreversible for the rest of the request, so it
        # needs a real sample — not the 12-cycle warmup used for the reversible
        # depth steps.  Measured 2026-08-18 on Qwen3.8-27B-JANG_4D-CRACK, same
        # bundle and same prompt: the FIRST request after load read 7/12 =
        # 58.3% while the MTP cache was still cold and demoted the whole
        # request to AR; the next request scored 2033/2104 = 96.6% and held MTP
        # for 34.1 t/s.  Twelve cycles was 0.57% of that request.  The genuine
        # sub-breakeven case this gate exists for ran 483 cycles at 58.6%, so
        # it still trips well inside its own request at this sample size.
        ar_min_sample = _restore_scale * _native_mtp_env_int(
            64,
            "VMLINUX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE",
            "VMLX_NATIVE_MTP_AR_FALLBACK_MIN_SAMPLE",
            minimum=1,
        )
        if (
            drafted_d1 >= max(warmup, ar_min_sample)
            and rate_d1 is not None
            and rate_d1 < min_d1
        ):
            state.depth = 1
            state.ar_fallback_pending = True
            state.ar_fallback_reason = (
                f"d1_acceptance={rate_d1:.3f}<min={min_d1:.3f}"
            )
            logger.info(
                "MLLM MTP[%s] adaptive depth D%d -> AR after cycles=%d "
                "acceptance[d1=%.3f,d2=%s,d3=%s]",
                request_id,
                current,
                state.stats.cycles,
                rate_d1,
                (
                    f"{_native_mtp_depth_rate(state.stats, 2):.3f}"
                    if _native_mtp_depth_rate(state.stats, 2) is not None
                    else "n/a"
                ),
                (
                    f"{_native_mtp_depth_rate(state.stats, 3):.3f}"
                    if _native_mtp_depth_rate(state.stats, 3) is not None
                    else "n/a"
                ),
            )
            return

    if target < current:
        logger.info(
            "MLLM MTP[%s] adaptive depth D%d -> D%d after cycles=%d "
            "acceptance[d2=%s,d3=%s]",
            request_id,
            current,
            target,
            state.stats.cycles,
            (
                f"{_native_mtp_depth_rate(state.stats, 2):.3f}"
                if _native_mtp_depth_rate(state.stats, 2) is not None
                else "n/a"
            ),
            (
                f"{_native_mtp_depth_rate(state.stats, 3):.3f}"
                if _native_mtp_depth_rate(state.stats, 3) is not None
                else "n/a"
            ),
        )
        # Acceptance remains a safety gate, but no longer destroys capability.
        # A later workload phase may make the adjacent depth profitable again;
        # the rolling wall-value controller can re-probe it after its cooldown.
        if _native_mtp_value_policy_enabled() and hasattr(
            state, "adaptive_value"
        ):
            note_forced_depth_change(
                state.adaptive_value,
                origin=current,
                target=target,
                cycle=int(state.stats.cycles),
                reason="acceptance_gate",
            )
            state.stats.adaptive_depth_value = adaptive_value_snapshot(
                state.adaptive_value,
                minimum_samples=_native_mtp_value_min_samples(),
            )
        state.depth = target
        return

    if _native_mtp_maybe_choose_value_depth(request_id, state, current):
        return

    # Explicitly disabling the wall-value policy restores the older cumulative
    # acceptance-only promotion path for controlled A/Bs.
    if not _native_mtp_value_policy_enabled():
        _native_mtp_maybe_raise_depth(request_id, state, current)


def _native_mtp_maybe_raise_depth(
    request_id: str,
    state: MLLMNativeMTPState,
    current: int,
) -> None:
    """Climb to a deeper draft chain when the shallow one is nearly perfect.

    The controller could only ever LOWER depth, so a bundle whose tuning
    sidecar says depth 1 stays at depth 1 no matter how well its head performs
    — capping it at (1 + acceptance) tokens per cycle.  A head accepting ~0.95
    is worth roughly 3.46 tokens per cycle at depth 3 (1 + .95 + .95*.88 +
    .95*.88*.80), which is the entire gap between a 1.5x and a 2.5x speedup.

    Raising is deliberately timid: it needs a near-perfect shallow rate over a
    real sample, climbs one step at a time, and never exceeds the request's
    capability ceiling.  This path is retained only for controlled A/Bs with
    the rolling wall-value policy disabled.
    """
    if not _native_mtp_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_ADAPTIVE_RAISE",
        "VMLX_NATIVE_MTP_ADAPTIVE_RAISE",
    ):
        return

    ceiling = max(1, min(3, int(getattr(state, "depth_ceiling", 3) or 3)))
    if current >= ceiling:
        return

    min_raise = _native_mtp_env_float(
        0.90,
        "VMLINUX_NATIVE_MTP_RAISE_MIN_ACCEPT",
        "VMLX_NATIVE_MTP_RAISE_MIN_ACCEPT",
    )
    raise_sample = _native_mtp_env_int(
        64,
        "VMLINUX_NATIVE_MTP_RAISE_MIN_SAMPLE",
        "VMLX_NATIVE_MTP_RAISE_MIN_SAMPLE",
        minimum=1,
    )

    drafted = (
        int(state.stats.drafted_by_depth[current - 1])
        if len(state.stats.drafted_by_depth) >= current
        else 0
    )
    rate = _native_mtp_depth_rate(state.stats, current)
    if drafted < raise_sample or rate is None or rate < min_raise:
        return

    state.depth = current + 1
    logger.info(
        "MLLM MTP[%s] adaptive depth D%d -> D%d after cycles=%d "
        "d%d_acceptance=%.3f>=raise_min=%.3f ceiling=D%d",
        request_id,
        current,
        state.depth,
        state.stats.cycles,
        current,
        rate,
        min_raise,
        ceiling,
    )


def _native_mtp_materialize_draft_ids(state: MLLMNativeMTPState) -> None:
    if len(state.draft_ids) == len(state.drafts):
        return
    if state.drafts:
        mx.eval(*state.drafts)
    state.draft_ids = [int(draft_tok.tolist()[0]) for draft_tok in state.drafts]


def _native_mtp_scalar_id(token: Any) -> Optional[int]:
    try:
        if isinstance(token, int):
            return int(token)
        value = token.tolist() if hasattr(token, "tolist") else token
        while isinstance(value, list):
            if len(value) != 1:
                return None
            value = value[0]
        return int(value)
    except Exception:
        return None


def _native_mtp_ar_fallback_ready(
    cache: List[Any],
    state: MLLMNativeMTPState,
    last_token_id: int,
) -> Tuple[bool, str]:
    """Check the cycle-boundary invariants required for MTP -> AR handoff."""
    if state.queue:
        return False, "pending_queue"
    next_id = _native_mtp_scalar_id(state.next_main)
    if next_id is None:
        return False, "missing_next_main"
    if next_id != int(last_token_id):
        return False, "next_main_mismatch"
    for layer in cache:
        if getattr(layer, "rollback_state", None) is not None:
            return False, "pending_rollback_state"
    return True, "ready"


def _native_mtp_request_has_tools(request: "MLLMBatchRequest") -> bool:
    """Return the effective tool-presence bit carried by the batch request.

    ``MLLMBatchRequest`` intentionally keeps render/decode policy in
    ``extra_kwargs`` rather than exposing the public API model's ``tools``
    attribute.  Adaptive MTP profile keys must read that preserved contract;
    otherwise tool-heavy and no-tool workloads train the same session profile.
    """

    extra = getattr(request, "extra_kwargs", None) or {}
    return bool(
        extra.get("_vmlx_tools_present")
        or extra.get("_vmlx_template_tools")
        or extra.get("tools")
        or extra.get("_vmlx_tool_choice")
        or extra.get("tool_choice")
    )


@dataclass
class MLLMBatchRequest:
    """
    Request data for MLLM batch processing.

    Contains all information needed to process a multimodal request
    within the batch generator.
    """

    uid: int  # Unique identifier within the batch generator
    request_id: str  # External request ID
    prompt: str  # Text prompt
    images: Optional[List[str]] = None  # Image paths/URLs/base64
    videos: Optional[List[str]] = None  # Video inputs
    audio: Optional[List[Any]] = None  # Audio inputs
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None
    max_prompt_tokens: int = 0
    enable_thinking: Optional[bool] = None

    # Video processing parameters (per-request overrides)
    image_token_budget: Optional[int] = None
    video_fps: Optional[float] = None
    video_max_frames: Optional[int] = None

    # Processed inputs (set after vision preprocessing)
    input_ids: Optional[mx.array] = None
    pixel_values: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    image_grid_thw: Optional[mx.array] = None
    video_pixel_values: Optional[mx.array] = None
    video_grid_thw: Optional[mx.array] = None
    audio_codes: Optional[mx.array] = None
    audio_embeds: Optional[mx.array] = None
    audio_features: Optional[mx.array] = None
    audio_features_mask: Optional[mx.array] = None
    audio_features_are_raw_input_features: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Generation state
    num_tokens: int = 0  # Tokens generated so far
    output_tokens: List[int] = field(default_factory=list)

    # Vision state
    vision_encoded: bool = False

    # Prefix cache state
    prompt_cache: Optional[List[Any]] = None  # Pre-filled KV cache from Prefix Cache or Disk Cache


@dataclass
class MLLMBatchResponse:
    """
    Response from a batch generation step.

    Contains the generated token and metadata for a single request.
    """

    uid: int  # Batch generator UID
    request_id: str  # External request ID
    token: int  # Generated token
    logprobs: Optional[mx.array]  # Log probabilities when explicitly supported
    # "stop", "length", "error", or None. "error" is used for prefill failures
    # that the batched engine catches and converts into a client-visible error
    # (see Issue #56 Bug 1). Without a distinct reason, silent prefill crashes
    # look like normal empty completions to the client.
    finish_reason: Optional[str] = None
    prompt_cache: Optional[Callable[[], List[Any]]] = None  # Cache extraction function
    prompt_token_ids: Optional[List[int]] = None  # Original tokenized prompt for prefix key
    cached_tokens: int = 0  # Number of prompt tokens served from cache
    cache_detail: str = ""  # e.g. "paged+ssm", "paged+ssm+disk", "disk"
    # Request-associated cache execution truth. This is carried on every
    # response (including misses/discarded hits) so concurrent batches cannot
    # accidentally project another request's generator-global snapshot.
    cache_execution: Optional[Dict[str, Any]] = None
    cache_extra_keys: Optional[Dict[str, str]] = None
    # Generation-prefix tokens captured from the untruncated prompt tail
    # (e.g. [<|im_start|>, assistant, \n, <think>, \n] for Qwen 3.6 thinking-on).
    # Thinking models occasionally re-emit these as their first output tokens
    # when prior assistant history lacks a reasoning_content wrapper; the
    # scheduler uses this list to suppress the echoed prefix from the output
    # stream. Empty when no gen-prefix was stripped.
    gen_prefix_tokens: Optional[List[int]] = None
    # Token list the CLEAN-STORE lane must be keyed by, when it differs
    # from prompt_token_ids. The media clean boundary is deliberately
    # block-aligned so the KV chain and the SSM companion land on the
    # SAME token count; prompt_token_ids cannot carry that, because the
    # scheduler also derives usage.prompt_tokens from its length and
    # shortening it would misreport the request to the user.
    clean_store_token_ids: Optional[List[int]] = None
    # Optional human-readable error message attached when finish_reason="error".
    # Scheduler and server.py lift this into an HTTP error response so users
    # can see the actual mlx / mlx_vlm traceback instead of an empty 200.
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_prompt_tokens: Optional[int] = None
    error_max_prompt_tokens: Optional[int] = None
    error_source: Optional[str] = None


@dataclass
class MLLMBatch:
    """
    Represents an active batch of MLLM requests.

    Manages the batch state including tokens, caches, and metadata
    for all requests being processed together.
    """

    uids: List[int]
    request_ids: List[str]
    y: mx.array  # Current token(s) for each request [batch_size]
    logprobs: List[Optional[mx.array]]  # Log probs for each request
    max_tokens: List[int]  # Max tokens per request
    num_tokens: List[int]  # Tokens generated per request
    cache: List[Any]  # BatchKVCache for language model
    requests: List[MLLMBatchRequest]  # Full request data

    def __len__(self) -> int:
        return len(self.uids)

    def index_of(self, uid: int) -> int:
        idx_map = getattr(self, "_uid_index", None)
        if idx_map is None or len(idx_map) != len(self.uids):
            idx_map = {u: i for i, u in enumerate(self.uids)}
            self._uid_index = idx_map
        try:
            return idx_map[uid]
        except KeyError:
            idx_map = {u: i for i, u in enumerate(self.uids)}
            self._uid_index = idx_map
            return idx_map[uid]

    def has_uid(self, uid: int) -> bool:
        idx_map = getattr(self, "_uid_index", None)
        if idx_map is None or len(idx_map) != len(self.uids):
            idx_map = {u: i for i, u in enumerate(self.uids)}
            self._uid_index = idx_map
        return uid in idx_map

    def _invalidate_uid_index(self) -> None:
        if hasattr(self, "_uid_index"):
            self._uid_index = None

    def filter(self, keep_idx: List[int]) -> None:
        """
        Filter batch to keep only requests at specified indices.

        Args:
            keep_idx: Indices of requests to keep
        """
        self.uids = [self.uids[k] for k in keep_idx]
        self.request_ids = [self.request_ids[k] for k in keep_idx]
        self.logprobs = [self.logprobs[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        self.requests = [self.requests[k] for k in keep_idx]
        self._invalidate_uid_index()

        keep_idx_array = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx_array]

        # Filter cache entries
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None
        for c in self.cache:
            if _CacheList is not None and isinstance(c, _CacheList):
                # CacheList (MoE models): filter each sub-cache independently
                for sc in c.caches:
                    if hasattr(sc, "filter"):
                        sc.filter(keep_idx_array)
            elif hasattr(c, "filter"):
                c.filter(keep_idx_array)

    def extend(self, other: "MLLMBatch") -> None:
        """
        Extend this batch with another batch's requests.

        Merges all metadata lists and extends cache layers. Both batches
        must have batch-aware caches (BatchKVCache/BatchMambaCache) — raw
        KVCache/ArraysCache cannot be extended.

        Args:
            other: Another MLLMBatch to merge into this one
        """
        self.uids.extend(other.uids)
        self.request_ids.extend(other.request_ids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs.extend(other.logprobs)
        self.max_tokens.extend(other.max_tokens)
        self.num_tokens.extend(other.num_tokens)
        self.requests.extend(other.requests)
        self._invalidate_uid_index()
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None
        for c, o in zip(self.cache, other.cache):
            if _CacheList is not None and isinstance(c, _CacheList):
                # CacheList (MoE models): extend each sub-cache independently
                for sc, so in zip(c.caches, o.caches):
                    sc.extend(so)
            else:
                c.extend(o)

    def extract_cache(self, idx: int) -> List[Any]:
        """
        Extract cache for a single request (for caching).

        Args:
            idx: Index of request in batch

        Returns:
            Cache state for that request
        """
        extracted = []
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None
        for c in self.cache:
            if _CacheList is not None and isinstance(c, _CacheList):
                # CacheList (MoE models): extract from each sub-cache
                sub_extracted = []
                for sc in c.caches:
                    if hasattr(sc, "extract"):
                        layer = sc.extract(idx)
                        if hasattr(layer, "keys") and layer.keys is not None:
                            layer.keys = mx.contiguous(layer.keys)
                            layer.values = mx.contiguous(layer.values)
                        sub_extracted.append(layer)
                    elif idx == 0:
                        sub_extracted.append(sc)
                    else:
                        sub_extracted.append(None)
                extracted.append(_CacheList(*sub_extracted))
            elif hasattr(c, "extract"):
                # Batched cache (BatchKVCache, BatchMambaCache) — extract single request
                layer = c.extract(idx)
                # Make extracted keys/values contiguous: BatchKVCache.extract()
                # returns sliced views that reference the full batch tensor.
                # Without contiguous(), the full batch tensor stays alive in memory
                # even after the batch is freed.
                if hasattr(layer, "keys") and layer.keys is not None:
                    layer.keys = mx.contiguous(layer.keys)
                    layer.values = mx.contiguous(layer.values)
                extracted.append(layer)
            elif idx == 0:
                # Unbatched cache (KVCache, ArraysCache) from single-request path —
                # return the cache itself since there's only one request
                extracted.append(c)
            else:
                extracted.append(None)
        return extracted


class MLLMBatchStats:
    """Statistics for MLLM batch generation."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.prompt_time: float = 0
        self.generation_tokens: int = 0
        self.generation_time: float = 0
        self.vision_encoding_time: float = 0
        self.num_images_processed: int = 0
        self.peak_memory: float = 0
        self.hybrid_kv_without_ssm_hits: int = 0
        self.hybrid_kv_without_ssm_tokens: int = 0
        self.last_hybrid_kv_without_ssm: Optional[Dict[str, Any]] = None
        self.last_cache_execution: Optional[Dict[str, Any]] = None
        self.last_native_mtp: Optional[Dict[str, Any]] = None
        self.last_native_mtp_skip: Optional[Dict[str, Any]] = None
        self.last_prefill_trace: Optional[Dict[str, Any]] = None
        self.last_turboquant_cache: Optional[Dict[str, Any]] = None

    def record_native_mtp(
        self,
        *,
        request_id: str,
        stats: MLLMNativeMTPStats,
        finish_reason: str,
        final_depth: int,
        fallback_reason: Optional[str] = None,
    ) -> None:
        self.last_native_mtp = stats.to_dict(
            request_id=request_id,
            finish_reason=finish_reason,
            final_depth=final_depth,
            fallback_reason=fallback_reason,
        )

    @property
    def prompt_tps(self) -> float:
        if self.prompt_time == 0:
            return 0
        return self.prompt_tokens / self.prompt_time

    @property
    def generation_tps(self) -> float:
        if self.generation_time == 0:
            return 0
        return self.generation_tokens / self.generation_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_time": self.prompt_time,
            "prompt_tps": self.prompt_tps,
            "generation_tokens": self.generation_tokens,
            "generation_time": self.generation_time,
            "generation_tps": self.generation_tps,
            "vision_encoding_time": self.vision_encoding_time,
            "num_images_processed": self.num_images_processed,
            "peak_memory": self.peak_memory,
            "hybrid_kv_without_ssm_hits": self.hybrid_kv_without_ssm_hits,
            "hybrid_kv_without_ssm_tokens": self.hybrid_kv_without_ssm_tokens,
            "last_hybrid_kv_without_ssm": self.last_hybrid_kv_without_ssm,
            "last_cache_execution": self.last_cache_execution,
            "last_native_mtp": self.last_native_mtp,
            "last_native_mtp_skip": self.last_native_mtp_skip,
            "last_prefill_trace": self.last_prefill_trace,
            "last_turboquant_cache": self.last_turboquant_cache,
        }



def _merge_caches(caches: List[List[Any]]) -> List[Any]:
    """
    Merge a list of per-request caches into batch-aware caches.

    Handles KVCache→BatchKVCache, RotatingKVCache→BatchRotatingKVCache,
    QuantizedKVCache→BatchKVCache (dequantize first),
    MambaCache/ArraysCache→BatchMambaCache, CacheList (recursive),
    and any type with a compatible .merge() class method.
    """
    from mlx_lm.models.cache import BatchKVCache, KVCache, RotatingKVCache, ArraysCache
    try:
        from mlx_lm.models.cache import MambaCache as _MambaCache
    except ImportError:
        _MambaCache = ArraysCache
    try:
        from mlx_lm.models.cache import QuantizedKVCache as _QuantizedKVCache
    except ImportError:
        _QuantizedKVCache = None
    try:
        from mlx_lm.models.cache import CacheList as _CacheList
    except ImportError:
        _CacheList = None
    try:
        from mlx_lm.generate import BatchRotatingKVCache
    except ImportError:
        BatchRotatingKVCache = None
    from .models.minimax_m3.cache import (
        BatchMiniMaxM3SparseCache,
        MiniMaxM3SparseCache,
    )

    batch_cache = []
    for i in range(len(caches[0])):
        layer_cache = caches[0][i]
        layer_caches = [c[i] for c in caches]

        try:
            if isinstance(layer_cache, MiniMaxM3SparseCache):
                batch_cache.append(
                    BatchMiniMaxM3SparseCache.merge(layer_caches)
                )
            elif _QuantizedKVCache is not None and isinstance(layer_cache, _QuantizedKVCache):
                # Dequantize all layers before merging as regular KVCache
                dequantized = []
                for qkv in layer_caches:
                    if qkv.keys is None:
                        dequantized.append(KVCache())
                    else:
                        kv = KVCache()
                        kv.keys = mx.dequantize(
                            qkv.keys[0], qkv.keys[1], qkv.keys[2],
                            qkv.group_size, qkv.bits,
                        )
                        kv.values = mx.dequantize(
                            qkv.values[0], qkv.values[1], qkv.values[2],
                            qkv.group_size, qkv.bits,
                        )
                        kv.offset = qkv.offset
                        dequantized.append(kv)
                batch_cache.append(BatchKVCache.merge(dequantized))
            elif isinstance(layer_cache, RotatingKVCache):
                if BatchRotatingKVCache is not None:
                    batch_cache.append(BatchRotatingKVCache.merge(layer_caches))
                else:
                    logger.warning(f"Layer {i}: RotatingKVCache but BatchRotatingKVCache unavailable")
                    batch_cache.append(BatchKVCache([0] * len(caches)))
            elif _is_tq_batch_api(layer_cache):
                merged = layer_cache
                for source in layer_caches[1:]:
                    merged.extend(source)
                batch_cache.append(merged)
            elif _is_kv_like(layer_cache):
                # TQ: .keys buffer is over-allocated in 256-token chunks.
                # BatchKVCache.merge() reads .keys directly, so convert to
                # properly sliced KVCache via .state before merging.
                if type(layer_cache).__name__ == _TQ_CLASS_NAME:
                    converted = []
                    for tq in layer_caches:
                        kv = KVCache()
                        k, v = tq.state
                        if k is not None and hasattr(k, 'shape'):
                            kv.keys = k
                            kv.values = v
                            kv.offset = tq.offset
                        converted.append(kv)
                    batch_cache.append(BatchKVCache.merge(converted))
                else:
                    batch_cache.append(BatchKVCache.merge(layer_caches))
            elif isinstance(layer_cache, (_MambaCache, ArraysCache)):
                from .utils.mamba_cache import BatchMambaCache
                batch_cache.append(BatchMambaCache.merge(layer_caches))
            elif _CacheList is not None and isinstance(layer_cache, _CacheList):
                # CacheList: merge each sub-cache independently across requests
                num_sub = len(layer_cache.caches)
                merged_subs = []
                for j in range(num_sub):
                    # Collect sub-cache j from all requests' CacheList at this layer
                    sub_caches = [[c.caches[j]] for c in layer_caches]
                    sub_merged = _merge_caches(sub_caches)[0]
                    merged_subs.append(sub_merged)
                batch_cache.append(_CacheList(*merged_subs))
            elif hasattr(layer_cache, "merge"):
                batch_cache.append(type(layer_cache).merge(layer_caches))
            else:
                logger.warning(f"Layer {i}: {type(layer_cache).__name__} has no merge(), using empty BatchKVCache")
                batch_cache.append(BatchKVCache([0] * len(caches)))
        except Exception as e:
            if isinstance(layer_cache, MiniMaxM3SparseCache):
                # Dropping the index lane would make every subsequent decode
                # retry fail less clearly. Let the prompt batch fail intact.
                raise
            logger.warning(f"Layer {i} merge failed ({type(layer_cache).__name__}), using fallback empty cache: {e}")
            if isinstance(layer_cache, (_MambaCache, ArraysCache)):
                from .utils.mamba_cache import BatchMambaCache
                batch_cache.append(BatchMambaCache(size=2, left_padding=[0] * len(caches)))
            else:
                batch_cache.append(BatchKVCache([0] * len(caches)))
    return batch_cache


def _ensure_batch_cache(cache: List[Any]) -> List[Any]:
    """Convert unbatched caches to batch-aware format for a single request.

    When a batch was created with a single request, _process_prompts() keeps
    raw KVCache/ArraysCache to preserve integer offsets (needed by Qwen3.5).
    When a second request needs to extend() into this batch, the cache must
    be converted to BatchKVCache/BatchMambaCache first.

    This wraps each layer cache using merge([cache]) which creates the
    batch-aware version with batch_size=1.
    """
    from mlx_lm.models.cache import KVCache, ArraysCache, BatchKVCache
    try:
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        QuantizedKVCache = None
    try:
        from mlx_lm.models.cache import RotatingKVCache
    except ImportError:
        RotatingKVCache = None
    try:
        from mlx_lm.models.cache import CacheList as _CacheList
    except ImportError:
        _CacheList = None
    try:
        from mlx_lm.generate import BatchRotatingKVCache
    except ImportError:
        BatchRotatingKVCache = None
    from .models.minimax_m3.cache import (
        BatchMiniMaxM3SparseCache,
        MiniMaxM3SparseCache,
    )

    converted = []
    for c in cache:
        if isinstance(c, BatchMiniMaxM3SparseCache):
            converted.append(c)
        elif isinstance(c, MiniMaxM3SparseCache):
            converted.append(BatchMiniMaxM3SparseCache.merge([c]))
        elif isinstance(c, BatchKVCache):
            converted.append(c)  # Already batch-aware
        elif QuantizedKVCache is not None and isinstance(c, QuantizedKVCache):
            # QuantizedKVCache (sibling of KVCache, both extend _BaseCache)
            # must be dequantized before merge — .keys is a tuple, not array
            dq = _dequantize_cache([c])
            if dq and len(dq) == 1:
                converted.append(BatchKVCache.merge([dq[0]]))
            else:
                # Dequant failed — use fresh KVCache
                converted.append(BatchKVCache.merge([KVCache()]))
        elif _is_tq_batch_api(c):
            converted.append(c)
        elif RotatingKVCache is not None and isinstance(c, RotatingKVCache):
            if BatchRotatingKVCache is not None:
                converted.append(BatchRotatingKVCache.merge([c]))
            else:
                converted.append(BatchKVCache.merge([c]))
        elif _is_kv_like(c):
            # TQ: convert via .state to avoid over-allocated buffer
            if type(c).__name__ == _TQ_CLASS_NAME:
                kv = KVCache()
                k, v = c.state
                if k is not None and hasattr(k, 'shape'):
                    kv.keys = k
                    kv.values = v
                    kv.offset = c.offset
                converted.append(BatchKVCache.merge([kv]))
            else:
                converted.append(BatchKVCache.merge([c]))
        elif isinstance(c, ArraysCache):
            from .utils.mamba_cache import BatchMambaCache
            converted.append(BatchMambaCache.merge([c]))
        elif _CacheList is not None and isinstance(c, _CacheList):
            # Recursively convert each sub-cache
            inner = _ensure_batch_cache(list(c.caches))
            converted.append(_CacheList(*inner))
        elif hasattr(c, "merge"):
            converted.append(type(c).merge([c]))
        else:
            # Unknown type — wrap as single-element merge via _merge_caches
            converted.append(c)
    return converted


class _BatchOffsetSafeCache:
    """Proxy that ensures cache.offset returns a scalar int, not mx.array.

    Several VL model attention layers (Qwen3.5, Qwen2.5-VL, Qwen2-VL, Qwen3-VL,
    Qwen3-VL-MoE, Qwen3-Omni-MoE) use cache.offset directly in slice operations
    like ``mask[..., :kv_seq_len]`` which requires a Python int. When multiple
    requests are batched, BatchKVCache.offset is an mx.array of per-request
    offsets, causing "Slice indices must be integers or None".

    This proxy wraps a BatchKVCache and intercepts .offset reads to return
    the **maximum** offset as a scalar int. Using max (not first element) ensures
    the attention mask is wide enough for ALL sequences in the batch, preventing
    broadcast shape mismatches when sequences have different lengths. The mask's
    built-in left_padding handling already masks out invalid positions for shorter
    sequences.

    Only applied during _step() when batch_size > 1. Single-request batches
    keep original KVCache objects (which already have int offsets).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    @property
    def offset(self):
        raw = self._inner.offset
        if isinstance(raw, mx.array):
            return (raw if raw.ndim == 0 else raw.max()).item()
        return raw

    @offset.setter
    def offset(self, value):
        self._inner.offset = value

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in _BatchOffsetSafeCache.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def __bool__(self):
        # BatchKVCache is always truthy when it exists (used in `if cache:` checks)
        return True

    def __len__(self):
        if hasattr(self._inner, "__len__"):
            return len(self._inner)
        # BatchKVCache doesn't implement __len__; return batch size from offset
        raw = self._inner.offset
        if isinstance(raw, mx.array) and raw.ndim > 0:
            return raw.shape[0]
        return 1

    def __iter__(self):
        if hasattr(self._inner, "__iter__"):
            return iter(self._inner)
        raise TypeError(f"'{type(self._inner).__name__}' object is not iterable")

    def __repr__(self):
        return f"_BatchOffsetSafeCache({self._inner!r})"


def _wrap_batch_caches(cache: List[Any]) -> List[Any]:
    """Wrap BatchKVCache objects with offset-safe proxies for VL model compat.

    Returns the list with BatchKVCache entries wrapped in _BatchOffsetSafeCache.
    Non-BatchKVCache entries (MambaCache, ArraysCache, etc.) pass through unchanged.
    """
    try:
        from mlx_lm.models.cache import BatchKVCache
    except ImportError:
        return cache

    wrapped = []
    for c in cache:
        if isinstance(c, BatchKVCache):
            wrapped.append(_BatchOffsetSafeCache(c))
        else:
            wrapped.append(c)
    return wrapped


def _offset_proxy_needed_for_model_type(model_type: str) -> bool:
    """Scalar-offset proxying is ONLY for Qwen-style VL language models.

    Qwen attention layers slice with ``cache.offset`` (requires a Python int)
    and take explicit position_ids, so their rope never reads cache.offset —
    flattening it is safe there. Every mlx_lm-style family served through this
    generator (gemma4, step3p5/step3p7, mimo_v2, zaya) instead ROPES from
    ``cache.offset`` and needs the raw per-row array: a flattened scalar
    silently ropes co-batched rows at the max row's position (F18, proven
    joined-vs-solo greedy divergence on Step-3.7 while fixed-gemma4 is
    byte-identical) and can hit the mx.fast.rope scalar-offset batch>1 kernel
    corruption (F16). Gate on the explicit config model_type (no name regex).
    """
    mt = str(model_type or "").lower()
    if mt.startswith("qwen"):
        return True
    # gemma4 stays wrapped: its patched attention reads the TRUE per-row
    # offset through the proxy (_gemma4_cache_rope_offset -> _inner), while
    # its model code has int(c.offset) sites (per-layer-inputs on E-models)
    # that require the scalar. Wrapping is therefore both safe and needed.
    if mt.startswith("gemma"):
        return True
    # Verified array-safe (offset used ONLY for rope, mlx_lm-style):
    # step3p5/step3p7, mimo_v2/mimo_v2_flash. Unknown families default to
    # UNWRAPPED because mlx_lm's own batch generation contract is per-row
    # array offsets; a family that needs the int proxy should be added
    # explicitly above.
    return False


class MLLMBatchGenerator:
    """Batch generator for Vision Language Models on Apple Metal.

    This is the low-level generation engine. The MLLMScheduler creates one
    instance and delegates all prefill/decode work to it.

    **Two-phase generation:**

    1. **Prefill** (``_process_prompts``):
       Vision encoding + language model forward pass, per-request.
       Each request gets its own KVCache/ArraysCache, then all are merged
       into batch-aware caches (BatchKVCache/BatchMambaCache) for decode.

    2. **Decode** (``step``):
       Language model generates one token for ALL active requests at once.
       Uses batched cache for efficient parallel generation.

    **Cache integration:**

    Receives cache objects (paged, memory-aware, legacy, disk) from the
    scheduler. Handles cache fetch in _process_prompts (before prefill)
    and exposes cache extraction via MLLMBatchResponse.prompt_cache
    (after generation, for store by scheduler).

    **Hybrid model support:**

    Pre-computes ``_hybrid_kv_positions`` and ``_hybrid_num_layers`` at init.
    Maintains ``HybridSSMStateCache`` for SSM state at prompt boundary.
    Uses ``_fix_hybrid_cache()`` to expand KV-only reconstructed caches.

    **Metal memory:**

    Sets ``mx.metal.set_cache_limit()`` at 25% of max working set,
    uses ``mx.async_eval()`` in prefill, ``mx.contiguous()`` on extracted
    cache. Restores old limits in ``close()``.

    Example::

        generator = MLLMBatchGenerator(model, processor)
        uids = generator.insert([request1, request2])
        while responses := generator.next():
            for resp in responses:
                print(f"Request {resp.request_id}: token={resp.token}")
    """

    # Generation stream for async eval
    _stream = None

    # Class-level defaults so partially-constructed instances (tests build
    # via __new__) can run _next without the media cache-limit machinery.
    _steady_cache_limit = None
    _vlm_cache_limit_tightened = False

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        max_tokens: int = 256,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        prefill_batch_size: int = 4,  # Smaller for MLLM due to vision overhead
        completion_batch_size: int = 16,  # Can be larger for text generation
        prefill_step_size: int = 1024,
        enable_vision_cache: bool = False,
        vision_cache_size: int = 16,
        paged_cache_manager: Optional[Any] = None,
        block_aware_cache: Optional[Any] = None,
        memory_aware_cache: Optional[Any] = None,
        prefix_cache: Optional[Any] = None,
        disk_cache: Optional[Any] = None,
        kv_cache_bits: int = 0,
        kv_cache_group_size: int = 64,
        ssm_state_cache_size: int = DEFAULT_SSM_COMPANION_ENTRIES,
        ssm_state_cache_max_mb: Optional[int] = 0,
        ssm_state_disk_store: Optional[Any] = None,
        ssm_state_cache_model_key: str = "",
        enable_prefix_cache: bool = True,
        uses_zaya_cache: Optional[bool] = None,
        mixed_attention_cache_model: bool = False,
    ):
        """
        Initialize MLLM batch generator.

        Args:
            model: The VLM model (must have model.language_model)
            processor: The VLM processor for tokenization and image processing
            max_tokens: Default max tokens per request
            stop_tokens: Set of stop token IDs
            sampler: Sampling function (default: argmax)
            prefill_batch_size: Max requests to prefill together
            completion_batch_size: Max requests for completion batching
            prefill_step_size: Tokens to process per prefill step
            enable_vision_cache: Enable vision embedding caching
            vision_cache_size: Max entries in vision cache
            paged_cache_manager: Optional PagedCacheManager
            block_aware_cache: Optional BlockAwarePrefixCache
            memory_aware_cache: Optional MemoryAwarePrefixCache
            prefix_cache: Optional PrefixCacheManager (legacy)
            disk_cache: Optional DiskCacheManager (L2)
            kv_cache_bits: Quantization bits (0=none, 4=q4, 8=q8)
            kv_cache_group_size: Quantization group size
            ssm_state_cache_size: Max retained RAM entries in
                HybridSSMStateCache (LRU); 0 keeps companion state SSD-only.
            ssm_state_cache_max_mb: Approximate retained-memory budget for
                companion SSM state; 0 keeps it SSD-only. Large hybrid/VLM
                entries still persist to L2 but are not backfilled into RAM.
            ssm_state_disk_store: Optional scheduler-owned L2 store for SSM
                companion states. Required for true hybrid cache restore after
                server restart; paged KV blocks alone are not enough.
            ssm_state_cache_model_key: Opaque model/cache identity mixed into
                companion keys so same-token prompts from different bundles or
                cache topologies cannot collide.
            enable_prefix_cache: Enables SSM companion cache work for hybrid
                prefix-cache hits/stores. When false, no hidden companion
                lookup/store/re-derive work is performed.
            uses_zaya_cache: True when the model uses ZAYA's typed CCA cache
                contract. ZAYA is hybrid-shaped but must use zaya_cca_v1
                paged blocks, not the generic SSM companion cache.
            mixed_attention_cache_model: True for Gemma-style mixed sliding
                window/full attention cache layouts. These use rotating-window
                metadata, not SSM state, and must not report paged+ssm telemetry.
        """
        self.model = model
        self.processor = processor
        # Cross-span prefill transient model, learned from completed spans and
        # consumed by the whole-span admission check. Instance state, not
        # module state: it is a property of THIS loaded model, and the
        # scheduler drops the generator when it swaps models.
        self._span_peak_model: tuple[float, float] | None = None
        self._span_peak_samples: int = 0
        self._span_largest_peak: int = 0
        # Largest context the fit was actually measured over, so the check can
        # refuse to extrapolate far past its own evidence.
        self._span_peak_max_context: int = 0
        # Cross-TURN peak walk: one (final context, absolute peak) point per
        # completed span. The within-span fit above cannot see the between-turn
        # allocator/fragmentation walk (+5.0-5.6GB per ~5.6k-token turn,
        # measured) that aborts a deep incremental conversation; this deque
        # feeds the turn-peak admission valve. Bounded to recent turns so a
        # residency change (eviction, unload, pool reconfig) ages out of the
        # fit instead of permanently inflating it.
        self._turn_peak_walk: deque[tuple[int, int]] = deque(
            maxlen=_TURN_PEAK_WALK_MAXLEN
        )
        # The span whose peak the NEXT deep turn's gauge reading belongs to.
        # Zeroed on a refusal so the retry's no-forward-ran reading is not
        # recorded as a walk point (it would drag the fit down and re-admit
        # the turn that was just declined).
        self._last_deep_span_tokens: int = 0
        self.paged_cache_manager = paged_cache_manager
        self.block_aware_cache = block_aware_cache
        self.memory_aware_cache = memory_aware_cache
        self.prefix_cache = prefix_cache
        self.disk_cache = disk_cache
        self._kv_cache_bits = kv_cache_bits
        self._kv_cache_group_size = kv_cache_group_size
        self._uses_zaya_cache = bool(
            uses_zaya_cache
            if uses_zaya_cache is not None
            else _model_uses_zaya_cache_contract(
                getattr(model, "language_model", model)
            )
        )
        self._mixed_attention_cache_model = bool(mixed_attention_cache_model)
        # Session-scoped learned MTP depth profiles. Lifetime == this
        # generator (== one loaded model in one process), so bundle/config
        # reloads and new PIDs invalidate them for free.
        self._native_mtp_profiles = NativeMTPProfileStore()

        self._prefix_cache_enabled = bool(enable_prefix_cache)
        self._ssm_companion_enabled = bool(
            self._prefix_cache_enabled
            and not self._uses_zaya_cache
            and (
                block_aware_cache is not None
                or memory_aware_cache is not None
                or prefix_cache is not None
                or disk_cache is not None
            )
        )

        # Companion SSM state cache for hybrid models (MambaCache + KVCache).
        # Stores SSM layer states at prompt boundary so hybrid cache HITs can
        # skip the full prefix instead of wasting the KV cache hit. It is
        # intentionally absent when prefix cache is disabled, so benchmark and
        # cache-bypass runs do not pay hidden SSM clone/re-derive costs.
        self._ssm_state_cache = (
            HybridSSMStateCache(
                max_entries=ssm_state_cache_size,
                model_key=ssm_state_cache_model_key,
                disk_store=ssm_state_disk_store,
                max_bytes=(
                    int(ssm_state_cache_max_mb) * 1024 * 1024
                    if ssm_state_cache_max_mb is not None
                    else None
                ),
            )
            if self._ssm_companion_enabled
            else None
        )

        # Async rederive queue for MLLM thinking models. When we capture SSM
        # state post-full-prefill (is_complete=False, gpl-contaminated), we
        # queue a deferred clean-prefill task that runs during idle cycles
        # (no active requests). The clean prefill processes just prompt[:-gpl]
        # tokens so the captured state matches its key — future fetches hit
        # with is_complete=True. Capped at 20 entries.
        self._ssm_rederive_queue: List[Tuple[Any, ...]] = []
        self._ssm_rederive_queue_max = 20

        # Get language model for text generation
        self.language_model = getattr(model, "language_model", model)
        self._model_type = _runtime_model_type(model)

        # Check if this is actually a VLM with separate language model
        self.is_vlm = hasattr(model, "language_model")
        if self.is_vlm:
            logger.info(
                "MLLMBatchGenerator: Using VLM's language_model for batched generation"
            )
        else:
            logger.warning(
                "MLLMBatchGenerator: Model does not have language_model, using model directly"
            )

        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self._decode_trace = os.environ.get("VMLINUX_DECODE_TRACE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._decode_trace_every = max(
            1, int(os.environ.get("VMLINUX_DECODE_TRACE_EVERY", "64") or "64")
        )
        self._decode_trace_count = 0
        self._decode_trace_model_s = 0.0
        self._decode_trace_sample_s = 0.0

        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.prefill_step_size = prefill_step_size

        # Kimi K2.6 (kimi_k25) — 191 GB 2-bit MoE bundles need a much smaller
        # prefill chunk to stay under Metal's ~60 s command-buffer watchdog.
        # See research/KIMI-K2.6-VMLX-INTEGRATION.md §1 — the Python reference
        # `jang_tools.kimi_prune.generate_vl` uses 32 tokens/chunk; larger
        # chunks trigger kIOGPUCommandBufferCallbackErrorTimeout on first
        # VL forward. Detection is from config.model_type at outer OR text
        # level so both the bundle's Kimi_K25ForConditionalGeneration wrapper
        # AND an inner text_config variant trip the override.
        try:
            _mt_outer = getattr(getattr(model, "config", None), "model_type", None)
            _mt_inner = getattr(
                getattr(getattr(model, "config", None), "text_config", None),
                "model_type",
                None,
            )
            if "kimi_k25" in (_mt_outer, _mt_inner):
                _kimi_step = 32
                if self.prefill_step_size > _kimi_step:
                    logger.info(
                        "Kimi K2.6 detected — clamping prefill_step_size %d → %d "
                        "to stay under Metal command-buffer watchdog",
                        self.prefill_step_size, _kimi_step,
                    )
                    self.prefill_step_size = _kimi_step
        except Exception:
            # Non-fatal: if detection fails, default chunk size still avoids
            # the worst of the OOM guard via mlxstudio#83 auto-chunk fallback.
            pass

        # Request management
        self.unprocessed_requests: List[MLLMBatchRequest] = []
        self.active_batch: Optional[MLLMBatch] = None
        self.uid_counter = 0
        self._prefill_errors: List[MLLMBatchResponse] = []  # Failed requests from prefill

        # Statistics
        self._stats = MLLMBatchStats()

        # Pre-compute hybrid cache template info (avoids make_cache() per request).
        # The callable may live behind a JANG-affine compatibility wrapper, so
        # resolve and retain the real owner instead of checking only the first
        # language_model object.
        self._hybrid_kv_positions: Optional[List[int]] = None
        self._hybrid_num_layers: Optional[int] = None
        (
            self._cache_model,
            self._cache_model_path,
            _hybrid_template,
            _hybrid_positions,
            _hybrid_detection_error,
        ) = _hybrid_cache_layout(model, self.language_model)
        if _hybrid_template is not None and _hybrid_positions is not None:
            self._hybrid_num_layers = len(_hybrid_template)
            self._hybrid_kv_positions = _hybrid_positions
            if self._cache_model is not self.language_model:
                logger.info(
                    "MLLMBatchGenerator: resolved make_cache through wrapper path %s (%s)",
                    self._cache_model_path,
                    type(self._cache_model).__name__,
                )
        elif _hybrid_detection_error and self._cache_model is not None:
            logger.warning(
                "Failed to pre-compute hybrid cache info from %s: %s",
                self._cache_model_path,
                _hybrid_detection_error,
            )

        # Pre-computed bool: is this a hybrid model (SSM + attention)?
        # Used throughout _process_prompts and _run_vision_encoding to gate
        # hybrid-specific logic (SSM companion cache, chunked prefill skip, etc.)
        self._is_hybrid: bool = _uses_ssm_companion_cache(
            self._hybrid_kv_positions,
            self._hybrid_num_layers,
            mixed_attention=self._mixed_attention_cache_model,
        )
        _warn_if_hybrid_detection_disabled(
            model=model,
            language_model=self.language_model,
            is_hybrid=self._is_hybrid,
            owner_path=self._cache_model_path,
            template=_hybrid_template,
            error=_hybrid_detection_error,
        )

        # Vision embedding cache for repeated images
        self.vision_cache = VisionEmbeddingCache(
            max_pixel_entries=vision_cache_size,
            enabled=enable_vision_cache,
        )
        if enable_vision_cache:
            logger.info(
                f"MLLMBatchGenerator: Vision cache enabled (size={vision_cache_size})"
            )

        # Generation stream
        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())

        # Memory management
        self._old_wired_limit = None
        self._old_cache_limit = None
        self._steady_cache_limit = None
        self._vlm_cache_limit_tightened = False
        self._tight_memory_prefill_drain = False
        if mx.metal.is_available():
            # Use non-deprecated API when available (MLX ≥ 0.25)
            _set_cache = getattr(mx, 'set_cache_limit', None) or mx.metal.set_cache_limit
            if True:  # Always set Metal limits (smelt mode doesn't need special limits)
                active_mem, max_ws = get_effective_metal_working_set_bytes(mx)
                self._old_wired_limit = mx.set_wired_limit(max_ws) if max_ws > 0 else None
                # Set Metal allocator cache limit.
                # mlxstudio#78: previously was a hard `max_ws * 0.25`, which
                # on a 64GB M4 Max loading Gemma-4-31B (~41GB active) would
                # reserve 12GB for cache on top of 41GB model → 53GB required
                # vs 48GB max working set → Metal command buffer OOM on the
                # FIRST request before a single token is generated.
                #
                # New policy: cap at min(25% of max_ws, 50% of FREE memory
                # after model load). Floor at 512MB. This keeps the original
                # behavior on machines with plenty of headroom (bounds the
                # free-list so the OS can reclaim memory when pressured)
                # while adapting on tight-memory systems where the model
                # already consumed most of the budget.
                try:
                    active = active_mem
                    if active <= 0:
                        active = (
                            getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
                        )()
                    free = max(0, max_ws - active) if max_ws > 0 else 0
                    # Base policy: 25% of max_ws (unchanged semantics on
                    # big-headroom systems).
                    base_limit = int(max_ws * 0.25)
                    # Safety cap: don't reserve more than 50% of FREE
                    # memory, so activations + KV + attention_scores have
                    # the other half to live in without forcing the
                    # allocator to release pooled blocks back to Metal.
                    safety_limit = int(free * 0.5)
                    cache_limit = max(
                        512 * 1024 * 1024, min(base_limit, safety_limit)
                    )
                    if max_ws > 0:
                        self._old_cache_limit = _set_cache(cache_limit)
                        self._steady_cache_limit = cache_limit
                        self._tight_memory_prefill_drain = safety_limit < base_limit
                        logger.info(
                            f"Metal cache limit set to {cache_limit / (1024**3):.2f}GB "
                            f"(max_ws={max_ws / (1024**3):.1f}GB, "
                            f"active={active / (1024**3):.1f}GB, "
                            f"free={free / (1024**3):.1f}GB; "
                            f"base={base_limit / (1024**3):.2f}GB "
                            f"safety={safety_limit / (1024**3):.2f}GB; "
                            f"mlxstudio#78)"
                        )
                    if safety_limit < base_limit:
                        logger.warning(
                            "Tight-memory configuration detected: model is "
                            "using a large fraction of max working set. "
                            "Cache limit adjusted downward. If requests OOM, "
                            "try a more aggressively quantized model or "
                            "reduce prompt length. (mlxstudio#78)"
                        )
                except Exception as e:
                    logger.debug(f"Metal cache limit not available: {e}")
            else:
                logger.info("Disk-streaming mode: skipping wired limit + cache limit override")

    def _drain_tight_memory_allocator(self, reason: str) -> Optional[str]:
        """Synchronize and clear MLX allocator state on tight-memory MLLM paths.

        Large MLLM/JANG bundles can leave only a few GB of Metal working-set
        headroom after weights are resident. In that regime the normal bounded
        allocator free-list is not enough between back-to-back prefills: stale
        queued work or reusable buffers from request N can make request N+1 die
        in Metal before Python can raise a recoverable prefill error.

        This is a lifecycle drain only. It does not alter prompts, sampling,
        cache keys, cache content, or model outputs.
        """
        if not self._tight_memory_prefill_drain:
            return None
        if os.environ.get("VMLINUX_MLLM_TIGHT_MEMORY_DRAIN", "1").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return None
        try:
            if MLLMBatchGenerator._stream is not None:
                try:
                    mx.synchronize(MLLMBatchGenerator._stream)
                except RuntimeError as exc:
                    if "There is no Stream" not in str(exc):
                        raise
                    mx.synchronize()
            else:
                mx.synchronize()
        except Exception as exc:
            logger.debug("Tight-memory MLLM prefill drain synchronize skipped: %s", exc)
        method = clear_mlx_memory_cache(mx=mx, log=logger)
        try:
            active, max_ws = get_effective_metal_working_set_bytes(mx)
            free = max(0, max_ws - active) if max_ws > 0 else 0
            logger.info(
                "Tight-memory MLLM allocator drain (%s): method=%s "
                "active=%.1fGB max_ws=%.1fGB free=%.1fGB",
                reason,
                method or "none",
                active / (1024**3),
                max_ws / (1024**3) if max_ws else 0.0,
                free / (1024**3),
            )
        except Exception:
            logger.info(
                "Tight-memory MLLM allocator drain (%s): method=%s",
                reason,
                method or "none",
            )
        return method

    def close(self) -> None:
        """Release resources and reset wired/cache limits."""
        if self._old_wired_limit is not None:
            try:
                if MLLMBatchGenerator._stream is not None:
                    mx.synchronize(MLLMBatchGenerator._stream)
                else:
                    mx.synchronize()
            except RuntimeError as e:
                # Shutdown can run outside the generation stream's owner
                # thread. MLX then rejects the stream handle as thread-local;
                # a bare synchronize drains pending work without resolving that
                # stale handle and still lets us restore global Metal limits.
                if "There is no Stream" not in str(e):
                    raise
                mx.synchronize()
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None
        if self._old_cache_limit is not None:
            try:
                _set_cache = getattr(mx, 'set_cache_limit', None) or mx.metal.set_cache_limit
                _set_cache(self._old_cache_limit)
            except Exception:
                pass
            self._old_cache_limit = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def insert(
        self,
        requests: List[MLLMBatchRequest],
        caches: Optional[List[Optional[List[Any]]]] = None,
    ) -> List[int]:
        """
        Insert requests for batch processing with optional prompt caches.

        Args:
            requests: List of MLLMBatchRequest to process
            caches: Optional list of prompt caches, one per request. None means no cache.

        Returns:
            List of UIDs assigned to requests
        """
        if caches is None:
            caches = [None] * len(requests)

        uids = []
        for req, c in zip(requests, caches):
            req.uid = self.uid_counter
            self.uid_counter += 1
            if c is not None:
                req.prompt_cache = c
            self.unprocessed_requests.append(req)
            uids.append(req.uid)

        # Sort by estimated complexity (no images = simpler)
        self.unprocessed_requests = sorted(
            self.unprocessed_requests,
            key=lambda x: (
                0 if not x.images and not x.videos else 1,
                len(x.images or []) + len(x.videos or []),
            ),
        )

        logger.debug(f"Inserted {len(requests)} requests, UIDs: {uids}")
        return uids

    def remove(self, uids: List[int]) -> None:
        """
        Remove requests from processing.

        Args:
            uids: List of UIDs to remove
        """
        uid_set = set(uids)

        # Remove from active batch
        if self.active_batch is not None:
            for index, uid in enumerate(self.active_batch.uids):
                if uid not in uid_set:
                    continue
                request = self.active_batch.requests[index]
                mtp_state = getattr(request, "_native_mtp_state", None)
                if mtp_state is None:
                    continue
                try:
                    self._abandon_pending_native_mtp_verify(
                        mtp_state, getattr(self.active_batch, "cache", None)
                    )
                    _native_mtp_log_stats(
                        request.request_id,
                        mtp_state.stats,
                        "cancelled",
                        mtp_state.mtp_cache,
                    )
                    self._stats.record_native_mtp(
                        request_id=request.request_id,
                        stats=mtp_state.stats,
                        finish_reason="cancelled",
                        final_depth=mtp_state.depth,
                        fallback_reason=mtp_state.ar_fallback_reason,
                    )
                    # Cancelled requests deliberately do NOT update the
                    # session profile — an aborted request proves nothing
                    # about depth value.
                except Exception:
                    logger.debug(
                        "MLLM MTP cancellation telemetry publication failed",
                        exc_info=True,
                    )
                try:
                    delattr(request, "_native_mtp_state")
                except AttributeError:
                    pass
            keep_idx = [
                i for i, uid in enumerate(self.active_batch.uids) if uid not in uid_set
            ]
            if keep_idx:
                self.active_batch.filter(keep_idx)
            else:
                self.active_batch = None

        # Remove from unprocessed
        self.unprocessed_requests = [
            r for r in self.unprocessed_requests if r.uid not in uid_set
        ]

    def request_graceful_stop(self, uid: int) -> bool:
        """Finish one active row on its next materialized decode step.

        API tool parsers may know that a native call is complete before the
        model emits EOS.  Removing that row is an abort and deliberately drops
        its prompt cache.  Marking it here instead lets ``_next()`` produce the
        normal finished response, including native-MTP rollback and the exact
        prompt-boundary cache handed to scheduler cleanup.

        The caller owns ``MLLMScheduler._batch_lock``.  A row that has not yet
        entered the active batch cannot have emitted a parseable tool call, so
        it is intentionally not searched in ``unprocessed_requests``.
        """
        batch = self.active_batch
        if batch is None:
            return False
        for index, active_uid in enumerate(batch.uids):
            if int(active_uid) != int(uid):
                continue
            request = batch.requests[index]
            request._vmlx_graceful_stop_requested = True
            return True
        return False

    def _preprocess_request(self, request: MLLMBatchRequest) -> None:
        """
        Preprocess a single MLLM request (vision encoding).

        This prepares the inputs by:
        1. Processing images/videos through the processor
        2. Tokenizing the prompt with image tokens
        3. Running vision encoder to get features

        Uses vision cache to skip processing for repeated images.
        **Fast-path**: text-only requests (no images/videos) skip the full
        VLM processor pipeline and use the tokenizer directly, avoiding
        ~100ms+ of vision processor overhead per request.

        Args:
            request: Request to preprocess
        """
        tic = time.perf_counter()
        preserved_private_kwargs = {
            key: value
            for key, value in (request.extra_kwargs or {}).items()
            if isinstance(key, str) and key.startswith("_vmlx_")
        }

        # FAST PATH: text-only requests skip VLM processor entirely.
        # For API-driven workloads (e.g. MMLU benchmarks with 14k short requests),
        # per-request VLM processor overhead dominates. The tokenizer alone is
        # sufficient when no images or videos are present.
        is_text_only = not request.images and not request.videos and not request.audio
        if is_text_only:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            # Use add_special_tokens=False because the prompt has already been
            # through apply_chat_template() which embeds BOS/EOS tokens.
            # Calling encode() with default add_special_tokens=True would
            # double-add BOS, misaligning prefix cache keys.
            try:
                token_ids = tokenizer.encode(request.prompt, add_special_tokens=False)
            except TypeError:
                # Tokenizer doesn't support add_special_tokens kwarg.
                # Fall through to the full VLM processor path to avoid
                # double-BOS from default add_special_tokens=True on
                # an already-formatted prompt.
                logger.debug(
                    f"Tokenizer {type(tokenizer).__name__} does not support "
                    f"add_special_tokens; falling back to VLM processor path"
                )
                is_text_only = False  # force slow path
            if is_text_only:
                request.input_ids = mx.array(token_ids)
                request.pixel_values = None
                request.attention_mask = None
                request.image_grid_thw = None
                request.video_pixel_values = None
                request.video_grid_thw = None
                request.extra_kwargs = dict(preserved_private_kwargs)
                self._raise_if_prompt_over_limit(
                    request,
                    source="tokenized VLM text prompt",
                )
                processing_time = time.perf_counter() - tic
                logger.debug(
                    f"Text-only fast-path for {request.request_id}: "
                    f"{len(token_ids)} tokens ({processing_time*1000:.1f}ms)"
                )
                return

        from mlx_vlm.utils import prepare_inputs

        # Collect media for native processor ingestion. Videos must stay on the
        # processor's ``videos=`` path so Qwen-style processors return
        # pixel_values_videos + video_grid_thw matching the <|video_pad|> token.
        all_images = []
        video_inputs = []
        video_cache_sources = []
        video_sample_fps: List[float] = []
        video_sample_timestamps: List[List[float]] = []
        all_audio = []

        if request.images:
            from .models.mllm import process_image_input

            for img in request.images:
                try:
                    path = process_image_input(img)
                    all_images.append(path)
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")

        if request.videos:
            from .models.mllm import (
                DEFAULT_FPS,
                MAX_FRAMES,
                process_video_input,
            )
            from mlx_vlm.video_generate import fetch_video

            fps = request.video_fps or DEFAULT_FPS
            max_frames = request.video_max_frames or MAX_FRAMES

            for video in request.videos:
                try:
                    video_path = process_video_input(video)
                    video_cache_sources.append(video_path)
                    video_result = fetch_video(
                        {"video": video_path, "fps": fps, "max_frames": max_frames},
                        return_video_sample_fps=True,
                    )
                    if isinstance(video_result, tuple) and len(video_result) == 2:
                        video_input, sample_fps = video_result
                    else:
                        video_input, sample_fps = video_result, fps
                    video_inputs.append(video_input)
                    video_sample_fps.append(float(sample_fps))
                    video_sample_timestamps.append(
                        _sampled_video_timestamps(
                            video_path,
                            len(video_input),
                            float(sample_fps),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to process video: {e}")
            if request.videos and not video_inputs:
                raise ValueError("All video inputs failed to process")

        if request.audio:
            from .models.mllm import process_audio_input

            for audio in request.audio:
                try:
                    all_audio.append(process_audio_input(audio))
                except Exception as e:
                    logger.warning(f"Failed to process audio: {e}")
            if request.audio and not all_audio:
                raise ValueError("All audio inputs failed to process")

        if all_images or video_inputs or all_audio:
            if _apply_vlm_image_request_cache_limit():
                self._vlm_cache_limit_tightened = True
            mx.clear_cache()

        # Check pixel cache first
        media_cache_sources = all_images + video_cache_sources + all_audio
        pixel_cache_prompt = request.prompt
        if request.image_token_budget is not None:
            pixel_cache_prompt += (
                f"\n\x00vmlx:image_token_budget={int(request.image_token_budget)}"
            )
        _mllm_bypass = bool(getattr(request, "_bypass_prefix_cache", False))
        cached_pixels = None
        if not _mllm_bypass:
            cached_pixels = self.vision_cache.get_pixel_cache(
                media_cache_sources, pixel_cache_prompt
            )
        if cached_pixels is not None:
            # Cache hit - use cached pixel values
            request.input_ids = cached_pixels.input_ids
            request.pixel_values = cached_pixels.pixel_values
            request.attention_mask = cached_pixels.attention_mask
            request.image_grid_thw = cached_pixels.image_grid_thw
            request.video_pixel_values = cached_pixels.video_pixel_values
            request.video_grid_thw = cached_pixels.video_grid_thw
            request.extra_kwargs = dict(cached_pixels.extra_kwargs)
            self._raise_if_prompt_over_limit(
                request,
                source="cached tokenized VLM media prompt",
            )

            logger.debug(
                f"Pixel cache HIT for request {request.request_id}: "
                f"saved {cached_pixels.processing_time:.2f}s"
            )
            return

        # Cache miss - process images
        # Get model config
        model_config = getattr(self.model, "config", None)
        image_token_index = (
            getattr(model_config, "image_token_index", None) if model_config else None
        )

        # Prepare inputs using mlx_vlm.
        # prepare_inputs has a BaseImageProcessor path that hardcodes split("<image>").
        # Models using a different image token (e.g. Gemma 4 uses "<|image|>") never
        # get their images processed through that path. When the prompt has images but
        # no "<image>" literal, bypass prepare_inputs and call process_inputs directly
        # which invokes the processor's native __call__ (handles any image token format).
        if _should_use_safe_processor_path(
            self.processor,
            has_image_literal="<image>" in request.prompt,
            has_images=bool(all_images),
        ) or bool(video_inputs) or bool(all_audio):
            inputs = _call_processor_direct(
                self.processor,
                prompts=request.prompt,
                images=all_images,
                videos=video_inputs,
                video_fps=video_sample_fps,
                video_timestamps=video_sample_timestamps,
                audio=all_audio,
                add_special_tokens=False,
                image_token_budget=request.image_token_budget,
            )
        else:
            inputs = _as_input_mapping(prepare_inputs(
                self.processor,
                images=all_images if all_images else None,
                prompts=request.prompt,
                image_token_index=image_token_index,
            ))

        # Issue #56 Bug 1 root fix — normalize input_ids to mx.int32.
        # mlx_vlm's various processors (Mistral3 / Pixtral / Gemma4 / Qwen3.5-VL)
        # return input_ids as numpy arrays, torch tensors, or mx arrays with
        # varying dtypes depending on the upstream transformers version. When a
        # quantized embedding (`nn.QuantizedEmbedding` packed uint32 weights)
        # receives a numpy/torch index or an mx int64, MLX raises
        # `ValueError: Cannot index mlx array using the given type` — and the
        # batched engine's outer prefill-except then silently queued an empty
        # "stop" response (#56). The SimpleEngine path never hit this because
        # mlx_vlm.generate() does its own dtype normalization before forward.
        def _ensure_mx_array(x, target_dtype=None):
            """Normalize numpy / torch / list / mx inputs to an mx.array,
            optionally casting to a specific dtype. Pixtral / Mistral 3 /
            Qwen3.5-VL processors all return different wire formats; the
            batched engine then passes the raw value straight into forward,
            which chokes with either `Cannot index mlx array using the given
            type` (QuantizedEmbedding) or `Cannot interpret mlx.core.bfloat16
            as a data type` (numpy.astype against an mx dtype). Normalizing
            once here makes all downstream layer calls match the SimpleEngine
            path."""
            if x is None:
                return None
            if not isinstance(x, mx.array):
                try:
                    if hasattr(x, "tolist"):
                        x = mx.array(x.tolist())
                    else:
                        x = mx.array(x)
                except Exception:
                    return x  # give up gracefully, downstream will error
            if target_dtype is not None and x.dtype != target_dtype:
                try:
                    x = x.astype(target_dtype)
                except Exception:
                    pass
            return x

        # Issue #56 — normalize input_ids + pixel_values + attention_mask
        # before storing on the request. Covers Mistral 3 / Pixtral,
        # Qwen3.5-VL, Gemma 4, and future VLM families without having to
        # special-case each processor's output format.
        request.input_ids = _ensure_mx_array(inputs.get("input_ids"), mx.int32)
        pixel_values = inputs.get("pixel_values")
        video_pixel_values = inputs.get("pixel_values_videos")
        request.pixel_values = _ensure_mx_array(pixel_values)
        request.video_pixel_values = _ensure_mx_array(video_pixel_values)
        request.attention_mask = _ensure_mx_array(inputs.get("attention_mask"))

        # Extract extra kwargs
        request.extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "pixel_values_videos", "attention_mask"]
        }
        request.extra_kwargs.update(preserved_private_kwargs)
        request.image_grid_thw = _ensure_mx_array(
            request.extra_kwargs.pop("image_grid_thw", None), mx.int32
        )
        if "video_grid_thw" in request.extra_kwargs:
            request.video_grid_thw = _ensure_mx_array(
                request.extra_kwargs.pop("video_grid_thw"), mx.int32
            )
        else:
            request.video_grid_thw = None
        request.audio_codes = _ensure_mx_array(
            request.extra_kwargs.pop("audio_codes", None), mx.int32
        )
        request.audio_embeds = _ensure_mx_array(
            request.extra_kwargs.pop("audio_embeds", None)
        )
        input_features = request.extra_kwargs.pop("input_features", None)
        input_features_mask = request.extra_kwargs.pop("input_features_mask", None)
        audio_features = request.extra_kwargs.pop("audio_features", None)
        # Chunked-audio metadata (dots3-style processors): padded mel chunks
        # are useless to the tower without the true per-chunk sample/token
        # lengths and the chunks-per-audio grouping — deriving them from the
        # PADDED shape produced a loud 38-vs-750 scatter mismatch.
        request.audio_chunk_meta = {
            key: request.extra_kwargs.pop(key)
            for key in (
                "chunk_sample_lens",
                "chunk_token_lens",
                "audio_chunk_counts",
                "chunk_audio_indices",
            )
            if key in request.extra_kwargs
        } or None
        request.audio_features = _ensure_mx_array(
            input_features if input_features is not None else audio_features
        )
        request.audio_features_mask = _ensure_mx_array(
            input_features_mask if input_features is not None else None, mx.bool_
        )
        request.audio_features_are_raw_input_features = input_features is not None
        if (
            all_audio
            and self._model_type == "mimo_v2"
            and not any(
                item is not None
                for item in (
                    request.audio_codes,
                    request.audio_embeds,
                    request.audio_features,
                )
            )
        ):
            request.audio_codes = _ensure_mx_array(
                _build_mimo_audio_codes_from_paths(
                    model=self.model,
                    processor=self.processor,
                    audio_paths=all_audio,
                    input_ids=request.input_ids,
                ),
                mx.int32,
            )
            if request.audio_codes is not None:
                audio_token_id = _mimo_audio_token_id(self.model)
                if audio_token_id is not None:
                    audio_groups = (
                        int(request.audio_codes.shape[0])
                        + _mimo_audio_group_size(self.model)
                        - 1
                    ) // _mimo_audio_group_size(self.model)
                    request.input_ids, request.attention_mask = (
                        _expand_mimo_audio_token_placeholders(
                            input_ids=request.input_ids,
                            attention_mask=request.attention_mask,
                            token_id=audio_token_id,
                            target_count=audio_groups,
                        )
                    )
        if all_audio and not any(
            item is not None
            for item in (
                request.audio_codes,
                request.audio_embeds,
                request.audio_features,
            )
        ):
            raise UnsupportedMediaModalityError(
                "audio",
                (
                    "raw audio reached the VLM processor, but the processor "
                    "returned no audio_codes, audio_embeds, or audio_features. "
                    "A real waveform-to-MiMo-audio-codes bridge is required; "
                    "continuing as text-only would hide an unsupported audio path."
                ),
                family=str(self._model_type or "mllm"),
                request_id=request.request_id,
            )

        self._raise_if_prompt_over_limit(
            request,
            source="tokenized VLM media prompt",
        )

        processing_time = time.perf_counter() - tic

        # Store in pixel cache for future reuse
        if (
            not _mllm_bypass
            and media_cache_sources
            and (
                request.pixel_values is not None
                or request.video_pixel_values is not None
            )
        ):
            self.vision_cache.set_pixel_cache(
                images=media_cache_sources,
                prompt=pixel_cache_prompt,
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                image_grid_thw=request.image_grid_thw,
                video_pixel_values=request.video_pixel_values,
                video_grid_thw=request.video_grid_thw,
                extra_kwargs=request.extra_kwargs,
                processing_time=processing_time,
            )

        self._stats.num_images_processed += len(media_cache_sources)
        self._stats.vision_encoding_time += processing_time

        logger.debug(
            f"Preprocessed request {request.request_id}: "
            f"{len(all_images)} images, {len(video_inputs)} videos, "
            f"{request.input_ids.size if request.input_ids is not None else 0} tokens "
            f"({processing_time:.2f}s)"
        )

    def _raise_if_prompt_over_limit(
        self,
        request: MLLMBatchRequest,
        *,
        source: str,
    ) -> None:
        prompt_tokens = _mllm_input_ids_token_count(request.input_ids)
        max_prompt_tokens = int(getattr(request, "max_prompt_tokens", 0) or 0)
        if max_prompt_tokens > 0 and prompt_tokens > max_prompt_tokens:
            raise PromptTooLongError(
                prompt_tokens,
                max_prompt_tokens,
                source=source,
                request_id=request.request_id,
            )
        # Bound output by the model's declared context (MLLM twin of the text
        # scheduler's admission clamp): prompt + output must not run past the
        # positional ceiling; a binding clamp logs a clear context-exhaustion
        # notice instead of silently degrading past-ceiling.
        sampling_params = getattr(request, "sampling_params", None)
        if sampling_params is not None and getattr(
            sampling_params, "max_tokens", None
        ) is not None:
            from vmlx_engine.context_limits import (
                clamp_output_to_declared_context,
            )

            sampling_params.max_tokens = clamp_output_to_declared_context(
                prompt_tokens,
                sampling_params.max_tokens,
                request_id=str(request.request_id),
            )

    def _maybe_capture_clean_ssm_boundary(
        self,
        request: "MLLMBatchRequest",
        cache: List[Any],
        all_tokens: List[int],
        boundary_len: int,
    ) -> bool:
        """vmlx#109 capture-during-prefill.

        Snapshot SSM layer state from a hybrid model's live ``cache`` at
        ``boundary_len`` tokens — i.e. BEFORE the gen-prompt suffix has
        been processed. The snapshot is stashed onto ``request`` so the
        post-prefill capture site stores it with ``is_complete=True``
        instead of queueing the slow deferred re-derive. Returns True on
        successful capture.

        Pre-conditions: hybrid model, gpl > 0, text-only request, valid
        ``boundary_len`` strictly less than the full input length. Image
        requests are skipped because the vision encoder needs the full
        sequence in one shot — splitting the prefill would corrupt
        vision token positions.
        """
        if not (self._is_hybrid and self._ssm_state_cache is not None):
            return False
        if not self._hybrid_kv_positions:
            return False
        if boundary_len <= 0:
            return False
        base_len = int(getattr(request, "_cached_tokens", 0) or 0)
        key_boundary = base_len + int(boundary_len)
        if key_boundary <= 0 or key_boundary > len(all_tokens):
            return False
        if not cache:
            return False
        # Idempotent: a request can carry multiple clean checkpoints. Hybrid
        # paged KV hits are block-aligned, while the full clean prompt boundary
        # can land inside a partial block. Store both boundaries when needed so
        # KV and SSM resume points can match exactly.
        prior_checkpoints = getattr(request, "_inline_ssm_checkpoints", None) or []
        if any(cp and cp[0] == key_boundary for cp in prior_checkpoints):
            return True
        try:
            # NOTE: do NOT call mx.eval() on cache.state arrays here.
            # Reading `.state` on BatchKVCache wrappers can create cross-
            # stream references that confuse the surrounding scheduler's
            # thread/stream context (RuntimeError: There is no Stream(gpu, 1)
            # in current thread). The deepcopy + mx.contiguous() loop below
            # materializes the SSM-only layers we actually need; the KV
            # layers are skipped via `_hybrid_kv_positions` anyway.
            kv_set = set(self._hybrid_kv_positions or [])
            ssm_layers: List[Any] = []
            _inline_materialize: List[Any] = []
            for layer_idx, c in enumerate(cache):
                if layer_idx in kv_set:
                    continue
                if _companion_exempt_cache(c):
                    continue
                if hasattr(c, "cache") and isinstance(c.cache, list):
                    from copy import deepcopy

                    cloned = deepcopy(c)
                    cloned_cache = []
                    for a in c.cache:
                        if a is None:
                            cloned_cache.append(None)
                            continue
                        materialized = mx.contiguous(a)
                        cloned_cache.append(materialized)
                        _inline_materialize.append(materialized)
                    cloned.cache = cloned_cache
                    ssm_layers.append(cloned)
                else:
                    ssm_layers.append(c)
            if not ssm_layers:
                return False
            if _inline_materialize:
                mx.eval(*_inline_materialize)
            checkpoint_tokens = list(all_tokens[:key_boundary])
            checkpoints = list(prior_checkpoints)
            checkpoints.append((key_boundary, checkpoint_tokens, ssm_layers))
            request._inline_ssm_checkpoints = checkpoints  # type: ignore[attr-defined]
            # Back-compat for older cleanup/test paths that still inspect the
            # singular inline checkpoint fields.
            request._inline_ssm_layers = ssm_layers  # type: ignore[attr-defined]
            request._inline_ssm_boundary = key_boundary  # type: ignore[attr-defined]
            request._inline_ssm_tokens = checkpoint_tokens  # type: ignore[attr-defined]
            logger.info(
                "vmlx#109: captured clean SSM at prefill boundary for %s "
                "(%d layers, key=%d tokens, is_complete=True — no re-derive needed)",
                getattr(request, "request_id", "?"),
                len(ssm_layers),
                key_boundary,
            )
            return True
        except Exception as e:
            logger.debug(
                "vmlx#109 inline SSM capture failed for %s: %s",
                getattr(request, "request_id", "?"), e,
            )
            return False

    @staticmethod
    def _clone_dots3_cache_at_boundary(
        layer: Any, boundary: int
    ) -> Optional[Tuple[Any, List[Any]]]:
        """Clone one native dots cache at an exact logical boundary."""
        if type(layer).__name__ != "Dots3LatentCache":
            return None
        from copy import copy as _shallow_copy

        clone = _shallow_copy(layer)
        trim = getattr(clone, "trim_to_boundary", None)
        if not callable(trim) or not trim(int(boundary)):
            return None
        if int(getattr(clone, "offset", 0) or 0) != int(boundary):
            return None
        materialize: List[Any] = []
        for attr in ("latent", "k_pe", "idx_k"):
            value = getattr(clone, attr, None)
            if value is None:
                continue
            value = mx.contiguous(value)
            setattr(clone, attr, value)
            materialize.append(value)
        return clone, materialize

    def _maybe_capture_dots3_media_boundary(
        self,
        request: "MLLMBatchRequest",
        cache: Optional[List[Any]],
    ) -> bool:
        """Snapshot only dots3's windowed native layers at real prefill end.

        Full/DSA layers are positional streams and can be sliced exactly from
        the finish-time cache regardless of reply length. Windowed layers cannot
        rewind once decode outruns their bounded overhang, so retain only their
        exact block-aligned state here. This avoids both a second 95-GiB-model
        media prefill and a duplicate full-latent cache during generation.
        """
        if str(getattr(self, "_model_type", "") or "").lower() != "dots3_note":
            return False
        if not getattr(self, "_prefix_cache_enabled", False) or not cache:
            return False
        if getattr(request, "_aux_clean_path_prefill", False):
            return False
        orig_tokens = list(getattr(request, "_original_token_ids", None) or [])
        if len(orig_tokens) < 2:
            return False
        try:
            if not self._media_prefix_cache_allowed(request, orig_tokens):
                return False
        except Exception:
            return False
        boundary = self._ssm_block_aligned_boundary(len(orig_tokens) - 1)
        if boundary <= 0:
            boundary = len(orig_tokens) - 1
        if int(getattr(request, "_cached_tokens", 0) or 0) >= boundary:
            return False
        prior = getattr(request, "_dots3_media_boundary", None)
        if prior is not None and int(prior[0]) == boundary:
            return True
        request._dots3_media_boundary = None  # type: ignore[attr-defined]

        snapshots: Dict[int, Any] = {}
        materialize: List[Any] = []
        full_layers = 0
        try:
            for layer_idx, layer in enumerate(cache):
                if type(layer).__name__ != "Dots3LatentCache":
                    logger.info(
                        "dots3 media boundary capture declined for %s: layer %d "
                        "is %s, expected native Dots3LatentCache",
                        request.request_id,
                        layer_idx,
                        type(layer).__name__,
                    )
                    return False
                if getattr(layer, "window", None) is None:
                    full_layers += 1
                    continue
                cloned = self._clone_dots3_cache_at_boundary(layer, boundary)
                if cloned is None:
                    logger.info(
                        "dots3 media boundary capture declined for %s: windowed "
                        "layer %d cannot rewind exactly to %d",
                        request.request_id,
                        layer_idx,
                        boundary,
                    )
                    return False
                snapshot, arrays = cloned
                snapshots[layer_idx] = snapshot
                materialize.extend(arrays)
            if not snapshots or not full_layers or len(snapshots) + full_layers != len(cache):
                return False
            if materialize:
                mx.eval(*materialize)
            request._dots3_media_boundary = (  # type: ignore[attr-defined]
                boundary,
                snapshots,
            )
            logger.info(
                "dots3 native media boundary captured for %s: %d windowed + "
                "%d positional layers at block boundary=%d",
                request.request_id,
                len(snapshots),
                full_layers,
                boundary,
            )
            return True
        except Exception as exc:
            logger.warning(
                "dots3 native media boundary capture failed for %s (non-fatal): %s",
                getattr(request, "request_id", "?"),
                exc,
            )
            request._dots3_media_boundary = None  # type: ignore[attr-defined]
            return False

    def _assemble_dots3_media_boundary(
        self,
        request: "MLLMBatchRequest",
        raw_cache: Optional[List[Any]],
    ) -> Optional[Tuple[List[Any], int]]:
        """Combine captured windowed state with finish-sliced positional state."""
        capture = getattr(request, "_dots3_media_boundary", None)
        if capture is None or not raw_cache:
            return None
        try:
            boundary = int(capture[0])
            snapshots = dict(capture[1])
        except Exception:
            return None
        assembled: List[Any] = []
        materialize: List[Any] = []
        try:
            for layer_idx, layer in enumerate(raw_cache):
                if type(layer).__name__ != "Dots3LatentCache":
                    return None
                if getattr(layer, "window", None) is not None:
                    snapshot = snapshots.pop(layer_idx, None)
                    if snapshot is None or int(getattr(snapshot, "offset", 0) or 0) != boundary:
                        return None
                    assembled.append(snapshot)
                    continue
                cloned = self._clone_dots3_cache_at_boundary(layer, boundary)
                if cloned is None:
                    return None
                snapshot, arrays = cloned
                assembled.append(snapshot)
                materialize.extend(arrays)
            if snapshots or len(assembled) != len(raw_cache):
                return None
            if materialize:
                mx.eval(*materialize)
            request._dots3_media_boundary = None  # type: ignore[attr-defined]
            return assembled, boundary
        except Exception as exc:
            logger.warning(
                "dots3 native media boundary assembly failed for %s: %s",
                getattr(request, "request_id", "?"),
                exc,
            )
            return None

    def _maybe_capture_mixed_swa_boundary(
        self,
        request: "MLLMBatchRequest",
        cache: Optional[List[Any]],
    ) -> bool:
        """Capture the exact N-1 rotating-SWA state at the end of prefill.

        The cache available at request FINISH is post-generation: every
        RotatingKVCache ring has advanced beyond the prompt boundary. Text
        turns previously recovered that boundary with a second prompt forward;
        measured on Step-3.7, that redundant prefill took 21.7 seconds before
        the 3.1-second native SSD publication. Media prompts are stricter:
        rebuilding from token ids would also discard vision-conditioned state.

        This capture runs at the END OF PREFILL, before any decode step
        advances the rings. At that point every rotating layer's buffer is the
        post-``_update_concat`` state: TEMPORAL ORDER (``_idx == len``) with
        the concat overhang still present (``max_size - 1 + S`` tokens), and
        ``offset == prompt_len`` (all prompt tokens processed, including any
        gen-prompt suffix). The state the store needs is the window at
        ``B = len(_original_token_ids) - 1`` (the N-1 cache key). Because K/V
        rows are causal — a token's KV depends only on positions <= its own —
        slicing the newest ``offset - B`` rows off the temporal buffer and
        bounding to ``min(B, max_size)`` (keep-prefix preserved) yields the
        EXACT boundary window, not an approximation.

        The copy is bounded by the window size plus two cache blocks
        (``mx.contiguous`` breaks the lazy reference to the full prefill
        buffer).  The small retained concat overhang is intentional: the block
        store can cut exact rotating checkpoints at the two boundaries before
        a terminal partial block.  Truncating immediately to one window made
        every preceding block ``rotating_kv_pending``; an exact repeat hit the
        terminal partial, but the same long prefix with a changed suffix went
        fully cold after restart.  This remains transient request state, not a
        retained RAM prefix-cache mirror. Cleanup combines it with exact slices
        of the append-only full-attention KV, publishes it, and releases it.

        Declines honestly (returns False, leaves no capture) whenever the
        buffer is not in temporal order (e.g. a single-token tail went through
        ``_update_in_place``), the window at B is not fully present, a
        rotating layer has an unexpected shape/class, or the boundary math
        does not hold. A declined capture means the store falls back to the
        existing ``rotating_kv_pending`` path — visible, never corrupt.
        """
        if str(getattr(self, "_model_type", "") or "").lower() == "dots3_note":
            return self._maybe_capture_dots3_media_boundary(request, cache)
        if (
            not getattr(self, "_prefix_cache_enabled", False)
            or not getattr(self, "_mixed_attention_cache_model", False)
        ):
            return False
        if getattr(request, "_aux_clean_path_prefill", False):
            # Nested bookkeeping prefill (clean media prefix path) — never
            # capture from it, and never clobber the real request's capture.
            return False
        if not cache:
            return False
        orig_tokens = list(getattr(request, "_original_token_ids", None) or [])
        if len(orig_tokens) < 2:
            return False
        # Idempotent: the text-tail prefill lanes isolate the FINAL token in a
        # single-token forward (which rolls the ring in place), so they call
        # this BEFORE that forward; the post-prefill call in
        # _run_vision_encoding then finds the ring rolled. A capture already
        # taken at THIS boundary must be kept, not clobbered. A capture at a
        # DIFFERENT boundary is stale (aborted prefill) and is dropped.
        _prior = getattr(request, "_mixed_swa_boundary", None)
        if _prior is not None and int(_prior[0]) == len(orig_tokens) - 1:
            return True
        request._mixed_swa_boundary = None  # type: ignore[attr-defined]
        try:
            has_media_context = self._request_has_media_cache_context(
                request, orig_tokens
            )
            if (
                has_media_context
                and not self._media_prefix_cache_allowed(request, orig_tokens)
            ):
                return False
        except Exception:
            return False
        boundary = len(orig_tokens) - 1
        if int(getattr(request, "_cached_tokens", 0) or 0) >= boundary:
            # Cleanup skips a store when the durable chain already covers N-1.
            # Do not create a transient window capture that no consumer needs.
            return False
        try:
            snap_map: Dict[int, Any] = {}
            materialize: List[Any] = []
            from copy import copy as _shallow_copy

            for layer_idx, layer in enumerate(cache):
                if type(layer).__name__ != "RotatingKVCache":
                    continue
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    logger.info(
                        "mixed-SWA boundary capture declined for %s: "
                        "rotating layer %d has no buffer",
                        request.request_id,
                        layer_idx,
                    )
                    return False
                offset = int(getattr(layer, "offset", 0) or 0)
                idx = int(getattr(layer, "_idx", -1))
                max_size = int(getattr(layer, "max_size", 0) or 0)
                keep = int(getattr(layer, "keep", 0) or 0)
                physical = int(keys.shape[2])
                trim = offset - boundary
                required = min(boundary, max_size)
                if (
                    trim < 0
                    or idx != physical
                    or max_size <= 0
                    or required <= keep
                    or physical - trim < required
                ):
                    logger.info(
                        "mixed-SWA boundary capture declined for %s: "
                        "layer %d offset=%d physical=%d idx=%d trim=%d "
                        "required=%d keep=%d (boundary=%d) — not a temporal "
                        "post-prefill buffer covering the N-1 window",
                        request.request_id,
                        layer_idx,
                        offset,
                        physical,
                        idx,
                        trim,
                        required,
                        keep,
                        boundary,
                    )
                    return False
                avail = physical - trim
                try:
                    _resume_block_size = max(
                        1,
                        int(
                            getattr(
                                getattr(self, "block_aware_cache", None),
                                "block_size",
                                64,
                            )
                            or 64
                        ),
                    )
                except (TypeError, ValueError):
                    _resume_block_size = 64
                # Keep at most two block widths of the causal concat overhang.
                # _rotating_previous_block_window() consumes this at store time
                # to publish bounded exact changed-tail resume checkpoints.
                retained = min(avail, required + 2 * _resume_block_size)
                if retained == avail:
                    snap_k = keys[..., :avail, :]
                    snap_v = values[..., :avail, :]
                elif keep > 0:
                    recent = retained - keep
                    if recent <= 0:
                        logger.info(
                            "mixed-SWA boundary capture declined for %s: "
                            "retained=%d does not cover keep=%d",
                            request.request_id,
                            retained,
                            keep,
                        )
                        return False
                    snap_k = mx.concatenate(
                        [
                            keys[..., :keep, :],
                            keys[..., avail - recent : avail, :],
                        ],
                        axis=2,
                    )
                    snap_v = mx.concatenate(
                        [
                            values[..., :keep, :],
                            values[..., avail - recent : avail, :],
                        ],
                        axis=2,
                    )
                else:
                    snap_k = keys[..., avail - retained : avail, :]
                    snap_v = values[..., avail - retained : avail, :]
                # Break the lazy reference to the full prefill buffer so the
                # retained copy is window-sized, not prompt-sized.
                snap_k = mx.contiguous(snap_k)
                snap_v = mx.contiguous(snap_v)
                materialize.extend((snap_k, snap_v))
                snap = _shallow_copy(layer)
                snap.keys = snap_k
                snap.values = snap_v
                snap.offset = boundary
                snap._idx = int(snap_k.shape[2])
                snap_map[layer_idx] = snap
            if not snap_map:
                return False
            if materialize:
                mx.eval(*materialize)
            request._mixed_swa_boundary = (  # type: ignore[attr-defined]
                boundary,
                snap_map,
            )
            logger.info(
                "mixed-SWA boundary captured for %s: %d rotating layers "
                "at N-1 boundary=%d (window+resume-overhang<=%d tokens each) "
                "before decode",
                request.request_id,
                len(snap_map),
                boundary,
                max(int(s.keys.shape[2]) for s in snap_map.values()),
            )
            return True
        except Exception as e:
            logger.warning(
                "mixed-SWA boundary capture failed for %s (non-fatal, "
                "store will fall back to rotating_kv_pending): %s",
                getattr(request, "request_id", "?"),
                e,
            )
            try:
                request._mixed_swa_boundary = None  # type: ignore[attr-defined]
            except Exception:
                pass
            return False

    def _mark_required_ssm_checkpoint(
        self,
        request: "MLLMBatchRequest",
        cached_tokens: int,
        *,
        reset_cached_tokens: bool = True,
    ) -> None:
        """Remember the KV-only hit boundary that needs a matching SSM state."""
        try:
            n = int(cached_tokens or 0)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            return
        request._ssm_required_checkpoint_tokens = n  # type: ignore[attr-defined]
        if reset_cached_tokens:
            # The KV hit was unusable without SSM, so the following prefill
            # starts from token 0. Keep inline checkpoint keys absolute over
            # all_tokens.
            request._cached_tokens = 0  # type: ignore[attr-defined]

    def _adjust_paged_hit_credit(self, request_id: str, accepted_tokens: int) -> None:
        """Best-effort reconciliation for hybrid KV/companion acceptance."""

        adjust = getattr(self.block_aware_cache, "adjust_cache_hit_credit", None)
        if callable(adjust):
            adjust(request_id, accepted_tokens=accepted_tokens)

    def _discard_request_cache_hit(
        self,
        request: "MLLMBatchRequest",
        *,
        reason: str,
        attempted_cached_tokens: Optional[int] = None,
        release_paged: bool = True,
    ) -> None:
        """Rollback an unusable MLLM cache candidate and preserve truthful telemetry."""

        execution = dict(getattr(request, "_cache_execution", None) or {})
        accepted_before = int(getattr(request, "_cached_tokens", 0) or 0)
        attempted = max(
            int(execution.get("attempted_cached_tokens", 0) or 0),
            int(attempted_cached_tokens or 0),
            accepted_before,
        )
        if release_paged and self.block_aware_cache is not None:
            self._adjust_paged_hit_credit(request.request_id, 0)
            try:
                self.block_aware_cache.release_cache(request.request_id)
            except Exception:
                pass

        request.prompt_cache = None
        request._cached_tokens = 0  # type: ignore[attr-defined]
        request._cache_detail = None  # type: ignore[attr-defined]
        request.input_ids = mx.array(
            [
                list(getattr(request, "_original_token_ids", None) or [])
                + list(getattr(request, "_gen_prefix_tokens", None) or [])
            ]
        )
        execution.update(
            {
                "request_id": request.request_id,
                "attempted_cached_tokens": attempted,
                "cached_tokens": 0,
                "cache_outcome": "discarded" if attempted > 0 else "miss",
                "cache_reuse_applied": False,
                "fallback_reason": reason,
            }
        )
        request._cache_execution = execution  # type: ignore[attr-defined]

    def _clean_ssm_boundary_for(
        self, request: "MLLMBatchRequest", seq_len: int, has_images: bool
    ) -> int:
        """Compute the boundary token index for vmlx#109 capture-during-prefill.

        Returns 0 when no inline capture should be attempted (non-hybrid
        model, gpl=0, image request, boundary out of range, or env-disabled).
        Otherwise returns the token count to process before snapshotting
        SSM state. The boundary excludes the gen-prompt suffix so the
        captured state matches the cache key produced by the post-prefill
        store path (which strips ``gen_prompt_len`` from the key).

        Killswitch: ``VMLX_DISABLE_SSM_INLINE_CAPTURE=1`` forces boundary=0
        so the legacy deferred re-derive path runs. Useful when the inline
        path interacts badly with a model's forward stream layout (e.g.
        async_eval + multi-phase lm() calls leaving Stream(gpu, 1) refs
        the surrounding scheduler can't resolve).
        """
        if has_images or not self._is_hybrid:
            return 0
        if getattr(request, "_bypass_prefix_cache", False):
            return 0
        if os.environ.get("VMLX_DISABLE_SSM_INLINE_CAPTURE") in (
            "1", "true", "True", "yes", "on"
        ):
            return 0
        gpl = int(getattr(request, "_gen_prompt_len", 0) or 0)
        if gpl <= 0:
            return 0
        # Boundary aligns with the post-prefill store key:
        #     key = all_tokens[:N-1] (post-prefill store uses N-1)
        # but the store also strips gpl from that key, leaving:
        #     stored_key = all_tokens[:N-1-gpl]
        # Capture state at exactly N-1-gpl tokens to match.
        boundary = seq_len - 1 - gpl
        if boundary <= 0:
            return 0
        return boundary

    def _ssm_capture_boundaries_for(
        self,
        request: "MLLMBatchRequest",
        seq_len: int,
        has_images: bool,
        clean_boundary: int,
    ) -> List[int]:
        """Return clean SSM checkpoint boundaries to capture during prefill."""
        if not self._is_hybrid:
            return []
        # DEFENSE IN DEPTH, NOT THE MULTIMODAL REUSE FIX. Be precise about
        # this, because the first version of this comment claimed otherwise:
        # every one of this function's three call sites sits inside an
        # `if not has_media_payload:` branch, and has_media_payload is
        # `has_images or has_audio_payload`, so has_images is ALWAYS False
        # here today. A turn carrying pixels takes the one-shot VLM forward
        # and captures nothing, by design -- the vision encoder needs the
        # whole sequence in a single pass. The media arm below therefore does
        # not currently execute; it exists so that if a future call site ever
        # does pass has_images=True, the capture is bounded to the pure-text
        # prefix instead of silently snapshotting recurrent state that
        # absorbed vision embeddings.
        media_limit = None
        if has_images:
            media_limit = self._media_safe_capture_limit(
                list(getattr(request, "_original_token_ids", None) or [])
            )
            if media_limit <= 0:
                return []
        if os.environ.get("VMLX_DISABLE_SSM_INLINE_CAPTURE") in (
            "1", "true", "True", "yes", "on"
        ):
            return []

        boundaries: List[int] = []
        try:
            base_len = int(getattr(request, "_cached_tokens", 0) or 0)
        except (TypeError, ValueError):
            base_len = 0
        try:
            required = int(
                getattr(request, "_ssm_required_checkpoint_tokens", 0) or 0
            )
        except (TypeError, ValueError):
            required = 0
        required_local = required - base_len
        if 0 < required_local < seq_len:
            boundaries.append(required_local)

        if clean_boundary > 0:
            block_boundary = self._ssm_block_aligned_boundary(clean_boundary)
            if 0 < block_boundary < seq_len:
                boundaries.append(block_boundary)
            if clean_boundary < seq_len:
                boundaries.append(clean_boundary)

        if media_limit is not None:
            # Keep only boundaries inside the pre-media text, and re-align each
            # to a block boundary so the KV chain can actually pair with it.
            capped: List[int] = []
            for boundary in boundaries:
                usable = min(int(boundary), int(media_limit))
                aligned = self._ssm_block_aligned_boundary(usable)
                if 0 < aligned <= media_limit:
                    capped.append(aligned)
            boundaries = capped

        return sorted(set(boundaries))

    def _media_safe_capture_limit(self, token_ids: Optional[List[int]]) -> int:
        """Highest token index whose prefix contains no media placeholder.

        A companion snapshot taken PAST a media placeholder describes recurrent
        state that absorbed vision embeddings. Anything later restoring from it
        and continuing with a text-only forward would re-feed those positions
        without pixel values -- coherent-looking wrong output, not a failure.
        Everything strictly before the first placeholder is pure text and is
        safe to snapshot and to resume from.

        Returns len(token_ids) when there is no media at all, and 0 when the
        prompt opens with media (nothing is safe).
        """
        if not token_ids:
            return 0
        media_ids = self._media_placeholder_token_ids()
        if not media_ids:
            return len(token_ids)
        for index, token in enumerate(token_ids):
            if token in media_ids:
                return index
        return len(token_ids)

    def _ssm_block_aligned_boundary(self, boundary: int) -> int:
        """Return the largest positive paged-cache block boundary below boundary.

        Hybrid SSM companion state must exist at the same token count as the
        paged KV block hit. If the clean prompt boundary lands inside a partial
        block, the block cache can only reuse up to the previous full block.

        When the clean boundary is itself EXACTLY block-aligned, return one
        block below it: a future request that diverges INSIDE the final block
        matches KV up to the previous block boundary, and without a companion
        checkpoint there the hybrid guard must discard the whole match
        (measured live: probe matched 351 of 352 blocks while the only
        checkpoint sat at the aligned 352-block boundary — full re-prefill).
        """
        block_size = int(getattr(self.block_aware_cache, "block_size", 0) or 0)
        if block_size <= 0 or boundary <= block_size:
            return 0
        block_boundary = (int(boundary) // block_size) * block_size
        if block_boundary == int(boundary):
            block_boundary -= block_size
        if 0 < block_boundary < boundary:
            return block_boundary
        return 0

    def _media_clean_cache_boundary_for(
        self,
        request: "MLLMBatchRequest",
        token_ids: List[int],
    ) -> int:
        """Choose the media KV+SSM boundary, honoring a learned KV-only miss.

        A hybrid KV-only miss records the exact block-aligned boundary whose
        missing companion forced a full prefill.  Text-only prefills already
        capture that boundary on the repair pass.  Media prefills use the same
        learned boundary; `_prefill_for_clean_media_prefix_cache` detects a cut
        through a placeholder run and derives the FULL media-conditioned
        embedding sequence before forwarding only the requested prefix.  It
        never passes a full pixel/video tensor to truncated placeholders.
        """
        if len(token_ids) <= 1:
            return 0
        terminal = self._ssm_block_aligned_boundary(len(token_ids) - 1)
        if terminal <= 0:
            terminal = len(token_ids) - 1
        try:
            required = int(
                getattr(request, "_ssm_required_checkpoint_tokens", 0) or 0
            )
        except (TypeError, ValueError):
            required = 0
        block_size = int(getattr(self.block_aware_cache, "block_size", 0) or 0)
        if (
            required <= 0
            or required >= len(token_ids)
            or block_size <= 0
            or required % block_size != 0
        ):
            return terminal
        logger.info(
            "MLLM media prefix cache: repairing learned KV-only boundary "
            "for %s at %d tokens instead of terminal %d",
            getattr(request, "request_id", "?"),
            required,
            terminal,
        )
        return required

    def _media_placeholder_token_ids_by_modality(self) -> Dict[str, set[int]]:
        """Return exact configured placeholder ids grouped by modality."""
        ids: Dict[str, set[int]] = {
            "image": set(),
            "video": set(),
            "audio": set(),
        }

        def _visit(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, dict):
                getter = obj.get
                nested = obj.get("text_config")
            else:
                getter = lambda key, default=None: getattr(obj, key, default)
                nested = getattr(obj, "text_config", None)
            for modality in tuple(ids):
                for suffix in ("token_index", "token_id"):
                    value = getter(f"{modality}_{suffix}", None)
                    if isinstance(value, int) and value >= 0:
                        ids[modality].add(value)
            if nested is not None and nested is not obj:
                _visit(nested)

        _visit(getattr(self.model, "config", None))
        _visit(getattr(self.language_model, "config", None))
        return ids

    def _media_placeholder_token_ids(self) -> set[int]:
        """Return configured media placeholder token ids for the loaded VLM.

        Different mlx-vlm families name these differently. Qwen-style configs
        commonly use ``image_token_index`` while Gemma/Nemotron-family configs
        may expose image/video/audio ids separately. Treat all discovered ids
        as path-dependent media placeholders for prefix-cache safety.
        """
        grouped = self._media_placeholder_token_ids_by_modality()
        return set().union(*grouped.values())

    def _media_scoped_cache_extra_keys(
        self,
        request: "MLLMBatchRequest",
        token_ids: List[int],
    ) -> Optional[Dict[str, Any]]:
        """Scope exact media identities to their causal placeholder runs.

        A single aggregate request digest is safe but needlessly invalidates an
        unchanged earlier image when a later turn adds a video.  When the live
        processor exposes a one-to-one source/run mapping, give every media
        item its own monotonically numbered key at its own placeholder.  The
        parent hash then preserves the exact earlier media state while a new
        item partitions only its own block and descendants.

        Any ambiguous mapping falls back to the aggregate digest at the first
        placeholder.  That costs reuse but can never alias different pixels.
        """
        grouped_ids = self._media_placeholder_token_ids_by_modality()
        all_ids = set().union(*grouped_ids.values())
        runs = _media_placeholder_runs(token_ids, all_ids)
        source_items = _mllm_media_source_items(request)
        if not runs:
            aggregate = _mllm_media_cache_extra_keys(request)
            if not aggregate:
                return None
            request._media_cache_scope = {
                "mode": "global_fail_closed",
                "items": 0,
                "boundaries": [],
            }
            return aggregate

        assignment_runs = runs
        run_group_sizes: Optional[List[int]] = None
        assignments: Optional[List[Tuple[str, Any]]] = None
        dots_grouping = _dots3_media_item_runs(
            request,
            token_ids,
            grouped_ids,
            model_type=getattr(self, "_model_type", None),
        )
        muse_grouping = None
        if dots_grouping is not None:
            assignment_runs, run_group_sizes, assignments = dots_grouping
        else:
            muse_grouping = _muse_glimmer_media_item_runs(
                request,
                source_items,
                token_ids,
                runs,
                grouped_ids,
                model_type=getattr(self, "_model_type", None),
            )
            if muse_grouping is not None:
                assignment_runs, run_group_sizes = muse_grouping
            else:
                step_grouping = _step3p7_media_item_runs(
                    request,
                    source_items,
                    runs,
                    model_type=getattr(self, "_model_type", None),
                )
                if step_grouping is not None:
                    assignment_runs, run_group_sizes = step_grouping

        modalities = {modality for modality, _value in source_items}
        if (
            assignments is None
            and len(source_items) == len(assignment_runs)
            and len(modalities) == 1
        ):
            assignments = list(source_items)
        elif (
            assignments is None
            and len(source_items) == len(assignment_runs)
            and source_items
        ):
            queues = {
                modality: deque(
                    value
                    for item_modality, value in source_items
                    if item_modality == modality
                )
                for modality in grouped_ids
            }
            resolved: List[Tuple[str, Any]] = []
            for run_start, _run_end in assignment_runs:
                token_id = int(token_ids[run_start])
                candidates = [
                    modality
                    for modality, ids in grouped_ids.items()
                    if token_id in ids and queues[modality]
                ]
                if len(candidates) != 1:
                    resolved = []
                    break
                modality = candidates[0]
                resolved.append((modality, queues[modality].popleft()))
            if resolved and all(not queue for queue in queues.values()):
                assignments = resolved

        if assignments is None:
            aggregate = _mllm_media_cache_extra_keys(request)
            if not aggregate:
                return None
            scoped = dict(aggregate)
            for key in tuple(aggregate):
                scoped = scope_cache_extra_key(scoped, key, runs[0][0])
            request._media_cache_scope = {
                "mode": "aggregate_first_placeholder",
                "items": len(source_items),
                "placeholder_runs": len(runs),
                "boundaries": [int(runs[0][0])],
            }
            return scoped

        scoped: Dict[str, Any] = {}
        boundaries: List[int] = []
        for index, ((modality, source), (run_start, _run_end)) in enumerate(
            zip(assignments, assignment_runs)
        ):
            key = f"mllm_media_{index:04d}"
            scoped[key] = _mllm_media_item_digest(request, modality, source)
            scoped = scope_cache_extra_key(scoped, key, run_start)
            boundaries.append(int(run_start))
        media_cache_scope = {
            "mode": "per_media_placeholder",
            "items": len(assignments),
            "modalities": [modality for modality, _source in assignments],
            "boundaries": boundaries,
        }
        if muse_grouping is not None or dots_grouping is not None:
            media_cache_scope["item_ranges"] = [
                [int(run_start), int(run_end)]
                for run_start, run_end in assignment_runs
            ]
        if run_group_sizes is not None:
            media_cache_scope["placeholder_runs"] = sum(run_group_sizes)
            media_cache_scope["run_group_sizes"] = run_group_sizes
        request._media_cache_scope = media_cache_scope
        return scoped

    def _tokens_contain_media_placeholders(self, token_ids: List[int]) -> bool:
        media_ids = self._media_placeholder_token_ids()
        return bool(media_ids and any(t in media_ids for t in token_ids or []))

    def _request_has_media_cache_context(
        self,
        request: "MLLMBatchRequest",
        token_ids: Optional[List[int]] = None,
    ) -> bool:
        """Return True when token-only prefix caches are unsafe for request.

        Media embeddings depend on the image/video/audio payload, not just text
        token ids. If a request carries media inputs, processed pixel values, or
        media placeholder token ids from current or historical turns, every
        token-prefix cache tier must be skipped. Only prompts whose serialized
        tokens are pure text remain eligible for token-prefix reuse.
        """
        if getattr(request, "images", None) or getattr(request, "videos", None):
            return True
        if getattr(request, "pixel_values", None) is not None:
            return True
        if getattr(request, "audio_codes", None) is not None:
            return True
        if getattr(request, "audio_embeds", None) is not None:
            return True
        if getattr(request, "audio_features", None) is not None:
            return True
        if getattr(request, "audio", None) or getattr(request, "audios", None):
            return True
        if token_ids is None and getattr(request, "input_ids", None) is not None:
            try:
                arr = request.input_ids
                token_ids = arr.tolist() if arr.ndim == 1 else arr[0].tolist()
            except Exception:
                token_ids = None
        return bool(
            token_ids and self._tokens_contain_media_placeholders(list(token_ids))
        )

    def _media_prefix_cache_allowed(
        self,
        request: "MLLMBatchRequest",
        token_ids: Optional[List[int]] = None,
    ) -> bool:
        """Return True when media prompts may use media-keyed KV+SSM cache.

        Qwen3.5/3.6 VL, Qwen4Exp, Muse Glimmer, Gemma 4, Step 3.7, and dots3
        own a clean media-conditioned N-1 prefill path and are enabled by
        default. Gemma, Step, and Muse captured boundaries include native
        rotating-SWA state plus their compatible full-attention slots. Muse's
        exact runtime is 39 RotatingKVCache + 13 KVCache, and the
        per-placeholder side keys bind the stored KV to the image/video bytes.
        Muse's live K/V tensors are FP32 (with I32 rotating metadata); SSD
        preserves that exact native representation without applying a storage
        codec. Other families retain the old double opt-in until their cache
        topology has equivalent source and live proof. An explicit false value
        remains a kill switch for every default-enabled family.
        """
        model_type = str(getattr(self, "_model_type", "") or "").lower()
        if not _mllm_media_prefix_cache_family_enabled(model_type):
            return False
        if getattr(request, "_bypass_prefix_cache", False):
            return False
        if not getattr(request, "_cache_extra_keys", None):
            return False
        return self._request_has_media_cache_context(request, token_ids)

    def _supports_pure_text_prefix_with_media_tail(
        self,
        request: "MLLMBatchRequest",
        token_ids: List[int],
        cached_tokens: int,
    ) -> bool:
        """Whether this live wrapper can resume before *all* media placeholders.

        This is deliberately narrower than generic partial-media reuse.  If a
        hit already covers one image but not a later video, the processor-owned
        pixel arrays would have to be sliced to the remaining placeholders; we
        do not guess at that mapping.  A hit in the pure-text region before the
        first placeholder is different: every media placeholder and every
        processor payload remain together in the forwarded tail.

        The exact Gemma 4 runtime contract is capability-checked, not inferred
        from the family name alone.  Its wrapper names ``input_ids``,
        ``pixel_values`` and ``cache``; exposes ``get_input_embeddings``; and
        its language model names both an embeddings input and ``cache``.  That
        path encodes the tail's pixels, then builds masks from the restored KV
        offset.  Other families stay fail-closed until their actual wrapper is
        inspected and admitted separately.
        """
        runtime_type = str(getattr(self, "_model_type", "") or "").lower()
        if runtime_type not in {"gemma4", "gemma4_unified"}:
            return False
        # JANG Gemma VL artifacts are promoted from on-disk ``gemma4`` to the
        # vendored ``gemma4_unified`` wrapper. That runtime is shared with the
        # E2B/E4B audio artifacts, so the type label alone cannot admit audio
        # or mixed-media tails. This receipt is for the inspected image-only
        # 26B/31B wrapper path; audio and image+video remain separate gates.
        media_scope = getattr(request, "_media_cache_scope", None) or {}
        modalities = {
            str(value).lower()
            for value in (media_scope.get("modalities") or [])
            if value
        }
        if modalities != {"image"}:
            return False
        if cached_tokens <= 0:
            return False
        pure_text_limit = self._media_safe_capture_limit(token_ids)
        if pure_text_limit <= 0 or cached_tokens > pure_text_limit:
            return False
        if not self._media_prefix_cache_allowed(request, token_ids):
            return False

        wrapper_call = getattr(self.model, "__call__", None)
        wrapper_names = _named_params(wrapper_call) if wrapper_call else set()
        language_call = getattr(self.language_model, "__call__", None)
        language_names = (
            _named_params(language_call) if language_call else set()
        )
        return bool(
            callable(getattr(self.model, "get_input_embeddings", None))
            and {"input_ids", "pixel_values", "cache"}.issubset(wrapper_names)
            and "cache" in language_names
            and _media_embed_kwarg_name(self.language_model) is not None
        )

    def _prepare_qwen_hybrid_media_tail_for_cache_hit(
        self,
        request: "MLLMBatchRequest",
        token_ids: List[int],
        cached_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        """Admit a Qwen hybrid hit strictly before every media placeholder.

        The restored KV+SSM state owns the pure-text prefix. The forward path
        will still vision-encode the complete, untrimmed request so Qwen's
        merged embeddings and mRoPE positions remain exact, then feed only the
        uncached suffix over that native state. Hits after media begins stay
        fail-closed because they require item-level processor payload slicing.
        """
        family = str(getattr(self, "_model_type", "") or "").lower()
        if family not in {"qwen3_5", "qwen3_5_moe"}:
            return None
        if cached_tokens <= 0 or cached_tokens > len(token_ids):
            return None
        media_limit = self._media_safe_capture_limit(token_ids)
        if media_limit <= 0 or cached_tokens > media_limit:
            return None
        if not self._media_prefix_cache_allowed(request, token_ids):
            return None
        if not callable(getattr(self.model, "get_input_embeddings", None)):
            return None
        if _media_embed_kwarg_name(self.language_model) != "inputs_embeds":
            return None

        full_input_ids = getattr(request, "input_ids", None)
        if full_input_ids is None:
            return None
        if full_input_ids.ndim == 1:
            full_input_ids = full_input_ids[None, :]
        if full_input_ids.shape[1] < len(token_ids):
            return None
        if full_input_ids[0, :len(token_ids)].tolist() != list(token_ids):
            return None
        request._qwen_media_tail_full_input_ids = full_input_ids  # type: ignore[attr-defined]
        request._qwen_media_tail_cached_tokens = cached_tokens  # type: ignore[attr-defined]
        return {
            "kind": "qwen_hybrid_conditioned_media_tail",
            "conditioned_full_tokens": int(full_input_ids.shape[1]),
            "conditioned_tail_tokens": int(full_input_ids.shape[1]) - cached_tokens,
        }

    def _prepare_muse_media_tail_for_cache_hit(
        self,
        request: "MLLMBatchRequest",
        token_ids: List[int],
        cached_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        """Keep only Muse media items whose placeholders remain after a hit.

        Muse's processor owns one prompt-ordered raw-patch buffer plus one grid
        per source item. A prefix ending after an earlier image but before a new
        video can therefore reuse the image-conditioned native KV exactly when
        it removes the covered image's raw-patch span and grid before forwarding
        the tail. Hits inside a media item remain fail-closed: splitting a
        timestamped temporal group or a merged feature run would require
        feature-level reconstruction, not item-level slicing.
        """
        if str(getattr(self, "_model_type", "") or "").lower() != "muse_glimmer":
            return None
        if cached_tokens <= 0 or cached_tokens > len(token_ids):
            return None
        if not self._media_prefix_cache_allowed(request, token_ids):
            return None

        scope = getattr(request, "_media_cache_scope", None) or {}
        if scope.get("mode") != "per_media_placeholder":
            return None
        modalities = [str(value).lower() for value in scope.get("modalities") or []]
        raw_ranges = scope.get("item_ranges") or []
        ranges: List[Tuple[int, int]] = []
        for raw in raw_ranges:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                return None
            try:
                start, end = int(raw[0]), int(raw[1])
            except (TypeError, ValueError):
                return None
            if start < 0 or end <= start or end > len(token_ids):
                return None
            ranges.append((start, end))
        if not ranges or len(ranges) != len(modalities):
            return None
        if any(modality not in {"image", "video"} for modality in modalities):
            return None

        removed = 0
        for start, end in ranges:
            if start < cached_tokens < end:
                return None
            if end <= cached_tokens:
                removed += 1
                continue
            if start < cached_tokens:
                return None
            break
        if removed <= 0 or removed >= len(ranges):
            return None
        if any(end <= cached_tokens for start, end in ranges[removed:]):
            return None

        extra_kwargs = getattr(request, "extra_kwargs", None)
        if not isinstance(extra_kwargs, dict):
            return None
        grids = _mllm_grid_rows(extra_kwargs.get("grid_thw"))
        if grids is None or len(grids) != len(ranges):
            return None
        pixel_values = getattr(request, "pixel_values", None)
        if (
            pixel_values is None
            or getattr(request, "video_pixel_values", None) is not None
        ):
            return None
        spans = [grid_t * grid_h * grid_w for grid_t, grid_h, grid_w in grids]
        try:
            physical_rows = int(pixel_values.shape[0])
        except Exception:
            return None
        if sum(spans) != physical_rows:
            return None

        wrapper_call = getattr(self.model, "__call__", None)
        wrapper_names = _named_params(wrapper_call) if wrapper_call else set()
        language_call = getattr(self.language_model, "__call__", None)
        language_names = _named_params(language_call) if language_call else set()
        if not (
            callable(getattr(self.model, "get_input_embeddings", None))
            and {"input_ids", "pixel_values", "cache"}.issubset(wrapper_names)
            and "cache" in language_names
            and _media_embed_kwarg_name(self.language_model) is not None
        ):
            return None

        raw_cut = sum(spans[:removed])
        kept_grids = grids[removed:]
        kept_modalities = modalities[removed:]
        request.pixel_values = pixel_values[raw_cut:]
        extra_kwargs["grid_thw"] = kept_grids
        request.image_grid_thw = (
            mx.array(
                [
                    grid
                    for grid, modality in zip(kept_grids, kept_modalities)
                    if modality == "image"
                ],
                dtype=mx.int32,
            )
            if "image" in kept_modalities
            else None
        )
        request.video_grid_thw = (
            mx.array(
                [
                    grid
                    for grid, modality in zip(kept_grids, kept_modalities)
                    if modality == "video"
                ],
                dtype=mx.int32,
            )
            if "video" in kept_modalities
            else None
        )
        return {
            "kind": "muse_prompt_ordered_media_items",
            "removed_items": removed,
            "remaining_items": len(kept_grids),
            "removed_raw_patch_rows": raw_cut,
            "remaining_raw_patch_rows": physical_rows - raw_cut,
        }

    def _prepare_dots3_media_tail_for_cache_hit(
        self,
        request: "MLLMBatchRequest",
        token_ids: List[int],
        cached_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        """Slice dots3's native visual/audio payloads at whole-item boundaries.

        Dots video frames and images share one prompt-ordered pixel/grid buffer,
        while audio owns a separate chunk buffer. Processor-authored metadata
        identifies exact token, row, grid, and chunk spans for each logical item.
        A hit inside any item or any physical-count disagreement is declined.
        """
        if str(getattr(self, "_model_type", "") or "").lower() != "dots3_note":
            return None
        if cached_tokens <= 0 or cached_tokens > len(token_ids):
            return None
        if not self._media_prefix_cache_allowed(request, token_ids):
            return None
        scope = getattr(request, "_media_cache_scope", None) or {}
        if scope.get("mode") != "per_media_placeholder":
            return None
        extra_kwargs = getattr(request, "extra_kwargs", None)
        if not isinstance(extra_kwargs, dict):
            return None
        raw_items = extra_kwargs.get("_vmlx_dots3_media_items")
        if not isinstance(raw_items, (list, tuple)) or not raw_items:
            return None

        items: List[Dict[str, int | str]] = []
        prior_end = 0
        for raw in raw_items:
            if not isinstance(raw, dict):
                return None
            modality = str(raw.get("modality") or "").lower()
            if modality not in {"image", "video", "audio"}:
                return None
            try:
                item: Dict[str, int | str] = {
                    "modality": modality,
                    "token_start": int(raw["token_start"]),
                    "token_end": int(raw["token_end"]),
                }
                if modality in {"image", "video"}:
                    for key in (
                        "visual_row_start",
                        "visual_row_end",
                        "visual_grid_start",
                        "visual_grid_end",
                    ):
                        item[key] = int(raw[key])
                else:
                    item["audio_chunk_start"] = int(raw["audio_chunk_start"])
                    item["audio_chunk_end"] = int(raw["audio_chunk_end"])
            except (KeyError, TypeError, ValueError):
                return None
            start = int(item["token_start"])
            end = int(item["token_end"])
            if start < prior_end or end <= start or end > len(token_ids):
                return None
            prior_end = end
            items.append(item)

        scope_ranges = scope.get("item_ranges") or []
        if len(scope_ranges) != len(items):
            return None
        for raw_range, item in zip(scope_ranges, items):
            try:
                if [int(raw_range[0]), int(raw_range[1])] != [
                    int(item["token_start"]),
                    int(item["token_end"]),
                ]:
                    return None
            except (IndexError, TypeError, ValueError):
                return None

        removed = 0
        for item in items:
            start = int(item["token_start"])
            end = int(item["token_end"])
            if start < cached_tokens < end:
                return None
            if end <= cached_tokens:
                removed += 1
            elif start < cached_tokens:
                return None
            else:
                break
        if removed >= len(items):
            return None

        visual_items = [
            item for item in items if item["modality"] in {"image", "video"}
        ]
        audio_items = [item for item in items if item["modality"] == "audio"]
        removed_items = items[:removed]
        removed_visual = [
            item
            for item in removed_items
            if item["modality"] in {"image", "video"}
        ]
        removed_audio = [
            item for item in removed_items if item["modality"] == "audio"
        ]

        def _contiguous_spans(
            values: List[Dict[str, int | str]], start_key: str, end_key: str
        ) -> bool:
            cursor = 0
            for value in values:
                if int(value[start_key]) != cursor or int(value[end_key]) <= cursor:
                    return False
                cursor = int(value[end_key])
            return True

        if not _contiguous_spans(
            visual_items, "visual_row_start", "visual_row_end"
        ) or not _contiguous_spans(
            visual_items, "visual_grid_start", "visual_grid_end"
        ):
            return None
        if not _contiguous_spans(
            audio_items, "audio_chunk_start", "audio_chunk_end"
        ):
            return None

        total_visual_rows = (
            int(visual_items[-1]["visual_row_end"]) if visual_items else 0
        )
        total_visual_grids = (
            int(visual_items[-1]["visual_grid_end"]) if visual_items else 0
        )
        visual_row_cut = (
            int(removed_visual[-1]["visual_row_end"]) if removed_visual else 0
        )
        visual_grid_cut = (
            int(removed_visual[-1]["visual_grid_end"]) if removed_visual else 0
        )
        pixel_values = getattr(request, "pixel_values", None)
        image_grid = getattr(request, "image_grid_thw", None)
        if visual_items:
            try:
                if (
                    pixel_values is None
                    or int(pixel_values.shape[0]) != total_visual_rows
                    or image_grid is None
                    or int(image_grid.shape[0]) != total_visual_grids
                ):
                    return None
            except Exception:
                return None
            if visual_row_cut == total_visual_rows:
                request.pixel_values = None
                request.image_grid_thw = None
            else:
                request.pixel_values = pixel_values[visual_row_cut:]
                request.image_grid_thw = image_grid[visual_grid_cut:]
            request.video_pixel_values = None
            request.video_grid_thw = None

        total_audio_chunks = (
            int(audio_items[-1]["audio_chunk_end"]) if audio_items else 0
        )
        audio_chunk_cut = (
            int(removed_audio[-1]["audio_chunk_end"]) if removed_audio else 0
        )
        audio_features = getattr(request, "audio_features", None)
        audio_meta = getattr(request, "audio_chunk_meta", None)
        if audio_items:
            if audio_features is None or not isinstance(audio_meta, dict):
                return None
            try:
                if int(audio_features.shape[0]) != total_audio_chunks:
                    return None
                chunk_counts = audio_meta["audio_chunk_counts"]
                if int(chunk_counts.shape[0]) != len(audio_items):
                    return None
                if sum(int(value) for value in chunk_counts.tolist()) != total_audio_chunks:
                    return None
                for key in (
                    "chunk_sample_lens",
                    "chunk_token_lens",
                    "chunk_audio_indices",
                ):
                    if int(audio_meta[key].shape[0]) != total_audio_chunks:
                        return None
            except (KeyError, TypeError, ValueError):
                return None
            if audio_chunk_cut == total_audio_chunks:
                request.audio_features = None
                request.audio_features_mask = None
                request.audio_chunk_meta = None
                request.audio_features_are_raw_input_features = False
            else:
                request.audio_features = audio_features[audio_chunk_cut:]
                audio_features_mask = getattr(request, "audio_features_mask", None)
                if audio_features_mask is not None:
                    try:
                        if int(audio_features_mask.shape[0]) != total_audio_chunks:
                            return None
                        request.audio_features_mask = audio_features_mask[
                            audio_chunk_cut:
                        ]
                    except (AttributeError, TypeError, ValueError):
                        return None
                kept_meta = dict(audio_meta)
                for key in (
                    "chunk_sample_lens",
                    "chunk_token_lens",
                    "chunk_audio_indices",
                ):
                    kept_meta[key] = audio_meta[key][audio_chunk_cut:]
                kept_meta["audio_chunk_counts"] = audio_meta[
                    "audio_chunk_counts"
                ][len(removed_audio) :]
                if "chunk_audio_indices" in kept_meta:
                    kept_meta["chunk_audio_indices"] = (
                        kept_meta["chunk_audio_indices"] - len(removed_audio)
                    )
                request.audio_chunk_meta = kept_meta

        return {
            "kind": "dots3_prompt_ordered_media_items",
            "removed_items": removed,
            "remaining_items": len(items) - removed,
            "removed_visual_rows": visual_row_cut,
            "remaining_visual_rows": total_visual_rows - visual_row_cut,
            "removed_visual_grids": visual_grid_cut,
            "remaining_visual_grids": total_visual_grids - visual_grid_cut,
            "removed_audio_chunks": audio_chunk_cut,
            "remaining_audio_chunks": total_audio_chunks - audio_chunk_cut,
        }

    def _turn_peak_walk_admit(
        self, final_ctx: int, request: "MLLMBatchRequest | None" = None
    ) -> None:
        """Deferred-measure + admission for the cross-turn peak walk.

        Called immediately before EVERY forward, chunked OR single-shot —
        the branch choice below keys on the NEW-token count, so a deep
        hit-lane delta (~5.6k new tokens over a ~96k base) predicts a tiny
        attention buffer and runs single-shot; a valve living only inside
        the chunked branch never sees the forward that aborts.

        Measurement is DEFERRED one turn because MLX is lazy: a peak read
        right after a single-shot call reports the pre-forward state (the
        eval happens at sampling, in the caller). Reading the gauge at the
        NEXT deep turn's entry captures everything the previous span did —
        forward, materialise, decode — and resetting it here makes each
        reading per-span rather than a lifetime cumulative max (the chunked
        branch already resets per chunk, so lifetime monotonicity was never
        a property anyone could rely on).

        On refusal ``_last_deep_span_tokens`` is zeroed so the retry's tiny
        no-forward-ran gauge reading is NOT recorded — one poisoned
        (deep span, near-zero peak) point would drag the fit down and
        re-admit the very turn that was just declined.
        """
        if not _TURN_PEAK_ADMISSION:
            return
        # Auxiliary prefills (the clean-media-prefix store's N-1 re-prefill)
        # run NESTED inside a real turn, before its sampling eval: their gauge
        # reading would pair the real span's anchor with a partially
        # unmaterialized peak, their reset would clobber the real span's
        # measurement, and a refusal here would silently skip the media
        # prefix store. They are bookkeeping, not user turns — exempt.
        if request is not None and getattr(
            request, "_aux_clean_path_prefill", False
        ):
            return
        if not (0 < _DEEP_SPAN_CACHE_CLEAR_TOKENS <= final_ctx):
            return
        try:
            peak_now = int(mx.get_peak_memory())
        except Exception:  # noqa: BLE001
            return
        prev_span = int(getattr(self, "_last_deep_span_tokens", 0) or 0)
        if prev_span > 0 and peak_now > 0:
            self._turn_peak_walk.append((prev_span, peak_now))
        try:
            mx.reset_peak_memory()
        except Exception:  # noqa: BLE001
            pass
        # Fit only the longest strictly-increasing-context SUFFIX of the walk
        # — i.e. the current growing conversation. Adversarial review: the
        # deque is generator-global, so interleaving two deep conversations at
        # different depths produces ctx-vs-peak points with no correlation —
        # the fit's slope collapses to <=0 and the valve goes silent (the
        # abort returns), or a shallow conversation inherits a deep one's
        # residency and over-refuses. A depth switch truncates the suffix to
        # one point, which yields a one-turn protection gap in mixed traffic —
        # a gap, not a poisoned fit.
        _walk_pts = list(self._turn_peak_walk)
        _suffix_start = len(_walk_pts) - 1
        while (
            _suffix_start > 0
            and _walk_pts[_suffix_start - 1][0] < _walk_pts[_suffix_start][0]
        ):
            _suffix_start -= 1
        _walk_pts = _walk_pts[_suffix_start:]
        if len(_walk_pts) >= 2:
            _walk_fit = fit_peak_model(_walk_pts)
            if _walk_fit is not None:
                try:
                    _, _walk_max_ws = get_effective_metal_working_set_bytes(mx)
                except Exception:  # noqa: BLE001
                    _walk_max_ws = 0
                _walk_last_ctx, _walk_last_peak = _walk_pts[-1]
                # The observed-peak floor only applies when this turn is at
                # least as deep as the observation; flooring a shallower
                # request at a deeper turn's peak would refuse work the
                # device serves.
                _walk_floor = (
                    _walk_last_peak if final_ctx >= _walk_last_ctx else 0
                )
                logger.info(
                    "Turn-peak admission engaged: context=%d points=%d "
                    "walk=%.2fGB+%.4fGB/1k-tok last_peak=%.2fGB "
                    "limit=%.2fGB allowance=%.2fGB",
                    final_ctx,
                    len(_walk_pts),
                    _walk_fit[0] / (1024**3),
                    _walk_fit[1] * 1000 / (1024**3),
                    _walk_last_peak / (1024**3),
                    _walk_max_ws / (1024**3),
                    _TURN_PEAK_ALLOWANCE_BYTES / (1024**3),
                )
                try:
                    turn_peak_admission_check(
                        _walk_max_ws,
                        _walk_fit,
                        final_ctx,
                        last_observed_peak_bytes=_walk_floor,
                        allowance_bytes=_TURN_PEAK_ALLOWANCE_BYTES,
                        fitted_max_context=max(ctx for ctx, _ in _walk_pts),
                        model_label="hybrid delta",
                    )
                except Exception:
                    self._last_deep_span_tokens = 0
                    raise
        self._last_deep_span_tokens = final_ctx

    def _run_vision_encoding(self, request: MLLMBatchRequest, cache: Optional[List[Any]] = None) -> mx.array:
        """
        Run the initial VLM forward pass to encode vision and get first logits.

        For image requests: runs full VLM model (vision + language) in one shot
        (vision encoding cannot be chunked).

        For text-only requests or long prompts after cache hit: uses chunked prefill
        via prefill_step_size to reduce peak GPU memory and enable interleaving.

        Args:
            request: Preprocessed request with input_ids and pixel_values
            cache: Optional pre-initialized BatchKVCache list

        Returns:
            Logits from the forward pass
        """
        # Pin all forward + materialize work to the dedicated generation
        # stream — see module-level `_gen_stream()` docstring for the
        # JANGTQ Metal kernel + scheduler thread stream-isolation rationale.
        with _MaybeStream():
            logits = self._run_vision_encoding_inner(request, cache)
            # Mixed-SWA prompts: snapshot the N-1 prompt-boundary
            # rotating windows NOW, before any decode step advances the
            # rings. This is the only moment the exact boundary state
            # exists; the finish-time cache has rolled past it by the
            # reply length. Guarded inside to be a no-op for every other
            # family/request shape.
            self._maybe_capture_mixed_swa_boundary(request, cache)
            return logits

    def _run_vision_encoding_inner(self, request: "MLLMBatchRequest", cache: Optional[List[Any]] = None) -> "mx.array":
        kwargs = dict(request.extra_kwargs)
        # Only pass pixel_values when non-None. Smelt-loaded models use a
        # text-only wrapper whose __call__ does NOT accept pixel_values at
        # all — passing even None triggers `unexpected keyword argument
        # 'pixel_values'`. For standard VLM models, pixel_values=None is a
        # no-op that the vision encoder skips, so omitting it is safe.
        if request.pixel_values is not None:
            kwargs["pixel_values"] = request.pixel_values
        if request.video_pixel_values is not None:
            kwargs[_video_pixel_values_kwarg_name(self.model)] = (
                request.video_pixel_values
            )
        has_mimo_media_payload = (
            self._model_type == "mimo_v2"
            and (
                request.pixel_values is not None
                or request.video_pixel_values is not None
                or request.audio_codes is not None
                or request.audio_embeds is not None
                or request.audio_features is not None
            )
        )
        if request.attention_mask is not None and not has_mimo_media_payload:
            kwargs["mask"] = request.attention_mask
        if request.image_grid_thw is not None:
            kwargs["image_grid_thw"] = request.image_grid_thw
        if request.video_grid_thw is not None:
            kwargs["video_grid_thw"] = request.video_grid_thw
        if request.audio_codes is not None:
            kwargs["audio_codes"] = request.audio_codes
        if request.audio_embeds is not None:
            kwargs["audio_embeds"] = request.audio_embeds
        elif request.audio_features is not None:
            if getattr(request, "audio_features_are_raw_input_features", False):
                # Gemma4 processors return raw acoustic input features as
                # `input_features`/`input_features_mask`; the model's audio
                # embedder must still project them. Do not alias these to
                # MiMo-style precomputed embeddings.
                kwargs["input_features"] = request.audio_features
                if request.audio_features_mask is not None:
                    kwargs["input_features_mask"] = request.audio_features_mask
                if getattr(request, "audio_chunk_meta", None):
                    kwargs.update(request.audio_chunk_meta)
            else:
                # Some processors name already-computed audio embeddings
                # `audio_features`. Treat them as precomputed embeddings for the
                # model bridge. Raw waveform/mel-to-code tokenization is a separate
                # runtime component and must not be implied by this alias.
                kwargs["audio_embeds"] = request.audio_features
        if cache is not None:
            kwargs["cache"] = cache

        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        has_images = (
            request.pixel_values is not None
            or request.video_pixel_values is not None
        )
        has_audio_payload = (
            request.audio_codes is not None
            or request.audio_embeds is not None
            or request.audio_features is not None
        )
        has_media_payload = has_images or has_audio_payload
        seq_len = input_ids.shape[1]

        # Cross-turn peak-walk admission, BEFORE the branch choice below. The
        # first landing of this valve lived inside the chunked branch only —
        # and a ~5.6k hit-lane delta predicts a tiny attention buffer, stays
        # single-shot, and bypassed every check (zero engagement lines, caught
        # live by the engagement-line protocol). The fatal ~32s single command
        # buffer WAS the single-shot forward. Placed here it covers both.
        self._turn_peak_walk_admit(
            int(getattr(request, "_cached_tokens", 0) or 0) + seq_len,
            request=request,
        )

        # Media prefill is the one path that can die WITHOUT a catchable error:
        # a Metal command-buffer OOM raises std::runtime_error inside a
        # completion handler, so it reaches std::terminate and kills the engine
        # (observed 2026-08-17 in-app on dots3 — the user saw only "Model is not
        # running", ledger row 181). Nothing here can prevent that, so make it
        # ATTRIBUTABLE: name media as a factor with the numbers, at the moment
        # of risk. Warn only — never decline — because the measured media
        # working set was ~3.5 GB while that crash had 12.9 GB free, so the
        # mechanism is not yet understood and a guard keyed on the measurement
        # would fire in the wrong conditions.
        if has_media_payload:
            try:
                _active, _max_ws = get_effective_metal_working_set_bytes(mx)
                _free = max(0, _max_ws - _active) if _max_ws > 0 else 0
                _gb = 1024 ** 3
                # Measured on dots3: one 448x448 image peaks ~3.5 GB over the
                # resident baseline, and a SECOND image adds only ~0.1 GB — the
                # vision tower's working set is near-fixed, not per-image.
                _media_ws_hint_gb = 3.5
                _msg = (
                    "media prefill: images=%s audio=%s seq_len=%s "
                    "active=%.1fGB max_ws=%.1fGB free=%.1fGB "
                    "(measured vision-tower working set ~%.1fGB)"
                )
                _args = (
                    bool(has_images), bool(has_audio_payload), seq_len,
                    _active / _gb, (_max_ws / _gb) if _max_ws else 0.0,
                    _free / _gb, _media_ws_hint_gb,
                )
                if _max_ws > 0 and _free < _media_ws_hint_gb * 2 * _gb:
                    logger.warning(
                        _msg + " — headroom is thin for a media forward; a "
                        "Metal OOM here TERMINATES the engine and cannot be "
                        "caught. Raise iogpu.wired_limit_mb or close other GPU "
                        "apps.",
                        *_args,
                    )
                else:
                    logger.info(_msg, *_args)
            except Exception:
                pass  # never let observability break a request

        if int(getattr(request, "_qwen_media_tail_cached_tokens", 0) or 0) > 0:
            output = self._run_qwen_conditioned_media_tail(
                request, input_ids, cache, kwargs
            )
            request.vision_encoded = True
            if hasattr(output, "logits"):
                return output.logits
            return output

        # vmlx#89 / mlxstudio#83: chunked prefill for hybrid SSM models on
        # text-only requests.
        #
        # HISTORY: hybrids default to one-shot prefill. The original comment
        # blamed cache-position mask indexing (fa_idx/ssm_idx) "only tested
        # for full-sequence processing" — that was caution, not a defect: the
        # fa mask is KVCache.make_mask ("causal" with the cache offset — the
        # same mechanism every decode step and every non-hybrid chunked
        # prefill uses), the ssm mask is None for single-request serve, and
        # the GDN conv/ssm state carry across chunk boundaries is the exact
        # recurrence decode and the MTP n_confirmed verify split rely on
        # constantly. Mechanism equivalence is proven bit-for-bit in
        # tests/test_hybrid_chunked_prefill_equivalence.py.
        #
        # The default was flipped to chunked for the proven families and then
        # RETRACTED the same day on the answer-byte gate — see the decision
        # below for the measured MLX 0.32.1 A/B. MLX 0.32.2 plus the eligible-
        # tail fused Qwen path removes that measured rounding mechanism, but
        # the replacement live answer-byte gate has not passed yet.
        #
        # VMLX_ALLOW_HYBRID_CHUNKED_PREFILL: unset -> one-shot default;
        # truthy -> chunk every hybrid; falsy -> one-shot every hybrid.
        # Neither value ever refuses work — this only selects a path.
        _hybrid_chunk_env = (
            os.environ.get("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL")
            or os.environ.get("VMLINUX_ALLOW_HYBRID_CHUNKED_PREFILL")
        )
        _hybrid_text_model_type = "unknown"
        if self._is_hybrid:
            try:
                cfg_outer = getattr(self.model, "config", None)
                cfg_inner = (
                    _read_config_field(cfg_outer, "text_config")
                    if cfg_outer is not None
                    else None
                )
                _hybrid_text_model_type = str(
                    _read_config_field(cfg_inner, "model_type")
                    or _read_config_field(cfg_outer, "model_type")
                    or "unknown"
                )
            except Exception:  # noqa: BLE001
                pass
        if _hybrid_chunk_env is not None:
            _allow_hybrid_chunked = _hybrid_chunk_env in (
                "1", "true", "True", "yes", "on"
            )
            _hybrid_path_reason = (
                f"VMLX_ALLOW_HYBRID_CHUNKED_PREFILL={_hybrid_chunk_env!r}"
            )
        else:
            # DEFAULT STAYS ONE-SHOT — the flip was built, proven at the
            # mechanism level, and then RETRACTED on the answer-byte gate
            # (2026-08-23 live A/B, Qwen3.8-27B mtp16, temp 0, cold SSD both
            # arms): a 9,190-token prompt produced 1,388 output tokens chunked
            # vs 1,915 one-shot; a 29,080-token prompt 2,942 vs 2,663. The
            # same-lane control (chunked twice) was byte-identical in all four
            # files, so the stack is deterministic and the divergence was
            # lane-attributable. The old owner was head_dim=256 falling back to
            # a materialized softmax(QK^T), whose shape-dependent tiling rounded
            # one-shot and chunked lanes differently. MLX 0.32.2 fuses causal
            # D=256 spans >=1,024 on NAX; the vendored Qwen path also forces the
            # supported fused kernel for shorter non-divisor tails over long KV.
            # Synthetic off-boundary attention is now bit-identical, but the
            # replacement live answer-byte gate still owns any default flip.
            # The OOM escape hatch below remains available in the meantime.
            _allow_hybrid_chunked = False
            _hybrid_path_reason = (
                "hybrid default one-shot (replacement MLX 0.32.2 fused-D256 "
                "answer-byte gate pending — see the comment at this decision)"
            )
        _hybrid_blocks_chunk = self._is_hybrid and not _allow_hybrid_chunked
        _allow_native_mtp_hybrid_text_split = (
            os.environ.get("VMLINUX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT")
            or os.environ.get("VMLX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT")
        ) in ("1", "true", "True", "yes", "on")
        _native_mtp_hybrid_text_split = (
            _allow_native_mtp_hybrid_text_split
            and not has_media_payload
            and self._is_hybrid
            and _native_mtp_model_has_head(self.language_model)
            and _lm_supports_return_logits(self.language_model)
        )

        # mlxstudio#83: auto-force chunking when a one-shot forward would
        # exceed the Metal single-buffer cap (reported by QwenCode/Opencode
        # `/init` sending full-repo context through a hybrid model).
        # Estimate attention_scores = heads * seq_len^2 * 2 bytes. Use a
        # conservative 8 GB threshold (Metal cap is 9.5 GB on 64 GB Macs).
        _OOM_GUARD_BYTES = _HYBRID_ONE_SHOT_GUARD_BYTES
        _n_heads_guess = _infer_attention_heads_for_hybrid_oom_guard(
            self.language_model
        )
        _predicted_attn_bytes = _n_heads_guess * seq_len * seq_len * 2
        _fused_d256_owns_allocation = (
            _qwen_fused_d256_owns_attention_allocation(
                self.language_model, _hybrid_text_model_type
            )
        )
        # GLM-5.3's one-shot prefill peak is not described by the quadratic
        # attention-score estimate above.  Its KDA/DSA/MoE forward can exhaust
        # the remaining Metal working set on a modest agentic prompt even when
        # ``heads * seq_len**2 * 2`` is well below the generic 8 GiB guard.
        #
        # Live M5 Max receipt (glm5_next, 95.8 GiB resident, 15.2 GiB free):
        # a 2,867-token Electron tool prompt stayed on the one-shot lane,
        # peaked at 110,751 MiB, and failed before its first token.  The
        # generator had already classified the process as tight-memory and the
        # configured 2,048-token split-prefill step was available, but the
        # hybrid default blocked that path.  Treat crossing the configured
        # step as the GLM-specific OOM escape hatch in this measured regime.
        # This does not flip the normal hybrid default and does not affect
        # Qwen/GDN or other families.  The existing global auto-chunk kill
        # switch remains an exact operator override.
        _glm_tight_memory_requires_split = (
            _hybrid_blocks_chunk
            and not has_media_payload
            and _hybrid_text_model_type in {"glm5_next", "glm5_next_text"}
            and bool(getattr(self, "_tight_memory_prefill_drain", False))
            and seq_len > int(self.prefill_step_size)
            and os.environ.get("VMLX_DISABLE_HYBRID_AUTO_CHUNK")
            not in ("1", "true", "True", "yes", "on")
        )
        if _glm_tight_memory_requires_split:
            _hybrid_blocks_chunk = False
            _hybrid_path_reason = (
                "tight-memory GLM prefill exceeds the configured split step "
                "(OOM escape hatch)"
            )
        if (
            _hybrid_blocks_chunk
            and not has_media_payload
            and not _fused_d256_owns_allocation
            and _predicted_attn_bytes > _OOM_GUARD_BYTES
            and os.environ.get("VMLX_DISABLE_HYBRID_AUTO_CHUNK") not in ("1", "true", "True", "yes", "on")
        ):
            # Family-specific safety wording ONLY. Qwen3.5-family GatedDeltaNet
            # chunking is equivalence-proven at the MECHANISM level (bit-exact
            # KV/SSM/logits — see the proven set above and
            # tests/test_hybrid_chunked_prefill_equivalence.py), but it is NOT
            # the default: the answer-byte gate failed live on MLX 0.32.1. MLX
            # 0.32.2 plus the eligible-tail fused Qwen path removes that measured
            # mechanism, but these families remain here until the replacement
            # live gate passes. They still take chunking as an OOM escape hatch,
            # exactly like every other hybrid family; the proven set only decides
            # how confidently this log line is phrased.
            #
            # Hybrid families NOT in the set (Nemotron-Cascade, MiniMax M2,
            # Granite Hybrid, ...) get the same fallback with no equivalence
            # evidence at all — spot-check their output.
            _verified = (
                _hybrid_text_model_type == "qwen3_next"
                or _hybrid_text_model_type
                in _HYBRID_CHUNKED_PROVEN_TEXT_MODEL_TYPES
            )
            logger.info(
                "Hybrid model (family=%s) seq_len=%d: one-shot attention buffer "
                "%.1f GB exceeds Metal single-buffer limit (~9.5 GB). Enabling "
                "chunked prefill — %s. Set VMLX_DISABLE_HYBRID_AUTO_CHUNK=1 to "
                "raise an OOM error instead of chunking.",
                _hybrid_text_model_type, seq_len,
                _predicted_attn_bytes / (1024**3),
                (
                    "verified safe on Qwen3.5 GatedDeltaNet" if _verified
                    else "spot-check output for correctness on non-Qwen3.5 hybrid families"
                ),
            )
            _hybrid_blocks_chunk = False
            _hybrid_path_reason = (
                "one-shot attention buffer would exceed the Metal "
                "single-buffer limit (OOM escape hatch)"
            )

        # The chosen path must be VISIBLE, never silent: one line per hybrid
        # text prefill naming the lane and why. (Media prefill is one-shot by
        # design for every family and logs through the media path.)
        if self._is_hybrid and not has_media_payload:
            logger.info(
                "Hybrid prefill path=%s family=%s seq_len=%d cached=%d — %s",
                "one-shot" if _hybrid_blocks_chunk else "chunked",
                _hybrid_text_model_type,
                seq_len,
                int(getattr(request, "_cached_tokens", 0) or 0),
                _hybrid_path_reason,
            )

        # TEXT-ONLY FAST PATH: use language_model directly, skip VLM wrapper.
        # The VLM wrapper adds overhead from vision encoder path and some VLM
        # wrappers (e.g. Gemma 4 loaded via smelt) may not accept pixel_values.
        # Using language_model directly avoids this entirely.
        # Hybrid SSM families default to the one-shot full-model forward
        # (answer-byte parity — vmlx#89, decision + log above); the chunked
        # lanes below serve the env opt-in and the OOM escape hatch.
        # mlxstudio#83: self.language_model already falls back to self.model
        # when the wrapped model has no `.language_model` attr (see __init__).
        # Using `getattr(self.model, 'language_model', None)` here returned
        # None for text-only models routed through MLLM path (e.g., smelt) and
        # silently skipped chunking, falling through to the OOM-prone
        # single-shot `self.model(input_ids, **kwargs)` at the bottom.
        _tight_text_prefill_step_size = self.prefill_step_size
        _tight_adaptive_growth = False
        if (
            not has_images
            and not has_audio_payload
            and cache is not None
            and bool(getattr(self, "_tight_memory_prefill_drain", False))
        ):
            _tight_env_step = os.environ.get(
                "VMLINUX_TIGHT_MEMORY_PREFILL_STEP_SIZE"
            )
            try:
                if _tight_env_step is not None:
                    # Explicit operator override wins, exactly as written.
                    _tight_text_prefill_step_size = max(
                        16,
                        min(int(self.prefill_step_size), int(_tight_env_step)),
                    )
                else:
                    # PROJECT the safe step instead of collapsing to a flat 64.
                    #
                    # The step must bound the per-chunk attention-score buffer,
                    # which grows with the CONTEXT, not with the chunk — that
                    # is exactly what max_prefill_chunk_tokens() computes and
                    # why "a step size that is safe at 10k is fatal at 100k".
                    # Projecting it needs no probe chunks, and probing is not
                    # free: every chunk re-streams the whole expert set, so on
                    # a MoE the chunk COUNT is the cost. Measured on dots3
                    # (1928-token cold prefill, temp 0, answers identical):
                    # flat 64 -> 174.4 pp/s, probe-then-grow -> 199.9,
                    # projected/full 2048 -> 510.6 pp/s (2.93x). At 4096:
                    # 145.2 / 160.5 / 309.4.
                    # Context, NOT chunk width: attention scores are
                    # chunk x CONTEXT, and on a continuation the context is
                    # the restored prefix plus this span. Using bare seq_len
                    # here would under-count a 20k-cached turn by 10x and
                    # pick a step that is safe only for the first turn — the
                    # same distinction the peak-walk admission makes just
                    # above with _cached_tokens + seq_len.
                    _tight_ctx = max(
                        int(getattr(request, "_cached_tokens", 0) or 0)
                        + int(seq_len),
                        1,
                    )
                    _tight_heads = max(
                        1,
                        _infer_attention_heads_for_hybrid_oom_guard(
                            self.language_model
                        ),
                    )
                    try:
                        _t_active, _t_limit = (
                            get_effective_metal_working_set_bytes(mx)
                        )
                        _t_head = max(0, _t_limit - _t_active)
                    except Exception:  # noqa: BLE001
                        _t_head = 0
                    # Spend at most a QUARTER of live headroom on the score
                    # buffer. The per-chunk valve projects the NEXT chunk by
                    # the context ratio, so a step sized to just fit the
                    # current context gets declined one chunk later — measured:
                    # a 2048 step served 2015 tokens at 591.6 pp/s and then
                    # 413'd an 8k prompt at chunk [2048:4096). Leaving room for
                    # that forward projection is what keeps depth WORKING, and
                    # a decline is worse than a smaller chunk.
                    _t_budget = (
                        int(_t_head / 4) if _t_head > 0 else None
                    )
                    _tight_text_prefill_step_size = max(
                        _HYBRID_MIN_CHUNK,
                        min(
                            int(self.prefill_step_size),
                            _TIGHT_PROJECTED_STEP_CAP,
                            int(
                                max_prefill_chunk_tokens(
                                    _tight_heads,
                                    _tight_ctx,
                                    budget_bytes=_t_budget,
                                )
                            ),
                        ),
                    )
            except Exception:
                _tight_text_prefill_step_size = min(int(self.prefill_step_size), 64)
            # 🚨 A SMALLER CHUNK DOES NOT REDUCE WEIGHT STREAMING — IT
            # MULTIPLIES IT. The tight step bounds the terms that scale with
            # the chunk (activations, attention scores); the weights are
            # re-read in FULL every chunk regardless. On a MoE that is
            # catastrophic: dots3 restreams ~85GB of expert weights per
            # chunk, so 64-token chunks paid it 32x more often than the
            # configured 2048 and prefill measured 152 pp/s where the bare
            # forward does 806. Measured end to end (dots3, same serve args,
            # step the ONLY change): 2048 tok 151.7 -> 420.3 pp/s (2.77x),
            # 4096 144.7 -> 268.2, 6144 119.9 -> 221.0, answers identical.
            #
            # So the tight step stays as the conservative FIRST chunk, and
            # the measured adaptive fitter below is allowed to GROW it back
            # toward the configured step from observed per-chunk transients
            # and peak-aware headroom. Growth is measurement-driven, the
            # per-chunk valve still refuses anything that will not fit, and
            # VMLX_TIGHT_PREFILL_ADAPTIVE_GROWTH=0 restores the flat 64.
            # Default OFF: growth measured only 1.15x AND grew the second
            # chunk into a valve decline at 8k (chunk [1024:3072) refused).
            # The projected step above is the real fix; growth is kept for
            # study behind the flag.
            _tight_adaptive_growth = (
                os.environ.get("VMLX_TIGHT_PREFILL_ADAPTIVE_GROWTH", "0")
                .strip()
                .lower()
                not in {"0", "false", "no", "off"}
            )

        _mimo_tight_text_prefill_requires_chunking = (
            not has_images
            and not has_audio_payload
            and self._model_type == "mimo_v2"
            and _tight_text_prefill_step_size < self.prefill_step_size
            and seq_len > _tight_text_prefill_step_size + 1
        )
        _raise_if_mimo_tight_memory_text_prefill_exceeds_budget(
            model_type=self._model_type,
            has_media_payload=has_media_payload,
            tight_memory_prefill_drain=bool(
                getattr(self, "_tight_memory_prefill_drain", False)
            ),
            seq_len=seq_len,
            generation_tokens=int(getattr(request, "max_tokens", 0) or 0),
            request_id=request.request_id,
        )
        if (
            not has_media_payload
            and self._model_type == "mimo_v2"
            and not _mimo_tight_text_prefill_requires_chunking
        ):
            lm = self.language_model
            if lm is not None and cache is not None:
                # MiMo V2.5 uses a mixed full-attention/sliding-window KV
                # layout. Live JANG_2L probes showed the generic MLLM
                # split-prefill optimization (prefix without logits, final
                # token with logits) diverges from the stable mlx-lm/simple
                # route and leaks incorrect visible text. Keep MiMo text
                # prefill one-shot so cache positions and rotating-window
                # metadata advance exactly as the model runtime expects.
                kwargs: Dict[str, Any] = {"cache": cache}
                if _lm_supports_position_ids(lm):
                    position_ids = _absolute_text_position_ids(input_ids, cache, lm)
                    if position_ids is not None:
                        kwargs["position_ids"] = position_ids
                _seed_text_rope_delta_for_decode(lm, input_ids)
                output = lm(input_ids, **kwargs)
                request.vision_encoded = True
                if hasattr(output, "logits"):
                    return output.logits
                return output
        elif _mimo_tight_text_prefill_requires_chunking:
            logger.info(
                "MiMo-V2 text prefill using chunked path under tight memory: "
                "seq_len=%d chunk=%d configured_step=%d",
                seq_len,
                _tight_text_prefill_step_size,
                self.prefill_step_size,
            )

        if not has_media_payload and (not _hybrid_blocks_chunk or _native_mtp_hybrid_text_split):
            lm = self.language_model
            if lm is not None and cache is not None:
                _supports_position_ids = _lm_supports_position_ids(lm)
                _abs_position_ids = _absolute_text_position_ids(
                    input_ids, cache, lm
                ) if _supports_position_ids else None
                if cache is not None:
                    _seed_text_rope_delta_for_decode(lm, input_ids)

                def _lm_kwargs_for(start: int, end: int) -> Dict[str, Any]:
                    _kwargs: Dict[str, Any] = {"cache": cache}
                    if _abs_position_ids is not None:
                        _kwargs["position_ids"] = _abs_position_ids[:, :, start:end]
                    return _kwargs

                if seq_len <= self.prefill_step_size * 2:
                    # Short text-only hybrid prompts still avoid materializing
                    # lm_head for every prompt token. Prefill the prefix to
                    # update cache/state, then ask for logits only on the final
                    # token. This mirrors the text SingleBatch path and keeps
                    # Qwen3.6 native-MTP/VL PP from paying full-prompt output
                    # projection cost on "short" 1-4K prompts.
                    # vmlx#109: for hybrid+thinking models we split the
                    # prefill at the gen-prompt boundary so we can capture
                    # SSM state without contamination. Phase A processes
                    # tokens[:boundary] and snapshots SSM. Phase B processes
                    # the gen-prompt suffix; the final lm() call (now on the
                    # remaining tail through last token) returns logits.
                    boundary = self._clean_ssm_boundary_for(
                        request, seq_len, has_images
                    )
                    ssm_boundaries = self._ssm_capture_boundaries_for(
                        request, seq_len, has_images, boundary
                    )
                    final_start = max(seq_len - 1, 0)
                    if ssm_boundaries:
                        # Use pre-materialized Python list to avoid an
                        # mx.array → list eval cycle that can pin the
                        # current op group to a non-default stream.
                        all_tokens = (
                            getattr(request, "_original_token_ids", None)
                            or input_ids[0].tolist()
                        )
                        processed = 0

                        def _advance_split_prefix(target: int) -> None:
                            """Advance to an exact SSM boundary in bounded spans.

                            The old boundary branch submitted
                            ``input_ids[:, processed:target]`` in one call.
                            That silently defeated the tight-memory chunk size
                            whenever the template's clean boundary covered most
                            of the prompt (GLM live receipt: 2,864 tokens in one
                            call despite a 1,024-token projected step).  Chunk
                            toward the boundary, then snapshot only after the
                            exact target has materialized.
                            """
                            nonlocal processed
                            while processed < target:
                                prefix_end = min(
                                    target,
                                    processed + _tight_text_prefill_step_size,
                                )
                                _call_lm_prefix_without_logits(
                                    lm,
                                    input_ids[:, processed:prefix_end],
                                    _lm_kwargs_for(processed, prefix_end),
                                )
                                _materialize_prefill_cache_state(cache)
                                processed = prefix_end
                                if (
                                    _tight_text_prefill_step_size
                                    < self.prefill_step_size
                                    and not prefill_keep_alloc_enabled()
                                ):
                                    mx.clear_cache()

                        for capture_boundary in ssm_boundaries:
                            if capture_boundary > processed:
                                _advance_split_prefix(capture_boundary)
                            self._maybe_capture_clean_ssm_boundary(
                                request, cache, all_tokens, capture_boundary
                            )
                        if final_start > processed:
                            _advance_split_prefix(final_start)
                    elif final_start > 0:
                        _processed_prefix = 0
                        while _processed_prefix < final_start:
                            _prefix_end = min(
                                final_start,
                                _processed_prefix + _tight_text_prefill_step_size,
                            )
                            _call_lm_prefix_without_logits(
                                lm,
                                input_ids[:, _processed_prefix:_prefix_end],
                                _lm_kwargs_for(_processed_prefix, _prefix_end),
                            )
                            _materialize_prefill_cache_state(cache)
                            _processed_prefix = _prefix_end
                            if (
                                _tight_text_prefill_step_size < self.prefill_step_size
                                and not prefill_keep_alloc_enabled()
                            ):
                                mx.clear_cache()
                    # Mixed-SWA context: the final-token forward below
                    # is a single-token ring write that trims the rotating
                    # buffers' overhang — after it the N-1 boundary window is
                    # unrecoverable. Capture it NOW, while the buffer is still
                    # a temporal post-concat state.
                    self._maybe_capture_mixed_swa_boundary(request, cache)
                    output = lm(
                        input_ids[:, final_start:],
                        **_lm_kwargs_for(final_start, seq_len),
                    )
                    request.vision_encoded = True
                    if hasattr(output, "logits"):
                        return output.logits
                    return output

        # Chunked prefill for text-only VLM requests with long prompts.
        # Image requests must run in one shot (vision encoder needs full sequence).
        # Hybrid SSM families: one-shot by default (answer-byte parity, see
        # the decision/log above); this chunked lane serves
        # VMLX_ALLOW_HYBRID_CHUNKED_PREFILL=1 and the OOM escape hatch.
        if (
            not has_media_payload
            and (
                seq_len > self.prefill_step_size * 2
                or (
                    _tight_text_prefill_step_size < self.prefill_step_size
                    and seq_len > _tight_text_prefill_step_size + 1
                )
            )
            and (not _hybrid_blocks_chunk or _native_mtp_hybrid_text_split)
        ):
            # Use language_model directly for chunked text prefill
            lm = self.language_model
            if lm is not None and cache is not None:
                _supports_position_ids = _lm_supports_position_ids(lm)
                _abs_position_ids = _absolute_text_position_ids(
                    input_ids, cache, lm
                ) if _supports_position_ids else None
                if cache is not None:
                    _seed_text_rope_delta_for_decode(lm, input_ids)

                def _lm_kwargs_for(start: int, end: int) -> Dict[str, Any]:
                    _kwargs: Dict[str, Any] = {"cache": cache}
                    if _abs_position_ids is not None:
                        _kwargs["position_ids"] = _abs_position_ids[:, :, start:end]
                    return _kwargs

                processed = 0
                chunk_num = 0
                # vmlx#109: clean boundary for inline SSM capture. Land
                # one chunk on this boundary exactly (shrinking the chunk
                # if needed), snapshot SSM, then continue chunked prefill
                # of the gen-prompt suffix.
                ssm_boundary = self._clean_ssm_boundary_for(
                    request, seq_len, has_images
                )
                ssm_boundaries = self._ssm_capture_boundaries_for(
                    request, seq_len, has_images, ssm_boundary
                )
                _sorted_boundaries: List[int] = sorted(set(ssm_boundaries))
                _boundary_idx = 0
                ssm_captured_boundaries: set[int] = set()
                _hoisted_all_tokens: Optional[List[int]] = getattr(
                    request, "_original_token_ids", None
                )
                if _hoisted_all_tokens is None and _sorted_boundaries:
                    _hoisted_all_tokens = input_ids[0].tolist()
                # This chunked span resets the peak gauge per chunk below. If
                # the span is SUB-THRESHOLD for the turn-peak walk (a mid-size
                # fresh prompt: big enough to auto-chunk, too small for the
                # valve), those resets clobber the deferred measurement a deep
                # conversation left in the gauge — pairing its stale anchor
                # with this span's tail reading would append a deflated deep
                # point, collapse the fit's slope, and silence the valve on
                # the turns approaching the wall (adversarial review, finding
                # 1). Dropping the anchor converts that poisoning into a
                # one-turn recording gap. Single-shot sub-threshold turns need
                # no such handling: they only RAISE the gauge, which inflates
                # the next point in the conservative direction.
                if (
                    int(getattr(request, "_cached_tokens", 0) or 0) + seq_len
                ) < _DEEP_SPAN_CACHE_CLEAR_TOKENS:
                    self._last_deep_span_tokens = 0
                # Adaptive chunk sizing from MEASURED per-chunk transient.
                # A modelled budget was tried and failed (a3cedb29f): it assumed
                # fp16 scores only, while the real forward holds several fp32
                # intermediates, so the projection was 4-6x optimistic and the
                # process still aborted. DSV4's valve solves this by adapting
                # from OBSERVED peaks instead of predicting them; same idea here.
                _observed_chunk_transient = 0
                _observed_transient_at_ctx = 0
                _observed_transient_chunk_tokens = 0
                # Absorbed-latent attention (dots3) materializes chunk x ctx
                # score buffers, so its transient scales with the CHUNK as
                # well as the context — unlike the GDN hybrid the valve was
                # measured on, where halving the chunk did not move the abort
                # point. Without chunk scaling the halving ladder can never
                # rescue a declined span (measured: dots3 refused at every
                # size down to the 64-token floor with the same projection,
                # capping the model at ~17k context).
                _valve_chunk_scaled = bool(cache) and (
                    type(cache[0]).__name__ == "Dots3LatentCache"
                )
                # (context, absolute peak) pairs for the cross-span affine fit.
                # The per-chunk valve only needs the LARGEST transient and where
                # it was seen; fitting an intercept as well needs every point.
                _peak_samples: list[tuple[int, int]] = []
                _observed_chunk_peak_max = 0
                _prefill_valve_enabled = prefill_valve_enabled()
                _prefill_valve_min_margin = prefill_valve_min_margin_bytes()
                _max_active_seen = 0
                _adaptive_chunk_cap = _tight_text_prefill_step_size
                # Growth ceiling: the tight step is only the FIRST chunk when
                # adaptive growth is on; the fitter may climb back to the
                # configured step from measured headroom (see the tight-step
                # comment above for the measured 2.77x).
                _chunk_ceiling = (
                    int(self.prefill_step_size)
                    if _tight_adaptive_growth
                    else _tight_text_prefill_step_size
                )
                _adaptive_chunk_active = (
                    _HYBRID_ADAPTIVE_CHUNK or _tight_adaptive_growth
                )
                # (chunk_tokens, transient_bytes) samples for the AFFINE fit
                # below. A per-chunk transient is NOT proportional to the
                # chunk: a MoE restreams its whole expert set every chunk, so
                # a 64-token probe is almost all fixed cost. Dividing that by
                # 64 to get a per-token rate over-estimates it by orders of
                # magnitude and pins the chunk small forever — the same
                # dropped-constant error fit_peak_model() was written for.
                _chunk_transient_samples: list[tuple[int, int]] = list(
                    getattr(self, "_tight_chunk_samples", []) or []
                )
                _adaptive_growth_logged = False
                # A chunk costs a full re-stream of the expert set, so the
                # PROBE chunks are themselves expensive (measured: 3 chunks
                # instead of 1 cost ~5s on a 1928-token dots3 prompt —
                # effective MoE weight streaming is ~35GB/s, not peak
                # bandwidth). Only the FIRST request on this generator should
                # pay for probing: carry the fitted transient model forward
                # and start later requests at the fitted chunk directly.
                if _tight_adaptive_growth and _chunk_transient_samples:
                    try:
                        _seed_fit = fit_peak_model(_chunk_transient_samples)
                        _seed_active, _seed_limit = (
                            get_effective_metal_working_set_bytes(mx)
                        )
                        _seed_head = max(0, _seed_limit - _seed_active)
                        if _seed_fit is not None and _seed_fit[1] > 0:
                            _seed_budget = int(_seed_head / 1.25)
                            _seed_tokens = int(
                                (_seed_budget - max(0.0, _seed_fit[0]))
                                / _seed_fit[1]
                            )
                            if _seed_tokens > _HYBRID_MIN_CHUNK:
                                _adaptive_chunk_cap = max(
                                    _HYBRID_MIN_CHUNK,
                                    min(_chunk_ceiling, _seed_tokens),
                                )
                                logger.info(
                                    "Tight-memory prefill chunk seeded at %d "
                                    "from %d earlier transient sample(s) "
                                    "(headroom %.2fGB, ceiling %d)",
                                    _adaptive_chunk_cap,
                                    len(_chunk_transient_samples),
                                    _seed_head / (1024**3),
                                    _chunk_ceiling,
                                )
                    except Exception:  # noqa: BLE001
                        pass
                _prefill_keep_alloc = prefill_keep_alloc_enabled()
                # Pre-size the KV slots for the WHOLE span before chunking.
                #
                # KVCache grows by mx.concatenate whenever offset would exceed
                # capacity, in units of `step` (256, a CLASS attribute). A 2048-
                # token chunk is an exact multiple of 256, so capacity == offset
                # at every chunk boundary and EVERY chunk reallocates every
                # attention layer's full K and V, orphaning the old buffers.
                #
                # MEASURED: 16 attention layers x 2 (K,V) x 4 kv_heads x 256
                # head_dim x 2B = 65,536 B per token of context, so the garbage
                # is 4.85GB per chunk at 74k and 5.90GB at 92k — exactly the
                # 4.6-5.9GB/chunk observed, and it scales with CONTEXT, which is
                # why no chunk-size change ever helped. The slot accounting only
                # reads current .nbytes, so it reported the legitimate +0.125GB
                # and the orphans were invisible.
                #
                # Setting step per INSTANCE for the span makes the first chunk
                # allocate full width and every later chunk write in place: zero
                # reallocations, zero per-chunk garbage. It must be restored
                # afterwards or the first decode token would allocate another
                # full-width buffer.
                # Deep spans: return the allocator cache's dead buffers
                # BEFORE the full-span presize and reconstruction copies land.
                _maybe_clear_deep_span_cache(
                    int(getattr(request, "_cached_tokens", 0) or 0) + seq_len
                )

                _presized_kv_slots: List[Any] = []
                if _KV_PRESIZE_SPAN and seq_len > 0:
                    for _slot in (cache or []):
                        if type(_slot).__name__ != "KVCache":
                            continue
                        if "step" in getattr(_slot, "__dict__", {}):
                            continue  # already instance-scoped; leave it alone
                        try:
                            _slot.step = int(seq_len + _DECODE_PRESIZE_HEADROOM)
                            _presized_kv_slots.append(_slot)
                        except Exception:  # noqa: BLE001
                            pass
                    if _presized_kv_slots:
                        logger.info(
                            "Pre-sized %d KV slots to the full %d-token span "
                            "+%d decode headroom (avoids a full K/V realloc "
                            "per chunk AND at decode start).",
                            len(_presized_kv_slots),
                            seq_len,
                            _DECODE_PRESIZE_HEADROOM,
                        )

                def _restore_kv_step() -> None:
                    """Put `step` back on the class default.

                    Must run on EVERY exit from the prefill — normal completion,
                    a chunk failure, or an admission decline — because a slot
                    left at span width would allocate another full-width buffer
                    on the first decode token.
                    """
                    while _presized_kv_slots:
                        _slot_to_restore = _presized_kv_slots.pop()
                        try:
                            del _slot_to_restore.step
                        except Exception:  # noqa: BLE001
                            pass

                # WHOLE-SPAN ADMISSION, using a transient model fitted on a
                # PREVIOUS span. The per-chunk valve below is correct but late:
                # it only declines once active memory has climbed near the
                # ceiling, which on a ~100k span is ~46 chunks of GPU work
                # already spent. This decides once, before the first chunk.
                #
                # It cannot fire on the first big prefill after a load — a span
                # cannot measure itself before it starts — and that is by
                # design, not a gap: the per-chunk valve still covers that run,
                # and this one converts every SUBSEQUENT doomed span into an
                # immediate error. Unknown model => no rejection.
                if _prefill_valve_enabled and self._span_peak_model is not None:
                    try:
                        _span_active, _span_max_ws = (
                            get_effective_metal_working_set_bytes(mx)
                        )
                    except Exception:  # noqa: BLE001
                        _span_active, _span_max_ws = 0, 0
                    try:
                        span_admission_check(
                            _span_max_ws,
                            self._span_peak_model,
                            int(getattr(request, "_cached_tokens", 0) or 0) + seq_len,
                            fresh_tokens=seq_len,
                            model_label="hybrid prefill",
                            fitted_max_context=self._span_peak_max_context,
                            # This loop halves a declined chunk and retries,
                            # so a span over the fitted projection is not
                            # automatically unservable — it just runs in
                            # narrower pieces.
                            degradable_chunks=True,
                        )
                    except Exception:
                        _restore_kv_step()
                        raise

                # The cross-turn peak-walk admission runs at the TOP of this
                # function (see _turn_peak_walk_admit) — it must cover the
                # single-shot branch too, which is the one that actually
                # submitted the fatal command buffer.
                while processed < seq_len - 1:  # -1: keep last token for final logits
                    chunk_size = min(_chunk_ceiling, seq_len - 1 - processed)
                    if _adaptive_chunk_active:
                        chunk_size = min(
                            chunk_size, max(1, _adaptive_chunk_cap)
                        )
                    while (
                        _boundary_idx < len(_sorted_boundaries)
                        and (
                            _sorted_boundaries[_boundary_idx] <= processed
                            or _sorted_boundaries[_boundary_idx]
                            in ssm_captured_boundaries
                        )
                    ):
                        _boundary_idx += 1
                    next_ssm_boundary: Optional[int] = None
                    if (
                        _boundary_idx < len(_sorted_boundaries)
                        and _sorted_boundaries[_boundary_idx]
                        <= processed + chunk_size
                    ):
                        next_ssm_boundary = _sorted_boundaries[_boundary_idx]
                    if next_ssm_boundary is not None:
                        chunk_size = next_ssm_boundary - processed
                    if _adaptive_chunk_active and _adaptive_chunk_cap < chunk_size:
                        logger.info(
                            "Hybrid prefill chunk %d -> %d at processed=%d "
                            "(measured transient %.2fGB per chunk)",
                            chunk_size,
                            _adaptive_chunk_cap,
                            processed,
                            _observed_chunk_transient / (1024**3),
                        )
                        chunk_size = max(1, _adaptive_chunk_cap)
                    chunk = input_ids[:, processed:processed + chunk_size]
                    # Per-chunk peak baseline. Reset immediately before the
                    # forward so get_peak_memory() after the eval is THIS
                    # chunk's requirement rather than a running maximum. Always
                    # measured, not only under the trace flag, because the
                    # admission valve below adapts from it.
                    try:
                        _peak_base = int(mx.get_active_memory())
                        mx.reset_peak_memory()
                    except Exception:  # noqa: BLE001
                        _peak_base = 0
                    # ADMISSION: decline before submitting GPU work.
                    #
                    # This loop was the one prefill path with no valve, and it
                    # does not fail with a catchable Python error — it dies with
                    # "[METAL] Command buffer execution failed: Insufficient
                    # Memory", which libc++ turns into a process abort. There is
                    # no exception to handle, so the ONLY defence is to not
                    # submit the chunk.
                    #
                    # MEASURED on Qwen3.6-27B, cold 101,502-token prefill: the
                    # per-chunk transient is linear in context,
                    # transient = 2.82GB + 0.00015 * ctx (residuals <=0.02GB),
                    # growing only ~0.31GB per 2048-token chunk. So the largest
                    # transient observed so far, with DSV4's 1.25x margin, is a
                    # sound projection for the next chunk. The run aborted at
                    # chunk 47 needing ~110GB; this declines it instead.
                    if _prefill_valve_enabled and _observed_chunk_transient > 0:
                        try:
                            _valve_active, _valve_max_ws = (
                                get_effective_metal_working_set_bytes(mx)
                            )
                        except Exception:  # noqa: BLE001
                            _valve_active, _valve_max_ws = 0, 0
                        try:
                            hybrid_chunk_valve_check(
                                    _valve_active,
                                _valve_max_ws,
                                _observed_chunk_transient,
                                _observed_transient_at_ctx,
                                int(getattr(request, "_cached_tokens", 0) or 0)
                                + processed
                                + chunk_size,
                                _prefill_valve_min_margin,
                                chunk_start=processed,
                                chunk_end=processed + chunk_size,
                                model_label="hybrid prefill",
                                observed_chunk_tokens=_observed_transient_chunk_tokens,
                                next_chunk_tokens=chunk_size,
                                chunk_scaled=_valve_chunk_scaled,
                            )
                        except PrefillAdmissionError:
                            # A decline used to fail the whole request with a
                            # 413. Halving and retrying is strictly better:
                            # the span the device CAN serve is served, just in
                            # smaller pieces, and only a chunk already at the
                            # floor is genuinely unservable. Measured: an 8k
                            # prompt was refused outright at chunk
                            # [1024:3072) while the same span completes in
                            # narrower chunks.
                            if chunk_size > _HYBRID_MIN_CHUNK:
                                _halved = max(_HYBRID_MIN_CHUNK, chunk_size // 2)
                                logger.info(
                                    "Prefill chunk declined at [%d:%d) — "
                                    "halving %d -> %d and retrying",
                                    processed,
                                    processed + chunk_size,
                                    chunk_size,
                                    _halved,
                                )
                                _adaptive_chunk_cap = _halved
                                _chunk_ceiling = min(_chunk_ceiling, _halved)
                                _adaptive_chunk_active = True
                                continue
                            _restore_kv_step()
                            raise
                        except Exception:
                            # A decline leaves the prefill; restore step first or
                            # the slot keeps span width into decode.
                            _restore_kv_step()
                            raise
                    if _adaptive_chunk_active:
                        try:
                            _active_before_chunk = int(mx.get_active_memory())
                            mx.reset_peak_memory()
                        except Exception:  # noqa: BLE001
                            _active_before_chunk = 0
                    try:
                        _call_lm_prefix_without_logits(
                            lm,
                            chunk,
                            _lm_kwargs_for(processed, processed + chunk_size),
                        )
                    except Exception as chunk_err:
                        # Log cache state at failure point for diagnosis
                        _cache_diag = []
                        for ci, cc in enumerate(cache[:6]):
                            if hasattr(cc, 'keys') and cc.keys is not None:
                                _cache_diag.append(f"L{ci}:KV={cc.keys.shape}")
                            elif hasattr(cc, 'cache') and isinstance(cc.cache, list):
                                shapes = [a.shape if a is not None else 'None' for a in cc.cache]
                                _cache_diag.append(f"L{ci}:SSM={shapes}")
                            elif hasattr(cc, 'offset'):
                                _cache_diag.append(f"L{ci}:off={cc.offset}")
                            else:
                                _cache_diag.append(f"L{ci}:{type(cc).__name__}")
                        logger.error(
                            f"Chunked prefill failed at chunk {chunk_num} "
                            f"(processed={processed}, chunk_size={chunk_size}, "
                            f"total={seq_len}): {chunk_err} "
                            f"[cache: {', '.join(_cache_diag)}]"
                        )
                        _restore_kv_step()
                        raise
                    if _HYBRID_PREFILL_MEM_TRACE and chunk_num % 8 == 0:
                        try:
                            _m_fwd = mx.get_active_memory() / (1024**3)
                        except Exception:  # noqa: BLE001
                            _m_fwd = -1.0
                    _materialize_prefill_cache_state(cache)
                    # AFTER materialize, not before. _call_lm_prefix_without_logits
                    # only BUILDS the graph — MLX is lazy, so nothing has run yet at
                    # that point and get_peak_memory() returns the pre-forward value.
                    # Reading there reported transient=0.00GB for every chunk of a
                    # 35-chunk span, which is the measurement equivalent of an A/B
                    # where the new path never engaged, and it is why four sizing
                    # attempts were tuned against zeros.
                    # _materialize_prefill_cache_state is the eval point, so the
                    # peak is only meaningful after it.
                    try:
                        _chunk_peak = int(mx.get_peak_memory())
                    except Exception:  # noqa: BLE001
                        _chunk_peak = 0
                    # Skip chunk 0 when learning the transient. With the KV
                    # slots pre-sized, the FIRST chunk pays a one-time
                    # full-width allocation (measured 8.71GB for a 101,502-token
                    # span) that does NOT scale with context. The valve projects
                    # the observed transient linearly in context, so feeding it
                    # that one-time cost extrapolated to ~425GB at 100k and
                    # declined every request — including spans this now fits
                    # comfortably.
                    if _chunk_peak > _peak_base and chunk_num > 0:
                        _this_transient = _chunk_peak - _peak_base
                        # Feed the cross-span fit the ABSOLUTE peak, not this
                        # transient. The whole-span check compares against the
                        # device limit directly, and a transient-only model has
                        # to add an active reading — which, taken at span start,
                        # is exactly what made the old check admit the span that
                        # died (active climbs from ~21GB to ~95GB across a 100k
                        # span as KV accumulates). mx.get_peak_memory() after the
                        # per-chunk reset is already weights + KV + transient.
                        #
                        # EVERY sample feeds the fit, not just the maxima below:
                        # a least-squares intercept needs the low-context points
                        # too, and keeping only running maxima biases the slope.
                        # The chunk_num > 0 exclusion still applies for the same
                        # reason it does below — chunk 0's one-time full-width
                        # allocation is not a function of context.
                        _peak_samples.append(
                            (
                                max(
                                    1,
                                    int(getattr(request, "_cached_tokens", 0) or 0)
                                    + processed
                                    + chunk_size,
                                ),
                                _chunk_peak,
                            )
                        )
                        if _chunk_peak > _observed_chunk_peak_max:
                            _observed_chunk_peak_max = _chunk_peak
                        if _this_transient >= _observed_chunk_transient:
                            # Record the context this was observed AT, not just
                            # the magnitude: the valve scales it forward by the
                            # context ratio, so a transient without its context
                            # cannot be projected.
                            _observed_chunk_transient = _this_transient
                            # END context, not start. A chunk attends over the
                            # context it FINISHES at, and at chunk 0 the start
                            # context is 0 — projecting from that multiplied the
                            # transient by the whole next context and produced a
                            # 6448GB estimate that declined chunk 1 of a prompt
                            # the device serves comfortably. Observation and
                            # projection must use the same end-of-chunk basis.
                            _observed_transient_at_ctx = max(
                                1,
                                int(getattr(request, "_cached_tokens", 0) or 0)
                                + processed
                                + chunk_size,
                            )
                            # The chunk size it was observed at, for paths
                            # whose transient scales with the chunk too.
                            _observed_transient_chunk_tokens = int(chunk_size)
                    if _HYBRID_PREFILL_MEM_TRACE:
                        try:
                            logger.info(
                                "hybrid-prefill-peak chunk=%d processed=%d "
                                "ctx=%d chunk_size=%d base=%.2fGB peak=%.2fGB "
                                "transient=%.2fGB",
                                chunk_num, processed,
                                int(getattr(request, "_cached_tokens", 0) or 0) + processed,
                                chunk_size,
                                _peak_base / (1024**3),
                                _chunk_peak / (1024**3),
                                max(0, _chunk_peak - _peak_base) / (1024**3),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    if _HYBRID_PREFILL_MEM_TRACE and chunk_num % 8 == 0:
                        # WHERE does the memory fail to come back? Read active
                        # at each stage of one chunk. The stage whose delta is
                        # not released by clear_cache is the leak.
                        try:
                            _m_mat = mx.get_active_memory() / (1024**3)
                            _cache_nb = sum(
                                int(getattr(_s, "nbytes", 0) or 0) for _s in (cache or [])
                            ) / (1024**3)
                            logger.info(
                                "hybrid-prefill-stage chunk=%d processed=%d "
                                "after_fwd=%.2fGB after_materialize=%.2fGB "
                                "cache_slots=%.2fGB",
                                chunk_num, processed, _m_fwd, _m_mat, _cache_nb,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    if _adaptive_chunk_active and _active_before_chunk > 0:
                        try:
                            _peak = int(mx.get_peak_memory())
                            _, _limit = get_effective_metal_working_set_bytes(mx)
                            _transient = max(0, _peak - _active_before_chunk)
                            if _transient > _observed_chunk_transient:
                                _observed_chunk_transient = _transient
                            # Baseline must be PEAK-AWARE, not a single sample.
                            # Active swings 26 -> 75GB inside this loop, so a
                            # lone get_active_memory() reading is wrong by up to
                            # 3x depending on where in the swing it lands — which
                            # is why the first version of this still let the
                            # process abort. Use the highest active seen so far.
                            _active_now = int(mx.get_active_memory())
                            if _active_now > _max_active_seen:
                                _max_active_seen = _active_now
                            _headroom = _limit - _max_active_seen
                            if _limit > 0 and _transient > 0 and chunk_size > 0:
                                _chunk_transient_samples.append(
                                    (int(chunk_size), int(_transient))
                                )
                                # Keep a bounded, distinct-size history on the
                                # generator so later requests skip the probe.
                                try:
                                    _hist = {
                                        int(c): int(t)
                                        for c, t in _chunk_transient_samples
                                    }
                                    self._tight_chunk_samples = sorted(
                                        _hist.items()
                                    )[-8:]
                                except Exception:  # noqa: BLE001
                                    pass
                                _budget = int(_headroom / 1.25)
                                _affine = fit_peak_model(_chunk_transient_samples)
                                if _affine is not None and _affine[1] > 0:
                                    # transient(tokens) = fixed + slope*tokens
                                    _fixed, _slope = _affine
                                    _fit = int(
                                        (_budget - max(0.0, _fixed)) / _slope
                                    )
                                else:
                                    # Single sample: no way to separate the
                                    # fixed term, so stay conservative and
                                    # grow geometrically instead of dividing.
                                    _fit = (
                                        chunk_size * 4
                                        if _transient * 4 < _budget
                                        else max(
                                            _HYBRID_MIN_CHUNK,
                                            int(_budget)
                                            // max(1, _transient // chunk_size),
                                        )
                                    )
                                _adaptive_chunk_cap = max(
                                    _HYBRID_MIN_CHUNK,
                                    min(_chunk_ceiling, int(_fit)),
                                )
                                if (
                                    _tight_adaptive_growth
                                    and not _adaptive_growth_logged
                                    and _adaptive_chunk_cap > chunk_size
                                ):
                                    _adaptive_growth_logged = True
                                    logger.info(
                                        "Tight-memory prefill chunk grows "
                                        "%d -> %d (transient %.2fGB at %d tok, "
                                        "headroom %.2fGB, ceiling %d)",
                                        chunk_size,
                                        _adaptive_chunk_cap,
                                        _transient / (1024**3),
                                        chunk_size,
                                        _headroom / (1024**3),
                                        _chunk_ceiling,
                                    )
                        except Exception:  # noqa: BLE001
                            pass
                    processed += chunk_size
                    chunk_num += 1
                    # Advancing prefill progress for the liveness probes.
                    # `num_prompt_tokens` is set only when the FIRST output
                    # token arrives, so without this a long prefill reads as
                    # zero progress for its entire duration — and prefill is
                    # the only phase long enough to hit the request timeout.
                    # Measured live: a 196k-token span burned all bounded
                    # grace windows (900s) and was still killed as wedged
                    # while the GPU was legitimately chunking. This counter
                    # makes `request_progress` genuinely increase per chunk,
                    # so the extension logic needs no grace at all here.
                    try:
                        request._prefill_tokens_done = processed
                    except Exception:  # noqa: BLE001
                        pass
                    if processed in ssm_boundaries and processed not in ssm_captured_boundaries:
                        if self._maybe_capture_clean_ssm_boundary(
                            request,
                            cache,
                            _hoisted_all_tokens or input_ids[0].tolist(),
                            processed,
                        ):
                            ssm_captured_boundaries.add(processed)
                    if not _prefill_keep_alloc:
                        # Drain the generation stream BEFORE clearing.
                        #
                        # clear_cache() only returns buffers the runtime has
                        # already freed. This prefill runs inside
                        # `with mx.stream(MLLMBatchGenerator._stream)`, and
                        # _materialize_prefill_cache_state evals only the CACHE
                        # arrays — it blocks until those outputs exist, but the
                        # dead buffers from the chunk stay "active" until the
                        # runtime catches up, so clear_cache had nothing to
                        # reclaim and the garbage accumulated across chunks.
                        #
                        # MEASURED why this matters: KVCache.update_and_fetch
                        # grows by mx.concatenate, and with step=256 dividing a
                        # 2048-token chunk exactly, EVERY chunk reallocates
                        # every attention layer's full K and V. At 92k context
                        # that turns ~6GB of KV into garbage per chunk — which
                        # is exactly the ~5.9GB per chunk observed, against a
                        # legitimate increment of 0.125GiB (16 attention layers
                        # x 4 kv_heads x 256 head_dim x 2048 x 2 x 2B). 47x.
                        #
                        # The engine already knew this shape: the tight-memory
                        # drain above pairs synchronize+clear for the same
                        # reason. It just was never applied inside this loop.
                        if _HYBRID_PREFILL_DRAIN:
                            try:
                                if MLLMBatchGenerator._stream is not None:
                                    try:
                                        mx.synchronize(MLLMBatchGenerator._stream)
                                    except RuntimeError as _sync_exc:
                                        # Shutdown/thread-handoff can leave the
                                        # stream handle stale; a bare
                                        # synchronize still drains pending work.
                                        if "There is no Stream" not in str(_sync_exc):
                                            raise
                                        mx.synchronize()
                                else:
                                    mx.synchronize()
                            except Exception:  # noqa: BLE001
                                pass
                        mx.clear_cache()
                    if (_HYBRID_PREFILL_MEM_TRACE and chunk_num % 8 == 0) or (
                        chunk_num > 0 and chunk_num % 8 == 0
                    ):
                        # Attribute the growth to actual cache slots instead of
                        # inferring it. The trace showed active memory reaching
                        # 94.86GB at 73,728 tokens while a 16-layer KV cache
                        # that size should be ~5GB, so ~90GB was unaccounted for
                        # and every guess about WHERE was wrong. Always-on at a
                        # 32-chunk cadence: the dots3 span-retention hunt burned
                        # a day because the by-kind census was invisible on the
                        # spans that actually died (the trace env only reaches
                        # the engine on a full app restart).
                        try:
                            _by_kind: dict = {}
                            for _slot in (cache or []):
                                _kind = type(_slot).__name__
                                _nb = int(getattr(_slot, "nbytes", 0) or 0)
                                if not _nb:
                                    for _attr in ("keys", "values"):
                                        _arr = getattr(_slot, _attr, None)
                                        if _arr is not None and hasattr(_arr, "nbytes"):
                                            _nb += int(_arr.nbytes)
                                    _inner = getattr(_slot, "cache", None)
                                    if isinstance(_inner, list):
                                        for _arr in _inner:
                                            if _arr is not None and hasattr(_arr, "nbytes"):
                                                _nb += int(_arr.nbytes)
                                _agg = _by_kind.setdefault(_kind, [0, 0])
                                _agg[0] += 1
                                _agg[1] += _nb
                            try:
                                _census_active = mx.get_active_memory() / (1024**3)
                            except Exception:  # noqa: BLE001
                                _census_active = -1.0
                            logger.info(
                                "hybrid-prefill-slots chunk=%d processed=%d "
                                "active=%.2fGB %s",
                                chunk_num,
                                processed,
                                _census_active,
                                " ".join(
                                    f"{k}x{v[0]}={v[1] / (1024**3):.2f}GB"
                                    for k, v in sorted(_by_kind.items())
                                ),
                            )
                            if os.environ.get("VMLX_CENSUS_GC", "").strip() == "1":
                                # Name the holder: bucket EVERY live mx.array
                                # by shape. Slow (full gc walk) — diagnosis
                                # only. The cache-slot census already proved
                                # the slots innocent while active carried
                                # ~18GB above baseline between chunks.
                                import gc as _gc

                                _buckets: dict = {}
                                for _obj in _gc.get_objects():
                                    if isinstance(_obj, mx.array):
                                        _kb = (
                                            tuple(_obj.shape),
                                            str(_obj.dtype),
                                        )
                                        _agg2 = _buckets.setdefault(_kb, [0, 0])
                                        _agg2[0] += 1
                                        _agg2[1] += int(_obj.nbytes)
                                _top = sorted(
                                    _buckets.items(),
                                    key=lambda kv: -kv[1][1],
                                )[:8]
                                _tot_n = sum(v[0] for v in _buckets.values())
                                _tot_b = sum(v[1] for v in _buckets.values())
                                _prev = getattr(
                                    MLLMBatchGenerator, "_census_gc_prev", {}
                                )
                                _grown = sorted(
                                    (
                                        (k, v[1] - _prev.get(k, (0, 0))[1],
                                         v[0] - _prev.get(k, (0, 0))[0])
                                        for k, v in _buckets.items()
                                    ),
                                    key=lambda kv: -kv[1],
                                )[:6]
                                MLLMBatchGenerator._census_gc_prev = dict(_buckets)
                                # Walk referrers of one array from the largest
                                # GROWN bucket to NAME the holder chain.
                                try:
                                    _tk = next(
                                        (k for k, db, dn in _grown
                                         if db > 0 and dn > 0),
                                        None,
                                    )
                                    if _tk is not None:
                                        _sample = None
                                        for _o in _gc.get_objects():
                                            if (
                                                isinstance(_o, mx.array)
                                                and tuple(_o.shape) == _tk[0]
                                                and str(_o.dtype) == _tk[1]
                                            ):
                                                _sample = _o
                                                break
                                        _chain = []
                                        _node = _sample
                                        for _depth in range(4):
                                            if _node is None:
                                                break
                                            _refs = [
                                                r for r in _gc.get_referrers(_node)
                                                if not isinstance(r, dict)
                                                or r is not locals()
                                            ]
                                            _named = None
                                            for _r in _refs:
                                                if isinstance(_r, (list, tuple, dict)):
                                                    _named = _r
                                                    break
                                            if _named is None:
                                                _chain.append(
                                                    ";".join(
                                                        type(_r).__name__
                                                        for _r in _refs[:4]
                                                    )
                                                )
                                                break
                                            _owners = [
                                                r2 for r2 in _gc.get_referrers(_named)
                                                if hasattr(r2, "__dict__")
                                                or isinstance(r2, (list, tuple, dict))
                                            ]
                                            _attr = ""
                                            for _r2 in _owners:
                                                if hasattr(_r2, "__dict__"):
                                                    for _an, _av in vars(_r2).items():
                                                        if _av is _named:
                                                            _attr = (
                                                                type(_r2).__name__
                                                                + "." + _an
                                                            )
                                                            break
                                                if _attr:
                                                    break
                                            _chain.append(
                                                f"{type(_named).__name__}"
                                                f"(len={len(_named)})"
                                                + (f"<-{_attr}" if _attr else "")
                                            )
                                            if _attr:
                                                break
                                            _node = _named
                                        logger.info(
                                            "census-gc-referrer %s %s -> %s",
                                            _tk[0],
                                            _tk[1],
                                            " -> ".join(_chain) or "unknown",
                                        )
                                except Exception as _re:
                                    logger.info("census-gc-referrer failed: %s", _re)
                                logger.info(
                                    "census-gc chunk=%d TOTAL=%d arrays "
                                    "%.2fGB | GROWN: %s",
                                    chunk_num,
                                    _tot_n,
                                    _tot_b / (1024**3),
                                    " | ".join(
                                        f"{k[0]} {k[1]} +{db / (1024**3):.3f}GB"
                                        f" (+{dn})"
                                        for k, db, dn in _grown
                                        if db > 0
                                    ) or "none",
                                )
                        except Exception:  # noqa: BLE001
                            pass
                    if _HYBRID_PREFILL_MEM_TRACE:
                        # Per-chunk memory trace. The crash at 60-100k looked
                        # like per-iteration accumulation, but that was inferred
                        # from WHERE different runs died — and those runs began
                        # with different disk/RAM cache states, so they were not
                        # comparable. This measures growth inside ONE run.
                        try:
                            logger.info(
                                "hybrid-prefill-mem chunk=%d processed=%d/%d "
                                "active=%.2fGB peak=%.2fGB cache=%.2fGB",
                                chunk_num,
                                processed,
                                seq_len,
                                mx.get_active_memory() / (1024**3),
                                mx.get_peak_memory() / (1024**3),
                                mx.get_cache_memory() / (1024**3),
                            )
                        except Exception:  # noqa: BLE001
                            pass

                _restore_kv_step()

                # DECODE-SIDE PRESIZE. Restoring `step` puts the slots back on
                # the class default of 256, so a long generation re-grows every
                # 256 tokens — and growth is mx.concatenate of the WHOLE buffer,
                # which after a 40k prompt copies 40k x heads x dim per layer,
                # per K and V, every time. That is O(n^2 / step) bytes moved
                # across a generation for no reason: the prefill path already
                # solved the identical problem by pre-sizing the span.
                #
                # Sizing `step` to the remaining output length collapses those
                # ~12 full-buffer copies into one.
                #
                # MEASURED, and it does NOT pay off — stays OFF. Muse-Glimmer-30B,
                # 27,078-token prompt, 1200 output tokens, one run per arm:
                #
                #   OFF  decode 22.21 t/s  TTFT 31.73s  total 85.76s  (fired 0x)
                #   ON   decode 24.77 t/s  TTFT 39.74s  total 88.19s  (fired 1x)
                #
                # Decode rate improves ~11% and TOTAL WALL CLOCK GETS WORSE, so
                # this stays off. Judging it on decode t/s alone would have
                # shipped a regression as a speedup — the same shape as the fp16
                # indexer that was 2.26x in isolation and -22% end to end.
                #
                # Mechanically the saving is real but smaller than it looks. At
                # a 256-token step, 1200 output tokens means OFF pays about
                # ceil(1199/256) = 5 growths, not twelve, each copying a buffer
                # that starts at the full 27k prompt and grows; ON pays one.
                # So presizing does eliminate the later copies rather than merely
                # relocating identical work, which is consistent with the decode
                # gain. What it cannot explain is the 8s TTFT rise.
                #
                # Caveat, stated honestly and NOT to be quoted as a result: this
                # is ONE run per arm. The 2.4s total delta is ~2.8%, inside
                # plausible run-to-run noise, and a single sample cannot
                # establish the TTFT rise as signal either. The only defensible
                # claim is "no demonstrated win". Re-run both arms several times
                # before concluding anything stronger.
                # VMLX_DECODE_KV_PRESIZE=1 enables it for that experiment.
                if os.environ.get("VMLX_DECODE_KV_PRESIZE", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
                    _decode_headroom = int(getattr(request, "max_tokens", 0) or 0)
                    if _decode_headroom > 0:
                        _presized_decode = 0
                        for _slot in cache:
                            # RotatingKVCache is already bounded by max_size and
                            # rotates in place — growing its step buys nothing
                            # and could over-allocate past the window.
                            if type(_slot).__name__ == "RotatingKVCache":
                                continue
                            if not _is_attention_cache_slot(_slot):
                                continue
                            try:
                                _slot.step = max(256, _decode_headroom)
                                _presized_decode += 1
                            except Exception:  # noqa: BLE001
                                pass
                        if _presized_decode:
                            logger.info(
                                "decode-kv-presize slots=%d step=%d (was 256)",
                                _presized_decode,
                                max(256, _decode_headroom),
                            )

                # Persist what this span learned so the NEXT one can be declined
                # up front. Only refit when this span produced more points than
                # the fit already in hand: a short span's two samples must not
                # displace a long span's forty, which span the context range the
                # intercept is actually determined by.
                # ALWAYS refit from the most recent span, not only when it has
                # more samples than the fit in hand. Keeping the widest-ever fit
                # sounds conservative but it is a staleness trap: peak depends on
                # what else is resident, so a fit learned while a second model
                # was loaded stays permanently inflated once that model unloads,
                # and nothing could ever displace it. Recency tracks current
                # conditions; the extrapolation bound above is what protects a
                # narrow fit from being pushed past its evidence, so the widest
                # fit no longer has to be retained to be safe.
                if len(_peak_samples) >= 2:
                    _fitted = fit_peak_model(_peak_samples)
                    if _fitted is not None:
                        self._span_peak_model = _fitted
                        self._span_peak_samples = len(_peak_samples)
                        # THIS span's largest peak, not the all-time max, for the
                        # same staleness reason as the fit itself: an all-time
                        # floor learned under heavier residency would outlive the
                        # conditions that produced it and keep inflating the
                        # projection forever.
                        self._span_largest_peak = _observed_chunk_peak_max
                        self._span_peak_max_context = max(
                            (ctx for ctx, _ in _peak_samples), default=0
                        )
                        logger.info(
                            "span-peak-fit samples=%d intercept=%.2fGB "
                            "slope=%.4fGB/1k-tok largest_observed_peak=%.2fGB",
                            len(_peak_samples),
                            _fitted[0] / (1024**3),
                            _fitted[1] * 1000 / (1024**3),
                            self._span_largest_peak / (1024**3),
                        )

                # Mixed-SWA context: capture the N-1 boundary window
                # BEFORE the single-token final forward trims the rotating
                # buffers' overhang (see the short-prompt lane above).
                self._maybe_capture_mixed_swa_boundary(request, cache)
                # Final chunk: get logits from last token
                last_chunk = input_ids[:, processed:]
                output = lm(last_chunk, **_lm_kwargs_for(processed, seq_len))
                request.vision_encoded = True
                if hasattr(output, "logits"):
                    return output.logits
                return output

        # Standard single-shot VLM forward (image requests or short text).
        # For text-only requests, try language_model first to avoid passing
        # pixel_values to models that may not accept it (e.g. smelt-loaded VLM
        # where the VLM wrapper's __call__ signature differs from standard mlx-vlm).
        # mlxstudio#83: use self.language_model (fallback-handled in __init__)
        # instead of getattr(self.model, 'language_model', None) which returns
        # None and silently falls through to the OOM-prone full-model forward.
        if not has_media_payload:
            lm = self.language_model
            if lm is not None and lm is not self.model:
                # Qwen3.5/3.6-VL hybrid (and similar mRoPE VL models) cache
                # `_position_ids` / `_rope_deltas` on the language_model across
                # calls. The VL wrapper's get_input_embeddings() text-only path
                # explicitly resets these (qwen3_5.py:36-38) before delegating;
                # bypassing the wrapper inherits stale state. Reset to match.
                # NOTE: this fix alone does NOT resolve the
                # Qwen3.6-27B-JANG_4M-CRACK garbage-output bug — preserved as
                # a defensive correctness measure mirroring wrapper semantics.
                if hasattr(lm, "_position_ids"):
                    lm._position_ids = None
                if hasattr(lm, "_rope_deltas"):
                    lm._rope_deltas = None
                lm_kwargs = {}
                if cache is not None:
                    lm_kwargs["cache"] = cache
                _supports_position_ids = _lm_supports_position_ids(lm)
                _abs_position_ids = _absolute_text_position_ids(
                    input_ids, cache, lm
                ) if cache is not None and _supports_position_ids else None
                if cache is not None:
                    _seed_text_rope_delta_for_decode(lm, input_ids)

                def _lm_kwargs_for(start: int, end: int) -> Dict[str, Any]:
                    _kwargs = dict(lm_kwargs)
                    if _abs_position_ids is not None:
                        _kwargs["position_ids"] = _abs_position_ids[:, :, start:end]
                    return _kwargs

                # vmlx#109: split prefill at clean boundary for hybrid
                # thinking models so SSM state is captured before the
                # gen-prompt suffix taints it.
                boundary = (
                    self._clean_ssm_boundary_for(request, seq_len, has_images)
                    if cache is not None else 0
                )
                ssm_boundaries = self._ssm_capture_boundaries_for(
                    request, seq_len, has_images, boundary
                ) if cache is not None else []
                if ssm_boundaries:
                    all_tokens = (
                        getattr(request, "_original_token_ids", None)
                        or input_ids[0].tolist()
                    )
                    processed = 0
                    for capture_boundary in ssm_boundaries:
                        if capture_boundary > processed:
                            lm(
                                input_ids[:, processed:capture_boundary],
                                **_lm_kwargs_for(processed, capture_boundary),
                            )
                            processed = capture_boundary
                        self._maybe_capture_clean_ssm_boundary(
                            request, cache, all_tokens, capture_boundary
                        )
                    output = lm(input_ids[:, processed:], **_lm_kwargs_for(processed, seq_len))
                else:
                    output = lm(input_ids, **_lm_kwargs_for(0, seq_len))
                request.vision_encoded = True
                if hasattr(output, "logits"):
                    return output.logits
                return output

        if has_images or has_audio_payload:
            # Media-expanded prompts must use the one-shot VLM wrapper path.
            # Drop allocator free-list memory and reject impossible requests
            # before Metal executes a command buffer that can kill the server.
            if _apply_vlm_image_request_cache_limit():
                self._vlm_cache_limit_tightened = True
            mx.clear_cache()
        _raise_if_image_prefill_exceeds_budget(
            has_images=has_images,
            has_audio_payload=has_audio_payload,
            seq_len=seq_len,
            language_model=self.language_model,
        )
        output = self._media_forward(
            request, input_ids, seq_len, cache, kwargs
        )
        request.vision_encoded = True

        if hasattr(output, "logits"):
            return output.logits
        return output

    def _run_qwen_conditioned_media_tail(
        self,
        request: "MLLMBatchRequest",
        input_ids: Any,
        cache: Optional[List[Any]],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Forward a Qwen media tail over a restored pure-text KV+SSM prefix."""
        cached_tokens = int(
            getattr(request, "_qwen_media_tail_cached_tokens", 0) or 0
        )
        full_input_ids = getattr(request, "_qwen_media_tail_full_input_ids", None)
        attempted = int(getattr(request, "_cached_tokens", 0) or cached_tokens)
        try:
            if cache is None or full_input_ids is None or cached_tokens <= 0:
                raise ValueError("missing restored cache or full Qwen media request")
            if full_input_ids.ndim == 1:
                full_input_ids = full_input_ids[None, :]
            if full_input_ids.shape[1] <= cached_tokens:
                raise ValueError("cached prefix consumes the full Qwen media request")
            if full_input_ids[:, cached_tokens:].tolist() != input_ids.tolist():
                raise ValueError("Qwen conditioned tail does not match trimmed input IDs")

            get_embeds = getattr(self.model, "get_input_embeddings", None)
            if not callable(get_embeds):
                raise ValueError("Qwen wrapper has no embedding seam")
            embed_kwargs = dict(kwargs)
            embed_kwargs.pop("cache", None)
            features = get_embeds(full_input_ids, **embed_kwargs)
            embeds = getattr(features, "inputs_embeds", None)
            if embeds is None or getattr(embeds, "ndim", 0) < 3:
                raise ValueError("Qwen wrapper returned no merged embeddings")
            feature_dict = (
                features.to_dict()
                if callable(getattr(features, "to_dict", None))
                else {}
            )
            unsupported = sorted(
                key for key, value in feature_dict.items()
                if key != "inputs_embeds" and value is not None
            )
            if unsupported:
                raise ValueError(
                    "Qwen wrapper returned unsliced auxiliary features: "
                    + ",".join(unsupported)
                )
            position_ids = getattr(self.language_model, "_position_ids", None)
            if position_ids is None or position_ids.shape[-1] != full_input_ids.shape[1]:
                raise ValueError("Qwen wrapper returned incomplete mRoPE positions")

            output = None
            tail_len = int(input_ids.shape[1])
            chunk = max(1, int(self._media_prefill_chunk_tokens(tail_len)))
            for start in range(0, tail_len, chunk):
                end = min(start + chunk, tail_len)
                full_start = cached_tokens + start
                full_end = cached_tokens + end
                output = self.language_model(
                    input_ids[:, start:end],
                    inputs_embeds=embeds[:, full_start:full_end],
                    mask=None,
                    cache=cache,
                    position_ids=position_ids[..., full_start:full_end],
                )
            if output is None:
                raise ValueError("Qwen conditioned tail produced no output")
            logger.info(
                "Qwen HYBRID conditioned media tail forwarded for %s: "
                "%d cached + %d conditioned tokens",
                getattr(request, "request_id", "?"),
                cached_tokens,
                tail_len,
            )
            _clear_mllm_request_media_payloads(request)
            for attr in (
                "_qwen_media_tail_full_input_ids",
                "_qwen_media_tail_cached_tokens",
            ):
                try:
                    delattr(request, attr)
                except AttributeError:
                    pass
            return output
        except Exception as ex:
            logger.warning(
                "Qwen HYBRID conditioned media tail failed for %s; "
                "retrying a full media prefill: %s",
                getattr(request, "request_id", "?"),
                ex,
            )
            for attr in (
                "_qwen_media_tail_full_input_ids",
                "_qwen_media_tail_cached_tokens",
            ):
                try:
                    delattr(request, attr)
                except AttributeError:
                    pass
            self._discard_request_cache_hit(
                request,
                reason="qwen_conditioned_media_tail_failed",
                attempted_cached_tokens=attempted,
            )
            cache_model = getattr(self, "_cache_model", None)
            fresh_cache = (
                cache_model.make_cache() if cache_model is not None else None
            )
            if fresh_cache is None:
                make_cache = getattr(self.language_model, "make_cache", None)
                fresh_cache = make_cache() if callable(make_cache) else None
            if fresh_cache is None or cache is None:
                raise
            cache[:] = fresh_cache
            request.attention_mask = None
            return self._run_vision_encoding_inner(request, cache)

    def _media_forward(
        self,
        request: "MLLMBatchRequest",
        input_ids: Any,
        seq_len: int,
        cache: Optional[List[Any]],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Run the media-expanded prefill, CHUNKED when the family allows it.

        The one-shot forward is what made long VL chats impossible: the whole
        media-expanded prompt goes through the language model in a single
        command buffer, so peak memory grows with the full sequence and a
        conversation dies once it crosses the machine's limit. Measured on an
        M5 Max: a 28,483-token prompt returned
        kIOGPUCommandBufferCallbackErrorOutOfMemory with 89GB free, because
        the problem is one enormous allocation, not total memory.

        The vision tower genuinely needs the whole image, but the LANGUAGE
        model does not need the whole sequence in one call. Every VL wrapper
        here already exposes the seam: `get_input_embeddings` merges pixels
        into an embedding sequence, and the language model accepts those
        embeddings plus a cache. qwen3_5 even pre-computes its mRoPE
        `position_ids` under the comment "Pre-calculate position_ids for
        chunked prefill" -- the plumbing was built for this and simply was
        not used.

        Capability detection, not a family list: the LM must NAME an
        embeddings parameter. `**kwargs` does not count -- several wrappers
        swallow `position_ids` into `**kwargs` and never read it, so counting
        that as support would build a chunked prefill on a model that ignores
        the positions. Anything undetected, unsupported, or raising falls
        straight back to the one-shot call.
        """
        one_shot = lambda: self.model(input_ids, **kwargs)

        if os.environ.get("VMLX_DISABLE_MEDIA_CHUNKED_PREFILL") in (
            "1", "true", "True", "yes", "on"
        ):
            return one_shot()
        # Wrappers may declare themselves unchunkable (gemma4 stamps
        # no_chunked_prefill when its config asks for bidirectional vision
        # attention). Honour it as a kill switch even though the MLX language
        # model does not implement that mask today.
        # `no_chunked_prefill` is an INTENT marker, not a capability answer.
        # gemma4 sets it from `use_bidirectional_attention == "vision"`, and
        # that config DEFAULTS to "vision", so honouring it as an absolute
        # kill switch made every gemma4 media prompt one-shot -- which is how
        # an 80,611-token media conversation reached
        #   [metal::malloc] Attempting to allocate 207,940,266,272 bytes
        #   which is greater than the maximum allowed buffer size of
        #   86,586,540,032 bytes
        # and every later turn failed.
        #
        # What the flag actually protects is vision spans: a bidirectional
        # mask over an image would break if the span were split across
        # forwards. (The MLX language model implements no such mask today --
        # `_make_masks` is causal-only -- so a split is currently output
        # identical, but that is a fact about today, not a licence.)
        #
        # `_media_chunk_boundaries` already guarantees no boundary lands
        # INSIDE a media run: a run that starts after the cut moves whole to
        # the next chunk, and a run already open at the cut extends the chunk
        # to cover it. So the intent is satisfiable without refusing to chunk
        # -- and it is VERIFIED below rather than assumed, falling back to
        # one-shot if any run would be split.
        _protect_media_spans = bool(
            getattr(self.model, "no_chunked_prefill", False)
        )
        if cache is None or seq_len <= 0:
            return one_shot()

        lm = self.language_model
        embed_kwarg = _media_embed_kwarg_name(lm)
        get_embeds = getattr(self.model, "get_input_embeddings", None)
        if lm is None or embed_kwarg is None or not callable(get_embeds):
            return one_shot()

        chunk = 0
        try:
            chunk = int(self._media_prefill_chunk_tokens(seq_len))
        except Exception:
            chunk = 0
        # Only chunk when the prompt is actually big. A short media prompt is
        # the common case and one-shot is strictly better for it: one pass
        # over the weights instead of several, and the peak was never the
        # problem at that size.
        if chunk <= 0 or seq_len <= max(chunk, _MEDIA_PREFILL_CHUNK_MIN_SEQ):
            return one_shot()

        try:
            features = get_embeds(input_ids, **kwargs)
        except Exception as exc:
            logger.info(
                "media chunked prefill unavailable for %s (embedding merge "
                "failed: %s); using the one-shot forward",
                getattr(request, "request_id", "?"),
                exc,
            )
            return one_shot()

        embeds = getattr(features, "inputs_embeds", None)
        if embeds is None:
            embeds = features if hasattr(features, "shape") else None
        if embeds is None or getattr(embeds, "ndim", 0) < 2:
            return one_shot()

        # Per-chunk extras that are NOT derivable from the cache offset.
        per_layer_inputs = getattr(features, "per_layer_inputs", None)
        image_mask = getattr(features, "image_mask", None)
        position_ids = getattr(lm, "_position_ids", None)
        lm_names = _named_params(getattr(lm, "__call__", None))

        media_ids: set = set()
        try:
            media_ids = self._media_placeholder_token_ids()
        except Exception:
            media_ids = set()
        token_list = None
        try:
            token_list = input_ids[0].tolist()
        except Exception:
            token_list = None
        runs = _media_placeholder_runs(token_list, media_ids)
        bounds = _media_chunk_boundaries(seq_len, chunk, runs)

        # Verify the invariant the wrapper asked for instead of trusting it.
        _split_run = None
        _prev = 0
        for _end in bounds[:-1]:
            for _rs, _re in runs:
                if _rs < _end < _re:
                    _split_run = (_rs, _re, _end)
                    break
            if _split_run:
                break
            _prev = _end
        if _split_run is not None:
            logger.info(
                "media chunked prefill declined for %s: a chunk boundary at "
                "%d would split the media run [%d, %d). Falling back to the "
                "one-shot forward.",
                getattr(request, "request_id", "?"),
                _split_run[2], _split_run[0], _split_run[1],
            )
            return one_shot()

        logger.info(
            "media chunked prefill for %s: %d tokens in %d chunks (step %d, "
            "media spans kept whole%s), peak now scales with the chunk "
            "instead of the whole prompt",
            getattr(request, "request_id", "?"),
            seq_len,
            len(bounds),
            chunk,
            "; wrapper requested span protection" if _protect_media_spans else "",
        )

        output = None
        start = 0
        for end in bounds:
            call_kwargs: Dict[str, Any] = {"cache": cache}
            call_kwargs[embed_kwarg] = embeds[:, start:end]
            # Every family in this tree builds its masks from the cache
            # offset, and muse actively overrides a caller-supplied bare mask
            # for its sliding layers. A (B, seq) padding mask must never
            # reach a chunked call.
            if "mask" in lm_names:
                call_kwargs["mask"] = None
            if position_ids is not None and "position_ids" in lm_names:
                call_kwargs["position_ids"] = position_ids[..., start:end]
            if per_layer_inputs is not None and "per_layer_inputs" in lm_names:
                # gemma4's language model slices this itself by cache offset.
                call_kwargs["per_layer_inputs"] = per_layer_inputs
            if image_mask is not None and "image_mask" in lm_names:
                call_kwargs["image_mask"] = image_mask[:, start:end]
            output = lm(input_ids[:, start:end], **call_kwargs)
            try:
                mx.clear_cache()
            except Exception:
                pass
            start = end
        return output

    def _media_prefill_chunk_tokens(self, seq_len: int) -> int:
        """Chunk size for a media-expanded prefill. BIGGER IS BETTER HERE.

        A SMALLER CHUNK DOES NOT REDUCE WEIGHT STREAMING -- IT MULTIPLIES IT.
        The chunk bounds only the terms that scale with it (activations,
        attention scores, masks); the model weights are re-read IN FULL on
        every chunk. On a MoE that is the dominant cost: dots3 restreams
        ~85GB of expert weights per chunk, so a 64-token chunk paid that 32x
        more often than a 2048-token one.

        So this deliberately floors well above the text step. The goal is the
        LARGEST chunk that still keeps peak allocation off the cliff, not the
        smallest chunk that fits.
        """
        step = int(getattr(self, "prefill_step_size", 0) or 0)
        step = max(step, _MEDIA_PREFILL_CHUNK_FLOOR)
        override = os.environ.get("VMLX_MEDIA_PREFILL_CHUNK_TOKENS")
        if override:
            try:
                step = max(1, int(override))
            except (TypeError, ValueError):
                pass
        return step

    def _process_prompts(
        self, requests: List[MLLMBatchRequest], force_batch_cache: bool = False
    ) -> MLLMBatch:
        """Prefill all requests: vision encoding, cache fetch, and batch merge.

        This is the most complex method in the batch generator. For each request:

        1. **Preprocess**: tokenize prompt + process pixel values via mlx-vlm processor.
           Save original token IDs before any cache mutation.
        2. **Cache fetch** (3 tiers + disk L2 fallback):
           - Paged: block_aware_cache.fetch_cache() -> reconstruct -> hybrid check
           - Memory-aware/Legacy: cache_obj.fetch() -> hybrid check
           - Disk L2: disk_cache.fetch() (only if in-memory missed)
           On HIT: set req.prompt_cache, trim req.input_ids, clear pixel_values.
           On HIT (hybrid + SSM companion): inject SSM state into full cache.
        3. **Vision encoding**: _run_vision_encoding() does full VLM forward pass.
           Uses req.prompt_cache if available (cache HIT = shorter prefix).
        4. **Async submit**: mx.async_eval() submits sampled token + cache states
           to GPU without blocking, enabling CPU/GPU overlap across requests.
        5. **SSM state capture** (hybrid models only): after prefill of fresh
           prompts, deep-copy SSM layer states into HybridSSMStateCache. Uses
           _original_token_ids (pre-mutation) for consistent keying.
        6. **Cache merge**: merge per-request caches into batch-aware caches
           (KVCache->BatchKVCache, MambaCache->BatchMambaCache). Single request
           optimization: keep original caches to preserve integer offsets.

        Returns:
            MLLMBatch with merged cache, first tokens, and request metadata.
        """
        tic = time.perf_counter()
        prefill_traces: Dict[str, _MLLMPrefillTrace] = {}
        language_model_cls = type(self.language_model)
        language_model_class = f"{language_model_cls.__module__}.{language_model_cls.__qualname__}"

        self._drain_tight_memory_allocator("before_prefill")

        for req in requests:
            trace = _MLLMPrefillTrace(
                request_id=req.request_id,
                prompt_tokens=0,
                has_images=False,
                is_hybrid=self._is_hybrid,
                native_mtp=_native_mtp_model_has_head(self.language_model),
                prefix_cache_enabled=self._prefix_cache_enabled,
                language_model_class=language_model_class,
                force_text_rope_1d=bool(
                    getattr(self.language_model, "_vmlx_force_text_rope_1d", False)
                ),
                supports_return_logits=_lm_supports_return_logits(self.language_model),
            )
            prefill_traces[req.request_id] = trace
            try:
                trace.start("preprocess")
                self._preprocess_request(req)
                trace.stop("preprocess")
            except PromptTooLongError as prompt_err:
                trace.stop("preprocess")
                logger.info(
                    "Rejected VLM prompt for %s before cache lookup/store: %s",
                    req.request_id,
                    prompt_err,
                )
                self._prefill_errors.append(
                    MLLMBatchResponse(
                        uid=req.uid,
                        request_id=req.request_id,
                        token=0,
                        logprobs=mx.zeros((1,)),
                        finish_reason="error",
                        error=str(prompt_err),
                        error_code="prompt_too_long",
                        error_prompt_tokens=prompt_err.prompt_tokens,
                        error_max_prompt_tokens=prompt_err.max_prompt_tokens,
                        error_source=prompt_err.source,
                    )
                )
                continue
            # Save full token list BEFORE cache fetch can mutate req.input_ids.
            # Used later for SSM state cache keying (must be consistent with fetch key).
            _all_tokens = (
                req.input_ids.tolist()
                if req.input_ids is not None and req.input_ids.ndim == 1
                else req.input_ids[0].tolist()
                if req.input_ids is not None
                else []
            )
            # Media identity used to salt EVERY block, including root text.
            # Scope exact item digests to their own causal placeholder runs so
            # an image->video chain reuses the unchanged image-conditioned
            # history and partitions only when the new video begins.
            _media_extra_keys = self._media_scoped_cache_extra_keys(
                req, _all_tokens
            )
            req._cache_extra_keys = _merge_mllm_cache_extra_keys(
                getattr(req, "_cache_extra_keys", None),
                _media_extra_keys,
            )
            # Strip generation prompt tokens from the cache key.
            # Chat templates append assistant role tokens (e.g. <|im_start|>assistant\n<think>\n)
            # at the end. The store path in mllm_scheduler._cleanup_finished() strips these
            # before storing block hashes. The fetch key here MUST match.
            _gpl = getattr(req, '_gen_prompt_len', 0)
            if _gpl > 0 and _gpl < len(_all_tokens):
                # Capture the gen-prefix tokens BEFORE trimming so the
                # scheduler's output-side re-emit suppressor can compare
                # against them. Without this, thinking models on dense
                # multi-turn history (no reasoning_content wrapper in prior
                # assistant messages) re-emit `<|im_start|>assistant\n<think>\n`
                # as their first output tokens, corrupting the reasoning
                # stream for the user.
                req._gen_prefix_tokens = list(_all_tokens[-_gpl:])
                _all_tokens = _all_tokens[:-_gpl]
            else:
                req._gen_prefix_tokens = []
            req._original_token_ids = _all_tokens
            # Track how many prompt tokens were served from cache (for usage reporting)
            req._cached_tokens = 0
            req._cache_execution_started = time.perf_counter()
            req._cache_execution = {
                "request_id": normalize_ssm_telemetry_request_id(
                    req.request_id
                ),
                "cache_detail": None,
                # API usage counts the cache-key prompt. The template-owned
                # generation suffix is tracked separately and still forwarded.
                "prompt_tokens": len(_all_tokens),
                "cache_key_tokens": len(_all_tokens),
                "generation_prompt_suffix_tokens": len(
                    getattr(req, "_gen_prefix_tokens", None) or []
                ),
                "attempted_cached_tokens": 0,
                "cached_tokens": 0,
                "uncached_prompt_tokens": len(_all_tokens),
                "prefill_tokens": _mllm_input_ids_token_count(req.input_ids),
                "selection": "miss",
                "cache_outcome": "miss",
                "cache_reuse_applied": False,
                "reconstructed": False,
                "dequantized": False,
                "reconstruction_seconds": 0.0,
                "dequantization_seconds": 0.0,
                "total_worker_cache_seconds": 0.0,
                "media_cache_scope": getattr(
                    req, "_media_cache_scope", None
                ),
            }
            trace.set(
                prompt_tokens=len(_all_tokens),
                has_images=req.pixel_values is not None,
            )
            trace.start("cache_lookup")
            # After preprocessing, the prompt is fully tokenized including image patches.
            # Query the BlockAwarePrefixCache for reusable KV blocks.
            # fetch_cache returns (block_table, remaining_tokens) — NOT cache objects!
            # IMPORTANT: Skip prefix cache for requests WITH images — image placeholder
            # tokens are identical for same-sized images regardless of content, so a
            # cache hit would serve KV states from a different image's vision encoding.
            # Text-only follow-up requests (no new images) can safely use prefix cache.
            has_images = req.pixel_values is not None
            # Per-request cache bypass (cache_salt / skip_prefix_cache).
            # When set, skip every VLM prefix-cache layer — paged, memory-aware,
            # legacy prefix, disk L2, SSM companion — so benchmark runs get
            # fresh execution without pollution from prior multimodal requests.
            _mllm_bypass = bool(getattr(req, "_bypass_prefix_cache", False))
            if _mllm_bypass:
                req._cache_execution.update(
                    {
                        "selection": "bypass",
                        "cache_outcome": "bypass",
                        "fallback_reason": "skip_prefix_cache",
                    }
                )
                _record_fetch_bypass = getattr(
                    self.block_aware_cache, "record_fetch_bypass", None
                )
                if callable(_record_fetch_bypass):
                    _record_fetch_bypass(
                        req.request_id,
                        attempted_tokens=len(_all_tokens),
                    )
            _media_context = self._request_has_media_cache_context(req)
            _media_cache_allowed = (
                _media_context and self._media_prefix_cache_allowed(req)
            )
            # Once an image enters a conversation, EVERY later text-only turn
            # used to re-prefill the entire history, forever. The two halves of
            # the gate disagree by construction: _media_context is TOKEN-based
            # (placeholders anywhere in the prompt, so True for the rest of the
            # chat), while _media_cache_allowed needs req._cache_extra_keys,
            # which is PAYLOAD-derived and therefore None the moment the user
            # stops re-attaching the picture. A text-only turn can never
            # reproduce a hash of image bytes it does not have, so it fell off
            # the allow-list and the gate skipped the fetch outright -- not
            # even the pure-text prefix sitting unsalted in the store from
            # turn 1. Measured cost in a VL document chat: full re-prefill of
            # the whole history on every single turn.
            #
            # Everything strictly BEFORE the first media placeholder is pure
            # text, is token-deterministic, and was stored unsalted. Reusing
            # exactly that region is safe by construction -- recurrent state
            # cannot pair with any image because the boundary precedes every
            # placeholder -- so this turn gets a positional cap instead of a
            # blanket skip.
            _media_payload_present = bool(
                getattr(req, "pixel_values", None) is not None
                or getattr(req, "video_pixel_values", None) is not None
                or getattr(req, "audio_codes", None) is not None
                or getattr(req, "audio_embeds", None) is not None
                or getattr(req, "audio_features", None) is not None
            )
            _media_text_prefix_only = bool(
                _media_context
                and not _media_cache_allowed
                and not _media_payload_present
            )
            if (
                self._prefix_cache_enabled
                and self.block_aware_cache is not None
                and req.prompt_cache is None
                and (
                    not _media_context
                    or _media_cache_allowed
                    or _media_text_prefix_only
                )
                and not _mllm_bypass
            ):
                if req.input_ids is not None:
                    try:
                        _full_token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()
                        # Strip gen_prompt_len from fetch key to match store key.
                        # CRITICAL: keep the stripped gpl suffix — after fetching we MUST
                        # prepend it back to `remaining` so the model re-sees the
                        # `<|im_start|>assistant\n<think>\n` template tokens on turn 2.
                        # Without this the model's prefill skips the thinking marker and
                        # jumps straight to content, producing "1 completion token → EOS"
                        # symptoms on hybrid thinking models (Qwen 3.5/3.6 VL, Nemotron
                        # Cascade, MiniMax). See bug trace 2026-04-21.
                        _gpl = getattr(req, '_gen_prompt_len', 0)
                        if _gpl > 0 and _gpl < len(_full_token_list):
                            token_list = _full_token_list[:-_gpl]
                            _gpl_suffix = _full_token_list[-_gpl:]
                        else:
                            token_list = _full_token_list
                            _gpl_suffix = []
                        _paged_disk_hits_before = 0
                        try:
                            _paged_disk_hits_before = int(
                                getattr(
                                    getattr(
                                        self.block_aware_cache.paged_cache,
                                        "stats",
                                        None,
                                    ),
                                    "disk_hits",
                                    0,
                                )
                                or 0
                            )
                        except Exception:
                            _paged_disk_hits_before = 0
                        _cache_extra_keys = getattr(req, "_cache_extra_keys", None)
                        # Positional cap for a text-only turn whose HISTORY
                        # holds media: match only the pure-text region that
                        # precedes the first placeholder. Nothing in that
                        # region depends on pixels, so the unsalted key that
                        # stored it is the correct key to read it back with.
                        _fetch_token_list = token_list
                        _media_prefix_cap = None
                        if _media_text_prefix_only:
                            _media_prefix_cap = self._media_safe_capture_limit(
                                list(token_list)
                            )
                            _min_worth = int(
                                getattr(self.block_aware_cache, "block_size", 64)
                                or 64
                            )
                            if _media_prefix_cap < _min_worth:
                                logger.info(
                                    "media-history turn %s: only %d pure-text "
                                    "tokens precede the first placeholder "
                                    "(< one %d-token block) -- not worth a "
                                    "fetch, full prefill",
                                    req.request_id,
                                    _media_prefix_cap,
                                    _min_worth,
                                )
                                _media_prefix_cap = None
                                _fetch_token_list = []
                            else:
                                _fetch_token_list = token_list[:_media_prefix_cap]
                                _cache_extra_keys = None
                                logger.info(
                                    "media-history turn %s: capping the prefix "
                                    "fetch at %d/%d tokens (the pure-text "
                                    "region before the first media "
                                    "placeholder) instead of skipping the "
                                    "cache entirely",
                                    req.request_id,
                                    _media_prefix_cap,
                                    len(token_list),
                                )
                        block_table, remaining = (
                            self.block_aware_cache.fetch_cache(
                                req.request_id,
                                _fetch_token_list,
                                cache_extra_keys=_cache_extra_keys,
                            )
                            if _fetch_token_list
                            else (None, list(token_list))
                        )
                        if _media_prefix_cap is not None and block_table is not None:
                            # fetch_cache computed `remaining` against the
                            # CAPPED list, so it stops at the cap. The real
                            # uncached tail is everything after the hit in the
                            # FULL prompt -- including the media region, which
                            # this turn must forward itself.
                            _hit_len = int(
                                getattr(block_table, "num_tokens", 0) or 0
                            )
                            remaining = list(token_list[_hit_len:])
                        if os.environ.get("VMLX_CACHE_HASH_DEBUG") == "1":
                            # Dump the ACTUAL engine token stream, because an
                            # offline re-render of the same conversation
                            # produces a different media-placeholder count and
                            # will point at the wrong divergence boundary.
                            _tl = list(token_list)
                            # Count the media placeholders too. If the SAME
                            # image expands to a different number of tokens
                            # between two turns of one conversation, every
                            # position after the first placeholder shifts and
                            # no block hash can ever match again -- which
                            # looks exactly like "the cache is broken" while
                            # the cache is working perfectly on a prompt that
                            # genuinely is not the same prompt.
                            try:
                                _media_ids = self._media_placeholder_token_ids()
                            except Exception:
                                _media_ids = set()
                            _media_count = (
                                sum(1 for _t in _tl if _t in _media_ids)
                                if _media_ids else 0
                            )
                            _first_media = next(
                                (
                                    _i
                                    for _i, _t in enumerate(_tl)
                                    if _t in _media_ids
                                ),
                                -1,
                            ) if _media_ids else -1
                            logger.info(
                                "mm-restore-debug TOKENS req=%s n=%d "
                                "media_tokens=%d first_media_at=%d "
                                "head=%s tail=%s",
                                req.request_id, len(_tl), _media_count,
                                _first_media, _tl[:8], _tl[-8:],
                            )
                            # Segment fingerprints. head/tail cannot show WHERE
                            # two prompts stop agreeing, and that is the only
                            # question worth asking when a chain matches fewer
                            # blocks than were stored: comparing two of these
                            # lines shows the first 512-token window that
                            # differs, without dumping thousands of ids.
                            _segs = []
                            for _off in range(0, len(_tl), 512):
                                _chunk = _tl[_off:_off + 512]
                                _segs.append(
                                    "%d:%s"
                                    % (
                                        _off,
                                        hashlib.sha256(
                                            repr(_chunk).encode()
                                        ).hexdigest()[:8],
                                    )
                                )
                            logger.info(
                                "mm-restore-debug SEGMENTS req=%s n=%d %s",
                                req.request_id, len(_tl), " ".join(_segs),
                            )
                            _block_size = max(
                                1,
                                int(
                                    getattr(
                                        self.block_aware_cache,
                                        "block_size",
                                        64,
                                    )
                                    or 64
                                ),
                            )
                            _block_digests = [
                                "%d:%s"
                                % (
                                    _off,
                                    hashlib.sha256(
                                        repr(_tl[_off : _off + _block_size]).encode()
                                    ).hexdigest()[:12],
                                )
                                for _off in range(0, len(_tl), _block_size)
                            ]
                            _side_key_digest = hashlib.sha256(
                                repr(_cache_extra_keys).encode()
                            ).hexdigest()[:12]
                            logger.info(
                                "mm-restore-debug BLOCKS req=%s block_size=%d "
                                "side_key=%s %s",
                                req.request_id,
                                _block_size,
                                _side_key_digest,
                                " ".join(_block_digests),
                            )
                            logger.info(
                                "mm-restore-debug FETCHED req=%s block_table=%s "
                                "num_tokens=%s prompt_tokens=%d media_allowed=%s",
                                req.request_id,
                                "None" if block_table is None else "present",
                                getattr(block_table, "num_tokens", None),
                                len(token_list),
                                _media_cache_allowed,
                            )
                        _paged_disk_hit = False
                        try:
                            _paged_disk_hits_after = int(
                                getattr(
                                    getattr(
                                        self.block_aware_cache.paged_cache,
                                        "stats",
                                        None,
                                    ),
                                    "disk_hits",
                                    0,
                                )
                                or 0
                            )
                            _paged_disk_hit = _paged_disk_hits_after > _paged_disk_hits_before
                        except Exception:
                            _paged_disk_hit = False
                        if block_table is not None:
                                req._cache_execution["attempted_cached_tokens"] = int(
                                    getattr(block_table, "num_tokens", 0) or 0
                                )
                                req._cache_execution["blocks"] = (
                                    _block_table_block_count(block_table)
                                )
                                req._cache_execution["selection"] = (
                                    "block-disk"
                                    if bool(
                                        getattr(
                                            self.paged_cache_manager,
                                            "disk_only",
                                            False,
                                        )
                                    )
                                    else "paged"
                                )
                                # Hybrid models (SSM + attention, e.g. Qwen3.5-VL):
                                # Prefix cache stores only KVCache (attention) layers.
                                # SSM layers are cumulative state that must process ALL tokens.
                                # For hybrid models without companion SSM state, the cached
                                # KV blocks are useless — skip reconstruction entirely to
                                # avoid allocating huge tensors that will be thrown away.
                                is_hybrid = (
                                    self._is_hybrid
                                    and self._ssm_state_cache is not None
                                )

                                if is_hybrid:
                                    # Check companion SSM state cache BEFORE reconstruction.
                                    # Use actual prompt token count (not block-aligned) to match
                                    # the store key which also uses len(all_tokens).
                                    # REQ-A3-001: fetch now returns Optional[Tuple[List[Any], bool]].
                                    # is_complete unused here (live MLLM fetch path, not the trie
                                    # path) — Agent 1's PrefixCacheManager consumes it.
                                    _fetch_num = block_table.num_tokens
                                    _ssm_extra_keys = (
                                        _ssm_companion_cache_extra_keys(req)
                                    )
                                    _entry = self._ssm_state_cache.fetch(
                                        token_list,
                                        _fetch_num,
                                        cache_extra_keys=_ssm_extra_keys,
                                    ) if _fetch_num > 0 else None
                                    _record_request_ssm_exact_lookup(
                                        req,
                                        self._ssm_state_cache,
                                        max_len=int(_fetch_num or 0),
                                        matched=_entry is not None,
                                        is_complete=(
                                            bool(_entry[1])
                                            if _entry is not None
                                            else False
                                        ),
                                    )
                                    if os.environ.get("VMLX_CACHE_HASH_DEBUG") == "1":
                                        logger.info(
                                            "mm-restore-debug SSM-EXACT req=%s asked_N=%s "
                                            "hit=%s (store writes at prompt_len, this asks "
                                            "block-aligned)",
                                            req.request_id, _fetch_num,
                                            _entry is not None,
                                        )
                                    if _entry is None:
                                        ssm_states = None
                                    else:
                                        ssm_states, _is_complete = _entry
                                        # is_complete=False means the stored SSM state
                                        # was captured AFTER processing the full prompt
                                        # including the gen_prompt_len (<think>\n) suffix.
                                        # The state represents more tokens than the key
                                        # claims — re-using it while re-feeding the gpl
                                        # suffix double-applies those tokens and causes
                                        # <think></think> generation loops. Reject the
                                        # hit and fall back to full prefill until we have
                                        # a proper pre-gpl capture path.
                                        if not _is_complete:
                                            logger.info(
                                                f"SSM companion for {req.request_id}: "
                                                f"is_complete=False (gpl-contaminated), "
                                                f"rejecting hit — full prefill"
                                            )
                                            ssm_states = None
                                    if ssm_states is None:
                                        # vmlx#91: exact SSM state miss — resume from the
                                        # longest stored checkpoint whose tokens are a strict
                                        # prefix of the current query. Default ON in v1.3.66;
                                        # set VMLX_DISABLE_SSM_PREFIX_RESUME=1 to force the
                                        # legacy full-prefill path.
                                        import os as _os
                                        _enable_resume = _os.environ.get(
                                            "VMLX_DISABLE_SSM_PREFIX_RESUME"
                                        ) not in ("1", "true", "True", "yes", "on")
                                        if _enable_resume:
                                            (
                                                _missed_ck,
                                                _ssm_prefix_lookup,
                                            ) = _fetch_request_ssm_longest_prefix(
                                                req,
                                                self._ssm_state_cache,
                                                enabled=True,
                                                token_ids=token_list,
                                                max_len=int(_fetch_num or 0),
                                                cache_extra_keys=_ssm_extra_keys,
                                                exact_boundary_already_missed=True,
                                            )
                                        else:
                                            _missed_ck = None
                                            _ssm_prefix_lookup = dict(
                                                req._cache_execution[
                                                    "ssm_prefix_lookup"
                                                ]
                                            )

                                        if os.environ.get("VMLX_CACHE_HASH_DEBUG") == "1":
                                            logger.info(
                                                "mm-restore-debug SSM-RESUME req=%s max_len=%s "
                                                "checkpoint=%s",
                                                req.request_id, int(_fetch_num or 0),
                                                "None" if _missed_ck is None
                                                else "len=%s complete=%s" % (
                                                    _missed_ck[0], _missed_ck[2]),
                                            )
                                        if _enable_resume and _missed_ck is not None:
                                            _ck_len, _ck_states, _ck_complete = _missed_ck
                                            if not _ck_complete:
                                                # Checkpoint was captured post-gpl-prefill
                                                # — state reflects more tokens than the
                                                # stored key. Reject to avoid the same
                                                # <think></think> loop seen on direct hits.
                                                self._stats.hybrid_kv_without_ssm_hits += 1
                                                self._stats.hybrid_kv_without_ssm_tokens += int(
                                                    getattr(block_table, "num_tokens", 0) or 0
                                                )
                                                self._stats.last_hybrid_kv_without_ssm = {
                                                    "request_id": normalize_ssm_telemetry_request_id(
                                                        req.request_id
                                                    ),
                                                    "cached_tokens": int(
                                                        getattr(block_table, "num_tokens", 0) or 0
                                                    ),
                                                    "reason": "checkpoint_incomplete",
                                                    "checkpoint_tokens": int(_ck_len or 0),
                                                }
                                                self._stats.last_hybrid_kv_without_ssm[
                                                    "ssm_prefix_lookup"
                                                ] = dict(_ssm_prefix_lookup)
                                                logger.info(
                                                    f"vmlx#91 RESUME skipped for {req.request_id}: "
                                                    f"checkpoint at {_ck_len} has is_complete=False "
                                                    f"(gpl-contaminated) — full prefill"
                                                )
                                                self._mark_required_ssm_checkpoint(
                                                    req,
                                                    int(getattr(block_table, "num_tokens", 0) or 0),
                                                )
                                                self._adjust_paged_hit_credit(req.request_id, 0)
                                                self.block_aware_cache.release_cache(req.request_id)
                                                continue
                                            _requested_kv_tokens = int(
                                                getattr(block_table, "num_tokens", 0) or 0
                                            )
                                            # Advancing the companion UP to the hit
                                            # boundary has to happen BEFORE the trim.
                                            # The seed needs attention KV covering
                                            # [0, checkpoint), and the checkpoint sits
                                            # ABOVE the block-aligned floor the trim
                                            # drops to -- reconstructing afterwards
                                            # yields KV that stops short of the very
                                            # state it has to pair with, and the
                                            # offset guard then correctly refuses it.
                                            # Measured live: checkpoint 38467, trim to
                                            # 38464, full hit 38528. Only the untrimmed
                                            # hit can be sliced to 38467.
                                            _delta_states = None
                                            if int(_ck_len or 0) < _requested_kv_tokens:
                                                _delta_states = (
                                                    self._derive_hybrid_companion_delta(
                                                        req,
                                                        token_list,
                                                        int(_fetch_num or 0),
                                                        int(_ck_len or 0),
                                                        block_table,
                                                        cache_extra_keys=_ssm_extra_keys,
                                                    )
                                                )
                                            if _delta_states:
                                                ssm_states = _delta_states
                                                remaining = token_list[int(_fetch_num or 0):]
                                                self._adjust_paged_hit_credit(
                                                    req.request_id, int(_fetch_num or 0)
                                                )
                                                logger.info(
                                                    f"vmlx#91 DELTA accepted for "
                                                    f"{req.request_id}: kept the full "
                                                    f"{_fetch_num}-token KV hit, companion "
                                                    f"advanced from {_ck_len}. Prefill "
                                                    f"tail: {len(remaining)} tokens"
                                                )
                                                trimmed = None
                                            else:
                                                # Trim the block_table down to block-aligned
                                                # <= _ck_len so KV + SSM stay aligned.
                                                trimmed = self.block_aware_cache.trim_block_table(
                                                    req.request_id, _ck_len
                                                )
                                            # KV block tables trim to WHOLE blocks while SSM
                                            # state is cumulative at exactly _ck_len. Pairing
                                            # KV@aligned<_ck_len with SSM@_ck_len re-feeds the
                                            # gap tokens through layers whose state already
                                            # absorbed them -- the double-application class the
                                            # LLM scheduler refuses via its checkpoint_len ==
                                            # aligned_len contract (a 1-token version of this
                                            # caused the v1.3.77 think-loop). Accept the resume
                                            # ONLY on exact alignment; otherwise full prefill.
                                            if (
                                                not _delta_states
                                                and trimmed is not None
                                                and trimmed.num_tokens > 0
                                                and int(trimmed.num_tokens) != int(_ck_len or 0)
                                            ):
                                                self._stats.hybrid_kv_without_ssm_hits += 1
                                                self._stats.hybrid_kv_without_ssm_tokens += int(
                                                    getattr(block_table, "num_tokens", 0) or 0
                                                )
                                                self._stats.last_hybrid_kv_without_ssm = {
                                                    "request_id": normalize_ssm_telemetry_request_id(
                                                        req.request_id
                                                    ),
                                                    "cached_tokens": int(
                                                        getattr(block_table, "num_tokens", 0) or 0
                                                    ),
                                                    "reason": "kv_ssm_checkpoint_misaligned",
                                                    "checkpoint_tokens": int(_ck_len or 0),
                                                    "kv_aligned_tokens": int(trimmed.num_tokens),
                                                }
                                                self._stats.last_hybrid_kv_without_ssm[
                                                    "ssm_prefix_lookup"
                                                ] = dict(_ssm_prefix_lookup)
                                                logger.info(
                                                    f"vmlx#91 RESUME skipped for {req.request_id}: "
                                                    f"checkpoint at {_ck_len} is not block-aligned "
                                                    f"(KV trims to {trimmed.num_tokens}) — pairing "
                                                    f"them would double-apply the gap through the "
                                                    f"SSM state; full prefill"
                                                )
                                                self._mark_required_ssm_checkpoint(
                                                    req,
                                                    int(getattr(block_table, "num_tokens", 0) or 0),
                                                )
                                                self._adjust_paged_hit_credit(req.request_id, 0)
                                                self.block_aware_cache.release_cache(req.request_id)
                                                continue
                                            if trimmed is not None and trimmed.num_tokens > 0:
                                                self._adjust_paged_hit_credit(
                                                    req.request_id, trimmed.num_tokens
                                                )
                                                block_table = trimmed
                                                ssm_states = _ck_states
                                                remaining = token_list[trimmed.num_tokens:]
                                                logger.info(
                                                    f"vmlx#91 RESUME for {req.request_id}: "
                                                    f"trimmed KV to {trimmed.num_tokens} tokens "
                                                    f"(block-aligned from checkpoint at {_ck_len}), "
                                                    f"SSM state reused from checkpoint. "
                                                    f"Prefill tail: {len(remaining)} tokens"
                                                )
                                                if _requested_kv_tokens > trimmed.num_tokens:
                                                    self._mark_required_ssm_checkpoint(
                                                        req,
                                                        _requested_kv_tokens,
                                                        reset_cached_tokens=False,
                                                    )
                                                # Fall through to reconstruct with the trimmed
                                                # block_table + ssm_states.
                                            else:
                                                # Trim returned None (e.g. checkpoint below
                                                # one block) — fall back to full prefill.
                                                self._stats.hybrid_kv_without_ssm_hits += 1
                                                self._stats.hybrid_kv_without_ssm_tokens += int(
                                                    getattr(block_table, "num_tokens", 0) or 0
                                                )
                                                self._stats.last_hybrid_kv_without_ssm = {
                                                    "request_id": normalize_ssm_telemetry_request_id(
                                                        req.request_id
                                                    ),
                                                    "cached_tokens": int(
                                                        getattr(block_table, "num_tokens", 0) or 0
                                                    ),
                                                    "reason": "checkpoint_below_one_block",
                                                    "checkpoint_tokens": int(_ck_len or 0),
                                                }
                                                self._stats.last_hybrid_kv_without_ssm[
                                                    "ssm_prefix_lookup"
                                                ] = dict(_ssm_prefix_lookup)
                                                logger.info(
                                                    f"vmlx#91 RESUME skipped for {req.request_id}: "
                                                    f"checkpoint at {_ck_len} below one block — "
                                                    f"full prefill required"
                                                )
                                                self._mark_required_ssm_checkpoint(
                                                    req,
                                                    int(getattr(block_table, "num_tokens", 0) or 0),
                                                )
                                                self._adjust_paged_hit_credit(req.request_id, 0)
                                                self.block_aware_cache.release_cache(req.request_id)
                                                continue
                                        else:
                                            self._stats.hybrid_kv_without_ssm_hits += 1
                                            self._stats.hybrid_kv_without_ssm_tokens += int(
                                                getattr(block_table, "num_tokens", 0) or 0
                                            )
                                            self._stats.last_hybrid_kv_without_ssm = {
                                                "request_id": normalize_ssm_telemetry_request_id(
                                                    req.request_id
                                                ),
                                                "cached_tokens": int(
                                                    getattr(block_table, "num_tokens", 0) or 0
                                                ),
                                                "reason": (
                                                    "ssm_prefix_resume_disabled"
                                                    if not _enable_resume
                                                    else "no_ssm_companion_state"
                                                ),
                                            }
                                            self._stats.last_hybrid_kv_without_ssm[
                                                "ssm_prefix_lookup"
                                            ] = dict(_ssm_prefix_lookup)
                                            if _missed_ck is not None:
                                                _ck_len, _, _ = _missed_ck
                                                self._stats.last_hybrid_kv_without_ssm[
                                                    "checkpoint_tokens"
                                                ] = int(_ck_len or 0)
                                                logger.info(
                                                    f"VLM prefix cache MISS for {req.request_id}: "
                                                    f"{block_table.num_tokens} KV blocks found but "
                                                    f"resume path disabled (VMLX_DISABLE_SSM_PREFIX_RESUME=1); "
                                                    f"stored checkpoint at {_ck_len} tokens was available. "
                                                    f"Full prefill required."
                                                )
                                            else:
                                                logger.info(
                                                    f"VLM prefix cache MISS for {req.request_id}: "
                                                    f"{block_table.num_tokens} KV blocks found but "
                                                    f"no SSM companion state — full prefill required"
                                                )
                                            # Release the block refs that fetch_cache incremented
                                            # to prevent ref_count leak → OOM on subsequent requests.
                                            self._mark_required_ssm_checkpoint(
                                                req,
                                                int(getattr(block_table, "num_tokens", 0) or 0),
                                            )
                                            self._adjust_paged_hit_credit(req.request_id, 0)
                                            self.block_aware_cache.release_cache(req.request_id)
                                            continue  # Skip reconstruction

                                # Either non-hybrid OR hybrid with SSM state — reconstruct
                                _block_disk_only = bool(
                                    getattr(self.paged_cache_manager, "disk_only", False)
                                )
                                _cache_execution = dict(
                                    getattr(req, "_cache_execution", None) or {}
                                )
                                _cache_execution.update(
                                    {
                                        "request_id": normalize_ssm_telemetry_request_id(
                                            req.request_id
                                        ),
                                        "cache_detail": None,
                                        "attempted_cached_tokens": int(
                                            getattr(block_table, "num_tokens", 0) or 0
                                        ),
                                        "blocks": _block_table_block_count(block_table),
                                        "selection": (
                                            "block-disk" if _block_disk_only else "paged"
                                        ),
                                        "disk_hit": bool(_paged_disk_hit),
                                    }
                                )
                                _cache_execution_started = float(
                                    getattr(
                                        req,
                                        "_cache_execution_started",
                                        time.perf_counter(),
                                    )
                                )
                                # The mixed-SWA clean store reconstructs the
                                # stored chain AGAIN a moment later to build its
                                # base (mllm_scheduler._clean_store_base_from_stored_chain),
                                # so the same chain is walked twice per turn.
                                # Retain a pristine copy here so that second walk
                                # is a lookup — but only opportunistically: the
                                # memo declines itself when a second copy would
                                # not fit under the working-set budget, because
                                # a mixed-SWA L1 is ~4.6GB at 86k and an OOM
                                # costs far more than the reconstruction.
                                _arm_memo = getattr(
                                    self.block_aware_cache, "arm_reconstruct_memo", None
                                )
                                if callable(_arm_memo):
                                    try:
                                        _arm_memo(True)
                                    except Exception:  # noqa: BLE001
                                        pass
                                _reconstruct_started = time.perf_counter()
                                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                                _cache_execution["reconstruction_seconds"] = round(
                                    max(0.0, time.perf_counter() - _reconstruct_started), 6
                                )
                                _cache_execution["reconstructed"] = reconstructed is not None
                                _cache_execution["reconstruction_ok"] = reconstructed is not None
                                _paged_disk_hit, _reconstruct_disk_blocks = (
                                    _paged_reconstruct_disk_source(
                                        fetch_disk_hit=_paged_disk_hit,
                                        block_aware_cache=self.block_aware_cache,
                                        reconstructed=reconstructed,
                                    )
                                )
                                _cache_execution["disk_hit"] = bool(_paged_disk_hit)
                                if _reconstruct_disk_blocks > 0:
                                    _cache_execution["disk_blocks"] = (
                                        _reconstruct_disk_blocks
                                    )
                                _tq_native_blocks = int(
                                    getattr(
                                        self.block_aware_cache,
                                        "_last_reconstruct_tq_blocks",
                                        0,
                                    )
                                    or 0
                                )
                                if _tq_native_blocks > 0:
                                    _cache_execution["tq_native_blocks"] = (
                                        _tq_native_blocks
                                    )
                                if reconstructed is None:
                                    # Never fall through with a credited hit
                                    # and no cache: the request would silently
                                    # full-prefill while the fetch's block refs
                                    # and hit credit stay live (the text
                                    # scheduler already rolls this back —
                                    # scheduler._release_unusable_paged_hit).
                                    logger.info(
                                        "VLM paged cache hit for %s "
                                        "reconstructed to nothing (%d tokens, "
                                        "%d blocks) — declining the hit: "
                                        "releasing blocks and zeroing hit "
                                        "credit before the full prefill",
                                        req.request_id,
                                        int(
                                            getattr(
                                                block_table, "num_tokens", 0
                                            )
                                            or 0
                                        ),
                                        _block_table_block_count(block_table),
                                    )
                                    req._cache_execution = dict(
                                        _cache_execution
                                    )
                                    self._discard_request_cache_hit(
                                        req,
                                        reason="paged_reconstruction_failed",
                                        attempted_cached_tokens=int(
                                            getattr(
                                                block_table, "num_tokens", 0
                                            )
                                            or 0
                                        ),
                                    )
                                    continue
                                if reconstructed is not None:
                                    _dequant_started = time.perf_counter()
                                    reconstructed = _dequantize_cache(reconstructed)
                                    _cache_execution["dequantization_seconds"] = round(
                                        max(0.0, time.perf_counter() - _dequant_started), 6
                                    )
                                    _cache_execution["dequantized"] = reconstructed is not None
                                    _cache_execution["dequantization_ok"] = reconstructed is not None
                                    if reconstructed is None:
                                        # Dequantize failed — release block refs to prevent leak
                                        self.block_aware_cache.release_cache(req.request_id)
                                        continue
                                    if not _validate_prompt_cache(
                                        reconstructed,
                                        source=f"mllm-paged-fetch:{req.request_id}",
                                    ):
                                        self.block_aware_cache.release_cache(req.request_id)
                                        continue
                                if reconstructed is not None and not is_hybrid:
                                    # A COMPACTED partial reconstruction (KV
                                    # layers densely packed, cumulative layers
                                    # deferred) is only restorable for hybrid
                                    # families whose SSM state arrives from the
                                    # external companion. For everyone else the
                                    # layer-index mapping is simply LOST — the
                                    # repair stage would wipe it to fresh
                                    # templates and the request would burn a
                                    # hit-then-reject cycle before its cold
                                    # prefill (measured live on dots3, ledger
                                    # row 152). Miss honestly instead.
                                    _cm = getattr(self, "_cache_model", None) or self.language_model
                                    _expected = getattr(self, "_expected_cache_layer_count", None)
                                    if _expected is None:
                                        try:
                                            _expected = len(_cm.make_cache())
                                        except Exception:
                                            _expected = 0
                                        self._expected_cache_layer_count = _expected
                                    if _expected and len(reconstructed) < _expected:
                                        logger.info(
                                            "Compacted partial reconstruction (%d/%d "
                                            "layers) for non-companion family — "
                                            "declining hit for %s, cold prefill",
                                            len(reconstructed),
                                            _expected,
                                            req.request_id,
                                        )
                                        self.block_aware_cache.release_cache(req.request_id)
                                        continue
                                if is_hybrid and ssm_states is not None and reconstructed is not None:
                                    # Full hybrid cache reconstruction:
                                    # KV from paged cache + SSM from companion cache
                                    full_cache = _fix_hybrid_cache(
                                        reconstructed,
                                        getattr(self, "_cache_model", None)
                                        or self.language_model,
                                        kv_positions=self._hybrid_kv_positions,
                                        num_model_layers=self._hybrid_num_layers,
                                    )
                                    # Inject stored SSM states at non-KV positions
                                    kv_set = set(self._hybrid_kv_positions or [])
                                    ssm_idx = 0
                                    for layer_idx in range(len(full_cache)):
                                        if layer_idx in kv_set:
                                            continue
                                        # Positional full-latent slots keep
                                        # the block-reconstructed (adopted)
                                        # content — companion entries no
                                        # longer carry them.
                                        if _companion_exempt_cache(
                                            full_cache[layer_idx]
                                        ):
                                            continue
                                        if ssm_idx < len(ssm_states):
                                            full_cache[layer_idx] = ssm_states[ssm_idx]
                                            ssm_idx += 1
                                    if not _validate_prompt_cache(
                                        full_cache,
                                        source=f"mllm-hybrid-paged-fetch:{req.request_id}",
                                    ):
                                        self.block_aware_cache.release_cache(req.request_id)
                                        continue
                                    # TQ recompress safe: blocks now store original float16
                                    req.prompt_cache = full_cache
                                    req._cached_tokens = block_table.num_tokens
                                    req._cache_detail = _paged_hybrid_cache_detail(
                                        disk_hit=_paged_disk_hit,
                                        mixed_attention=self._mixed_attention_cache_model,
                                        disk_only=_block_disk_only,
                                        tq_native=_tq_native_blocks > 0,
                                    )
                                    _cache_execution["cache_detail"] = req._cache_detail
                                    _cache_execution["total_worker_cache_seconds"] = round(
                                        max(0.0, time.perf_counter() - _cache_execution_started),
                                        6,
                                    )
                                    req._cache_execution = dict(_cache_execution)
                                    # Re-attach the gen-prompt suffix: fetch key was
                                    # gpl-stripped but the model MUST see the template
                                    # suffix (<|im_start|>assistant\n<think>\n) to enter
                                    # thinking mode on turn 2.
                                    _full_remaining = (remaining or []) + list(_gpl_suffix)
                                    # Dropping pixel_values below is what makes
                                    # a warm media turn cheap -- the vision
                                    # tower is skipped entirely because the
                                    # image's KV is already in the restored
                                    # prefix. That is only true when the hit
                                    # COVERS the media span. If the tail still
                                    # contains placeholders, forwarding it with
                                    # no pixels embeds them as ordinary text
                                    # tokens: the model answers fluently about
                                    # an image it never saw, and nothing
                                    # raises. Refuse the hit instead -- a full
                                    # prefill is slow, a confidently wrong
                                    # answer is worse.
                                    try:
                                        _tail_has_media = (
                                            self._tokens_contain_media_placeholders(
                                                list(_full_remaining)
                                            )
                                        )
                                    except Exception:
                                        _tail_has_media = False
                                    if _tail_has_media:
                                        _qwen_tail = (
                                            self._prepare_qwen_hybrid_media_tail_for_cache_hit(
                                                req,
                                                list(token_list),
                                                int(block_table.num_tokens),
                                            )
                                        )
                                        if _qwen_tail is None:
                                            logger.info(
                                                "VLM cache hit DECLINED for %s: the "
                                                "%d-token tail still contains media "
                                                "placeholders, so the %d-token hit "
                                                "does not cover the image. Full "
                                                "prefill (an image-blind answer "
                                                "would be worse).",
                                                req.request_id,
                                                len(_full_remaining),
                                                int(block_table.num_tokens),
                                            )
                                            self._adjust_paged_hit_credit(req.request_id, 0)
                                            self.block_aware_cache.release_cache(
                                                req.request_id
                                            )
                                            req._cached_tokens = 0
                                            req.prompt_cache = None
                                            continue
                                        req.input_ids = mx.array([_full_remaining])
                                        req.attention_mask = None
                                        _cache_execution.update(
                                            {
                                                "media_tail_reencoded": True,
                                                "media_tail_prefix_kind": _qwen_tail["kind"],
                                            }
                                        )
                                        _cache_execution.update(
                                            {
                                                key: value
                                                for key, value in _qwen_tail.items()
                                                if key != "kind"
                                            }
                                        )
                                        req._cache_execution = dict(_cache_execution)
                                        logger.info(
                                            "VLM HYBRID media-tail HIT for %s: %d "
                                            "cached KV+SSM tokens, forwarding %d-token "
                                            "tail via %s",
                                            req.request_id,
                                            int(block_table.num_tokens),
                                            len(_full_remaining),
                                            _qwen_tail["kind"],
                                        )
                                    elif _full_remaining:
                                        req.input_ids = mx.array([_full_remaining])
                                        _clear_mllm_request_media_payloads(req)
                                        req.attention_mask = None
                                        logger.info(
                                            f"VLM HYBRID cache HIT for {req.request_id}: "
                                            f"{block_table.num_tokens} cached (KV+SSM), "
                                            f"{len(_full_remaining)} remaining "
                                            f"(incl. {len(_gpl_suffix)}-token gen-prompt suffix)"
                                        )
                                    else:
                                        req.input_ids = mx.array([token_list[-1:]])
                                        _clear_mllm_request_media_payloads(req)
                                        req.attention_mask = None
                                        logger.info(
                                            f"VLM HYBRID cache FULL HIT for {req.request_id}: "
                                            f"{block_table.num_tokens} cached (KV+SSM)"
                                        )
                                    # The delta forward over a deep base is
                                    # exactly the peak that aborted Metal at
                                    # ~94-96k; free the accumulated allocator
                                    # garbage before it starts. Span from the
                                    # block table, NOT req._cached_tokens
                                    # (reset_cached_tokens paths zero that).
                                    # NOTE: do NOT presize the
                                    # reconstructed cache here — measured
                                    # r6: an up-front span+headroom step on
                                    # the hit lane materialized an EXTRA
                                    # full-span allocation early in every
                                    # turn and moved the deep-span Metal
                                    # OOM wall TWO turns earlier (t17->t15).
                                    _maybe_clear_deep_span_cache(
                                        int(block_table.num_tokens or 0)
                                        + len(_full_remaining or [0])
                                    )
                                elif not is_hybrid and reconstructed is not None:
                                    if not _validate_prompt_cache(
                                        reconstructed,
                                        source=f"mllm-attn-paged-fetch:{req.request_id}",
                                    ):
                                        self.block_aware_cache.release_cache(req.request_id)
                                        continue
                                    # Pure attention VLM: TQ recompress safe (original float16)
                                    req.prompt_cache = reconstructed
                                    req._cached_tokens = block_table.num_tokens
                                    if self._uses_zaya_cache:
                                        req._cache_detail = "paged+zaya_cca"
                                    else:
                                        req._cache_detail = _paged_attention_cache_detail(
                                            disk_hit=_paged_disk_hit,
                                            mixed_attention=self._mixed_attention_cache_model,
                                            disk_only=_block_disk_only,
                                            tq_native=_tq_native_blocks > 0,
                                        )
                                    _cache_execution["cache_detail"] = req._cache_detail
                                    _cache_execution["total_worker_cache_seconds"] = round(
                                        max(0.0, time.perf_counter() - _cache_execution_started),
                                        6,
                                    )
                                    req._cache_execution = dict(_cache_execution)
                                    # Re-attach gen-prompt suffix (see hybrid branch above
                                    # for full rationale — same correctness requirement
                                    # for attention-only thinking VLMs).
                                    _full_remaining = (remaining or []) + list(_gpl_suffix)
                                    if _full_remaining:
                                        # A tail with media is reusable only when
                                        # the hit ends before the FIRST media
                                        # placeholder. In that narrow case all
                                        # processor payloads still correspond to
                                        # placeholders in this tail and the live
                                        # wrapper can merge fresh media embeddings
                                        # over the restored pure-text cache. A hit
                                        # covering only some media would require
                                        # slicing processor-owned arrays and stays
                                        # fail-closed.
                                        has_images = self._tokens_contain_media_placeholders(
                                            _full_remaining
                                        )
                                        if has_images:
                                            _hit_tokens = int(
                                                getattr(
                                                    block_table,
                                                    "num_tokens",
                                                    0,
                                                )
                                                or 0
                                            )
                                            _media_tail = None
                                            if self._supports_pure_text_prefix_with_media_tail(
                                                req, list(token_list), _hit_tokens
                                            ):
                                                _media_tail = {
                                                    "kind": "pure_text_before_first_placeholder"
                                                }
                                            else:
                                                _media_tail = (
                                                    self._prepare_muse_media_tail_for_cache_hit(
                                                        req,
                                                        list(token_list),
                                                        _hit_tokens,
                                                    )
                                                )
                                                if _media_tail is None:
                                                    _media_tail = (
                                                        self._prepare_dots3_media_tail_for_cache_hit(
                                                            req,
                                                            list(token_list),
                                                            _hit_tokens,
                                                        )
                                                    )
                                            if _media_tail is not None:
                                                req.input_ids = mx.array(
                                                    [_full_remaining]
                                                )
                                                # The processor mask describes
                                                # the original full prompt. Let
                                                # the language model rebuild its
                                                # causal/sliding masks from the
                                                # restored cache offset instead.
                                                req.attention_mask = None
                                                _cache_execution.update(
                                                    {
                                                        "media_tail_reencoded": True,
                                                        "media_tail_prefix_kind": _media_tail[
                                                            "kind"
                                                        ],
                                                    }
                                                )
                                                _cache_execution.update(
                                                    {
                                                        key: value
                                                        for key, value in _media_tail.items()
                                                        if key != "kind"
                                                    }
                                                )
                                                req._cache_execution = dict(
                                                    _cache_execution
                                                )
                                                logger.info(
                                                    "VLM media-tail prefix HIT for %s: "
                                                    "%d cached tokens, forwarding %d-token "
                                                    "tail via %s for fresh embedding",
                                                    req.request_id,
                                                    _hit_tokens,
                                                    len(_full_remaining),
                                                    _media_tail["kind"],
                                                )
                                            else:
                                                self._discard_request_cache_hit(
                                                    req,
                                                    reason="media_placeholders_in_uncached_tail",
                                                    attempted_cached_tokens=_hit_tokens,
                                                )
                                                logger.info(
                                                    f"VLM prefix cache HIT for {req.request_id}: "
                                                    f"{block_table.num_tokens} cached tokens, "
                                                    f"remaining has images — full prefill"
                                                )
                                        else:
                                            req.input_ids = mx.array([_full_remaining])
                                            _clear_mllm_request_media_payloads(req)
                                            req.attention_mask = None
                                            logger.info(
                                                f"VLM prefix cache HIT for {req.request_id}: "
                                                f"{block_table.num_tokens} cached, "
                                                f"{len(_full_remaining)} remaining "
                                                f"(incl. {len(_gpl_suffix)}-token gen-prompt suffix)"
                                            )
                                    else:
                                        # All tokens cached. Need at least the last token
                                        # for a forward pass to get logits for sampling.
                                        req.input_ids = mx.array([token_list[-1:]])
                                        _clear_mllm_request_media_payloads(req)
                                        req.attention_mask = None
                                        logger.info(
                                            f"VLM prefix cache FULL HIT for {req.request_id}: "
                                            f"{block_table.num_tokens} cached tokens"
                                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch paged cache for {req.request_id}: {e}")

            # Memory-aware or legacy prefix cache fetch (non-paged paths)
            elif (
                self._prefix_cache_enabled
                and (self.memory_aware_cache is not None or self.prefix_cache is not None)
                and req.prompt_cache is None
                and not self._request_has_media_cache_context(req)
                and not _mllm_bypass
            ):
                if req.input_ids is not None:
                    try:
                        _full_token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()
                        # Strip gen_prompt_len from fetch key, keep suffix to re-attach
                        # to `remaining` so the model re-sees the template suffix
                        # (<|im_start|>assistant\n<think>\n). See paged-path comment.
                        _gpl = getattr(req, '_gen_prompt_len', 0)
                        if _gpl > 0 and _gpl < len(_full_token_list):
                            token_list = _full_token_list[:-_gpl]
                            _gpl_suffix = _full_token_list[-_gpl:]
                        else:
                            token_list = _full_token_list
                            _gpl_suffix = []

                        # Try memory-aware cache first, then legacy
                        cache_obj = self.memory_aware_cache or self.prefix_cache
                        fetch_fn = getattr(cache_obj, 'fetch', None) or getattr(cache_obj, 'fetch_cache', None)
                        if fetch_fn is not None:
                            cache, remaining = fetch_fn(token_list)
                            if cache:
                                # Dequantize if KV cache quantization is active
                                if self._kv_cache_bits:
                                    cache = _dequantize_cache(cache)
                                    if cache is None:
                                        continue  # Dequantize failed, full prefill
                                if not _validate_prompt_cache(
                                    cache,
                                    source=f"mllm-memory-fetch:{req.request_id}",
                                ):
                                    continue

                                # Hybrid model check (same logic as paged path)
                                is_hybrid = (
                                    self._is_hybrid
                                    and not self._uses_zaya_cache
                                )
                                if is_hybrid:
                                    logger.info(
                                        f"VLM memory/legacy cache HIT for {req.request_id}: "
                                        f"{len(token_list) - len(remaining)} cached tokens "
                                        f"(hybrid model — full prefill required)"
                                    )
                                else:
                                    req.prompt_cache = cache
                                    (
                                        _full_remaining,
                                        num_cached,
                                    ) = _prefix_hit_tail_and_cached_tokens(
                                        token_list=token_list,
                                        remaining=remaining or [],
                                        gen_prompt_suffix=list(_gpl_suffix),
                                    )
                                    if _full_remaining:
                                        has_images = self._tokens_contain_media_placeholders(
                                            _full_remaining
                                        )
                                        if has_images:
                                            req.prompt_cache = None
                                            logger.info(
                                                f"VLM cache HIT for {req.request_id}: "
                                                f"remaining has images — full prefill"
                                            )
                                        else:
                                            req._cached_tokens = num_cached
                                            req._cache_detail = "memory" if self.memory_aware_cache is not None else "prefix"
                                            req.input_ids = mx.array([_full_remaining])
                                            req.pixel_values = None
                                            req.attention_mask = None
                                            req.image_grid_thw = None
                                            logger.info(
                                                f"VLM cache HIT for {req.request_id}: "
                                                f"{num_cached} cached, "
                                                f"{len(_full_remaining)} remaining "
                                                f"(incl. {len(_gpl_suffix)}-token gen-prompt suffix)"
                                            )
                                    else:
                                        req._cached_tokens = num_cached
                                        req._cache_detail = "memory" if self.memory_aware_cache is not None else "prefix"
                                        req.input_ids = mx.array([_full_token_list[-1:]])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        logger.info(
                                            f"VLM cache FULL HIT for {req.request_id}: "
                                            f"{num_cached} cached tokens"
                                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch VLM cache for {req.request_id}: {e}")

            # L2: Disk cache fallback when in-memory cache missed.
            # Prefer longest-prefix lookup so fresh-process MLLM restore has
            # the same prefix-cache semantics as the text scheduler. Exact
            # fetch remains the fallback for older DiskCacheManager instances.
            if (
                self._prefix_cache_enabled
                and req.prompt_cache is None
                and self.disk_cache is not None
                and not self._request_has_media_cache_context(req)
                and not _mllm_bypass
            ):
                if req.input_ids is not None:
                    try:
                        _full_token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()
                        # Strip gen_prompt_len from fetch key, keep suffix to re-feed.
                        _gpl = getattr(req, '_gen_prompt_len', 0)
                        if _gpl > 0 and _gpl < len(_full_token_list):
                            token_list = _full_token_list[:-_gpl]
                            _gpl_suffix = _full_token_list[-_gpl:]
                        else:
                            token_list = _full_token_list
                            _gpl_suffix = []
                        disk_matched_tokens = list(token_list)
                        if hasattr(self.disk_cache, "fetch_longest_prefix"):
                            disk_result, disk_matched_tokens = (
                                self.disk_cache.fetch_longest_prefix(token_list)
                            )
                            disk_matched_tokens = list(disk_matched_tokens or [])
                        else:
                            disk_result = self.disk_cache.fetch(token_list)
                        if disk_result is not None:
                            if not self._is_hybrid:
                                # Check for image tokens in remaining suffix
                                has_images = self._tokens_contain_media_placeholders(
                                    token_list
                                )
                                if has_images:
                                    logger.info(
                                        f"VLM disk cache (L2) HIT for {req.request_id}: "
                                        f"has images — full prefill"
                                    )
                                else:
                                    # Dequantize if KV cache quantization is active
                                    if self._kv_cache_bits:
                                        disk_result = _dequantize_cache(disk_result)
                                    if disk_result is None:
                                        pass  # Dequantize failed, full prefill
                                    elif not _validate_prompt_cache(
                                        disk_result,
                                        source=f"mllm-disk-fetch:{req.request_id}",
                                    ):
                                        pass
                                    else:
                                        req.prompt_cache = disk_result
                                        if (
                                            disk_matched_tokens
                                            and len(disk_matched_tokens)
                                            < len(token_list)
                                        ):
                                            (
                                                _tail,
                                                num_cached,
                                            ) = _disk_prefix_hit_tail_and_cached_tokens(
                                                token_list=token_list,
                                                matched_tokens=disk_matched_tokens,
                                                gen_prompt_suffix=list(_gpl_suffix),
                                            )
                                        else:
                                            if not disk_matched_tokens:
                                                disk_matched_tokens = list(token_list)
                                            (
                                                _tail,
                                                num_cached,
                                            ) = _prefix_hit_tail_and_cached_tokens(
                                                token_list=disk_matched_tokens,
                                                remaining=[],
                                                gen_prompt_suffix=list(_gpl_suffix),
                                            )
                                        req._cached_tokens = num_cached
                                        # Disk cache is exact-match on the gpl-stripped
                                        # prefix. Feed gpl suffix + last stripped token
                                        # so the model sees <|im_start|>assistant\n<think>\n
                                        # before sampling.
                                        req.input_ids = mx.array([_tail or _full_token_list[-1:]])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        # Annotate cache_detail: "disk+tq" for TQ-native files,
                                        # "disk" for standard float16 format.
                                        _tq_disk = (
                                            hasattr(self.disk_cache, '_last_fetch_tq_native')
                                            and self.disk_cache._last_fetch_tq_native
                                        )
                                        req._cache_detail = "disk+tq" if _tq_disk else "disk"
                                        logger.info(
                                            f"VLM disk cache (L2) HIT for {req.request_id}: "
                                            f"{num_cached} cached tokens"
                                            f"{' (TQ-native)' if _tq_disk else ''}"
                                        )
                    except Exception as e:
                        logger.debug(f"VLM disk cache fetch failed for {req.request_id}: {e}")
            trace.stop("cache_lookup")
            _lookup_execution = dict(
                getattr(req, "_cache_execution", None) or {}
            )
            _lookup_execution["total_worker_cache_seconds"] = round(
                max(
                    0.0,
                    time.perf_counter()
                    - float(
                        getattr(
                            req,
                            "_cache_execution_started",
                            time.perf_counter(),
                        )
                    ),
                ),
                6,
            )
            req._cache_execution = _lookup_execution

        # Get token sequences and lengths
        input_ids_list = [
            req.input_ids.tolist() if req.input_ids is not None else [0]
            for req in requests
        ]
        # Processor inputs are commonly shaped [1, seq]. ``len(ids)`` reports
        # 1 for that form and made MLLM prompt throughput/statistics false.
        lengths = [
            _mllm_input_ids_token_count(req.input_ids)
            for req in requests
        ]

        self._stats.prompt_tokens += sum(lengths)

        per_request_caches = []
        first_tokens = []
        all_logprobs = []
        succeeded_requests = []

        for i, req in enumerate(requests):
          try:
            with mx.stream(MLLMBatchGenerator._stream):
                trace = prefill_traces.get(req.request_id)
                if trace is not None:
                    trace.stop("cache_lookup")
                    trace.start("cache_prepare")
                # Reset stale per-batch module state on the language model before
                # each request's forward pass.
                #
                # Some mlx_vlm language models (Qwen3.5 / Qwen3.5-Moe hybrid SSM
                # family) cache `_rope_deltas` and `_position_ids` at module level
                # as an optimization for multi-step generation within a single
                # request. The upstream code only clears them when `pixel_values`
                # is not None (new vision request). On text-only follow-ups,
                # request N reuses request N-1's cached position_ids which has
                # the WRONG seq_length, producing broadcast errors like
                # `(1, 16, 56, 64) vs (1, 1, 20, 64)` at the first linear_attention
                # layer — because the slice `_position_ids[:, :, 0:L_new]` ends
                # up shorter than `L_new` and then broadcasts against the full
                # queries tensor. Fresh per-request clears fix this universally
                # and are a no-op for models that don't use these attributes.
                try:
                    _lm = self.language_model
                    for _attr in ("_rope_deltas", "_position_ids"):
                        if hasattr(_lm, _attr):
                            setattr(_lm, _attr, None)
                except Exception:
                    pass

                if req.prompt_cache is not None:
                    # Dequantize before _fix_hybrid_cache (it checks KVCache,
                    # not QuantizedKVCache which inherits from _BaseCache)
                    if self._kv_cache_bits:
                        cache_for_fix = _dequantize_cache(req.prompt_cache)
                        if cache_for_fix is None:
                            self._discard_request_cache_hit(
                                req,
                                reason="cache_dequantization_failed",
                                attempted_cached_tokens=int(
                                    getattr(req, "_cached_tokens", 0) or 0
                                ),
                            )
                    else:
                        cache_for_fix = req.prompt_cache
                if req.prompt_cache is not None:
                    if not _validate_prompt_cache(
                        cache_for_fix,
                        source=f"mllm-prefill-cache:{req.request_id}",
                    ):
                        self._discard_request_cache_hit(
                            req,
                            reason="cache_validation_failed",
                            attempted_cached_tokens=int(
                                getattr(req, "_cached_tokens", 0) or 0
                            ),
                        )
                        cache_for_fix = None
                if req.prompt_cache is not None:
                    # Family-owned re-typing of restored caches (duck-typed):
                    # positional reconstruction returns plain KVCache objects,
                    # and _fix_hybrid_cache's type-mismatch repair would
                    # otherwise replace them with fresh template slots for
                    # families whose native cache class is structurally
                    # non-attention (dots3 latent caches; the Gemma 4
                    # wiped-prefix failure shape).
                    _adopt_model = (
                        getattr(self, "_cache_model", None) or self.language_model
                    )
                    _adopt = getattr(_adopt_model, "adopt_prompt_cache", None)
                    if callable(_adopt) and isinstance(cache_for_fix, list):
                        try:
                            cache_for_fix = _adopt(cache_for_fix) or cache_for_fix
                        except Exception as _ae:
                            logger.warning(
                                "adopt_prompt_cache failed for %s: %s",
                                req.request_id,
                                _ae,
                            )
                    req_cache = _fix_hybrid_cache(
                        cache_for_fix,
                        getattr(self, "_cache_model", None) or self.language_model,
                        kv_positions=self._hybrid_kv_positions,
                        num_model_layers=self._hybrid_num_layers,
                    )
                    # Paged/memory/disk cache reconstruction returns plain
                    # KVCache objects. For JANG/JANGTQ VLMs whose loader
                    # patched make_cache() to TurboQuantKVCache, re-wrap the
                    # fetched KV layers before the prefill tail so the live
                    # decode path keeps the same TQ memory profile as cold
                    # prefill. If the model is not TQ-backed this is a no-op.
                    req_cache = _recompress_to_tq(
                        req_cache,
                        getattr(self, "_cache_model", None) or self.language_model,
                    )
                    if not _cache_has_materialized_state(req_cache):
                        logger.warning(
                            "Rejecting empty reconstructed VLM prefix for %s "
                            "(claimed_cached_tokens=%d); retrying full prefill",
                            req.request_id,
                            int(getattr(req, "_cached_tokens", 0) or 0),
                        )
                        self._discard_request_cache_hit(
                            req,
                            reason="empty_reconstructed_cache",
                            attempted_cached_tokens=int(
                                getattr(req, "_cached_tokens", 0) or 0
                            ),
                        )
                        cache_model = getattr(self, "_cache_model", None)
                        if cache_model is not None:
                            req_cache = cache_model.make_cache()
                        else:
                            from mlx_lm.models.cache import KVCache

                            req_cache = [
                                KVCache() for _ in self.language_model.layers
                            ]
                else:
                    try:
                        cache_model = getattr(self, "_cache_model", None)
                        if cache_model is not None:
                            req_cache = cache_model.make_cache()
                        else:
                            from mlx_lm.models.cache import KVCache
                            req_cache = [KVCache() for _ in self.language_model.layers]
                    except Exception as e:
                        logger.warning(f"model.make_cache() failed, falling back to KVCache: {e}")
                        from mlx_lm.models.cache import KVCache
                        req_cache = [KVCache() for _ in self.language_model.layers]

                if trace is not None:
                    trace.stop("cache_prepare")
                    trace.set(
                        cached_tokens=getattr(req, "_cached_tokens", 0),
                        cache_detail=getattr(req, "_cache_detail", "none"),
                    )
                    if self._decode_trace:
                        trace.set(
                            cache_before_forward=_cache_layer_debug_summary(req_cache)
                        )
                try:
                    if trace is not None:
                        trace.start("forward")
                    _cache_in_offset = (
                        getattr(req_cache[0], "offset", None) if req_cache else None
                    )
                    logger.info(
                        "MLLM prefill cache-in for %s: layer0=%s offset=%s "
                        "layers=%d cached_tokens=%s",
                        req.request_id,
                        type(req_cache[0]).__name__ if req_cache else None,
                        _cache_in_offset,
                        len(req_cache or []),
                        getattr(req, "_cached_tokens", None),
                    )
                    # A hit that reconstructs to NOTHING must never be quiet.
                    # Measured on Gemma-4-26B: the paged lane reported a
                    # 2,638-token hit across 42 blocks, the rotating restore
                    # produced an EMPTY cache (offset 0), and the turn fell
                    # back to a full prefill with no log line anywhere saying
                    # so. From the outside it is indistinguishable from "the
                    # cache never had it" -- which is what made the multimodal
                    # investigation take as long as it did.
                    try:
                        _bt = self.block_aware_cache.paged_cache.get_block_table(
                            req.request_id
                        )
                        _hit_tokens = int(getattr(_bt, "num_tokens", 0) or 0)
                    except Exception:
                        _hit_tokens = 0
                    if (
                        _hit_tokens > 0
                        and not int(getattr(req, "_cached_tokens", 0) or 0)
                        and not _cache_in_offset
                    ):
                        logger.warning(
                            "Cache hit EVAPORATED for %s: the paged lane "
                            "matched %d tokens but the restored %s came back "
                            "at offset %s, so this turn re-prefills in full. "
                            "This is a restore failure, not a cache miss.",
                            req.request_id,
                            _hit_tokens,
                            type(req_cache[0]).__name__ if req_cache else None,
                            _cache_in_offset,
                        )
                    self._prepare_native_mtp_prompt_priming(req)
                    logits = self._run_vision_encoding(req, cache=req_cache)
                    from .utils.turboquant_config import turboquant_cache_telemetry

                    self._stats.last_turboquant_cache = turboquant_cache_telemetry(
                        req_cache
                    )
                    if self._stats.last_turboquant_cache.get("object_layers", 0):
                        logger.info(
                            "TurboQuant live telemetry: %s",
                            self._stats.last_turboquant_cache,
                        )
                    if trace is not None:
                        trace.stop("forward")
                        if self._decode_trace:
                            trace.set(
                                cache_after_forward=_cache_layer_debug_summary(req_cache)
                            )

                    # A media prefix cannot be reconstructed later from token
                    # ids alone. Build its clean N-1 cache now, while the
                    # request still owns pixel/video tensors and grids. The
                    # older deferred call ran after those fields were cleared,
                    # silently turning the supposedly media-conditioned cache
                    # into a text-only prefill. Do this only on a cold request;
                    # a restored media prefix already owns the clean boundary.
                    _media_tokens = list(
                        getattr(req, "_original_token_ids", None) or []
                    )
                    # Mixed-SWA families already hold the exact N-1 boundary
                    # from the end-of-prefill rotating snapshot — the aux
                    # clean media prefill would be a redundant SECOND full
                    # forward pass (the cost class profiled at 40.8% of
                    # engine time on the text lane).
                    _native_media_boundary_captured = bool(
                        getattr(req, "_dots3_media_boundary", None) is not None
                        or (
                            getattr(self, "_mixed_attention_cache_model", False)
                            and getattr(req, "_mixed_swa_boundary", None)
                            is not None
                        )
                    )
                    if (
                        _native_media_boundary_captured
                        and int(getattr(req, "_cached_tokens", 0) or 0) == 0
                    ):
                        logger.info(
                            "MLLM media prefix cache: skipping aux clean media "
                            "prefill for %s — architecture-native boundary "
                            "snapshot already captured at end of prefill",
                            req.request_id,
                        )
                    if (
                        int(getattr(req, "_cached_tokens", 0) or 0) == 0
                        and len(_media_tokens) > 1
                        and not _native_media_boundary_captured
                        and self._media_prefix_cache_allowed(req, _media_tokens)
                    ):
                        # BLOCK-ALIGN the clean media boundary. Capturing at
                        # N-1 is what made every media turn re-prefill from
                        # scratch: the paged chain can only ever be MATCHED on
                        # block boundaries, so a companion stored at 7607 is
                        # invisible to the next turn's 7488-token block hit,
                        # and the whole found hit gets discarded with "KV
                        # blocks found but no SSM companion state". Measured
                        # live on Qwen3.8 VL: 7,488 tokens found and thrown
                        # away on turn 3, 7,488 again on turn 4, TTFT climbing
                        # 35s -> 60s -> 85s until the prompt hit a hard guard
                        # and the conversation died. Giving up <= block_size-1
                        # tokens of stored prefix buys back all of it.
                        _clean_media_len = self._media_clean_cache_boundary_for(
                            req, _media_tokens
                        )
                        clean_media_cache = (
                            self._prefill_for_clean_media_prefix_cache(
                                req, _media_tokens[:_clean_media_len]
                            )
                        )
                        if clean_media_cache is None:
                            # Exact in-media embedding capture is capability-
                            # gated. Preserve the safe terminal snapshot when
                            # a wrapper cannot expose all conditioned state.
                            _terminal_media_len = self._ssm_block_aligned_boundary(
                                len(_media_tokens) - 1
                            )
                            if _terminal_media_len <= 0:
                                _terminal_media_len = len(_media_tokens) - 1
                            if _terminal_media_len != _clean_media_len:
                                logger.info(
                                    "MLLM media prefix cache: exact learned "
                                    "boundary %d unavailable for %s; retaining "
                                    "safe terminal boundary %d",
                                    _clean_media_len,
                                    req.request_id,
                                    _terminal_media_len,
                                )
                                clean_media_cache = (
                                    self._prefill_for_clean_media_prefix_cache(
                                        req, _media_tokens[:_terminal_media_len]
                                    )
                                )
                                if clean_media_cache is not None:
                                    _clean_media_len = _terminal_media_len
                        if clean_media_cache is not None:
                            req._media_clean_prefix_cache = clean_media_cache  # type: ignore[attr-defined]
                            req._media_clean_prefix_len = _clean_media_len  # type: ignore[attr-defined]
                            logger.info(
                                "MLLM media prefix cache: captured clean media "
                                "boundary for %s (%d tokens, block-aligned from "
                                "N-1=%d) before tensor release",
                                req.request_id,
                                _clean_media_len,
                                len(_media_tokens) - 1,
                            )
                except ValueError as ve:
                    if trace is not None:
                        trace.stop("forward")
                    if "broadcast" in str(ve).lower():
                        # Cache shape mismatch (e.g., GQA head count differs between
                        # stored cache and model's current KV projection — root cause:
                        # BatchKVCache.merge() inflates H to max across all caches,
                        # so if any cache had expanded n_heads, extracted caches
                        # inherit inflated H and get stored in blocks with wrong shape).
                        # Discard cached prefix and retry with full prefill.
                        logger.warning(
                            f"Cache shape mismatch for {req.request_id}, "
                            f"retrying without prefix cache: {ve}"
                        )
                        # Release stale blocks so they can be evicted/overwritten —
                        # without this, next turn would hit the same stale block
                        # and retry every single turn
                        self._discard_request_cache_hit(
                            req,
                            reason="cache_shape_broadcast_mismatch",
                            attempted_cached_tokens=int(
                                getattr(req, "_cached_tokens", 0) or 0
                            ),
                        )
                        req.attention_mask = None
                        # vmlx#109 hardening: drop stale inline SSM stash
                        # captured against the aborted prefill — see
                        # broadcast-retry path below for full rationale.
                        if hasattr(req, "_inline_ssm_layers"):
                            req._inline_ssm_layers = None
                        if hasattr(req, "_inline_ssm_tokens"):
                            req._inline_ssm_tokens = None
                        if hasattr(req, "_inline_ssm_boundary"):
                            req._inline_ssm_boundary = 0
                        if hasattr(req, "_inline_ssm_checkpoints"):
                            req._inline_ssm_checkpoints = None
                        # Same hardening for the mixed-SWA boundary snapshot:
                        # it may have been taken over the corrupt restored
                        # cache the retry is discarding.
                        if hasattr(req, "_mixed_swa_boundary"):
                            req._mixed_swa_boundary = None
                        try:
                            cache_model = getattr(self, "_cache_model", None)
                            if cache_model is not None:
                                req_cache = cache_model.make_cache()
                            else:
                                from mlx_lm.models.cache import KVCache
                                req_cache = [KVCache() for _ in self.language_model.layers]
                        except Exception:
                            # make_cache() failed — use KVCache as last resort.
                            # For hybrid models this will likely fail too
                            # (SSM layers need ArraysCache), but at least we tried.
                            from mlx_lm.models.cache import KVCache
                            req_cache = [KVCache() for _ in self.language_model.layers]
                        if trace is not None:
                            trace.start("forward")
                        self._prepare_native_mtp_prompt_priming(req)
                        logits = self._run_vision_encoding(req, cache=req_cache)
                        if trace is not None:
                            trace.stop("forward")
                    else:
                        raise
                execution = dict(getattr(req, "_cache_execution", None) or {})
                _prompt_tokens = len(
                    getattr(req, "_original_token_ids", None) or []
                )
                _final_cached_tokens = int(
                    getattr(req, "_cached_tokens", 0) or 0
                )
                _attempted_cached_tokens = max(
                    int(execution.get("attempted_cached_tokens", 0) or 0),
                    _final_cached_tokens,
                )
                if bool(getattr(req, "_bypass_prefix_cache", False)):
                    _cache_outcome = "bypass"
                elif _final_cached_tokens > 0:
                    _cache_outcome = "hit"
                elif _attempted_cached_tokens > 0:
                    _cache_outcome = "discarded"
                else:
                    _cache_outcome = "miss"
                execution.update(
                    {
                        "request_id": normalize_ssm_telemetry_request_id(
                            req.request_id
                        ),
                        "cache_detail": getattr(req, "_cache_detail", None),
                        "prompt_tokens": _prompt_tokens,
                        "cache_key_tokens": _prompt_tokens,
                        "generation_prompt_suffix_tokens": len(
                            getattr(req, "_gen_prefix_tokens", None) or []
                        ),
                        "attempted_cached_tokens": _attempted_cached_tokens,
                        "cached_tokens": _final_cached_tokens,
                        "uncached_prompt_tokens": max(
                            _prompt_tokens - _final_cached_tokens, 0
                        ),
                        # Actual token tail submitted to the model. It includes
                        # any template-owned generation suffix and, on exact
                        # hits, the single kickoff token.
                        "prefill_tokens": _mllm_input_ids_token_count(
                            req.input_ids
                        ),
                        "cache_outcome": _cache_outcome,
                        "cache_reuse_applied": bool(
                            req.prompt_cache is not None
                            and _final_cached_tokens > 0
                        ),
                    }
                )
                if (
                    _final_cached_tokens == 0
                    and _attempted_cached_tokens > 0
                    and "fallback_reason" not in execution
                ):
                    execution["fallback_reason"] = "cache_candidate_discarded"
                req._cache_execution = {
                    key: value
                    for key, value in execution.items()
                    if value is not None
                }
                self._stats.last_cache_execution = dict(req._cache_execution)
                per_request_caches.append(req_cache)

                # Free pixel_values and vision tensors after encoding —
                # they're never needed again and can be very large for
                # high-res multi-image requests (fixes OOM on 122B + images)
                req.pixel_values = None
                req.attention_mask = None
                req.image_grid_thw = None
                # Keep small request-policy metadata for decode-time processors.
                # MiMo required-tool guided decoding needs `_vmlx_tool_choice`
                # and `_vmlx_template_tools` after prefill; clearing
                # `extra_kwargs` here made the sampler blind to tool policy.
                keep_extra_keys = {
                    "_vmlx_tool_choice",
                    "_vmlx_template_tools",
                    "_vmlx_tools_present",
                    "tool_choice",
                    "tools",
                }
                req.extra_kwargs = {
                    key: value
                    for key, value in (req.extra_kwargs or {}).items()
                    if key in keep_extra_keys
                }

                # All post-prefill materialization (logits eval, sampler,
                # cache state submission, .item()) MUST run inside the
                # dedicated generation stream context. JANGTQ Metal kernels
                # in jang_tools (P3/P15/P17/P18) dispatch async work onto
                # an internal stream — running these ops outside the stream
                # context raises:
                #   RuntimeError: There is no Stream(gpu, 1) in current thread.
                # Wrapping in `_MaybeStream()` makes every op share the
                # same Stream handle so cross-thread resolution works.
                with _MaybeStream():
                    if trace is not None:
                        trace.start("sample")
                    last_logits = logits[:, -1, :]
                    if trace is not None:
                        trace.start("logits_eval")
                    mx.eval(last_logits)
                    if trace is not None:
                        trace.stop("logits_eval")
                        trace.start("clear_cache")
                    del logits
                    mx.clear_cache()
                    if trace is not None:
                        trace.stop("clear_cache")
                    req_sampler = self._make_request_sampler(req)
                    if trace is not None:
                        trace.start("sample_call")
                    sampled, logprobs = _sample_mllm_prefill_logits(
                        last_logits,
                        req_sampler,
                    )
                    if trace is not None:
                        trace.stop("sample_call")

                    # Async submit cache states to GPU for CPU/GPU overlap
                    try:
                        if trace is not None:
                            trace.start("cache_submit")
                        cache_states = []
                        for c in req_cache:
                            if hasattr(c, 'state'):
                                st = c.state
                                if isinstance(st, (list, tuple)):
                                    cache_states.extend(x for x in st if x is not None)
                                elif st is not None:
                                    cache_states.append(st)
                            elif hasattr(c, 'cache'):
                                cache_states.extend(x for x in c.cache if x is not None)
                        _native_mtp_async_eval(sampled, logprobs, *cache_states)
                        if trace is not None:
                            trace.stop("cache_submit")
                    except Exception as e:
                        if trace is not None:
                            trace.stop("cache_submit")
                        logger.warning(f"Cache state submission error (non-fatal): {e}")
                        if trace is not None:
                            trace.start("cache_submit")
                        _native_mtp_async_eval(sampled, logprobs)
                        if trace is not None:
                            trace.stop("cache_submit")

                    if trace is not None:
                        trace.start("token_item")
                    _sampled_value = sampled.item()
                    _trace_mimo_v2_generated_token(
                        self,
                        req,
                        _sampled_value,
                        phase="prefill_sample",
                        logprobs=logprobs,
                    )
                    if trace is not None:
                        trace.stop("token_item")
                    if trace is not None:
                        trace.stop("sample")
                first_tokens.append(_sampled_value)
                all_logprobs.append(logprobs.squeeze(0) if logprobs is not None else None)
                succeeded_requests.append(req)

                if trace is not None:
                    trace.start("ssm_capture")
                # Capture SSM state at prompt boundary for hybrid models.
                # Must fire on BOTH cache-miss AND cache-hit turns: on a hit,
                # the SSM state advances during prefill of remaining tokens,
                # so the next turn needs a fresh companion keyed on the longer
                # token list.  (Fixes alternating miss/hit pattern — #45)
                if (
                    self._is_hybrid
                    and self._ssm_companion_enabled
                    and not getattr(req, '_bypass_prefix_cache', False)
                ):
                    # Guard: skip SSM capture+rederive on tokens containing
                    # image/video context. Rederive's text-only forward pass
                    # would produce wrong state at vision positions, corrupting
                    # text-only follow-up resume. Explicit media-prefix cache
                    # experiments carry a media side-key through both KV and
                    # SSM companion hashes, so they may store media-conditioned
                    # state without aliasing token-only entries.
                    _tp = getattr(req, '_original_token_ids', None) or input_ids_list[i]
                    _media_context_for_ssm = self._request_has_media_cache_context(
                        req, _tp
                    )
                    _media_cache_allowed_for_ssm = (
                        _media_context_for_ssm
                        and self._media_prefix_cache_allowed(req, _tp)
                    )
                    if _media_context_for_ssm and not _media_cache_allowed_for_ssm:
                        continue
                    _ssm_extra_keys = _ssm_companion_cache_extra_keys(req)
                    if _media_cache_allowed_for_ssm:
                        # The clean N-1 media cache was captured immediately
                        # after the real media forward, before pixel/grid tensor
                        # release. Use that same boundary for both attention KV
                        # and native SSM/GDN companion state. Never fall through
                        # to text-only async rederive for a media prompt.
                        clean_media_cache = getattr(
                            req, "_media_clean_prefix_cache", None
                        )
                        if clean_media_cache is None:
                            if int(getattr(req, "_cached_tokens", 0) or 0) > 0:
                                # A WARM media turn has no clean-boundary
                                # capture (that only runs on a cold request),
                                # but it MUST still leave a companion behind.
                                # Calling this store "redundant" and skipping
                                # it was the last link in the multimodal
                                # reuse chain: the KV chain advances to the
                                # new turn's boundary while the companion
                                # stays at the OLD one, so the NEXT turn finds
                                # its blocks and no companion and re-prefills
                                # everything. Measured live on Qwen3.8 VL: the
                                # warm turn restored 2432 tokens and stored KV
                                # out to 2477, and the turn after it asked for
                                # a companion at 2477, missed, and fell back
                                # to a full prefill.
                                #
                                # The inline capture already ran during this
                                # turn's prefill (vmlx#109 logs it at the new
                                # boundary), so the state exists. Fall through
                                # to the inline-checkpoint store below instead
                                # of discarding it.
                                logger.info(
                                    "MLLM media prefix cache: warm media turn "
                                    "for %s -- storing the companion at the "
                                    "NEW boundary so the next turn can pair "
                                    "with it",
                                    req.request_id,
                                )
                                # Fall through to the inline-checkpoint store
                                # below. `continue` here is what stranded the
                                # companion at the previous boundary.
                                clean_media_cache = None
                            else:
                                logger.info(
                                    "MLLM media prefix cache: no clean media boundary "
                                    "for %s; full prefill will remain required",
                                    req.request_id,
                                )
                                continue
                        if clean_media_cache is not None:
                            clean_ssm_layers: List[Any] = []
                            kv_set = set(self._hybrid_kv_positions or [])
                            for layer_idx, cache_obj in enumerate(clean_media_cache):
                                if layer_idx in kv_set:
                                    continue
                                if _companion_exempt_cache(cache_obj):
                                    continue
                                if hasattr(cache_obj, "cache") and isinstance(
                                    cache_obj.cache, list
                                ):
                                    from copy import deepcopy

                                    cloned = deepcopy(cache_obj)
                                    cloned.cache = [
                                        mx.contiguous(arr) if arr is not None else None
                                        for arr in cache_obj.cache
                                    ]
                                    clean_ssm_layers.append(cloned)
                                else:
                                    clean_ssm_layers.append(cache_obj)
                            all_tokens = list(
                                getattr(req, "_original_token_ids", None)
                                or input_ids_list[i]
                            )
                            # Must be the SAME length the clean cache actually
                            # covers, or the companion claims state it does not
                            # have and the KV/SSM pair is off by up to a block.
                            prompt_len = int(
                                getattr(req, "_media_clean_prefix_len", 0) or 0
                            )
                            if prompt_len <= 0:
                                prompt_len = (
                                    len(all_tokens) - 1
                                    if len(all_tokens) > 1
                                    else len(all_tokens)
                                )
                            if clean_ssm_layers and prompt_len > 0:
                                self._ssm_state_cache.store(
                                    all_tokens[:prompt_len],
                                    prompt_len,
                                    clean_ssm_layers,
                                    is_complete=True,
                                    cache_extra_keys=_ssm_extra_keys,
                                )
                                logger.info(
                                    "MLLM media prefix cache: stored clean media SSM "
                                    "companion for %s (%d layers, %d-token key)",
                                    req.request_id,
                                    len(clean_ssm_layers),
                                    prompt_len,
                                )
                            continue
                    # vmlx#109: if capture-during-prefill already snapshotted
                    # a clean SSM state at the gpl boundary, store it now
                    # with is_complete=True and skip the deferred re-derive
                    # path entirely. The post-prefill cache is post-gpl and
                    # would otherwise either be wrong (gpl>0) or queue a
                    # second prefill pass.
                    _inline_checkpoints = getattr(req, "_inline_ssm_checkpoints", None)
                    if not _inline_checkpoints:
                        _inline_layers = getattr(req, "_inline_ssm_layers", None)
                        _inline_boundary = getattr(req, "_inline_ssm_boundary", 0)
                        _inline_tokens = getattr(req, "_inline_ssm_tokens", None)
                        if _inline_layers and _inline_boundary > 0 and _inline_tokens:
                            _inline_checkpoints = [
                                (_inline_boundary, _inline_tokens, _inline_layers)
                            ]
                    if _inline_checkpoints:
                        try:
                            for _inline_boundary, _inline_tokens, _inline_layers in _inline_checkpoints:
                                self._ssm_state_cache.store(
                                    _inline_tokens,
                                    _inline_boundary,
                                    _inline_layers,
                                    is_complete=True,
                                    cache_extra_keys=_ssm_extra_keys,
                                )
                                logger.info(
                                    "vmlx#109: stored inline-captured SSM for %s "
                                    "(%d layers, %d-token key, no re-derive)",
                                    req.request_id,
                                    len(_inline_layers),
                                    _inline_boundary,
                                )
                        except Exception as e:
                            logger.debug(
                                "vmlx#109 inline store failed for %s: %s",
                                req.request_id, e,
                            )
                        # Hand the boundary snapshot to the prefix-cache store
                        # before dropping it. Without this the scheduler's
                        # path-dependent store finds nothing and falls back to
                        # a SECOND full prefill (profiled at 40.8% of engine
                        # time, ~28s at 15.4k, blocking the next request).
                        # ArraysCache state is fixed-size per layer (~0.15GB
                        # for 48 layers here), so holding it until the store
                        # completes is cheap; the scheduler clears it after use.
                        try:
                            req._clean_boundary_recurrent = list(_inline_checkpoints)
                            # The scheduler works with its own request wrapper,
                            # so hand off by request_id rather than by object.
                            snaps = getattr(self, "_clean_boundary_snapshots", None)
                            if snaps is None:
                                snaps = {}
                                self._clean_boundary_snapshots = snaps
                            snaps[str(req.request_id)] = list(_inline_checkpoints)
                            # Bound the map: these are per-request and consumed
                            # by the store, but a dropped request must not leak.
                            if len(snaps) > 8:
                                for _stale in list(snaps)[:-8]:
                                    snaps.pop(_stale, None)
                        except Exception:
                            req._clean_boundary_recurrent = None
                        # Drop refs so per-request memory isn't held longer
                        # than necessary.
                        req._inline_ssm_layers = None
                        req._inline_ssm_tokens = None
                        req._inline_ssm_checkpoints = None
                        continue
                    try:
                        kv_set = set(self._hybrid_kv_positions)
                        ssm_layers = []
                        for layer_idx, c in enumerate(req_cache):
                            if layer_idx not in kv_set:
                                if _companion_exempt_cache(c):
                                    continue
                                if hasattr(c, 'cache') and isinstance(c.cache, list):
                                    from copy import deepcopy
                                    cloned = deepcopy(c)
                                    # Ensure MLX arrays are fully materialized copies
                                    cloned.cache = [
                                        mx.contiguous(a) if a is not None else None
                                        for a in c.cache
                                    ]
                                    ssm_layers.append(cloned)
                                else:
                                    ssm_layers.append(c)
                        if ssm_layers:
                            all_tokens = getattr(req, '_original_token_ids', None)
                            if all_tokens is None:
                                all_tokens = input_ids_list[i]
                            # MLLM paged cache stores at N-1 (truncated for re-feed)
                            # and block_table.num_tokens returns N-1 on fetch.
                            # SSM companion key must match.
                            prompt_len = len(all_tokens) - 1 if len(all_tokens) > 1 else len(all_tokens)
                            if prompt_len > 0:
                                # When gen_prompt_len > 0 (thinking models with
                                # a `<think>\n` template suffix), the captured
                                # SSM state covers the FULL prompt including
                                # those template tokens. Subsequent turns with
                                # the same exact template are fine, but mark
                                # is_complete=False so the fetch path can
                                # decide whether to trust the entry for
                                # differently-templated prefixes.
                                _gpl_for_flag = getattr(req, '_gen_prompt_len', 0)
                                _is_complete_flag = (_gpl_for_flag == 0)
                                if _is_complete_flag:
                                    # gpl=0 path — post-prefill state matches
                                    # its key, safe to store directly.
                                    self._ssm_state_cache.store(
                                        all_tokens, prompt_len, ssm_layers,
                                        is_complete=True,
                                        cache_extra_keys=_ssm_extra_keys,
                                    )
                                else:
                                    # gpl>0 (thinking models): queue deferred
                                    # clean re-prefill. Queue the FIRST
                                    # prompt_len tokens (not the full
                                    # all_tokens list), so _prefill_for_clean_ssm
                                    # produces state-at-prompt_len that matches
                                    # the key exactly. Passing full all_tokens
                                    # here produces state-at-N while the key is
                                    # N-1 → T2 HIT re-feeds token N-1 + gpl on
                                    # top of an already-advanced SSM state,
                                    # causing infinite generation loops
                                    # (v1.3.77 regression, v1.3.78 fix).
                                    if self._ssm_state_cache is not None and self._ssm_state_cache.has_complete(
                                        all_tokens,
                                        prompt_len,
                                        cache_extra_keys=_ssm_extra_keys,
                                    ):
                                        # A complete companion already sits at
                                        # this exact key (typical after a cache
                                        # HIT, which restored from it). The
                                        # deferred clean prefill would recompute
                                        # byte-identical state — and its full
                                        # prompt-length prefill starves the next
                                        # request's TTFT.
                                        logger.info(
                                            "MLLM SSM re-derive skipped for %s: "
                                            "complete companion already stored "
                                            "at %d-token key",
                                            req.request_id,
                                            prompt_len,
                                        )
                                    else:
                                        _rq = self._ssm_rederive_queue
                                        if len(_rq) >= self._ssm_rederive_queue_max:
                                            _rq.pop(0)
                                        _rq.append(
                                            (
                                                list(all_tokens[:prompt_len]),
                                                prompt_len,
                                                req.request_id,
                                                _ssm_extra_keys,
                                            )
                                        )
                                logger.info(
                                    f"Captured SSM state for "
                                    f"{req.request_id}: {len(ssm_layers)} layers, "
                                    f"{prompt_len}-token key, "
                                    f"is_complete={_is_complete_flag} "
                                    f"(gen_prompt_len={_gpl_for_flag})"
                                )
                    except Exception as e:
                        # WARNING, not DEBUG. This except wraps the entire
                        # companion capture AND store. When it fires, the turn
                        # still answers, so nothing looks wrong -- but no
                        # companion is written, and every later turn then
                        # reports "KV blocks found but no SSM companion state
                        # - full prefill required" with no hint as to why.
                        # Observed live on Qwen3.8 VL: the capture logged
                        # success at 14,272 tokens and the store line never
                        # appeared, because the failure in between was
                        # invisible at DEBUG.
                        logger.warning(
                            "SSM companion capture/store FAILED for %s: %s "
                            "(the turn still answers, but this prefix will "
                            "have no companion and every later turn will "
                            "re-prefill in full)",
                            req.request_id,
                            e,
                            exc_info=True,
                        )
                if trace is not None:
                    trace.stop("ssm_capture")
          except Exception as prefill_err:
                # Broadcast shape errors from stale cache (prefix, paged blocks, or
                # residual batch state) — retry with completely fresh cache.
                # Don't require req.prompt_cache to be set: the stale shapes can come
                # from paged cache blocks that were fetched but didn't set prompt_cache,
                # or from batch KV cache state left over from the previous generation.
                if "broadcast" in str(prefill_err).lower():
                    # Log diagnostic info to identify stale shape source
                    _diag_parts = []
                    _diag_parts.append(f"prompt_cache={'set' if req.prompt_cache is not None else 'None'}")
                    _diag_parts.append(f"input_ids_shape={req.input_ids.shape if req.input_ids is not None else 'None'}")
                    _diag_parts.append(f"attn_mask={'set' if req.attention_mask is not None else 'None'}")
                    _diag_parts.append(f"pixel_values={'set' if req.pixel_values is not None else 'None'}")
                    if req.prompt_cache is not None:
                        for ci, cc in enumerate(req.prompt_cache[:3]):
                            if hasattr(cc, 'keys') and cc.keys is not None:
                                _diag_parts.append(f"cache[{ci}].keys={cc.keys.shape}")
                                break
                    logger.warning(
                        f"Cache shape mismatch for {req.request_id}, "
                        f"retrying without prefix cache: {prefill_err} "
                        f"[diag: {', '.join(_diag_parts)}]"
                    )
                    if self.block_aware_cache is not None:
                        try:
                            self.block_aware_cache.release_cache(req.request_id)
                        except Exception:
                            pass
                    req.prompt_cache = None
                    req.input_ids = mx.array([req._original_token_ids])
                    # The retry prefills the FULL prompt from scratch — the
                    # stale hit-lane value would double-count the context
                    # (cached + full prompt) in every consumer of
                    # _cached_tokens + seq_len: the turn-peak walk, the
                    # deep-span clear, and the whole-span admission all see a
                    # span up to ~2x the real one (adversarial review,
                    # finding 3).
                    req._cached_tokens = 0
                    # Reset vision fields — on broadcast retry we do full prefill from scratch
                    req.pixel_values = None
                    req.attention_mask = None
                    req.image_grid_thw = None
                    req.extra_kwargs = {}
                    # vmlx#109 hardening: any inline SSM stash captured
                    # against the now-aborted prefill is stale — drop it
                    # so the retry path captures fresh state instead of
                    # storing layers that reference a discarded cache.
                    if hasattr(req, "_inline_ssm_layers"):
                        req._inline_ssm_layers = None
                    if hasattr(req, "_inline_ssm_tokens"):
                        req._inline_ssm_tokens = None
                    if hasattr(req, "_inline_ssm_boundary"):
                        req._inline_ssm_boundary = 0
                    if hasattr(req, "_inline_ssm_checkpoints"):
                        req._inline_ssm_checkpoints = None
                    # Same hardening for the mixed-SWA boundary snapshot: it
                    # may reference the discarded cache state.
                    if hasattr(req, "_mixed_swa_boundary"):
                        req._mixed_swa_boundary = None
                    # Flush stale GPU state before retry
                    mx.clear_cache()
                    try:
                        from mlx_lm.models.cache import KVCache
                        try:
                            req_cache = (
                                getattr(self, "_cache_model", None)
                                or self.language_model
                            ).make_cache()
                        except Exception:
                            req_cache = [KVCache() for _ in self.language_model.layers]
                        logits = self._run_vision_encoding(req, cache=req_cache)
                        per_request_caches.append(req_cache)
                        with _MaybeStream():
                            last_logits = logits[:, -1, :]
                            mx.eval(last_logits)
                            del logits
                            mx.clear_cache()
                            req_sampler = self._make_request_sampler(req)
                            sampled, logprobs = _sample_mllm_prefill_logits(
                                last_logits,
                                req_sampler,
                            )
                            _native_mtp_async_eval(sampled, logprobs)
                            _sampled_value = sampled.item()
                            _trace_mimo_v2_generated_token(
                                self,
                                req,
                                _sampled_value,
                                phase="prefill_sample",
                                logprobs=logprobs,
                            )
                        first_tokens.append(_sampled_value)
                        all_logprobs.append(logprobs.squeeze(0) if logprobs is not None else None)
                        succeeded_requests.append(req)
                        continue  # Successfully retried
                    except Exception as retry_err:
                        # Nuclear retry: clear ALL paged cache blocks and try once more.
                        # Hybrid models (Mamba+Attention) can have stale state that
                        # persists even through make_cache() and mx.clear_cache().
                        if "broadcast" in str(retry_err).lower() and self.block_aware_cache is not None:
                            logger.warning(f"Retry failed with broadcast — clearing ALL paged cache and retrying once more: {retry_err}")
                            try:
                                self.block_aware_cache.clear(force=True)
                            except Exception:
                                pass
                            # A3→A2-001 (audit 2026-04-08): do NOT call
                            # _ssm_state_cache.clear() here. The previous nuclear
                            # clear dropped EVERY active session's SSM companion
                            # entries on a single failed prefill (multi-tenant
                            # blast radius). The retry below uses make_cache()
                            # which constructs a fresh hybrid cache instance —
                            # it does not read from the SSM companion. Other
                            # requests' stored entries are isolated by the
                            # deep-copy fetch contract (session 2026-03-28b
                            # root-cause fix), so leaving them in place is safe.
                            mx.clear_cache()
                            try:
                                # MUST use make_cache() for hybrid models — it returns
                                # the correct mix of KVCache + ArraysCache. Plain
                                # [KVCache() for _] breaks hybrid models that need
                                # ArraysCache.create_attention_mask().
                                cache_model = getattr(self, "_cache_model", None)
                                if cache_model is not None:
                                    req_cache = cache_model.make_cache()
                                else:
                                    req_cache = [KVCache() for _ in self.language_model.layers]
                                logits = self._run_vision_encoding(req, cache=req_cache)
                                per_request_caches.append(req_cache)
                                with _MaybeStream():
                                    last_logits = logits[:, -1, :]
                                    mx.eval(last_logits)
                                    del logits
                                    mx.clear_cache()
                                    req_sampler = self._make_request_sampler(req)
                                    sampled, logprobs = _sample_mllm_prefill_logits(
                                        last_logits,
                                        req_sampler,
                                    )
                                    _native_mtp_async_eval(sampled, logprobs)
                                    _sampled_value = sampled.item()
                                    _trace_mimo_v2_generated_token(
                                        self,
                                        req,
                                        _sampled_value,
                                        phase="prefill_sample",
                                        logprobs=logprobs,
                                    )
                                first_tokens.append(_sampled_value)
                                all_logprobs.append(logprobs.squeeze(0) if logprobs is not None else None)
                                succeeded_requests.append(req)
                                continue
                            except Exception as nuclear_err:
                                logger.error(f"Nuclear retry also failed for {req.request_id}: {nuclear_err}")
                        else:
                            logger.error(f"Retry also failed for {req.request_id}: {retry_err}")
                # Per-request prefill failure (bad image, OOM, etc.)
                # Clean up vision tensors — without this, pixel_values and partial
                # cache from make_cache() leak until GC collects them (hundreds of MB
                # for 122B+ models with high-res images).
                req.pixel_values = None
                req.attention_mask = None
                req.image_grid_thw = None
                req.extra_kwargs = {}
                mx.clear_cache()
                # A deep span that FAILED did not produce the peak its anchor
                # promises — the turn-peak valve zeroes this on its own
                # refusal, but a failure from any other source (media error,
                # sibling valve, broadcast) reaches here with the anchor still
                # set, and the next deep admit would record a poisoned
                # (full ctx, partial-span peak) walk point (adversarial
                # review, finding 4).
                self._last_deep_span_tokens = 0
                # Queue an immediate error response instead of killing the entire batch.
                # Issue #56 Bug 1: use finish_reason="error" + error string so the
                # scheduler → server path can raise an HTTP 500 with the real
                # traceback instead of a silent 200 with empty content. Previously
                # the client saw `{"content": null, "finish_reason": "stop",
                # "prompt_tokens": 0}` and had no way to distinguish "empty
                # completion" from "prefill crashed".
                if isinstance(prefill_err, VLMImagePrefillBudgetError):
                    _err_code = VLMImagePrefillBudgetError.code
                elif isinstance(prefill_err, UnsupportedMediaModalityError):
                    _err_code = UnsupportedMediaModalityError.code
                elif isinstance(prefill_err, PromptTooLongError):
                    _err_code = "prompt_too_long"
                elif isinstance(prefill_err, PrefillAdmissionError):
                    # A span the device cannot serve is the CALLER's problem to
                    # fix by shortening the prompt, not an engine fault. Without
                    # a code it fell through as a generic RuntimeError and the UI
                    # rendered a device-capacity limit as an internal crash.
                    _err_code = "prefill_admission_declined"
                else:
                    _err_code = None
                _err_detail = (
                    str(prefill_err)
                    if _err_code
                    else f"{type(prefill_err).__name__}: {prefill_err}"
                )
                if _err_code in {
                    VLMImagePrefillBudgetError.code,
                    UnsupportedMediaModalityError.code,
                    "prompt_too_long",
                    "prefill_admission_declined",
                }:
                    logger.warning(
                        f"Prefill rejected for {req.request_id}: {_err_detail} "
                        f"— other requests in batch will continue"
                    )
                else:
                    import traceback as _tb

                    logger.error(
                        f"Prefill failed for {req.request_id}: {_err_detail}\n"
                        f"{_tb.format_exc()}\n"
                        f"— other requests in batch will continue"
                    )
                self._prefill_errors.append(MLLMBatchResponse(
                    uid=req.uid,
                    request_id=req.request_id,
                    token=0,
                    logprobs=mx.zeros((1,)),
                    finish_reason="error",
                    error=_err_detail,
                    error_code=_err_code,
                    error_prompt_tokens=(
                        prefill_err.prompt_tokens
                        if isinstance(prefill_err, PromptTooLongError)
                        else None
                    ),
                    error_max_prompt_tokens=(
                        prefill_err.max_prompt_tokens
                        if isinstance(prefill_err, PromptTooLongError)
                        else None
                    ),
                    error_source=(
                        prefill_err.source
                        if isinstance(prefill_err, PromptTooLongError)
                        else None
                    ),
                ))

        # Use only the successfully prefilled requests for the batch
        requests = succeeded_requests

        y = mx.array(first_tokens)

        # If all requests failed prefill, return empty batch
        if not succeeded_requests:
            self._stats.prompt_time += time.perf_counter() - tic
            return None

        # Merge per-request caches into batch-aware caches for batched decode.
        # Handles KVCache→BatchKVCache, MambaCache→BatchMambaCache, etc.
        try:
            for req in requests:
                trace = prefill_traces.get(req.request_id)
                if trace is not None:
                    trace.start("cache_merge")
            if len(per_request_caches) == 1 and not force_batch_cache:
                # Single request with no active batch: keep raw KVCache/ArraysCache
                # to preserve integer offsets (Qwen3.5 needs cache.offset as int).
                # If force_batch_cache is True, we're extending into an active batch
                # and MUST produce batch-aware caches for extend() compatibility.
                batch_cache = per_request_caches[0]
            else:
                batch_cache = _merge_caches(per_request_caches)
            for req in requests:
                trace = prefill_traces.get(req.request_id)
                if trace is not None:
                    trace.stop("cache_merge")
        except Exception as e:
            for req in requests:
                trace = prefill_traces.get(req.request_id)
                if trace is not None:
                    trace.stop("cache_merge")
            logger.error(f"Cache merge failed: {e}")
            for req in requests:
                self._prefill_errors.append(
                    MLLMBatchResponse(
                        uid=req.uid,
                        request_id=req.request_id,
                        token=0,
                        logprobs=mx.zeros((1,)),
                        finish_reason="stop",
                    )
                )
            return None

        self._stats.prompt_time += time.perf_counter() - tic
        for req in requests:
            trace = prefill_traces.get(req.request_id)
            if trace is not None:
                logged = trace.log()
                if logged is not None:
                    self._stats.last_prefill_trace = logged

        batch = MLLMBatch(
            uids=[req.uid for req in requests],
            request_ids=[req.request_id for req in requests],
            y=y,
            logprobs=all_logprobs,
            max_tokens=[req.max_tokens for req in requests],
            num_tokens=[0] * len(requests),
            cache=batch_cache,
            requests=requests,
        )
        if len(requests) == 1 and not force_batch_cache:
            try:
                self._seed_native_mtp_from_prefill(
                    requests[0],
                    batch.cache,
                    batch.y,
                    batch.logprobs,
                )
            except Exception as exc:
                logger.debug(
                    "MLLM native MTP seed skipped for %s: %s",
                    requests[0].request_id,
                    exc,
                )
                if hasattr(requests[0], "_native_mtp_state"):
                    delattr(requests[0], "_native_mtp_state")
        return batch

    def _make_request_sampler(self, request: MLLMBatchRequest) -> Callable[[mx.array], mx.array]:
        """Create a sampler for a specific request's sampling parameters.

        Each request can have different temperature/top_p/top_k/min_p.
        Repetition penalty is applied via logits_processors when set.
        Samplers are cached on the request to avoid per-step reconstruction.
        """
        cached = getattr(request, '_cached_sampler', None)
        if cached is not None:
            return cached

        from .sampling import make_sampler
        base_sampler = make_sampler(
            temp=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else 0,
            min_p=request.min_p if request.min_p > 0 else 0.0,
            seed=getattr(request, "seed", None),
        )

        logits_processors = []

        if self._model_type == "mimo_v2" and request.enable_thinking is False:
            logits_processors.extend(self._mimo_v2_thinking_off_logits_processors(request))
        if self._model_type == "mimo_v2":
            logits_processors.extend(self._mimo_v2_required_tool_prefix_processors(request))

        # Apply repetition penalty if set for this request
        rep_penalty = getattr(request, "repetition_penalty", 1.0)
        if rep_penalty is not None and rep_penalty != 1.0:
            from mlx_lm.sample_utils import make_logits_processors

            logits_processors.extend(make_logits_processors(repetition_penalty=rep_penalty))

        from .utils.token_logits_processors import (
            make_openai_token_penalty_processor,
        )

        openai_processor = make_openai_token_penalty_processor(
            logit_bias=getattr(request, "logit_bias", None),
            frequency_penalty=getattr(request, "frequency_penalty", 0.0),
            presence_penalty=getattr(request, "presence_penalty", 0.0),
        )
        if openai_processor is not None:
            logits_processors.append(openai_processor)

        if logits_processors:
            # Use _original_token_ids (saved before cache fetch trims input_ids)
            # so repetition penalty covers the full prompt, not just uncached tokens.
            prompt_list = getattr(request, '_original_token_ids', None)
            if prompt_list is None:
                ids = request.input_ids
                prompt_list = ids[0].tolist() if ids is not None and ids.ndim > 1 else (
                    ids.tolist() if ids is not None else []
                )

            base_accepts_logits = _native_mtp_sampler_accepts_logits(base_sampler)

            def sampler_with_processors(logits, _req=request, _prompt_list=prompt_list):
                # Build full token sequence (prompt + generated) so penalty
                # applies to already-generated tokens, not just the prompt,
                # and MiMo can distinguish first-token EOS from natural stop.
                all_tokens = mx.array(_prompt_list + _req.output_tokens)
                processed = logits
                for proc in logits_processors:
                    processed = proc(all_tokens, processed)
                # Logits processors follow mlx-lm's contract: they transform
                # raw logits before the one log-softmax consumed by generic
                # stochastic samplers.  Compact/greedy samplers intentionally
                # stay in logit space.
                if not base_accepts_logits:
                    processed = _native_mtp_logprobs(processed)
                return base_sampler(processed)

            # This wrapper now owns the raw-logit -> processed -> normalized
            # transition, so callers must not normalize before invoking it.
            sampler_with_processors._vmlx_accepts_logits = True
            request._cached_sampler = sampler_with_processors
            return sampler_with_processors

        request._cached_sampler = base_sampler
        return base_sampler

    def _mimo_v2_thinking_off_logits_processors(
        self,
        request: MLLMBatchRequest,
    ) -> list[Callable[[mx.array, mx.array], mx.array]]:
        """Return MiMo V2 thinking-off decode processors for batched MLLM.

        Mirrors SimpleEngine's MiMo policy for the continuous-batching/cache
        route: suppress native thinking delimiters whenever API thinking is
        off, and suppress the primary EOS marker only before the first
        generated token. The first-token test must use request output state,
        not the processor's token vector, because batched processors see prompt
        tokens too. This avoids the proven first-token ``<|im_end|>`` stop
        without preventing natural stop later.
        """

        token_ids = getattr(self, "_mimo_v2_thinking_off_token_ids", None)
        if token_ids is None:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)

            def _encode_single(token: str) -> Optional[int]:
                try:
                    encoded = tokenizer.encode(token, add_special_tokens=False)
                except TypeError:
                    encoded = tokenizer.encode(token)
                except Exception:
                    encoded = []
                if len(encoded) != 1:
                    return None
                try:
                    return int(encoded[0])
                except Exception:
                    return None

            think_ids = {
                token_id
                for token_id in (_encode_single("<think>"), _encode_single("</think>"))
                if token_id is not None
            }
            eos_ids = {token_id for token_id in (_encode_single("<|im_end|>"),) if token_id is not None}
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos_id, int):
                eos_ids.add(eos_id)
            for token_id in getattr(self, "stop_tokens", set()) or set():
                try:
                    eos_ids.add(int(token_id))
                except Exception:
                    continue
            token_ids = {
                "think_ids": think_ids,
                "eos_ids": eos_ids,
            }
            self._mimo_v2_thinking_off_token_ids = token_ids
        think_ids = token_ids["think_ids"]
        eos_ids = token_ids["eos_ids"]

        processors: list[Callable[[mx.array, mx.array], mx.array]] = []
        if think_ids:

            def _suppress_thinking_tags(_, logits):
                indices = mx.array(sorted(think_ids))
                return logits.at[:, indices].add(-float("inf"))

            processors.append(_suppress_thinking_tags)

        if eos_ids:

            def _suppress_first_token_eos(tokens, logits, _request=request):
                first_generation_token = len(getattr(_request, "output_tokens", [])) == 0
                if not first_generation_token:
                    return logits
                indices = mx.array(sorted(eos_ids))
                return logits.at[:, indices].add(-float("inf"))

            processors.append(_suppress_first_token_eos)

        return processors

    def _mimo_v2_required_tool_prefix_processors(
        self,
        request: MLLMBatchRequest,
    ) -> list[Callable[[mx.array, mx.array], mx.array]]:
        """Constrain MiMo required-tool decode to the native XML prefix.

        This is guided decoding for the structural XML scaffold only. It does
        not infer tool arguments from user text and does not synthesize a tool
        call after generation. Once the model has emitted:

            <tool_call>
            <function=name>
            <parameter=field>

        the processor releases logits so the model must still generate the
        parameter value and closing XML itself.
        """

        extra = getattr(request, "extra_kwargs", {}) or {}
        if (
            extra.get("_vmlx_tools_present")
            or extra.get("_vmlx_template_tools")
            or extra.get("_vmlx_tool_choice")
            or extra.get("tool_choice")
        ):
            logger.debug(
                "MiMo required XML tool prefix inputs for %s: keys=%s choice=%r tools=%d",
                request.request_id,
                sorted(str(key) for key in extra.keys()),
                extra.get("tool_choice", extra.get("_vmlx_tool_choice")),
                len(extra.get("_vmlx_template_tools") or extra.get("tools") or []),
            )
        tool_choice = extra.get("tool_choice", extra.get("_vmlx_tool_choice"))
        if isinstance(tool_choice, dict):
            required = tool_choice.get("type") == "required"
        else:
            required = str(tool_choice or "").lower() == "required"
        if not required:
            return []

        tools = extra.get("_vmlx_template_tools") or extra.get("tools") or []
        if not tools:
            return []

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        def _function_payload(tool: Any) -> dict[str, Any]:
            if not isinstance(tool, dict):
                return {}
            nested = tool.get("function")
            if isinstance(nested, dict):
                return nested
            if tool.get("type") == "function":
                return tool
            return {}

        def _encode_prefix(text: str) -> list[int]:
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
            except Exception:
                return []
            out: list[int] = []
            for token_id in encoded:
                try:
                    out.append(int(token_id))
                except Exception:
                    return []
            return out

        target_prefixes: list[list[int]] = []
        for tool in tools:
            fn = _function_payload(tool)
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required_fields = params.get("required", []) if isinstance(params, dict) else []
            ordered_fields = [
                field for field in required_fields
                if isinstance(field, str) and field in props
            ]
            if not ordered_fields:
                ordered_fields = [field for field in props if isinstance(field, str)]
            if not ordered_fields:
                ordered_fields = ["value"]
            for field in ordered_fields[:1]:
                prefix = _encode_prefix(
                    f"<tool_call>\n<function={name}>\n<parameter={field}>"
                )
                if prefix:
                    target_prefixes.append(prefix)

        if not target_prefixes:
            logger.debug(
                "MiMo required XML tool prefix constraint inactive for %s: "
                "no encodable target prefixes from %d tool(s)",
                request.request_id,
                len(tools),
            )
            return []

        logger.debug(
            "MiMo required XML tool prefix constraint active for %s: "
            "%d target prefix(es), first length=%d",
            request.request_id,
            len(target_prefixes),
            len(target_prefixes[0]),
        )

        def _force_xml_tool_prefix(_, logits, _request=request, _targets=target_prefixes):
            output = [int(token_id) for token_id in getattr(_request, "output_tokens", [])]
            current_input = getattr(_request, "_sampler_current_input_token", None)
            if current_input is not None:
                try:
                    current_input = int(current_input)
                    if not output or output[-1] != current_input:
                        output.append(current_input)
                except Exception:
                    pass
            allowed: set[int] = set()
            for target in _targets:
                if len(output) >= len(target):
                    continue
                if output == target[: len(output)]:
                    allowed.add(int(target[len(output)]))
            if not allowed:
                return logits
            vocab = int(logits.shape[-1])
            valid = sorted(token_id for token_id in allowed if 0 <= token_id < vocab)
            if not valid:
                return logits
            columns = mx.arange(vocab)
            mask = columns == int(valid[0])
            for token_id in valid[1:]:
                mask = mask | (columns == int(token_id))
            masked_out = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
            return mx.where(mask.reshape(1, -1), logits, masked_out)

        return [_force_xml_tool_prefix]

    def _native_mtp_disabled_reason_for_request(self, request: MLLMBatchRequest) -> Optional[str]:
        """Return a per-request native-MTP gate reason, or None when enabled."""
        from .native_mtp import native_mtp_disabled_by_env

        if native_mtp_disabled_by_env():
            return "disabled by VMLX_NATIVE_MTP=0/--disable-native-mtp"
        if not _native_mtp_model_has_head(self.language_model):
            return "loaded language model has no native MTP head"
        if (
            float(getattr(request, "temperature", 0.0) or 0.0) != 0.0
            and not _NATIVE_MTP_STOCHASTIC_ACCEPT
        ):
            return f"temperature={getattr(request, 'temperature', None)!r} is not deterministic"
        if float(getattr(request, "repetition_penalty", 1.0) or 1.0) != 1.0:
            return f"repetition_penalty={getattr(request, 'repetition_penalty', None)!r} is not 1.0"
        if float(getattr(request, "frequency_penalty", 0.0) or 0.0) != 0.0:
            return f"frequency_penalty={getattr(request, 'frequency_penalty', None)!r} is not 0.0"
        if float(getattr(request, "presence_penalty", 0.0) or 0.0) != 0.0:
            return f"presence_penalty={getattr(request, 'presence_penalty', None)!r} is not 0.0"
        if request is None:
            return "request missing"
        return None

    def _native_mtp_enabled_for_request(self, request: MLLMBatchRequest) -> bool:
        """Return true when vMLX can run native MTP on this MLLM request."""
        reason = self._native_mtp_disabled_reason_for_request(request)
        if reason is None:
            return True
        if request is not None:
            last_reason = getattr(request, "_native_mtp_gate_logged", None)
            if last_reason != reason:
                request._native_mtp_gate_logged = reason
                logger.info(
                    "MLLM native MTP skipped for request=%s: %s",
                    getattr(request, "request_id", "unknown"),
                    reason,
                )
            # Publish the skip like the text lane does: PerformancePanel reads
            # batch_generator.last_native_mtp_skip, and without this key an
            # MLLM session that only ever ran sampled requests shows a null
            # MTP tile with no way to tell "skipped by policy" from "broken".
            stats = getattr(self, "_stats", None)
            if stats is not None:
                stats.last_native_mtp_skip = {
                    "uid": str(getattr(request, "request_id", "unknown")),
                    "request_id": getattr(request, "request_id", None),
                    "reason": reason,
                }
        return False

    def _step_native_mtp_head(
        self,
        request: MLLMBatchRequest,
        hidden_state: mx.array,
        next_token: mx.array,
        mtp_cache: List[Any],
        *,
        return_hidden: bool = False,
        return_margin: bool = False,
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Run one MTP-head prediction and return sampled token/logprobs/hidden.

        With ``return_margin`` the tuple gains the head's top-1-minus-top-2
        logit gap for this position, so the caller can decide whether
        extending the draft chain is worth another forward.
        """
        sampler = self._make_request_sampler(request)
        try:
            mtp_output = self.language_model.mtp_forward(
                hidden_state,
                _native_mtp_ensure_uint32(next_token).reshape(1, -1),
                mtp_cache,
                return_hidden=return_hidden,
            )
        except TypeError:
            mtp_output = self.language_model.mtp_forward(
                hidden_state,
                _native_mtp_ensure_uint32(next_token).reshape(1, -1),
                mtp_cache,
            )
        if isinstance(mtp_output, tuple):
            mtp_logits, mtp_hidden = mtp_output
        elif hasattr(mtp_output, "logits") and hasattr(mtp_output, "hidden_states"):
            mtp_logits = mtp_output.logits
            mtp_hidden = _native_mtp_hidden_tensor(mtp_output.hidden_states)
        else:
            mtp_logits, mtp_hidden = mtp_output, None
        final_logits = mtp_logits[:, -1, :]
        draft_tok, draft_lp = _native_mtp_sample_one(final_logits, sampler)
        if return_margin:
            return draft_tok, draft_lp, mtp_hidden, _native_mtp_top2_margin(final_logits)
        return draft_tok, draft_lp, mtp_hidden

    def _draft_native_mtp_tokens(
        self,
        request: MLLMBatchRequest,
        hidden_state: mx.array,
        start_token: mx.array,
        mtp_cache: List[Any],
        depth: int,
        stats: Optional[MLLMNativeMTPStats] = None,
    ) -> Tuple[List[mx.array], List[mx.array], List[int]]:
        trace_t0 = _native_mtp_trace_start() if stats is not None else 0.0
        draft_head_before = (
            _native_mtp_draft_head_status(self.language_model)
            if stats is not None
            else {}
        )
        drafts: List[mx.array] = []
        draft_lps: List[mx.array] = []
        current_hidden = hidden_state
        current_token = start_token
        total_depth = max(1, int(depth))
        # Confidence gate. Read the head's top-2 gap ONCE, after the first
        # draft, and stop there when it is small. Deciding per level would cost
        # a device sync per level; deciding once costs a single sync to avoid
        # up to two further head forwards plus a wider verify, which is the
        # trade that actually pays on a low-acceptance position.
        margin_threshold = (
            _native_mtp_draft_margin_threshold(getattr(self, "_model_type", None))
            if total_depth > 1
            else 0.0
        )
        for level in range(total_depth):
            if stats is not None:
                stats.mtp_forwards += 1
            want_margin = margin_threshold > 0.0 and level == 0
            step = self._step_native_mtp_head(
                request,
                current_hidden,
                current_token,
                mtp_cache,
                return_hidden=level + 1 < depth,
                return_margin=want_margin,
            )
            if want_margin:
                draft_tok, draft_lp, mtp_hidden, margin = step
            else:
                draft_tok, draft_lp, mtp_hidden = step
                margin = None
            drafts.append(draft_tok)
            draft_lps.append(draft_lp)
            current_token = draft_tok
            if mtp_hidden is None:
                current_hidden = current_hidden
            else:
                current_hidden = mtp_hidden[:, -1:, :]
            if margin is not None:
                try:
                    gap = float(margin.item())
                except Exception:
                    gap = None
                if gap is not None and gap < margin_threshold:
                    # Never drop below one draft: the caller's verify path
                    # expects a chain, and a single draft still wins whenever
                    # the head is right.
                    if stats is not None:
                        stats.margin_truncated_cycles += 1
                    break
        _native_mtp_async_eval(*drafts)
        if stats is not None:
            _native_mtp_trace_stop(stats, "draft_ms", trace_t0)
            _record_native_mtp_draft_head_delta(
                stats,
                draft_head_before,
                _native_mtp_draft_head_status(self.language_model),
            )
        return drafts, draft_lps, []

    def _observe_native_mtp_profile(
        self, state: "MLLMNativeMTPState", finish_reason: str
    ) -> None:
        """Report a finished request's controller outcome to the session profile."""
        key = getattr(state, "profile_key", None)
        if key is None:
            return
        store = getattr(self, "_native_mtp_profiles", None)
        if store is None:
            return
        snapshot = state.stats.adaptive_depth_value
        values = counts = None
        if isinstance(snapshot, dict):
            values = snapshot.get("values_tok_s")
            counts = snapshot.get("sample_counts")
        ar_ms = float(getattr(state, "ar_step_ms", 0.0) or 0.0)
        ar_baseline_tps = (1000.0 / ar_ms) if ar_ms > 0 else None
        try:
            store.observe(
                key,
                final_depth=int(state.depth or 1),
                fallback_to_ar=bool(
                    state.ar_fallback_pending
                    or finish_reason == "fallback_to_ar"
                ),
                fallback_reason=state.ar_fallback_reason,
                finish_reason=finish_reason,
                values_tok_s=values,
                sample_counts=counts,
                ar_baseline_tps=ar_baseline_tps,
            )
        except Exception:
            logger.debug("native MTP profile observe failed", exc_info=True)

    def _prepare_native_mtp_prompt_priming(
        self, request: MLLMBatchRequest
    ) -> bool:
        """Arm proven Qwen prompt-history capture immediately before prefill."""
        from .native_mtp_prompt_priming import drop_context, prepare_prompt

        # Qwen4/Flash-Next and the patched dense Qwen3.5 family expose the
        # same exact head-input contract: pre-norm trunk hidden at token t plus
        # token t+1, with one KVCache per native MTP layer.  Other families
        # stay unprimed until their own hidden variant/cache topology is
        # proven independently.
        model_type = str(getattr(self, "_model_type", "") or "").lower()
        if model_type not in {
            "qwen4_exp",
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_vl",
        }:
            drop_context(self.language_model)
            return False
        if model_type.startswith("qwen3_5") and not _native_mtp_env_flag(
            False,
            "VMLINUX_QWEN35_MTP_PROMPT_PRIMING",
            "VMLX_QWEN35_MTP_PROMPT_PRIMING",
        ):
            # Exact-output A/B at 84d2c7c39 showed the dense 27B head accepts
            # more drafts with prompt history, but did not establish a wall
            # speed win: three fresh-process adaptive pairs were neutral to
            # ~3% slower and fixed-depth runs were thermally noisy.  Keep the
            # proven implementation available for controlled profiling, not
            # silently enabled as a claimed optimization.
            drop_context(self.language_model)
            return False
        if (
            request is None
            or int(getattr(request, "max_tokens", 0) or 0) <= 1
            or self._native_mtp_disabled_reason_for_request(request) is not None
        ):
            drop_context(self.language_model)
            return False
        tokens = list(getattr(request, "_original_token_ids", None) or [])
        if not tokens:
            drop_context(self.language_model)
            return False
        return prepare_prompt(
            self.language_model,
            request_id=request.request_id,
            prompt_tokens=tokens,
            cached_tokens=int(getattr(request, "_cached_tokens", 0) or 0),
            prefix_cache=self.block_aware_cache,
            # This is the canonical scoped cache-key structure already used by
            # fetch/store.  The sidecar hashes each block with the same helper.
            extra_keys=getattr(request, "_cache_extra_keys", None),
        )

    def _seed_native_mtp_from_prefill(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        first_tokens: mx.array,
        first_logprobs: List[mx.array],
    ) -> bool:
        """Seed native MTP after prompt prefill produced the first token."""
        if not self._native_mtp_enabled_for_request(request):
            return False
        if request.max_tokens <= 1:
            return False
        if first_tokens is None or int(first_tokens.shape[0]) != 1:
            return False

        first_tok = _native_mtp_ensure_uint32(first_tokens)
        first_id = int(first_tok.tolist()[0])
        if first_id in self.stop_tokens:
            return False

        # Resolve the AR-vs-MTP decision BEFORE the MTP seed forward below:
        # a skipped activation must not pay the extra hidden-state forward.
        from .native_mtp import native_mtp_effective_depth

        depth, depth_source = native_mtp_effective_depth()
        depth = max(1, min(3, depth))
        profile_seed = "configured"
        request_profile_key = None
        # An explicit VMLX/VMLINUX_NATIVE_MTP_DEPTH env override is a user
        # decision the session profile must not second-guess; the profile
        # governs only advisory sources (default/tuning/bundle stamps).
        depth_is_explicit_override = depth_source.startswith("VML")
        if not depth_is_explicit_override and _native_mtp_env_flag(
            True,
            "VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH",
            "VMLX_NATIVE_MTP_ADAPTIVE_DEPTH",
        ):
            # Adaptive policy normally keeps unseen workloads on AR.  The
            # qwen4_exp runtime is the narrow measured exception: live 2L
            # AR/D1/D2/D3 receipts on the installed app showed D3 at 82.6
            # tok/s vs 35.4 AR on predictable output, while sampled novel code
            # still improved to 41.4 vs 39.5 tok/s.  Seed unseen qwen4_exp
            # shapes at the configured/capability-clamped depth and let the
            # existing request-local value controller demote it when measured
            # acceptance or wall cost changes.  No other family inherits this.
            request_profile_key = native_mtp_profile_key(
                temperature=float(getattr(request, "temperature", 0.0) or 0.0),
                restored_prefix=bool(
                    int(getattr(request, "_cached_tokens", 0) or 0) > 0
                ),
                prompt_tokens=(
                    int(request.input_ids.shape[-1])
                    if getattr(request, "input_ids", None) is not None
                    else 0
                ),
                has_tools=_native_mtp_request_has_tools(request),
            )
            # Lazy fallback: some fixtures build the generator via __new__
            # and never run __init__ (same class of construction the
            # prefix-cache get_stats hardening covers).
            store = getattr(self, "_native_mtp_profiles", None)
            if store is None:
                store = self._native_mtp_profiles = NativeMTPProfileStore()
            sampled_profile = request_profile_key[0] == "sampled"
            depth, profile_seed = store.start_depth(
                request_profile_key,
                configured_depth=depth,
                capability_ceiling=_native_mtp_depth_ceiling_for_request(request),
                # Legacy tuning sidecars describe one benchmark-wide winner,
                # not a sampler/context/tool profile. A greedy counting/code
                # result must not auto-enable MTP for sampled prose. A profile
                # learned in this process still takes precedence in the store.
                tuning_validated=(
                    "vmlx_mtp_tuning" in depth_source and not sampled_profile
                ),
                unseen_start_depth=(
                    depth
                    if str(getattr(self, "_model_type", "") or "").lower()
                    == "qwen4_exp"
                    and not sampled_profile
                    else None
                ),
                unseen_start_source="qwen4_exp_measured_cold_start",
            )
            if depth <= 0:
                logger.info(
                    "MLLM native MTP stays AR for request=%s seed=%s key=%s",
                    request.request_id,
                    profile_seed,
                    request_profile_key,
                )
                return False

        sampler = self._make_request_sampler(request)
        logger.info(
            "MLLM native MTP sampler contract request=%s request_temp=%s "
            "request_top_p=%s request_top_k=%s accepts_logits=%s greedy=%s "
            "distribution=%s",
            request.request_id,
            getattr(request, "temperature", None),
            getattr(request, "top_p", None),
            getattr(request, "top_k", None),
            _native_mtp_sampler_accepts_logits(sampler),
            _native_mtp_sampler_is_greedy(sampler),
            bool(getattr(sampler, "_vmlx_acceptance_logprobs", None)),
        )
        seed_main_forwards = 1
        _seed_t0 = time.perf_counter()
        output = self.language_model(
            first_tok[:, None],
            cache=cache,
            return_hidden=True,
        )
        if isinstance(output, tuple):
            logits, hidden = output
        elif hasattr(output, "logits") and hasattr(output, "hidden_states"):
            logits = output.logits
            hidden = _native_mtp_hidden_tensor(output.hidden_states)
        else:
            logger.debug("Native MTP seed skipped: model did not return hidden states")
            return False

        next_tok, next_lp = _native_mtp_sample_one(logits[:, -1, :], sampler)
        # Materialize the seed forward NOW so the wall below is a real AR step,
        # not graph-build time.  First measurement shipped without this eval
        # and read 2.3ms for a ~30ms forward — a baseline that would demote
        # perfectly healthy MTP.
        mx.eval(next_tok)
        _seed_ar_ms = (time.perf_counter() - _seed_t0) * 1000.0
        from .native_mtp_prompt_priming import take_primed

        primed = take_primed(self.language_model, cache, first_tok)
        if primed is None:
            mtp_cache = self.language_model.make_mtp_cache()
            primed_pairs = 0
            prime_source = "unprimed"
        else:
            mtp_cache, primed_pairs = primed
            prime_source = (
                "restored_prefix_and_tail"
                if int(getattr(request, "_cached_tokens", 0) or 0) > 0
                else "cold_prompt"
            )
        draft_head_before = _native_mtp_draft_head_status(self.language_model)
        drafts, draft_lps, draft_ids = self._draft_native_mtp_tokens(
            request,
            hidden[:, -1:, :],
            next_tok,
            mtp_cache,
            depth,
        )
        mx.eval(first_tok, next_tok)

        state = MLLMNativeMTPState(
            mtp_cache=mtp_cache,
            next_main=next_tok,
            drafts=drafts,
            draft_lps=draft_lps,
            draft_ids=draft_ids,
            depth=depth,
            depth_ceiling=_native_mtp_depth_ceiling_for_request(request),
            head_chain_pairs=max(0, len(drafts) - 1),
            ar_step_ms=_seed_ar_ms,
            cycle_span_start=time.perf_counter(),
            restored_prefix=bool(
                int(getattr(request, "_cached_tokens", 0) or 0) > 0
            ),
            profile_key=request_profile_key,
        )
        state.stats.profile_seed = profile_seed
        state.stats.prompt_primed_pairs = int(primed_pairs)
        state.stats.prompt_prime_source = prime_source
        if request_profile_key is not None:
            state.stats.profile_key_label = "|".join(
                str(part) for part in request_profile_key
            )
        state.stats.seed_main_forwards += seed_main_forwards
        state.stats.mtp_forwards += len(drafts)
        _record_native_mtp_draft_head_delta(
            state.stats,
            draft_head_before,
            _native_mtp_draft_head_status(self.language_model),
        )
        if first_logprobs:
            first_lp = first_logprobs[0]
        elif _native_mtp_sampler_accepts_logits(sampler):
            first_lp = None
        else:
            first_lp = _native_mtp_logprobs(logits[:, -1, :]).squeeze(0)
        state.queue.append((first_id, first_lp, "init"))
        state.queue.append((int(next_tok.tolist()[0]), next_lp, "init"))
        request._native_mtp_state = state
        logger.info(
            "MLLM native MTP path activated for request=%s depth=%d seed=%s "
            "prompt_prime=%s folded_pairs=%d",
            request.request_id,
            depth,
            profile_seed,
            prime_source,
            int(primed_pairs),
        )
        return True

    def _replay_native_mtp_confirmed_tokens(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        confirmed_tokens: List[mx.array],
    ) -> mx.array:
        replay_tokens = [_native_mtp_ensure_uint32(tok) for tok in confirmed_tokens]
        replay_input = mx.concatenate(replay_tokens).reshape(1, len(replay_tokens))
        output = self.language_model(
            replay_input,
            cache=cache,
            return_hidden=True,
        )
        if isinstance(output, tuple):
            _logits, hidden = output
        elif hasattr(output, "hidden_states"):
            hidden = _native_mtp_hidden_tensor(output.hidden_states)
        else:
            raise RuntimeError("native MTP replay did not return hidden states")
        return hidden[:, -1:, :]

    def _submit_native_mtp_verify(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        state: MLLMNativeMTPState,
    ) -> Dict[str, Any]:
        """Build and launch the verify forward for the current drafts.

        The graph is submitted with ``mx.async_eval`` so the GPU can run it
        while the host emits the previous cycle's queued tokens.  The returned
        record carries the pre-verify snapshot needed to roll the appended
        positions back on rejection or abandonment.
        """
        verify_inputs = [state.next_main] + list(state.drafts)
        trace_t0 = _native_mtp_trace_start()
        replay_snapshot = _native_mtp_snapshot_replay_cache(
            cache, len(verify_inputs)
        )
        _native_mtp_trace_stop(state.stats, "snapshot_ms", trace_t0)
        inputs = mx.concatenate(
            [_native_mtp_ensure_uint32(tok) for tok in verify_inputs]
        )
        state.stats.verify_main_forwards += 1
        # n_confirmed=1 marks state.next_main as already-confirmed, so the
        # hybrid layers stash the post-confirmed (conv, ssm) pair in
        # cache.rollback_state AND a rollback_to closure that recomputes the
        # state after any accepted-draft count through the same chunk kernel.
        # Every rejection can then roll back instead of replaying, at every
        # depth — the depth-1-only gate made 32-61% of deeper cycles pay a
        # full main-model replay forward.
        verify_kwargs = {}
        if _NATIVE_MTP_SKIP_REPLAY and len(state.drafts) >= 1:
            verify_kwargs["n_confirmed"] = 1
        from .metal.native_mtp_verify_qmm import native_mtp_verify_qmm_scope

        with native_mtp_verify_qmm_scope() as verify_qmm_scope_stats:
            output = self.language_model(
                inputs[None, :],
                cache=cache,
                return_hidden=True,
                **verify_kwargs,
            )
        state.stats.verify_qmm_calls += int(
            verify_qmm_scope_stats.get("calls", 0)
        )
        if isinstance(output, tuple):
            logits, hidden = output
        elif hasattr(output, "logits") and hasattr(output, "hidden_states"):
            logits = output.logits
            hidden = _native_mtp_hidden_tensor(output.hidden_states)
        else:
            raise RuntimeError("native MTP verify did not return hidden states")
        _native_mtp_async_eval(logits, hidden)
        return {
            "snapshot": replay_snapshot,
            "logits": logits,
            "hidden": hidden,
            "n_inputs": len(verify_inputs),
        }

    def _abandon_pending_native_mtp_verify(
        self,
        state: Optional["MLLMNativeMTPState"],
        cache: List[Any],
    ) -> None:
        """Undo an in-flight prefetched verify's cache appends.

        Must run before the cache is reused (AR fallback step) or persisted
        (prefix-cache store on finish), or the unverified draft positions leak
        into it.
        """
        if state is None:
            return
        pending = getattr(state, "pending_verify", None)
        if not isinstance(pending, dict):
            return
        state.pending_verify = None
        if cache is None:
            return
        try:
            _native_mtp_restore_replay_cache(
                cache,
                pending["snapshot"],
                pending["n_inputs"],
            )
        except Exception:
            logger.warning(
                "native MTP pending-verify rollback failed; cache may hold "
                "unverified draft positions"
            )

    def _rewind_native_mtp_terminal_boundary(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        state: Optional["MLLMNativeMTPState"],
    ) -> None:
        """Rewind verified-but-unemitted drafts before terminal cache capture.

        A depth-2/3 verify may commit several accepted draft positions before
        the host emits them. If EOS or max_tokens lands in the middle of that
        queue, persisting the cache as-is stores tokens the client never saw.
        Restore the pre-verify native state and replay only the visible prefix.
        This owns the actual boundary hazard for tool and non-tool requests.
        """
        if state is None:
            return
        self._abandon_pending_native_mtp_verify(state, cache)
        residual_drafts = sum(
            1 for _token, _logprobs, source in state.queue if source == "draft"
        )
        if residual_drafts <= 0:
            return
        snapshot = state.terminal_snapshot
        base_token = state.terminal_base_token
        n_inputs = int(state.terminal_n_inputs or 0)
        if snapshot is None or base_token is None or n_inputs <= 0:
            raise RuntimeError(
                "native MTP terminal boundary lacks a restorable verify snapshot"
            )
        if not _native_mtp_restore_replay_cache(cache, snapshot, n_inputs):
            raise RuntimeError("native MTP terminal boundary cache rejected rollback")
        visible_drafts = [
            mx.array([int(token)], dtype=mx.uint32)
            for token, source in state.terminal_emitted
            if source == "draft"
        ]
        confirmed_tokens = [base_token] + visible_drafts
        state.stats.replay_main_forwards += 1
        self._replay_native_mtp_confirmed_tokens(
            request,
            cache,
            confirmed_tokens,
        )
        state.queue.clear()

    def _run_native_mtp_verify_cycle(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        state: MLLMNativeMTPState,
    ) -> None:
        if state.next_main is None or not state.drafts:
            raise RuntimeError("native MTP verify entered without pending drafts")

        sampler = self._make_request_sampler(request)
        pending = getattr(state, "pending_verify", None)
        state.pending_verify = None
        if pending is None:
            pending = self._submit_native_mtp_verify(request, cache, state)
        replay_snapshot = pending["snapshot"]
        logits = pending["logits"]
        hidden = pending["hidden"]
        state.terminal_snapshot = replay_snapshot
        state.terminal_n_inputs = int(pending["n_inputs"])
        state.terminal_base_token = state.next_main
        state.terminal_emitted = []
        trace_t0 = _native_mtp_trace_start()
        _native_mtp_trace_eval(logits, hidden)
        _native_mtp_trace_stop(state.stats, "verify_ms", trace_t0)

        depth = len(state.drafts)
        trace_t0 = _native_mtp_trace_start()
        # ONE device round-trip per cycle: sample the verify rows, compare them
        # against the drafts, and count the leading accepted run -- all on
        # device -- then read back a single small bundle.  The original path
        # paid two blocking syncs (sampled ids here, draft ids in
        # _native_mtp_materialize_draft_ids) and compared in Python with the
        # GPU idle.  VMLX_MTP_FUSED_SYNC=0 restores it for A/B.
        fused_decision = (
            _NATIVE_MTP_FUSED_SYNC
            and bool(state.drafts)
            and _native_mtp_sampler_is_greedy(sampler)
        )
        precomputed_accepted = None
        if fused_decision:
            (
                target_tokens,
                target_lps,
                target_ids,
                fused_draft_ids,
                precomputed_accepted,
            ) = _native_mtp_sample_and_decide(
                logits[:, -(depth + 1) :, :].reshape(depth + 1, -1),
                sampler,
                state.drafts,
                depth,
            )
            state.draft_ids = fused_draft_ids
            _native_mtp_trace_stop(state.stats, "sample_ms", trace_t0)
        else:
            target_tokens, target_lps, target_ids = _native_mtp_sample_rows(
                logits[:, -(depth + 1) :, :].reshape(depth + 1, -1),
                sampler,
                also_eval=(tuple(state.drafts) if _NATIVE_MTP_FUSED_SYNC else ()),
            )
            _native_mtp_trace_stop(state.stats, "sample_ms", trace_t0)
            trace_t0 = _native_mtp_trace_start()
            _native_mtp_materialize_draft_ids(state)
            _native_mtp_trace_stop(state.stats, "materialize_ms", trace_t0)
        # Measured 2026-08-15 (35B MXFP8 MTP, depth 2): with any cache tier
        # populated, per-cycle wall GROWS 19 -> ~30ms under normal async
        # execution while barrier-traced runs stay FLAT at ~21.6ms with the
        # SAME per-phase compute — lazy work accumulated between cycle evals
        # stalls later forwards. One fence per cycle bounds the outstanding
        # queue. Gated for A/B; flips default only on byte-equal + speedup
        # proof at the app-default cache shape.
        if _native_mtp_cycle_fence_enabled(
            depth,
            model_type=getattr(self, "_model_type", None),
        ):
            try:
                mx.synchronize()
            except Exception:
                pass

        if precomputed_accepted is not None:
            # Decided on device in the bundle above; no host-side comparison.
            accepted = precomputed_accepted
        else:
            decision_telemetry: Dict[str, int] = {}
            if any(lp is not None for lp in state.draft_lps[:depth]) and any(
                lp is not None for lp in target_lps[:depth]
            ):
                state.stats.stochastic_distribution_cycles += 1
            accepted = _native_mtp_accepted_count(
                state.draft_ids,
                target_ids,
                state.draft_lps,
                target_lps,
                sampler,
                telemetry=decision_telemetry,
            )
            state.stats.stochastic_ratio_checks += int(
                decision_telemetry.get("ratio_checks", 0)
            )
            state.stats.stochastic_ratio_accepts += int(
                decision_telemetry.get("ratio_accepts", 0)
            )

        state.stats.cycles += 1
        state.stats.drafted_tokens += depth
        state.stats.accepted_tokens += accepted
        for level in range(min(depth, len(state.stats.drafted_by_depth))):
            state.stats.drafted_by_depth[level] += 1
            if level < accepted:
                state.stats.accepted_by_depth[level] += 1
        if _native_mtp_debug_enabled():
            logger.info(
                "MLLM MTP[%s] cycle=%d next=%s drafts=%s targets=%s accepted=%d/%d",
                request.request_id,
                state.stats.cycles,
                int(state.next_main.tolist()[0]),
                list(state.draft_ids),
                list(target_ids),
                accepted,
                depth,
            )
        # Measure the real interval between completed speculative cycles.  It
        # includes verify prefetch overlap and queued-token draining, exactly
        # the wall effects that raw acceptance and trace-only phase timings
        # miss.  The first cycle merely arms the interval; a depth transition
        # cannot contaminate either depth because the armed/completed depths
        # must agree.
        _value_cycle_now = time.perf_counter()
        _native_mtp_finish_value_cycle(
            state,
            depth=depth,
            accepted=accepted,
            now=_value_cycle_now,
        )
        _native_mtp_maybe_adapt_depth(request.request_id, state)
        _native_mtp_arm_value_cycle(state, now=_value_cycle_now)
        if accepted == depth:
            state.stats.accepts += 1
            _native_mtp_clear_rollback(cache)
            for draft_id, draft_lp in zip(state.draft_ids, state.draft_lps):
                state.queue.append((draft_id, draft_lp, "draft"))
            bonus_tok = target_tokens[depth]
            bonus_id = int(target_ids[depth])
            state.queue.append((bonus_id, target_lps[depth], "bonus"))
            next_hidden = hidden[:, depth : depth + 1, :]
            state.next_main = bonus_tok
            if state.ar_fallback_pending:
                _native_mtp_capture_head_cache_before_discard(
                    state.stats,
                    state.mtp_cache,
                )
                state.mtp_cache = None
                state.drafts = []
                state.draft_lps = []
                state.draft_ids = []
                return
            if _NATIVE_MTP_ALIGNED_HEAD_CACHE:
                # PR #990's batched cache commit: trim the unverified chain
                # pairs, then run ONE head forward whose input pairs are every
                # token confirmed this cycle — (h0, d1) .. (h_{k-1}, dk),
                # (hk, bonus) — with backbone hiddens from the verify output.
                # The last position of that same forward IS the next level-0
                # draft, so alignment costs no extra head call.
                _native_mtp_trim_head_chain(state)
                state.mtp_cache = state.mtp_cache or self.language_model.make_mtp_cache()
                commit_hidden = hidden[:, 0 : depth + 1, :]
                commit_tokens = mx.concatenate(
                    [_native_mtp_ensure_uint32(t).reshape(1) for t in state.drafts]
                    + [_native_mtp_ensure_uint32(bonus_tok).reshape(1)]
                ).reshape(1, depth + 1)
                (
                    state.drafts,
                    state.draft_lps,
                    state.draft_ids,
                ) = self._draft_native_mtp_tokens(
                    request,
                    commit_hidden,
                    commit_tokens,
                    state.mtp_cache,
                    state.depth,
                    state.stats,
                )
                state.head_chain_pairs = max(0, len(state.drafts) - 1)
                return
            # The head cache accumulated across accepted cycles is GAPPY: each
            # cycle emits draft + bonus but only the draft passes through the
            # head, so the bonus token never enters the head's KV and its
            # context has a hole at every other position.  Upstream PR #990
            # fixes this with "batched MTP cache commits ... maintaining cache
            # alignment between backbone and MTP head"; we never did.  Measured
            # 2026-08-18 (Qwen3.8-27B-JANG_4D-CRACK, code workload, fair 48+
            # samples): retained-gappy cache d1 78.1% / d2 41.7%; fresh cache
            # per cycle d1 78.7% / d2 74.6%.  Fresh is neutral at depth 1 and
            # nearly doubles depth-2 acceptance, so it is the default.
            # VMLX_MTP_RETAIN_HEAD_CACHE=1 restores the old accumulation.
            if _NATIVE_MTP_RETAIN_HEAD_CACHE:
                state.mtp_cache = state.mtp_cache or self.language_model.make_mtp_cache()
            else:
                state.mtp_cache = self.language_model.make_mtp_cache()
            state.drafts, state.draft_lps, state.draft_ids = self._draft_native_mtp_tokens(
                request,
                next_hidden,
                bonus_tok,
                state.mtp_cache,
                state.depth,
                state.stats,
            )
            return

        state.stats.rejects += 1
        accepted_drafts = state.drafts[:accepted]
        skipped_replay = False
        if _NATIVE_MTP_SKIP_REPLAY and accepted < depth:
            # Roll the hybrid state back to the accepted point using the
            # states the verify forward captured (post-confirmed snapshot for
            # accepted==0, the rollback_to recompute for accepted>=1), trim
            # the rejected KV positions, and take the corrected token's
            # hidden straight out of the verify output. That is the same
            # state the replay would have rebuilt, without the extra
            # main-model forward.
            trace_t0 = _native_mtp_trace_start()
            skipped_replay = _native_mtp_rollback_to_confirmed(
                cache,
                depth - accepted,
                accepted_drafts=accepted,
            )
            _native_mtp_trace_stop(state.stats, "restore_ms", trace_t0)
            if skipped_replay:
                state.stats.replay_skips = (
                    int(getattr(state.stats, "replay_skips", 0) or 0) + 1
                )
                replay_hidden = hidden[:, accepted : accepted + 1, :]
        if not skipped_replay:
            trace_t0 = _native_mtp_trace_start()
            if not _native_mtp_restore_replay_cache(
                cache,
                replay_snapshot,
                depth + 1,
            ):
                raise RuntimeError("native MTP cache rejected rollback")
            _native_mtp_trace_stop(state.stats, "restore_ms", trace_t0)
        for draft_id, draft_lp in zip(state.draft_ids[:accepted], state.draft_lps[:accepted]):
            state.queue.append((draft_id, draft_lp, "draft"))
        correction = target_tokens[accepted]
        correction_id = int(target_ids[accepted])
        if accepted < len(state.draft_lps) and accepted < len(target_lps):
            if (
                _NATIVE_MTP_STOCHASTIC_ACCEPT
                and state.draft_lps[accepted] is not None
                and target_lps[accepted] is not None
            ):
                state.stats.stochastic_residual_corrections += 1
            correction, correction_id = _native_mtp_rejection_correction(
                correction,
                correction_id,
                target_lps[accepted],
                state.draft_lps[accepted],
                sampler,
            )
        state.queue.append((correction_id, target_lps[accepted], "verify"))
        if not skipped_replay:
            confirmed_tokens = [state.next_main] + accepted_drafts
            trace_t0 = _native_mtp_trace_start()
            state.stats.replay_main_forwards += 1
            replay_hidden = self._replay_native_mtp_confirmed_tokens(
                request,
                cache,
                confirmed_tokens,
            )
            _native_mtp_trace_stop(state.stats, "replay_ms", trace_t0)
        state.next_main = correction
        if state.ar_fallback_pending:
            _native_mtp_capture_head_cache_before_discard(
                state.stats,
                state.mtp_cache,
            )
            state.mtp_cache = None
            state.drafts = []
            state.draft_lps = []
            state.draft_ids = []
            return
        if _NATIVE_MTP_ALIGNED_HEAD_CACHE:
            # Trim the unverified chain pairs (one may carry a rejected token),
            # keep the valid history, and commit the confirmed prefix plus the
            # correction in the same forward that drafts the next token.
            _native_mtp_trim_head_chain(state)
            state.mtp_cache = state.mtp_cache or self.language_model.make_mtp_cache()
            state.stats.mtp_cache_retained_on_rejects += 1
            aligned_hidden = hidden[:, 0 : accepted + 1, :]
            aligned_tokens = mx.concatenate(
                [_native_mtp_ensure_uint32(t).reshape(1) for t in accepted_drafts]
                + [_native_mtp_ensure_uint32(correction).reshape(1)]
            ).reshape(1, accepted + 1)
            state.drafts, state.draft_lps, state.draft_ids = self._draft_native_mtp_tokens(
                request,
                aligned_hidden,
                aligned_tokens,
                state.mtp_cache,
                state.depth,
                state.stats,
            )
            state.head_chain_pairs = max(0, len(state.drafts) - 1)
            return
        if _NATIVE_MTP_RECREATE_HEAD_CACHE_ON_REJECT:
            state.mtp_cache = self.language_model.make_mtp_cache()
            state.stats.mtp_cache_recreated_on_rejects += 1
        else:
            # Retain the head cache across a rejection, matching the text path
            # (patches/mlx_lm_mtp/batch_generator.py): "The head cache is never
            # rolled back (loose history by design -- verify guarantees
            # correctness; the cache only shapes draft quality)."
            #
            # Recreating it means the head drafts the next token with ZERO
            # history. On predictable text rejections are rare so it never
            # shows, but on real prose ~37% of cycles reject, so the head is
            # repeatedly blinded and acceptance spirals down.
            state.stats.mtp_cache_retained_on_rejects += 1
        state.drafts, state.draft_lps, state.draft_ids = self._draft_native_mtp_tokens(
            request,
            replay_hidden,
            correction,
            state.mtp_cache,
            state.depth,
            state.stats,
        )

    def _next_native_mtp_token(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        state: MLLMNativeMTPState,
    ) -> Tuple[int, Any]:
        if not state.queue:
            self._run_native_mtp_verify_cycle(request, cache, state)
            if (
                _NATIVE_MTP_VERIFY_PREFETCH
                and state.drafts
                and not state.ar_fallback_pending
                and state.pending_verify is None
            ):
                # Launch the next verify now: the GPU runs it while the host
                # emits the tokens this cycle just queued.
                state.pending_verify = self._submit_native_mtp_verify(
                    request, cache, state
                )
        if not state.queue:
            raise RuntimeError("native MTP verify produced no emit token")
        token, logprobs, source = state.queue.popleft()
        if state.terminal_snapshot is not None:
            state.terminal_emitted.append((int(token), str(source)))
        _native_mtp_bump_emit(state, source)
        if _native_mtp_debug_enabled():
            logger.info(
                "MLLM MTP[%s] emit source=%s token=%s",
                request.request_id,
                source,
                token,
            )
        return token, logprobs

    def _step(
        self, input_tokens: mx.array, cache: List[Any]
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Run one generation step through the language model.

        Args:
            input_tokens: Input tokens [batch_size, 1] or [batch_size]
            cache: BatchKVCache for the language model

        Returns:
            Tuple of (sampled tokens, logprobs list)
        """
        # Ensure correct shape
        if input_tokens.ndim == 1:
            input_tokens = input_tokens[:, None]

        # Wrap BatchKVCache with offset-safe proxies ONLY for Qwen-style VL
        # language models: their attention slices with cache.offset (needs a
        # Python int; batch that filtered down to 1 still carries an mx.array
        # offset) and they position via explicit position_ids, so flattening
        # is safe. All other families rope FROM cache.offset and must see the
        # raw per-row array — see _offset_proxy_needed_for_model_type
        # (F16/F18, 2026-07-08).
        trace = self._decode_trace
        _wrap_t0 = time.perf_counter() if trace else 0.0
        if _offset_proxy_needed_for_model_type(self._model_type):
            cache = _wrap_batch_caches(cache)
        if trace:
            self._decode_trace_wrap_s = getattr(
                self, "_decode_trace_wrap_s", 0.0
            ) + (time.perf_counter() - _wrap_t0)

        model_t0 = time.perf_counter() if trace else 0.0

        # Run language model only (not full VLM). Qwen3.5/3.6 mRoPE language
        # models may leave `_rope_deltas` unset when text-only prefill used
        # explicit positions, so keep decode absolute too instead of relying on
        # module-level rope state.
        lm_kwargs: Dict[str, Any] = {"cache": cache}
        _posid_t0 = time.perf_counter() if trace else 0.0
        if _lm_supports_position_ids(self.language_model):
            position_ids = _absolute_text_position_ids(
                input_tokens,
                cache,
                self.language_model,
            )
            if position_ids is not None:
                lm_kwargs["position_ids"] = position_ids
        if trace:
            self._decode_trace_posid_s = getattr(
                self, "_decode_trace_posid_s", 0.0
            ) + (time.perf_counter() - _posid_t0)
        output = self.language_model(input_tokens, **lm_kwargs)
        if trace:
            mx.synchronize()
            model_s = time.perf_counter() - model_t0
            sample_t0 = time.perf_counter()

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        logits = logits[:, -1, :]

        # Per-request sampling using each request's sampling parameters.
        # VLM logprobs are rejected at the API layer, so do not materialize a
        # full-vocab logsoftmax every decode token on the default fast path.
        batch = self.active_batch
        if batch and len(batch.requests) == logits.shape[0]:
            try:
                current_input_tokens = input_tokens[:, -1].tolist()
            except Exception:
                current_input_tokens = []
            for req, token_id in zip(batch.requests, current_input_tokens):
                try:
                    req._sampler_current_input_token = int(token_id)
                except Exception:
                    pass
            if _batch_shares_sampler_params(batch.requests):
                shared_sampler = self._make_request_sampler(batch.requests[0])
                sampled, _ = _sample_mllm_prefill_logits(logits, shared_sampler)
            else:
                tokens = []
                for i, req in enumerate(batch.requests):
                    req_sampler = self._make_request_sampler(req)
                    token, _ = _sample_mllm_prefill_logits(
                        logits[i:i+1], req_sampler
                    )
                    tokens.append(token)
                sampled = mx.concatenate(tokens, axis=0)
        else:
            sampled, _ = _sample_mllm_prefill_logits(logits, self.sampler)

        if trace:
            mx.synchronize()
            sample_s = time.perf_counter() - sample_t0
            self._decode_trace_count += 1
            self._decode_trace_model_s += model_s
            self._decode_trace_sample_s += sample_s
            if self._decode_trace_count % self._decode_trace_every == 0:
                n = self._decode_trace_count
                logger.info(
                    "VMLINUX_DECODE_TRACE mllm steps=%d avg_model_ms=%.2f "
                    "avg_sample_ms=%.2f last_model_ms=%.2f last_sample_ms=%.2f "
                    "avg_wrap_ms=%.2f avg_posid_ms=%.2f batch=%d",
                    n,
                    (self._decode_trace_model_s / n) * 1000.0,
                    (self._decode_trace_sample_s / n) * 1000.0,
                    model_s * 1000.0,
                    sample_s * 1000.0,
                    (getattr(self, "_decode_trace_wrap_s", 0.0) / n) * 1000.0,
                    (getattr(self, "_decode_trace_posid_s", 0.0) / n) * 1000.0,
                    int(logits.shape[0]),
                )

        if _mimo_v2_token_trace_enabled():
            return sampled, [logits[i] for i in range(int(logits.shape[0]))]
        return sampled, [None] * int(logits.shape[0])

    def _next(self) -> List[MLLMBatchResponse]:
        """
        Internal next() with true continuous batching.

        New requests can join the active batch mid-generation:
        1. Finish pending async GPU work
        2. Prefill new requests (with batch-aware caches)
        3. Convert existing batch cache if needed
        4. Extend active batch with new requests
        5. Continue decode step for the merged batch

        Returns:
            List of MLLMBatchResponse for this step
        """
        tic = time.perf_counter()

        if self._vlm_cache_limit_tightened and self._steady_cache_limit is not None:
            # The media-request tighten protects vision encode + one-shot
            # prefill + the first decode step (all inside the previous _next
            # call). Restore the steady-state scheduler limit afterwards so a
            # ~1GB allocator ceiling doesn't outlive the media spike (A/B
            # measured this as hygiene, not a decode-speed lever); a new media
            # admission re-tightens.
            try:
                _set_cache = (
                    getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit
                )
                _set_cache(self._steady_cache_limit)
                logger.info(
                    "Restored steady-state Metal cache limit %.2fGB after "
                    "media prefill spike",
                    self._steady_cache_limit / (1024**3),
                )
            except Exception as exc:
                logger.debug("Steady-state cache limit restore skipped: %s", exc)
            self._vlm_cache_limit_tightened = False

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        if (
            batch is not None
            and any(
                getattr(req, "_native_mtp_state", None) is not None
                for req in batch.requests
            )
        ):
            # Native MTP owns private per-row draft/verify state. Keep standard
            # continuous-batch admission closed until this row finishes.
            num_to_add = 0

        # Process new prompts — fresh batch or extend into active one
        if num_to_add > 0 and self.unprocessed_requests:
            requests = self.unprocessed_requests[:num_to_add]

            if num_active == 0:
                # No active batch — create fresh
                new_batch = self._process_prompts(requests)
                self.unprocessed_requests = self.unprocessed_requests[len(requests):]
                self.active_batch = new_batch
                prompt_processing = True
            else:
                # Active batch exists — prefill new requests and extend.
                # Must finish pending async work before extending cache arrays.
                mx.synchronize()
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

                # force_batch_cache=True: even single new request produces
                # batch-aware caches (BatchKVCache/BatchMambaCache) for extend().
                new_batch = self._process_prompts(requests, force_batch_cache=True)
                self.unprocessed_requests = self.unprocessed_requests[len(requests):]

                if new_batch is not None:
                    # Convert existing batch cache from raw KVCache to BatchKVCache
                    # if it was a single-request batch (Qwen3.5 offset optimization).
                    from mlx_lm.models.cache import BatchKVCache, KVCache
                    needs_convert = any(
                        _is_kv_like(c) and not isinstance(c, BatchKVCache)
                        for c in batch.cache
                    )
                    if needs_convert:
                        batch.cache = _ensure_batch_cache(batch.cache)

                    batch.extend(new_batch)
                    # Free peak memory from prefill before continuing decode
                    mx.clear_cache()
                    prompt_processing = True

        elif num_active == 0:
            # No active batch and no pending requests
            self.active_batch = None
            return []

        # Drain any per-request prefill errors (from M6 per-request isolation)
        prefill_errors = list(self._prefill_errors)
        self._prefill_errors.clear()

        # Generate next token for active batch
        batch = self.active_batch
        if batch is None:
            return prefill_errors

        mtp_state = None
        if len(batch.requests) == 1:
            mtp_state = getattr(batch.requests[0], "_native_mtp_state", None)
        _next_trace = bool(getattr(self, "_decode_trace", False) and mtp_state is None)
        _step_s = 0.0
        _async_s = 0.0
        _materialize_t0 = time.perf_counter() if _next_trace else 0.0
        if mtp_state is not None:
            try:
                token, lp = self._next_native_mtp_token(
                    batch.requests[0],
                    batch.cache,
                    mtp_state,
                )
                y = [token]
                logprobs = [lp]
                batch.y = mx.array([token], dtype=mx.uint32)
                batch.logprobs = [lp]
                if (
                    getattr(mtp_state, "ar_fallback_pending", False)
                    and not mtp_state.queue
                ):
                    self._abandon_pending_native_mtp_verify(mtp_state, batch.cache)
                    ready, fallback_reason = _native_mtp_ar_fallback_ready(
                        batch.cache,
                        mtp_state,
                        token,
                    )
                    if not ready:
                        raise RuntimeError(
                            f"native MTP AR fallback unsafe: {fallback_reason}"
                        )
                    batch.y, batch.logprobs = self._step(batch.y[:, None], batch.cache)
                    _submit_decode_token_eval(batch.y, batch.cache)
                    _native_mtp_log_stats(
                        batch.requests[0].request_id,
                        mtp_state.stats,
                        "fallback_to_ar",
                        mtp_state.mtp_cache,
                    )
                    self._stats.record_native_mtp(
                        request_id=batch.requests[0].request_id,
                        stats=mtp_state.stats,
                        finish_reason="fallback_to_ar",
                        final_depth=mtp_state.depth,
                        fallback_reason=mtp_state.ar_fallback_reason,
                    )
                    self._observe_native_mtp_profile(
                        mtp_state, "fallback_to_ar"
                    )
                    logger.info(
                        "MLLM MTP[%s] fallback to AR after queue drain: %s",
                        batch.requests[0].request_id,
                        mtp_state.ar_fallback_reason or "adaptive policy",
                    )
                    if hasattr(batch.requests[0], "_native_mtp_state"):
                        delattr(batch.requests[0], "_native_mtp_state")
            except Exception as exc:
                logger.error(
                    "MLLM native MTP decode failed for %s: %s",
                    batch.requests[0].request_id,
                    exc,
                )
                _native_mtp_log_stats(
                    batch.requests[0].request_id,
                    mtp_state.stats,
                    "error",
                    mtp_state.mtp_cache,
                )
                self._stats.record_native_mtp(
                    request_id=batch.requests[0].request_id,
                    stats=mtp_state.stats,
                    finish_reason="error",
                    final_depth=mtp_state.depth,
                    fallback_reason=mtp_state.ar_fallback_reason,
                )
                if hasattr(batch.requests[0], "_native_mtp_state"):
                    delattr(batch.requests[0], "_native_mtp_state")
                self.active_batch = None
                return prefill_errors + [
                    MLLMBatchResponse(
                        uid=batch.uids[0],
                        request_id=batch.request_ids[0],
                        token=0,
                        logprobs=mx.zeros((1,)),
                        finish_reason="error",
                        error=f"NativeMTPError: {exc}",
                    )
                ]
        else:
            y, logprobs = batch.y, batch.logprobs
            _step_t0 = time.perf_counter() if _next_trace else 0.0
            batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
            _step_s = time.perf_counter() - _step_t0 if _next_trace else 0.0
            _async_t0 = time.perf_counter() if _next_trace else 0.0
            _submit_decode_token_eval(batch.y, batch.cache)
            _async_s = time.perf_counter() - _async_t0 if _next_trace else 0.0
            _materialize_t0 = time.perf_counter() if _next_trace else 0.0

        if hasattr(y, "tolist"):
            y = y.tolist()
        toc = time.perf_counter()
        if _next_trace:
            try:
                materialize_s = toc - _materialize_t0
                total_s = toc - tic
                if self._decode_trace_count % self._decode_trace_every == 0:
                    logger.info(
                        "VMLINUX_DECODE_TRACE_NEXT mllm steps=%d "
                        "last_total_ms=%.2f last_step_ms=%.2f "
                        "last_async_ms=%.2f last_materialize_ms=%.2f "
                        "prompt_processing=%s batch=%d",
                        self._decode_trace_count,
                        total_s * 1000.0,
                        _step_s * 1000.0,
                        _async_s * 1000.0,
                        materialize_s * 1000.0,
                        bool(prompt_processing),
                        len(batch.requests),
                    )
            except Exception:
                pass

        # Note: prompt_time is already counted in _process_prompts().
        # Only count the first decode step after prompt processing as generation time.
        if not prompt_processing:
            self._stats.generation_time += toc - tic

        # Build responses and track finished
        keep_idx = []
        end_idx = []
        responses = []

        for i, (token, uid, request_id, num_tok, max_tok, req) in enumerate(
            zip(
                y,
                batch.uids,
                batch.request_ids,
                batch.num_tokens,
                batch.max_tokens,
                batch.requests,
            )
        ):
            # Must be bound for EVERY request, not inside the finish branch.
            # The response is built on every path, so a branch-local binding
            # made an unfinished request raise UnboundLocalError -- which the
            # scheduler surfaced as "1 requests failed permanently", i.e. the
            # whole turn died and every ctx/cached reading came back 0.
            _clean_store_tokens: Optional[List[int]] = None
            num_tok += 1
            batch.num_tokens[i] = num_tok
            req.num_tokens = num_tok
            req.output_tokens.append(token)

            finish_reason = None
            cache_fn = None

            if getattr(req, "_vmlx_graceful_stop_requested", False):
                # The token computed by this step stays internal to the engine:
                # the API parser already retained the complete native tool-call
                # prefix and drains until this real terminal.  Treat the row as
                # a normal stop so cache extraction, MTP rollback, scheduler
                # cleanup and the durability barrier all run unchanged.
                finish_reason = "stop"
                end_idx.append(i)
            elif token in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(i)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(i)
            else:
                keep_idx.append(i)

            _trace_mimo_v2_generated_token(
                self,
                req,
                token,
                phase="response_step",
                finish_reason=finish_reason,
                logprobs=logprobs[i],
            )

            if finish_reason is not None:
                mtp_state_for_finish = getattr(req, "_native_mtp_state", None)
                if mtp_state_for_finish is not None:
                    self._rewind_native_mtp_terminal_boundary(
                        req, batch.cache, mtp_state_for_finish
                    )
                    _native_mtp_log_stats(
                        request_id,
                        mtp_state_for_finish.stats,
                        finish_reason,
                        mtp_state_for_finish.mtp_cache,
                    )
                    self._stats.record_native_mtp(
                        request_id=request_id,
                        stats=mtp_state_for_finish.stats,
                        finish_reason=finish_reason,
                        final_depth=mtp_state_for_finish.depth,
                        fallback_reason=mtp_state_for_finish.ar_fallback_reason,
                    )
                    self._observe_native_mtp_profile(
                        mtp_state_for_finish, finish_reason
                    )
                    try:
                        delattr(req, "_native_mtp_state")
                    except AttributeError:
                        pass
                # Extract cache NOW before batch.filter() invalidates indices.
                # Do NOT TQ-compress here — the scheduler needs original float16
                # for block extraction. TQ recompress happens on the fetch path.
                captured_cache = getattr(req, "_media_clean_prefix_cache", None)
                if captured_cache is None:
                    finish_cache = batch.extract_cache(i)
                    dots3_boundary = self._assemble_dots3_media_boundary(
                        req, finish_cache
                    )
                    if dots3_boundary is not None:
                        captured_cache, _clean_len = dots3_boundary
                        req._media_clean_prefix_len = _clean_len  # type: ignore[attr-defined]
                    else:
                        captured_cache = finish_cache
                if captured_cache is not None:
                    # The clean media cache covers a BLOCK-ALIGNED prefix, not
                    # N-1, so the store must be keyed by that prefix + 1 (the
                    # N-1 payload contract). prompt_token_ids stays full length
                    # because usage.prompt_tokens is derived from it.
                    _clean_len = int(
                        getattr(req, "_media_clean_prefix_len", 0) or 0
                    )
                    _orig = getattr(req, "_original_token_ids", None) or []
                    if 0 < _clean_len < len(_orig):
                        _clean_store_tokens = list(_orig[: _clean_len + 1])
                cache_fn = lambda c=captured_cache: c
                # Hand the mixed-SWA N-1 boundary snapshot (captured at end
                # of prefill, before decode advanced the rotating rings) to
                # the scheduler's paged store. The scheduler works with its
                # own request wrapper, so hand off by request_id — mirroring
                # _clean_boundary_snapshots.
                _swa_capture = getattr(req, "_mixed_swa_boundary", None)
                if _swa_capture is not None:
                    try:
                        _swa_snaps = getattr(
                            self, "_mixed_swa_boundary_snapshots", None
                        )
                        if _swa_snaps is None:
                            _swa_snaps = {}
                            self._mixed_swa_boundary_snapshots = _swa_snaps
                        _swa_snaps[str(request_id)] = _swa_capture
                        # Bound the map: entries are consumed by the store,
                        # but a dropped request must not leak window copies.
                        if len(_swa_snaps) > 8:
                            for _stale in list(_swa_snaps)[:-8]:
                                _swa_snaps.pop(_stale, None)
                    except Exception:
                        pass
                    req._mixed_swa_boundary = None

            responses.append(
                MLLMBatchResponse(
                    uid=uid,
                    request_id=request_id,
                    token=token,
                    logprobs=logprobs[i],
                    finish_reason=finish_reason,
                    prompt_cache=cache_fn,
                    prompt_token_ids=(
                        getattr(req, '_original_token_ids', None)
                        or (req.input_ids[0].tolist() if req.input_ids is not None and req.input_ids.ndim > 1
                            else req.input_ids.tolist() if req.input_ids is not None
                            else [])
                    ),
                    cached_tokens=getattr(req, '_cached_tokens', 0),
                    cache_detail=getattr(req, '_cache_detail', "") or "",
                    cache_execution=dict(
                        getattr(req, "_cache_execution", None) or {}
                    ),
                    cache_extra_keys=getattr(req, '_cache_extra_keys', None),
                    gen_prefix_tokens=getattr(req, '_gen_prefix_tokens', None),
                    clean_store_token_ids=_clean_store_tokens,
                )
            )

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                self.active_batch = None
                # All requests done — release Metal cache to reclaim GPU memory.
                # Without this, MLX holds freed buffers in its allocator free-list
                # indefinitely, causing apparent memory bloat after long prefills.
                if getattr(self, "_tight_memory_prefill_drain", False):
                    self._drain_tight_memory_allocator("after_batch_finish")
                else:
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass

        self._stats.generation_tokens += len(responses)
        return prefill_errors + responses

    def next(self) -> List[MLLMBatchResponse]:
        """
        Generate next token for all requests in the batch.

        Returns:
            List of MLLMBatchResponse, one per active request
        """
        with mx.stream(MLLMBatchGenerator._stream):
            return self._next()

    def _can_native_mtp_burst(self) -> bool:
        if not _native_mtp_burst_enabled():
            return False
        batch = self.active_batch
        if batch is None or len(batch.requests) != 1:
            return False
        req = batch.requests[0]
        if getattr(req, "_native_mtp_state", None) is None:
            return False
        if getattr(req, "_stop_strings", None):
            # Scheduler-level string stop matching must see one token at a time
            # so it can remove the row before later queued MTP tokens leak.
            return False
        return True

    def next_burst(self) -> List[MLLMBatchResponse]:
        """Generate one step, then drain already verified native-MTP tokens.

        ``next()`` keeps the traditional one-response-per-active-request
        contract. The async MLLM scheduler can use this method to amortize
        executor/queue overhead for native MTP without launching extra verifier
        work: after the first token, only tokens already present in the private
        verified MTP queue are drained.
        """
        with mx.stream(MLLMBatchGenerator._stream):
            responses = self._next()
            if not responses or any(resp.finish_reason is not None for resp in responses):
                return responses
            if not self._can_native_mtp_burst():
                return responses

            while self._can_native_mtp_burst():
                batch = self.active_batch
                if batch is None:
                    break
                state = getattr(batch.requests[0], "_native_mtp_state", None)
                if state is None or not state.queue:
                    break
                more = self._next()
                if not more:
                    break
                responses.extend(more)
                if any(resp.finish_reason is not None for resp in more):
                    break
            return responses

    def stats(self) -> MLLMBatchStats:
        """
        Get generation statistics.

        Returns:
            MLLMBatchStats with timing and token counts
        """
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def get_vision_cache_stats(self) -> Dict[str, Any]:
        """Get vision cache statistics."""
        return self.vision_cache.get_stats()

    def has_pending(self) -> bool:
        """Check if there are pending or active requests."""
        return bool(self.unprocessed_requests or self.active_batch)

    def _derive_hybrid_companion_delta(
        self,
        req: Any,
        token_list: List[int],
        fetch_num: int,
        ck_len: int,
        block_table: Any,
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[List[Any]]:
        """Advance companion state from a checkpoint to the KV hit boundary.

        The RESUME path needs a BLOCK-ALIGNED checkpoint because it trims the
        KV back to it, and SSM at an arbitrary length paired with KV floored to
        a block boundary re-feeds the gap through layers that already absorbed
        it. Companion stores land on gpl-stripped prompt boundaries, so they are
        almost never multiples of the block size, and the whole hit was being
        thrown away over the difference.

        Measured on Qwen3.8 at 51k context: a 51,328-token KV hit was discarded
        because the only companion sat at 51,297 -- 31 tokens away -- costing a
        138.9s full re-prefill where the neighbouring turns took 4-6s.

        Alignment is irrelevant here because this does the opposite of RESUME:
        the KV hit is kept whole and only the recurrent state is advanced up to
        it, so both describe ``fetch_num`` once the derive finishes. Returns the
        companion states, or None to fall back to the existing full prefill.
        """
        if not getattr(self, "_is_hybrid", False):
            return None
        fetch_num = int(fetch_num or 0)
        ck_len = int(ck_len or 0)
        if not (0 < ck_len < fetch_num) or block_table is None:
            return None
        if not self._hybrid_kv_positions:
            return None
        # The derive advances recurrent state with a TEXT-ONLY forward. If the
        # gap it has to cross contains media placeholders, those positions
        # would be re-fed without their pixel values and the resulting state
        # would be quietly wrong -- coherent output, wrong content, no error.
        # The store side already refuses this pairing; so must the fetch side.
        try:
            if self._tokens_contain_media_placeholders(
                list(token_list[ck_len:fetch_num])
            ):
                logger.info(
                    "vmlx#91 DELTA declined for %s: the %d-token gap from the "
                    "checkpoint crosses media placeholders, which a text-only "
                    "derive cannot reproduce.",
                    getattr(req, "request_id", "?"),
                    fetch_num - ck_len,
                )
                return None
        except Exception:
            return None
        try:
            from .utils.cache_extent import cache_offset
        except Exception:
            return None
        try:
            reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
        except Exception as exc:
            logger.debug("Companion delta: reconstruct failed: %s", exc)
            return None
        if not reconstructed:
            return None

        sliced: List[Any] = []
        for layer in reconstructed:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                return None
            # OFFSET is the token-count authority. Restored buffers are
            # zero-padded up to the cache step, so keys.shape routinely exceeds
            # the logical length; slicing off the shape is the mistake that
            # silently emptied dots3 answers.
            if int(cache_offset(layer) or 0) < ck_len:
                return None
            seq_axis = 1 if keys.ndim == 3 else 2
            if int(keys.shape[seq_axis]) < ck_len:
                return None
            try:
                clone = type(layer)()
            except Exception:
                # Needs constructor arguments, so it is a typed or windowed
                # cache and is not position-sliceable.
                return None
            if seq_axis == 1:
                clone.keys = keys[:, :ck_len, :]
                clone.values = values[:, :ck_len, :]
            else:
                clone.keys = keys[..., :ck_len, :]
                clone.values = values[..., :ck_len, :]
            clone.offset = ck_len
            sliced.append(clone)

        logger.info(
            "vmlx#91 DELTA for %s: keeping the %d-token KV hit and advancing "
            "the companion %d tokens from its checkpoint at %d, instead of "
            "discarding the hit for a full prefill.",
            getattr(req, "request_id", "?"),
            fetch_num,
            fetch_num - ck_len,
            ck_len,
        )
        try:
            derived = self._prefill_for_clean_ssm(
                list(token_list[:fetch_num]),
                sliced,
                ck_len,
                cache_extra_keys=cache_extra_keys,
            )
        except Exception as exc:
            logger.info(
                "vmlx#91 DELTA failed for %s (%s); falling back to full prefill",
                getattr(req, "request_id", "?"),
                exc,
            )
            return None
        if not derived:
            return None
        kv_set = set(self._hybrid_kv_positions or [])
        states = [c for i, c in enumerate(derived) if i not in kv_set]
        if not states:
            return None

        # Hand the attention transients back BEFORE returning.
        #
        # This derive materialises the whole prefix KV twice over -- once in
        # reconstruct_cache(), then again as the forward regrows attention KV up
        # to fetch_num -- and keeps NONE of it: only the recurrent slots survive
        # the kv_set filter above. `sliced` holds views into `reconstructed`, and
        # `derived`'s attention buffers are pure waste, so without an explicit
        # drop all of it stays resident and OVERLAPS the caller's own forward.
        #
        # Measured 2026-08-23 (Qwen3.8-27B, 17.5GB weights, SSD-only tier): Metal
        # peaks walked to 36.6GB inside a turn while the retained floor tracked
        # live KV correctly at 61 KB/token. utils/prefill_admission.py then
        # refuses the turn outright ("Message not sent" at 80,097 tokens). The
        # transient is the whole problem, so release it here rather than let a
        # guard decline work because of it.
        try:
            import mlx.core as mx

            # Force the recurrent state to be REAL before dropping its parents.
            # These are lazy graphs until evaluated; freeing the attention
            # buffers first would only re-materialise them on the next eval.
            to_eval = []
            for st in states:
                st_state = getattr(st, "state", None)
                if isinstance(st_state, (list, tuple)):
                    to_eval.extend(a for a in st_state if hasattr(a, "dtype"))
                elif hasattr(st_state, "dtype"):
                    to_eval.append(st_state)
            if to_eval:
                mx.eval(*to_eval)

            del derived, sliced, reconstructed

            from .mlx_memory import clear_mlx_memory_cache

            clear_mlx_memory_cache(mx=mx, log=logger)
        except Exception as exc:  # noqa: BLE001 - reclamation is best-effort
            logger.debug("Companion delta: could not release transients: %s", exc)

        return states

    def _complete_hybrid_base_from_companion(
        self,
        base_cache: Any,
        token_ids: Any,
        base_token_count: int,
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[List[Any]]:
        """Fill a paged-reconstructed base's recurrent slots from the companion.

        Returns None whenever the pairing cannot be made EXACTLY -- no companion,
        an incomplete one, or a state count that does not fill every
        non-attention slot. A partial fill would pair attention KV with recurrent
        state from a different token position, which is precisely the corruption
        this path exists to avoid, so it declines instead of guessing.
        """
        cache = getattr(self, "_ssm_state_cache", None)
        if cache is None or int(base_token_count) <= 0:
            return None
        if not getattr(self, "_is_hybrid", False):
            return None
        try:
            # The salt MUST travel here. Asking with the bare token key while
            # every media writer salts means the splice can never find the
            # companion for a media conversation, so it silently declines and
            # the caller falls back to a TEXT-ONLY re-derive of the whole
            # prompt -- across the image placeholder positions, with no pixels.
            # That produces wrong recurrent state which is then stored under
            # the CORRECT salted key with is_complete=True and used for the
            # live answer: coherent prose about the wrong picture, persisted,
            # self-reinforcing on every later turn.
            entry = cache.fetch(
                list(token_ids),
                int(base_token_count),
                cache_extra_keys=cache_extra_keys,
            )
        except Exception as exc:
            logger.debug("Companion fetch for clean-store base failed: %s", exc)
            return None
        if not entry:
            return None
        try:
            ssm_states, is_complete = entry
        except (TypeError, ValueError):
            return None
        # An incomplete companion was captured after the gen-prompt suffix, so it
        # does not describe this prefix boundary.
        if not ssm_states or not is_complete:
            return None
        try:
            fixed = _fix_hybrid_cache(
                base_cache,
                getattr(self, "_cache_model", None) or self.language_model,
                kv_positions=self._hybrid_kv_positions,
                num_model_layers=self._hybrid_num_layers,
            )
        except Exception as exc:
            logger.debug("Hybrid base layout repair failed: %s", exc)
            return None
        if not fixed:
            return None
        kv_positions = set(self._hybrid_kv_positions or [])
        injected = 0
        for layer_idx in range(len(fixed)):
            if layer_idx in kv_positions:
                continue
            if _companion_exempt_cache(fixed[layer_idx]):
                continue
            if injected < len(ssm_states):
                fixed[layer_idx] = ssm_states[injected]
                injected += 1
        if injected != len(ssm_states):
            return None
        if not _validate_prompt_cache(fixed, source="mllm-clean-store-base"):
            return None
        return fixed

    def _prefill_for_clean_path_dependent_cache(
        self,
        tokens: List[int],
        base_cache: Optional[List[Any]] = None,
        base_token_count: int = 0,
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[List[Any]]:
        """Run a clean prompt-only prefill matching a path-dependent cache key.

        ``base_cache``/``base_token_count`` let the caller hand in a cache that
        already covers ``tokens[:base_token_count]`` (typically reconstructed
        from the previously stored chain), so only the delta is forwarded. That
        turns the store from O(context) into O(new tokens). Ignored for caches
        that require a single contiguous pass, and ignored unless the base is
        itself chunk-safe, since resuming mid-sequence is the same operation a
        chunk boundary performs.

        Mirrors Scheduler._prefill_for_prompt_only_cache for the MLLM path.
        Returned cache covers exactly `tokens` worth of processing — no
        gen_prompt_len suffix, no generation output. Safe to store with
        is_complete=True.

        Only genuine recurrent state forces a one-shot pass. SSM re-derive
        requires contiguous state math across the full prompt: chunking the
        forward pass broke on the 2nd chunk for fresh ``make_cache()`` output
        because ArraysCache's offset/mask machinery (``lengths``/
        ``left_padding``) is only populated when the cache goes through
        ``BatchKVCache`` wrappers. For those models we one-shot when the
        attention buffer fits under the Metal single-buffer cap and skip
        gracefully otherwise (the live prefill's SSM stash still serves as a
        possibly-contaminated companion for thinking-model prompts).

        Attention-only mixed-SWA stacks (Gemma 4: RotatingKVCache for sliding
        layers + KVCache for full-attention layers) have no such constraint —
        the live prefill already builds them chunk by chunk via
        ``prefill_step_size``, so a chunked re-derive is the identical math.
        They previously shared the recurrent one-shot path and its O(seq_len^2)
        dense-attention estimate, which predicts 8.9 GB at only ~12k tokens on
        a 30-head backbone. The guard therefore rejected every Gemma 4
        conversation past ~11.5k tokens, and because a rejected clean prefill
        means "skip the store entirely", prefix caching was silently dead:
        measured 0 cached tokens on every turn of a 6-turn 74k conversation,
        with TTFT growing 3.1s -> 54.8s as each turn re-prefilled from scratch.
        """
        if not tokens or self.language_model is None:
            return None
        seq_len = len(tokens)
        try:
            resume_at = 0
            fresh_cache = None
            # A base rebuilt from paged blocks carries attention KV only, so a
            # hybrid model's linear layers would receive a KVCache and index it
            # ("'KVCache' object is not subscriptable"). The one-shot detector
            # cannot catch that: with no ArraysCache present the base looks
            # attention-only and passes. Compare the base against the layers it
            # will actually feed instead.
            _model_layers = getattr(self.language_model, "layers", None)
            if not isinstance(_model_layers, (list, tuple)):
                # Only a real layer sequence can be compared; anything else
                # (including a test double) must not veto a valid base.
                _model_layers = None
            _base_matches_layers = True
            if base_cache is not None and _model_layers is not None:
                if len(base_cache) != len(_model_layers):
                    _base_matches_layers = False
                else:
                    for _layer, _entry in zip(_model_layers, base_cache):
                        # `is True` on purpose: a stub/mock layer returns a
                        # truthy object for any attribute, which would make
                        # every layer look linear and refuse a perfectly good
                        # attention-only base.
                        if getattr(_layer, "is_linear", False) is True and type(
                            _entry
                        ).__name__ in ("KVCache", "QuantizedKVCache", "RotatingKVCache"):
                            _base_matches_layers = False
                            break
            if (
                base_cache is not None
                and 0 < int(base_token_count) < seq_len
                and _base_matches_layers
                and not _cache_requires_one_shot_rederive(
                    base_cache, ignore_chunk_override=True
                )
            ):
                fresh_cache = base_cache
                resume_at = int(base_token_count)
            elif base_cache is not None and not _base_matches_layers:
                _spliced = (
                    self._complete_hybrid_base_from_companion(
                        base_cache,
                        tokens,
                        int(base_token_count),
                        cache_extra_keys=cache_extra_keys,
                    )
                    if _HYBRID_BASE_SPLICE
                    else None
                )
                if _spliced is not None:
                    fresh_cache = _spliced
                    resume_at = int(base_token_count)
                    logger.info(
                        "MLLM clean prefill: SPLICED companion SSM state into the "
                        "reconstructed base at %d tokens; forwarding only the "
                        "%d-token delta (VMLX_HYBRID_BASE_SPLICE)",
                        int(base_token_count),
                        max(0, seq_len - int(base_token_count)),
                    )
                else:
                    logger.info(
                        "MLLM clean prefill: reconstructed base does not match the "
                        "hybrid layer layout; re-deriving the whole prompt instead"
                        "%s",
                        " (splice declined)" if _HYBRID_BASE_SPLICE else "",
                    )

            if fresh_cache is None:
                cache_model = getattr(self, "_cache_model", None)
                fresh_cache = (
                    cache_model.make_cache() if cache_model is not None else None
                )
                if fresh_cache is None:
                    from mlx_lm.models.cache import KVCache
                    fresh_cache = [
                        KVCache() for _ in range(len(self.language_model.layers))
                    ]

            if _cache_requires_one_shot_rederive(fresh_cache):
                _OOM_GUARD_BYTES = _HYBRID_ONE_SHOT_GUARD_BYTES
                _n_heads_guess = _infer_attention_heads_for_hybrid_oom_guard(
                    self.language_model
                )
                _predicted_attn_bytes = _n_heads_guess * seq_len * seq_len * 2
                if _predicted_attn_bytes > _OOM_GUARD_BYTES:
                    logger.info(
                        "MLLM SSM re-derive: skipping clean prefill for %d-token "
                        "prompt (predicted attention buffer %.1f GB exceeds Metal "
                        "single-buffer limit; re-derive requires contiguous state "
                        "math that chunking breaks). Live prefill's SSM stash "
                        "will be used as the companion.",
                        seq_len, _predicted_attn_bytes / (1024**3),
                    )
                    del fresh_cache
                    return None
                chunk_size = seq_len
            else:
                chunk_size = max(1, int(getattr(self, "prefill_step_size", 1024) or 1024))

            # Qwen3.5/3.6 hybrid language models keep mRoPE bookkeeping on the
            # module object. A clean prompt-only SSM re-derive must behave like a
            # new request, otherwise a prior cache-hit tail can leave an
            # 8-token `_position_ids` cache and the prompt-only 18-token prefill
            # fails in attention with broadcast_shapes. The normal request
            # prefill path already clears these attributes before each request;
            # mirror that contract here for the idle re-derive path.
            _saved_pos_state = {}
            for _attr in ("_rope_deltas", "_position_ids"):
                if hasattr(self.language_model, _attr):
                    _saved_pos_state[_attr] = getattr(self.language_model, _attr)
                    setattr(self.language_model, _attr, None)
            materialize: List[Any] = []
            def _collect_cache_arrays(cache_obj: Any) -> None:
                if hasattr(cache_obj, "keys") and cache_obj.keys is not None:
                    if isinstance(cache_obj.keys, tuple):
                        materialize.extend(cache_obj.keys)
                        materialize.extend(cache_obj.values)
                    else:
                        materialize.extend([cache_obj.keys, cache_obj.values])
                elif hasattr(cache_obj, "caches") and isinstance(
                    getattr(cache_obj, "caches", None), (list, tuple)
                ):
                    for sub_cache in cache_obj.caches:
                        _collect_cache_arrays(sub_cache)
                elif hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, list):
                    for arr in cache_obj.cache:
                        if hasattr(arr, "shape"):
                            materialize.append(arr)

            # One chunk for the recurrent path (chunk_size == seq_len keeps the
            # single contiguous pass those caches require); prefill_step_size
            # chunks for attention-only stacks, materializing after each so the
            # lazy graph — and peak Metal working set — stay bounded.
            # resume_at is non-zero only when a caller-supplied base already
            # covers that prefix, so those tokens are never re-forwarded.
            for _start in range(resume_at, seq_len, chunk_size):
                _ = self.language_model(
                    mx.array([tokens[_start:_start + chunk_size]]),
                    cache=fresh_cache,
                )
                materialize.clear()
                for c in fresh_cache:
                    _collect_cache_arrays(c)
                if materialize:
                    try:
                        mx.eval(materialize)
                    except RuntimeError as _eval_err:
                        if "Stream" in str(_eval_err):
                            mx.synchronize()
                        else:
                            raise
                # Return each chunk's transients to the allocator before the
                # next chunk. This pass runs post-turn in the background with
                # no admission valve; retained transients (~18GB per
                # 2048-token chunk on a 30-head hybrid) stack across chunks
                # into a Metal working-set wave that the wired limit turns
                # into an uncatchable process abort — measured killing the
                # serve process ~2 minutes after an 11k-token turn, and once
                # hard-resetting the whole machine.
                mx.clear_cache()
            self._store_companion_from_clean_pass(
                tokens, fresh_cache, cache_extra_keys=cache_extra_keys
            )
            return fresh_cache
        except Exception as ex:
            if isinstance(ex, (NameError, AttributeError, TypeError)):
                # Programming errors are not runtime turbulence — a swallowed
                # NameError here silently killed the hybrid base-splice branch
                # for every store. Non-fatal for the request, but loud.
                logger.error(
                    "MLLM clean SSM prefill hit a programming error "
                    "(non-fatal for the request, but this branch is broken): "
                    "%s",
                    ex,
                    exc_info=True,
                )
            else:
                logger.warning(
                    f"MLLM clean SSM prefill failed (non-fatal): {ex}"
                )
            return None
        finally:
            for _attr, _value in locals().get("_saved_pos_state", {}).items():
                try:
                    setattr(self.language_model, _attr, _value)
                except Exception:
                    pass

    def _store_companion_from_clean_pass(
        self,
        tokens: List[int],
        fresh_cache: Optional[List[Any]],
        cache_extra_keys: Optional[Any] = None,
    ) -> None:
        """Store the SSM companion produced by a clean prompt-only prefill.

        The clean pass computes exactly the typed hybrid state at the stored
        key (no gen_prompt_len suffix, no generation output — the same
        contract that makes its result safe to store with is_complete=True).
        That is byte-for-byte the state the idle re-derive queue would later
        recompute with ANOTHER O(prompt) background forward; storing the
        companion here makes ``run_idle_rederive``'s ``has_complete`` probe
        skip that second pass, halving the post-turn background GPU work for
        hybrid thinking models.

        Keys: ``cache_extra_keys`` MUST be whatever the reader will ask with.
        Text captures enqueue their re-derive entries with ``None``, so the
        default matches them. Media prompts DO reach here (the vmlx#91 delta
        derive routes through ``_prefill_for_clean_ssm``), and their reader
        asks with the media salt — storing those under the bare token key
        would be worse than useless: two turns whose placeholder token ids are
        identical but whose IMAGES differ hash the same, so the second turn
        would restore recurrent state that absorbed the FIRST image. Coherent
        output, wrong picture, no error anywhere. Callers on a media path pass
        their fetch-side ``_ssm_extra_keys`` here so store and fetch agree.
        """
        companion_cache = getattr(self, "_ssm_state_cache", None)
        if fresh_cache is None or companion_cache is None:
            return
        kv_positions = getattr(self, "_hybrid_kv_positions", None)
        if not kv_positions:
            return
        try:
            prompt_len = len(tokens)
            if prompt_len <= 0 or companion_cache.has_complete(
                tokens, prompt_len, cache_extra_keys=cache_extra_keys
            ):
                return
            kv_set = set(kv_positions)
            ssm_layers: List[Any] = []
            for layer_idx, c in enumerate(fresh_cache):
                if layer_idx in kv_set:
                    continue
                if _companion_exempt_cache(c):
                    # Positional full-latent slots (dots3) are excluded from
                    # every companion snapshot — the restore path rebuilds
                    # them as windowed shells. Including them here would
                    # resurrect the O(ctx)-per-checkpoint growth through a
                    # new door.
                    continue
                if hasattr(c, "cache") and isinstance(c.cache, list):
                    from copy import deepcopy

                    cloned = deepcopy(c)
                    cloned.cache = [
                        mx.contiguous(a) if a is not None else None
                        for a in c.cache
                    ]
                    ssm_layers.append(cloned)
                else:
                    ssm_layers.append(c)
            if ssm_layers:
                _unsalted_media = False
                if cache_extra_keys is None:
                    try:
                        _unsalted_media = (
                            self._tokens_contain_media_placeholders(
                                list(tokens)
                            )
                        )
                    except Exception:
                        # Detection failed, so we do not KNOW there is media.
                        # A failed probe must not become a restriction --
                        # decline only on a POSITIVE media reading.
                        _unsalted_media = False
                if _unsalted_media:
                    # Fail closed. A media prompt under the bare token key is
                    # a cross-image collision waiting to happen: the reader
                    # that asks with the media salt can never find it, and the
                    # reader that asks WITHOUT the salt would find it for a
                    # different image. Neither outcome is acceptable, so this
                    # store is declined rather than guessed at.
                    logger.info(
                        "MLLM clean prefill: declining the companion store at "
                        "%d tokens -- the prompt carries media placeholders "
                        "but no media cache key was supplied, and an unsalted "
                        "media companion collides across images.",
                        prompt_len,
                    )
                    return
                companion_cache.store(
                    tokens,
                    prompt_len,
                    ssm_layers,
                    is_complete=True,
                    cache_extra_keys=cache_extra_keys,
                )
                logger.info(
                    "MLLM clean prefill: stored complete SSM companion at the "
                    "%d-token key (media_salted=%s; idle re-derive will skip)",
                    prompt_len,
                    cache_extra_keys is not None,
                )
        except Exception as ex:
            logger.warning(
                "MLLM clean prefill: companion store failed (non-fatal): %s", ex
            )

    def _prefill_for_clean_ssm(
        self,
        tokens: List[int],
        base_cache: Optional[List[Any]] = None,
        base_token_count: int = 0,
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[List[Any]]:
        """Compatibility alias for hybrid SSM callers."""
        return self._prefill_for_clean_path_dependent_cache(
            tokens,
            base_cache,
            base_token_count,
            cache_extra_keys=cache_extra_keys,
        )

    def _prefill_for_clean_media_prefix_cache(
        self,
        request: "MLLMBatchRequest",
        tokens: List[int],
    ) -> Optional[List[Any]]:
        """Run a media-conditioned clean prefill for a media prefix key.

        Unlike `_prefill_for_clean_path_dependent_cache`, this keeps the VLM
        wrapper and pixel/video tensors in the forward path. It adds an extra
        prefill on a cold request but is the minimal safe way to create SSM
        state at the same media-keyed N-1 boundary as the paged KV blocks.
        """
        if not tokens or self.language_model is None:
            return None
        full_tokens = list(getattr(request, "_original_token_ids", None) or [])
        media_ids: set[int] = set()
        if full_tokens:
            try:
                media_ids = self._media_placeholder_token_ids()
            except Exception:
                pass
        media_runs = _media_placeholder_runs(full_tokens, media_ids)
        # A truncated wrapper call is invalid both INSIDE a media run and
        # BEFORE one: either way the shortened IDs do not contain every
        # placeholder required by the request's full pixel/video tensor.
        if media_runs and len(tokens) <= max(end for _start, end in media_runs):
            return self._prefill_for_exact_media_embedding_prefix_cache(
                request, tokens, full_tokens
            )
        _saved_pos_state: Dict[str, Any] = {}
        try:
            # Qwen VL keeps request-local RoPE state on the language model.
            # The auxiliary N-1 prefill must start clean, then restore the
            # original full-prompt state so the active request's decode step
            # continues from its real media prefill rather than the shorter
            # cache-building pass.
            for attr in ("_rope_deltas", "_position_ids"):
                if hasattr(self.language_model, attr):
                    _saved_pos_state[attr] = getattr(self.language_model, attr)
                    setattr(self.language_model, attr, None)
            cache_model = getattr(self, "_cache_model", None)
            fresh_cache = cache_model.make_cache() if cache_model is not None else None
            if fresh_cache is None:
                from mlx_lm.models.cache import KVCache

                fresh_cache = [
                    KVCache() for _ in range(len(self.language_model.layers))
                ]
            from copy import copy

            clean_req = copy(request)
            clean_req.input_ids = mx.array([tokens])
            if request.attention_mask is not None:
                try:
                    clean_req.attention_mask = request.attention_mask[:, : len(tokens)]
                except Exception:
                    clean_req.attention_mask = request.attention_mask
            clean_req.vision_encoded = False
            # Nested bookkeeping prefill, not a user turn: exempt from the
            # turn-peak walk (its mid-turn gauge read/reset would corrupt the
            # real span's deferred measurement, and a refusal here would
            # silently skip the media prefix store).
            clean_req._aux_clean_path_prefill = True
            logits = self._run_vision_encoding(clean_req, cache=fresh_cache)
            materialize: List[Any] = []

            def _collect_cache_arrays(cache_obj: Any) -> None:
                if hasattr(cache_obj, "keys") and cache_obj.keys is not None:
                    if isinstance(cache_obj.keys, tuple):
                        materialize.extend(cache_obj.keys)
                        materialize.extend(cache_obj.values)
                    else:
                        materialize.extend([cache_obj.keys, cache_obj.values])
                elif hasattr(cache_obj, "caches") and isinstance(
                    getattr(cache_obj, "caches", None), (list, tuple)
                ):
                    for sub_cache in cache_obj.caches:
                        _collect_cache_arrays(sub_cache)
                elif hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, list):
                    for arr in cache_obj.cache:
                        if hasattr(arr, "shape"):
                            materialize.append(arr)

            for c in fresh_cache:
                _collect_cache_arrays(c)
            if materialize:
                try:
                    mx.eval(materialize)
                except RuntimeError as _eval_err:
                    if "Stream" in str(_eval_err):
                        mx.synchronize()
                    else:
                        raise
            del logits
            return fresh_cache
        except Exception as ex:
            logger.warning(
                "MLLM clean media prefix prefill failed for %s (non-fatal): %s",
                getattr(request, "request_id", "?"),
                ex,
            )
            return None
        finally:
            for attr, value in _saved_pos_state.items():
                try:
                    setattr(self.language_model, attr, value)
                except Exception:
                    pass

    def _prefill_for_exact_media_embedding_prefix_cache(
        self,
        request: "MLLMBatchRequest",
        tokens: List[int],
        full_tokens: List[int],
    ) -> Optional[List[Any]]:
        """Snapshot a Qwen hybrid prefix at or before expanded media.

        The complete request is vision-encoded once. Only its exact merged
        embedding and mRoPE prefix is then forwarded into a fresh native cache.
        Wrappers with additional DeepStack/cross-attention/per-layer state are
        rejected so their caller can retain the safe terminal snapshot.
        """
        family = str(getattr(self, "_model_type", "") or "").lower()
        if family not in {"qwen3_5", "qwen3_5_moe"}:
            return None
        if not full_tokens or len(tokens) >= len(full_tokens):
            return None
        if full_tokens[:len(tokens)] != list(tokens):
            logger.warning(
                "MLLM exact media prefix declined for %s: requested tokens "
                "are not a prefix of the full media prompt",
                getattr(request, "request_id", "?"),
            )
            return None

        lm = self.language_model
        get_embeds = getattr(self.model, "get_input_embeddings", None)
        if not callable(get_embeds) or _media_embed_kwarg_name(lm) != "inputs_embeds":
            return None

        saved_position_state: Dict[str, Any] = {}
        try:
            for attr in ("_rope_deltas", "_position_ids"):
                if hasattr(lm, attr):
                    saved_position_state[attr] = getattr(lm, attr)
                    setattr(lm, attr, None)

            full_input_ids = request.input_ids
            if full_input_ids is None:
                return None
            if full_input_ids.ndim == 1:
                full_input_ids = full_input_ids[None, :]
            if full_input_ids.shape[1] < len(full_tokens):
                return None
            if full_input_ids[0, :len(tokens)].tolist() != list(tokens):
                return None

            kwargs = dict(request.extra_kwargs)
            if request.pixel_values is not None:
                kwargs["pixel_values"] = request.pixel_values
            if request.video_pixel_values is not None:
                kwargs[_video_pixel_values_kwarg_name(self.model)] = (
                    request.video_pixel_values
                )
            if request.attention_mask is not None:
                kwargs["mask"] = request.attention_mask
            if request.image_grid_thw is not None:
                kwargs["image_grid_thw"] = request.image_grid_thw
            if request.video_grid_thw is not None:
                kwargs["video_grid_thw"] = request.video_grid_thw

            features = get_embeds(full_input_ids, **kwargs)
            embeds = getattr(features, "inputs_embeds", None)
            if embeds is None or getattr(embeds, "ndim", 0) < 3:
                return None
            feature_dict = (
                features.to_dict()
                if callable(getattr(features, "to_dict", None))
                else {}
            )
            unsupported = sorted(
                key for key, value in feature_dict.items()
                if key != "inputs_embeds" and value is not None
            )
            if unsupported:
                logger.info(
                    "MLLM exact media prefix unavailable for %s: embedding "
                    "features require unsliced state %s",
                    getattr(request, "request_id", "?"),
                    ",".join(unsupported),
                )
                return None

            position_ids = getattr(lm, "_position_ids", None)
            if position_ids is None or position_ids.shape[-1] < len(tokens):
                return None
            cache_model = getattr(self, "_cache_model", None)
            fresh_cache = cache_model.make_cache() if cache_model is not None else None
            if fresh_cache is None:
                make_cache = getattr(lm, "make_cache", None)
                fresh_cache = make_cache() if callable(make_cache) else None
            if fresh_cache is None:
                return None

            chunk = max(1, int(self._media_prefill_chunk_tokens(len(tokens))))
            output = None
            for start in range(0, len(tokens), chunk):
                end = min(start + chunk, len(tokens))
                output = lm(
                    full_input_ids[:, start:end],
                    inputs_embeds=embeds[:, start:end],
                    mask=None,
                    cache=fresh_cache,
                    position_ids=position_ids[..., start:end],
                )

            materialize: List[Any] = []
            def collect(cache_obj: Any) -> None:
                if hasattr(cache_obj, "keys") and cache_obj.keys is not None:
                    if isinstance(cache_obj.keys, tuple):
                        materialize.extend(cache_obj.keys)
                        materialize.extend(cache_obj.values)
                    else:
                        materialize.extend([cache_obj.keys, cache_obj.values])
                elif hasattr(cache_obj, "caches") and isinstance(
                    getattr(cache_obj, "caches", None), (list, tuple)
                ):
                    for sub_cache in cache_obj.caches:
                        collect(sub_cache)
                elif hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, list):
                    materialize.extend(
                        arr for arr in cache_obj.cache if hasattr(arr, "shape")
                    )

            for cache_obj in fresh_cache:
                collect(cache_obj)
            if materialize:
                try:
                    mx.eval(materialize)
                except RuntimeError as eval_err:
                    if "Stream" in str(eval_err):
                        mx.synchronize()
                    else:
                        raise
            del output
            logger.info(
                "MLLM media prefix cache: captured exact media-conditioned "
                "embedding boundary for %s at %d tokens",
                getattr(request, "request_id", "?"),
                len(tokens),
            )
            return fresh_cache
        except Exception as ex:
            logger.warning(
                "MLLM exact media embedding prefix failed for %s (non-fatal): %s",
                getattr(request, "request_id", "?"),
                ex,
            )
            return None
        finally:
            for attr, value in saved_position_state.items():
                try:
                    setattr(lm, attr, value)
                except Exception:
                    pass

    def run_idle_rederive(self) -> bool:
        """Process one SSM rederive task from the queue (scheduler idle tick).

        Returns True if a task was processed, False if queue was empty.
        Caps at one task per tick so decode latency is unaffected.
        """
        if not self._ssm_rederive_queue or self._ssm_state_cache is None:
            return False
        if not self._hybrid_kv_positions:
            self._ssm_rederive_queue.clear()
            return False
        item = self._ssm_rederive_queue.pop(0)
        if len(item) == 4:
            tokens, prompt_len, orig_rid, cache_extra_keys = item
        else:
            tokens, prompt_len, orig_rid = item
            cache_extra_keys = None
        if self._ssm_state_cache.has_complete(
            tokens, prompt_len, cache_extra_keys=cache_extra_keys
        ):
            # Stored between enqueue and this idle tick (e.g. an identical
            # request completed first). Skip the redundant clean prefill.
            logger.info(
                "MLLM SSM re-derive skipped at idle for %s: complete "
                "companion already stored at %d-token key",
                orig_rid,
                prompt_len,
            )
            return True
        logger.info(
            f"MLLM SSM re-derive: clean prefill for {orig_rid} "
            f"({prompt_len} prompt tokens, {len(self._ssm_rederive_queue)} remaining)"
        )
        try:
            # The salt has to travel WITH the prefill. Without it the clean
            # pass stored its companion under the bare token key while this
            # function's has_complete() probe asks with the salt -- so the
            # probe below always missed, every media re-derive paid the
            # 50-200MB deep clone plus disk write TWICE, and the first copy
            # was an unsalted media companion that a different image with the
            # same placeholder tokens would happily restore.
            clean_cache = self._prefill_for_clean_ssm(
                list(tokens), cache_extra_keys=cache_extra_keys
            )
            if clean_cache is None:
                return True
            if self._ssm_state_cache.has_complete(
                tokens, prompt_len, cache_extra_keys=cache_extra_keys
            ):
                # The clean pass itself already stored the (filtered)
                # companion via _store_companion_from_clean_pass — a second
                # store here would both duplicate a 50-200MB deep clone +
                # disk write AND, before this guard, re-store the layers
                # UNFILTERED, overwriting the exemption-correct entry.
                logger.info(
                    "MLLM SSM re-derive: companion already stored by the "
                    "clean pass for %s (%d-token key)",
                    orig_rid,
                    prompt_len,
                )
                del clean_cache
                try:
                    mx.clear_cache()
                except Exception:
                    pass
                return True
            kv_set = set(self._hybrid_kv_positions or [])
            ssm_layers: List[Any] = []
            for layer_idx, c in enumerate(clean_cache):
                if layer_idx in kv_set:
                    continue
                if _companion_exempt_cache(c):
                    # Positional full-latent slots (dots3) are excluded from
                    # every companion snapshot; storing them here would both
                    # shift the fetch splice's sequential slot fill and
                    # resurrect the O(ctx)-per-checkpoint growth (1c282ae23).
                    continue
                if hasattr(c, "cache") and isinstance(c.cache, list):
                    from copy import deepcopy
                    cloned = deepcopy(c)
                    cloned.cache = [
                        mx.contiguous(a) if a is not None else None
                        for a in c.cache
                    ]
                    ssm_layers.append(cloned)
                else:
                    ssm_layers.append(c)
            if ssm_layers:
                self._ssm_state_cache.store(
                    tokens,
                    prompt_len,
                    ssm_layers,
                    is_complete=True,
                    cache_extra_keys=cache_extra_keys,
                )
                logger.info(
                    f"MLLM SSM re-derive: stored clean companion for {orig_rid}: "
                    f"{len(ssm_layers)} SSM layers, {prompt_len}-token key"
                )
            del clean_cache
            try:
                mx.clear_cache()
            except Exception:
                pass
        except Exception as ex:
            logger.warning(f"MLLM SSM re-derive failed for {orig_rid}: {ex}")
        return True
