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
import logging
import inspect
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .errors import (
    PromptTooLongError,
    UnsupportedMediaModalityError,
    VLMImagePrefillBudgetError,
)
from .vision_embedding_cache import VisionEmbeddingCache
from .utils.memory_limits import (
    get_effective_metal_working_set_bytes,
    get_metal_ws_guard_threshold,
)
from .mlx_memory import clear_mlx_memory_cache

logger = logging.getLogger(__name__)


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
        or getattr(request, "pixel_values", None) is not None
        or getattr(request, "image_grid_thw", None) is not None
    )
    if not has_media:
        return None

    hasher = hashlib.sha256()
    hasher.update(b"vmlx-mllm-media-cache-v1")
    media_sources = list(getattr(request, "images", None) or []) + list(
        getattr(request, "videos", None) or []
    )

    def _hash_text(label: str, value: Any) -> None:
        hasher.update(label.encode("utf-8"))
        hasher.update(hashlib.sha256(str(value).encode("utf-8")).digest())

    def _hash_array(label: str, value: Any) -> None:
        if value is None:
            return
        hasher.update(label.encode("utf-8"))
        try:
            hasher.update(str(getattr(value, "shape", "")).encode("utf-8"))
            hasher.update(str(getattr(value, "dtype", "")).encode("utf-8"))
            import numpy as np

            hasher.update(np.array(value).tobytes())
        except Exception:
            _hash_text(label, value)

    for source in getattr(request, "images", None) or []:
        _hash_text("image", source)
    for source in getattr(request, "videos", None) or []:
        _hash_text("video", source)
    if getattr(request, "videos", None):
        _hash_text("video_fps", getattr(request, "video_fps", None))
        _hash_text("video_max_frames", getattr(request, "video_max_frames", None))
    _hash_array("image_grid_thw", getattr(request, "image_grid_thw", None))
    if not media_sources:
        # Fallback for callers that hand us preprocessed pixel tensors without
        # source URLs/paths. Do not hash attention_mask: it changes with text
        # history length and would make the same image miss across turns.
        _hash_array("pixel_values", getattr(request, "pixel_values", None))

    return {"mllm_media": hasher.hexdigest()}


@dataclass(frozen=True)
class VLMImagePrefillBudgetDecision:
    should_reject: bool
    predicted_attention_bytes: int
    active_memory_bytes: int
    max_working_set_bytes: int
    detail: str


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


def _apply_vlm_image_request_cache_limit() -> None:
    """Tighten the Metal reusable cache before VLM media work.

    This is a preflight memory-safety control, not a model behavior change. It
    leaves text-only requests on the normal scheduler cache policy and only
    shrinks MLX's allocator free-list ceiling for media requests that otherwise
    have large transient tensors.
    """
    if os.environ.get("VMLX_VLM_IMAGE_CACHE_LIMIT", "1") == "0":
        return
    if not mx.metal.is_available():
        return
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
        return
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
            return
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
    except Exception as exc:
        logger.debug("VLM image request cache limit not applied: %s", exc)


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
            should_reject=False,
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
            should_reject=False,
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
            f"predicted attention buffer {gb(predicted):.1f}GB exceeds "
            f"single-buffer guard {gb(single_limit):.1f}GB"
        )
    if exceeds_working_set:
        reasons.append(
            f"projected Metal working set {projected_pct:.0f}% exceeds "
            f"threshold {reject_pct:.0f}%"
        )
    return VLMImagePrefillBudgetDecision(
        should_reject=True,
        predicted_attention_bytes=predicted,
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        detail=(
            "VLM image prefill rejected before Metal forward: "
            + "; ".join(reasons)
            + ". Image prefill cannot be chunked safely because the VLM "
            "wrapper needs the full media-expanded prompt. Reduce image "
            "resolution/prompt length, use a smaller model, or set "
            "VMLX_VLM_IMAGE_PREFILL_GUARD=0 to bypass this guard at OOM risk."
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


def _raise_if_image_prefill_exceeds_budget(
    *,
    has_images: bool,
    seq_len: int,
    language_model: Any,
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
        has_images=has_images,
        seq_len=seq_len,
        num_attention_heads=_infer_attention_heads_for_hybrid_oom_guard(
            language_model
        ),
        active_memory_bytes=active,
        max_working_set_bytes=max_ws,
        reject_pct=reject_pct,
        single_buffer_limit_bytes=single_buffer_limit,
        guard_enabled=guard_enabled,
    )
    if decision.should_reject:
        logger.warning(decision.detail)
        raise VLMImagePrefillBudgetError(decision.detail)


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
    if getattr(first_req, "repetition_penalty", 1.0) not in (None, 1.0):
        return False
    first_key = (
        getattr(first_req, "temperature", None),
        getattr(first_req, "top_p", None),
        getattr(first_req, "top_k", None),
        getattr(first_req, "min_p", None),
    )
    for req in requests[1:]:
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
    """Return prefill tail + cached-token count for an N-1 VLM prefix hit."""
    key_tokens = list(token_list or [])
    cache_remaining = list(remaining or [])
    suffix = list(gen_prompt_suffix or [])
    if suffix and not cache_remaining and key_tokens:
        cache_remaining = [key_tokens[-1]]
    if not cache_remaining and key_tokens:
        cached_tokens = max(len(key_tokens) - 1, 0)
    else:
        cached_tokens = max(len(key_tokens) - len(cache_remaining), 0)
    return cache_remaining + suffix, cached_tokens


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


def _gen_stream() -> Any:
    """Lazily create the dedicated GPU stream for prefill/decode work."""
    global _GENERATION_STREAM
    if _GENERATION_STREAM is None:
        try:
            _GENERATION_STREAM = mx.new_stream(mx.default_device())
        except Exception:
            try:
                _GENERATION_STREAM = mx.default_stream(mx.default_device())
            except Exception:
                _GENERATION_STREAM = None
    return _GENERATION_STREAM


def reset_generation_streams() -> None:
    """Drop thread-local MLX stream handles after MLLM engine teardown.

    MLX stream objects are bound to the worker thread that created them. Deep
    sleep and model switch tear down the scheduler/executor and create a new
    worker on wake. Keeping these globals across that lifecycle can make the
    next generation try to use a stream from the old thread.
    """
    global _GENERATION_STREAM
    _GENERATION_STREAM = None
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


def _call_processor_direct(
    processor: Any,
    *,
    prompts: Any,
    images: Optional[List[str]],
    videos: Optional[List[Any]] = None,
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
    if callable(process_method) and not videos:
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

    kwargs: Dict[str, Any] = {
        "text": prompts,
        "padding": True,
        "return_tensors": "mlx",
        "add_special_tokens": add_special_tokens,
    }
    if images:
        kwargs["images"] = _shape_images_for_processor_call(processor, images)
    if videos:
        kwargs["videos"] = videos
    try:
        params = inspect.signature(processor).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return _as_input_mapping(processor(**kwargs))

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


def _is_kv_like(c) -> bool:
    """Check if cache is attention-KV compatible.

    Gemma4 mixed-SWA uses RotatingKVCache for sliding-window attention layers
    and KVCache for full-attention layers. Both are attention cache slots; they
    must not be mistaken for SSM/ArraysCache hybrid state.
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    return isinstance(c, (KVCache, RotatingKVCache)) or type(c).__name__ == _TQ_CLASS_NAME


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


def _paged_hybrid_cache_detail(*, disk_hit: bool, mixed_attention: bool) -> str:
    base = "paged+mixed_swa" if mixed_attention else "paged+ssm"
    return f"{base}+disk" if disk_hit else base


def _paged_attention_cache_detail(*, disk_hit: bool, mixed_attention: bool) -> str:
    base = "paged+mixed_swa" if mixed_attention else "paged"
    return f"{base}+disk" if disk_hit else base


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


def _recompress_to_tq(cache: List[Any], language_model) -> List[Any]:
    """Re-wrap reconstructed KVCache layers into TurboQuantKVCache and compress.

    After paged cache reconstruction, KV layers are full-precision float16.
    If the model uses TurboQuant, this wastes 5.3x memory vs the 3-bit
    compressed form used during generation. This function:
    1. Gets a TQ template from model.make_cache() to extract TQ config
    2. For each KVCache layer, creates a matching TurboQuantKVCache
    3. Copies keys/values and calls .compress() to restore 3-bit storage

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
            sink_tokens=tpl.sink_tokens,
        )
        # Copy reconstructed float16 data into TQ
        tq.keys = layer.keys
        tq.values = layer.values
        tq.offset = layer.offset
        tq.step = getattr(layer, 'step', layer.keys.shape[2]) if layer.keys.ndim >= 3 else layer.offset
        # Compress to 3-bit — this creates _joined_k/_joined_v and clears .keys/.values
        tq.compress()
        result[i] = tq
        tq_count += 1

    if tq_count > 0:
        logger.info(f"Re-compressed {tq_count} KV layers to TurboQuant (5x memory savings)")
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
                if not _is_kv_like(tmpl) and _is_kv_like(cached):
                    result[i] = tmpl
                    fixed = True
            if fixed:
                logger.debug("Fixed hybrid cache: replaced KVCache at SSM positions")
            return result

        # Cache shorter than model — expand using template
        positions = kv_positions if kv_positions is not None else [
            i for i, t in enumerate(template) if _is_kv_like(t)
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
from .utils.ssm_companion_cache import HybridSSMStateCache, SSMCompanionCache  # noqa: F401


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
            "timings_ms": timings,
            "fallback_reason": fallback_reason,
        }


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


@dataclass
class _NativeMTPCacheObjectSnapshot:
    obj: Any
    attrs: Dict[str, Any]


def _native_mtp_depth() -> int:
    from .native_mtp import native_mtp_effective_depth

    depth, _source = native_mtp_effective_depth()
    return depth


def _native_mtp_tool_choice_requires_tools(tool_choice: Any) -> bool:
    if isinstance(tool_choice, dict):
        return True
    if isinstance(tool_choice, str):
        return tool_choice in {"required", "auto"}
    return False


def _native_mtp_request_has_tools(request: Any) -> bool:
    if request is None:
        return False

    prompt = getattr(request, "prompt", None)
    if isinstance(prompt, str) and any(
        marker in prompt
        for marker in (
            "<tools>",
            "<tool_call>",
            '"type": "function"',
            "'type': 'function'",
            "You have access to the following tools",
            "[Calling tool:",
        )
    ):
        return True

    for attr in ("tools", "_vmlx_effective_tools"):
        if getattr(request, attr, None):
            return True
    if _native_mtp_tool_choice_requires_tools(getattr(request, "tool_choice", None)):
        return True

    extra = getattr(request, "extra_kwargs", None)
    if not isinstance(extra, dict):
        return False
    if extra.get("_vmlx_tools_present"):
        return True
    if extra.get("tools") or extra.get("_vmlx_effective_tools"):
        return True
    if _native_mtp_tool_choice_requires_tools(extra.get("tool_choice")):
        return True

    template_kwargs = extra.get("chat_template_kwargs")
    if isinstance(template_kwargs, dict):
        if template_kwargs.get("tools"):
            return True
        if _native_mtp_tool_choice_requires_tools(template_kwargs.get("tool_choice")):
            return True

    return False


def _native_mtp_depth_for_request(request: Any) -> int:
    depth = _native_mtp_depth()
    if depth <= 1 or not _native_mtp_request_has_tools(request):
        return depth

    request_id = getattr(request, "request_id", "<unknown>")
    logged_depth = getattr(request, "_native_mtp_tool_depth_cap_logged", None)
    if logged_depth != depth:
        logger.info(
            "MLLM native MTP depth capped to D1 for request=%s because tools are present (configured D%d)",
            request_id,
            depth,
        )
        try:
            setattr(request, "_native_mtp_tool_depth_cap_logged", depth)
        except Exception:
            pass
    return 1


def _native_mtp_logprobs(logits_2d: mx.array) -> mx.array:
    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _native_mtp_sampler_accepts_logits(sampler: Callable[[mx.array], mx.array]) -> bool:
    return bool(getattr(sampler, "_vmlx_accepts_logits", False))


def _native_mtp_sample_one(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[mx.array, Optional[mx.array]]:
    if _native_mtp_sampler_accepts_logits(sampler):
        token = _native_mtp_ensure_uint32(sampler(logits_2d))
        return token, None
    logprobs = _native_mtp_logprobs(logits_2d)
    token = _native_mtp_ensure_uint32(sampler(logprobs))
    return token, logprobs.squeeze(0)


def _native_mtp_sample_rows(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[List[mx.array], List[Optional[mx.array]], List[int]]:
    if _native_mtp_sampler_accepts_logits(sampler):
        sampled = _native_mtp_ensure_uint32(sampler(logits_2d))
        mx.eval(sampled)
        sampled_ids = [int(value) for value in sampled.tolist()]
        rows = int(sampled.shape[0])
        return [sampled[i : i + 1] for i in range(rows)], [None] * rows, sampled_ids
    logprobs = _native_mtp_logprobs(logits_2d)
    sampled = _native_mtp_ensure_uint32(sampler(logprobs))
    mx.eval(sampled)
    sampled_ids = [int(value) for value in sampled.tolist()]
    rows = int(sampled.shape[0])
    return [sampled[i : i + 1] for i in range(rows)], [
        logprobs[i] for i in range(rows)
    ], sampled_ids


def _sample_mllm_prefill_logits(
    logits_2d: mx.array,
    sampler: Callable[[mx.array], mx.array],
) -> Tuple[mx.array, Optional[mx.array]]:
    """Sample the first MLLM decode token without logprobs when possible."""
    if _native_mtp_sampler_accepts_logits(sampler):
        sampled = _native_mtp_ensure_uint32(sampler(logits_2d))
        return sampled, None
    logprobs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
    sampled = _native_mtp_ensure_uint32(sampler(logprobs))
    return sampled, logprobs


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


def _mllm_decode_sync_eval_enabled() -> bool:
    return os.environ.get("VMLINUX_MLLM_DECODE_SYNC_EVAL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _submit_decode_token_eval(value: Any) -> None:
    if _mllm_decode_sync_eval_enabled():
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


def _native_mtp_should_snapshot_layer(layer: Any) -> bool:
    if layer is None:
        return False
    if hasattr(layer, "rollback_state"):
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
) -> List[Optional[_NativeMTPCacheObjectSnapshot]]:
    return [
        _native_mtp_snapshot_object(layer)
        if _native_mtp_should_snapshot_layer(layer)
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


def _native_mtp_restore_or_trim(cache: List[Any], draft_count: int) -> bool:
    """Rollback unaccepted native-MTP draft tokens after verifier rejection."""
    for layer in cache:
        rollback = getattr(layer, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            layer.conv_state = conv_snap
            layer.ssm_state = ssm_snap
            layer.rollback_state = None
        elif hasattr(layer, "is_trimmable") and layer.is_trimmable():
            layer.trim(max(1, int(draft_count)))
        elif draft_count > 0:
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


def _native_mtp_log_stats(request_id: str, stats: MLLMNativeMTPStats, reason: str) -> None:
    rate = (stats.accepted_tokens / stats.drafted_tokens * 100.0) if stats.drafted_tokens else 0.0
    logger.info(
        "MLLM MTP[%s] finish=%s cycles=%d accepted=%d/%d (%.1f%%) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d]",
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
    if int(state.stats.cycles) < warmup:
        return

    target = current
    if target >= 3:
        drafted_d3 = int(state.stats.drafted_by_depth[2]) if len(state.stats.drafted_by_depth) > 2 else 0
        rate_d3 = _native_mtp_depth_rate(state.stats, 3)
        min_d3 = _native_mtp_env_float(
            0.85,
            "VMLINUX_NATIVE_MTP_D3_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_D3_MIN_ACCEPT",
        )
        if drafted_d3 >= warmup and rate_d3 is not None and rate_d3 < min_d3:
            target = 2
    if target >= 2:
        drafted_d2 = int(state.stats.drafted_by_depth[1]) if len(state.stats.drafted_by_depth) > 1 else 0
        rate_d2 = _native_mtp_depth_rate(state.stats, 2)
        min_d2 = _native_mtp_env_float(
            0.75,
            "VMLINUX_NATIVE_MTP_D2_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_D2_MIN_ACCEPT",
        )
        if drafted_d2 >= warmup and rate_d2 is not None and rate_d2 < min_d2:
            target = 1

    if _native_mtp_maybe_cost_fallback(request_id, state, current):
        return

    if target <= 1 and _native_mtp_env_flag(
        False,
        "VMLINUX_NATIVE_MTP_AR_FALLBACK",
        "VMLX_NATIVE_MTP_AR_FALLBACK",
    ):
        drafted_d1 = int(state.stats.drafted_by_depth[0]) if state.stats.drafted_by_depth else 0
        rate_d1 = _native_mtp_depth_rate(state.stats, 1)
        min_d1 = _native_mtp_env_float(
            0.65,
            "VMLINUX_NATIVE_MTP_D1_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_D1_MIN_ACCEPT",
        )
        if drafted_d1 >= warmup and rate_d1 is not None and rate_d1 < min_d1:
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
        state.depth = target


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
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_prompt_tokens: int = 0
    enable_thinking: Optional[bool] = None

    # Video processing parameters (per-request overrides)
    video_fps: Optional[float] = None
    video_max_frames: Optional[int] = None

    # Processed inputs (set after vision preprocessing)
    input_ids: Optional[mx.array] = None
    pixel_values: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    image_grid_thw: Optional[mx.array] = None
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
    cache_extra_keys: Optional[Dict[str, str]] = None
    # Generation-prefix tokens captured from the untruncated prompt tail
    # (e.g. [<|im_start|>, assistant, \n, <think>, \n] for Qwen 3.6 thinking-on).
    # Thinking models occasionally re-emit these as their first output tokens
    # when prior assistant history lacks a reasoning_content wrapper; the
    # scheduler uses this list to suppress the echoed prefix from the output
    # stream. Empty when no gen-prefix was stripped.
    gen_prefix_tokens: Optional[List[int]] = None
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
        self.last_prefill_trace: Optional[Dict[str, Any]] = None

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
            "last_prefill_trace": self.last_prefill_trace,
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

    batch_cache = []
    for i in range(len(caches[0])):
        layer_cache = caches[0][i]
        layer_caches = [c[i] for c in caches]

        try:
            if _QuantizedKVCache is not None and isinstance(layer_cache, _QuantizedKVCache):
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

    converted = []
    for c in cache:
        if isinstance(c, BatchKVCache):
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
        enable_vision_cache: bool = True,
        vision_cache_size: int = 16,
        paged_cache_manager: Optional[Any] = None,
        block_aware_cache: Optional[Any] = None,
        memory_aware_cache: Optional[Any] = None,
        prefix_cache: Optional[Any] = None,
        disk_cache: Optional[Any] = None,
        kv_cache_bits: int = 0,
        kv_cache_group_size: int = 64,
        ssm_state_cache_size: int = 8,
        ssm_state_cache_max_mb: Optional[int] = 512,
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
            ssm_state_cache_size: Max entries in HybridSSMStateCache (LRU)
            ssm_state_cache_max_mb: Approximate resident-memory budget for
                companion SSM state. Large hybrid/VLM entries are skipped or
                LRU-evicted once this budget is exceeded.
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
        try:
            _cfg = getattr(model, "config", None)
            _text_cfg = getattr(_cfg, "text_config", None)
            self._model_type = str(
                getattr(_text_cfg, "model_type", None)
                or getattr(_cfg, "model_type", None)
                or (
                    _cfg.get("model_type")
                    if isinstance(_cfg, dict)
                    else ""
                )
                or ""
            ).lower()
        except Exception:
            self._model_type = ""

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

        # Pre-compute hybrid cache template info (avoids make_cache() per request)
        self._hybrid_kv_positions: Optional[List[int]] = None
        self._hybrid_num_layers: Optional[int] = None
        if hasattr(self.language_model, 'make_cache'):
            try:
                from mlx_lm.models.cache import KVCache
                template = self.language_model.make_cache()
                self._hybrid_num_layers = len(template)
                self._hybrid_kv_positions = [i for i, t in enumerate(template) if _is_kv_like(t)]
            except Exception as e:
                logger.warning(f"Failed to pre-compute hybrid cache info: {e}")

        # Pre-computed bool: is this a hybrid model (SSM + attention)?
        # Used throughout _process_prompts and _run_vision_encoding to gate
        # hybrid-specific logic (SSM companion cache, chunked prefill skip, etc.)
        self._is_hybrid: bool = (
            self._hybrid_kv_positions is not None
            and self._hybrid_num_layers is not None
            and len(self._hybrid_kv_positions) < self._hybrid_num_layers
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
        is_text_only = not request.images and not request.videos
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
                process_video_input,
                DEFAULT_FPS,
                MAX_FRAMES,
            )
            from mlx_vlm.video_generate import fetch_video

            fps = request.video_fps or DEFAULT_FPS
            max_frames = request.video_max_frames or MAX_FRAMES

            for video in request.videos:
                try:
                    video_path = process_video_input(video)
                    video_cache_sources.append(video_path)
                    video_input = fetch_video(
                        {"video": video_path, "fps": fps, "max_frames": max_frames}
                    )
                    video_inputs.append(video_input)
                except Exception as e:
                    logger.warning(f"Failed to process video: {e}")
            if request.videos and not video_inputs:
                raise ValueError("All video inputs failed to process")

        if all_images or video_inputs:
            _apply_vlm_image_request_cache_limit()
            mx.clear_cache()

        # Check pixel cache first
        media_cache_sources = all_images + video_cache_sources
        _mllm_bypass = bool(getattr(request, "_bypass_prefix_cache", False))
        cached_pixels = None
        if not _mllm_bypass:
            cached_pixels = self.vision_cache.get_pixel_cache(
                media_cache_sources, request.prompt
            )
        if cached_pixels is not None:
            # Cache hit - use cached pixel values
            request.input_ids = cached_pixels.input_ids
            request.pixel_values = cached_pixels.pixel_values
            request.attention_mask = cached_pixels.attention_mask
            request.image_grid_thw = cached_pixels.image_grid_thw
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
        ) or bool(video_inputs):
            inputs = _call_processor_direct(
                self.processor,
                prompts=request.prompt,
                images=all_images,
                videos=video_inputs,
                add_special_tokens=False,
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
        if pixel_values is None:
            pixel_values = inputs.get("pixel_values_videos")
        request.pixel_values = _ensure_mx_array(pixel_values)
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
            request.extra_kwargs["video_grid_thw"] = _ensure_mx_array(
                request.extra_kwargs["video_grid_thw"], mx.int32
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
            and request.pixel_values is not None
        ):
            self.vision_cache.set_pixel_cache(
                images=media_cache_sources,
                prompt=request.prompt,
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                image_grid_thw=request.image_grid_thw,
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
        max_prompt_tokens = int(getattr(request, "max_prompt_tokens", 0) or 0)
        if max_prompt_tokens <= 0:
            return
        prompt_tokens = _mllm_input_ids_token_count(request.input_ids)
        if prompt_tokens > max_prompt_tokens:
            raise PromptTooLongError(
                prompt_tokens,
                max_prompt_tokens,
                source=source,
                request_id=request.request_id,
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
        if has_images or not self._is_hybrid:
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

        return sorted(set(boundaries))

    def _ssm_block_aligned_boundary(self, boundary: int) -> int:
        """Return the largest positive paged-cache block boundary below boundary.

        Hybrid SSM companion state must exist at the same token count as the
        paged KV block hit. If the clean prompt boundary lands inside a partial
        block, the block cache can only reuse up to the previous full block.
        """
        block_size = int(getattr(self.block_aware_cache, "block_size", 0) or 0)
        if block_size <= 0 or boundary <= block_size:
            return 0
        block_boundary = (int(boundary) // block_size) * block_size
        if 0 < block_boundary < boundary:
            return block_boundary
        return 0

    def _media_placeholder_token_ids(self) -> set[int]:
        """Return configured media placeholder token ids for the loaded VLM.

        Different mlx-vlm families name these differently. Qwen-style configs
        commonly use ``image_token_index`` while Gemma/Nemotron-family configs
        may expose image/video/audio ids separately. Treat all discovered ids
        as path-dependent media placeholders for prefix-cache safety.
        """
        ids: set[int] = set()

        def _visit(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, dict):
                getter = obj.get
                nested = obj.get("text_config")
            else:
                getter = lambda key, default=None: getattr(obj, key, default)
                nested = getattr(obj, "text_config", None)
            for name in (
                "image_token_index",
                "image_token_id",
                "video_token_index",
                "video_token_id",
                "audio_token_index",
                "audio_token_id",
            ):
                value = getter(name, None)
                if isinstance(value, int) and value >= 0:
                    ids.add(value)
            if nested is not None and nested is not obj:
                _visit(nested)

        _visit(getattr(self.model, "config", None))
        _visit(getattr(self.language_model, "config", None))
        return ids

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
        """Return True when media prompts may use media-keyed KV+SSM cache."""
        enabled = os.environ.get("VMLINUX_MLLM_MEDIA_PREFIX_CACHE", "").strip()
        if enabled not in ("1", "true", "True", "yes", "on"):
            return False
        unsafe_ack = os.environ.get(
            "VMLINUX_MLLM_MEDIA_PREFIX_CACHE_UNSAFE_ACK", ""
        ).strip()
        if unsafe_ack not in ("1", "true", "True", "yes", "on"):
            return False
        if getattr(request, "_bypass_prefix_cache", False):
            return False
        if not getattr(request, "_cache_extra_keys", None):
            return False
        return self._request_has_media_cache_context(request, token_ids)

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
            return self._run_vision_encoding_inner(request, cache)

    def _run_vision_encoding_inner(self, request: "MLLMBatchRequest", cache: Optional[List[Any]] = None) -> "mx.array":
        kwargs = dict(request.extra_kwargs)
        # Only pass pixel_values when non-None. Smelt-loaded models use a
        # text-only wrapper whose __call__ does NOT accept pixel_values at
        # all — passing even None triggers `unexpected keyword argument
        # 'pixel_values'`. For standard VLM models, pixel_values=None is a
        # no-op that the vision encoder skips, so omitting it is safe.
        if request.pixel_values is not None:
            kwargs["pixel_values"] = request.pixel_values
        if request.attention_mask is not None:
            kwargs["mask"] = request.attention_mask
        if request.image_grid_thw is not None:
            kwargs["image_grid_thw"] = request.image_grid_thw
        if cache is not None:
            kwargs["cache"] = cache

        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        has_images = request.pixel_values is not None
        seq_len = input_ids.shape[1]

        # vmlx#89 / mlxstudio#83: opt-in chunked prefill for hybrid SSM models
        # on text-only requests. Hybrid models default to one-shot prefill
        # because their mask computation uses cache-position indexing
        # (fa_idx/ssm_idx) that was only tested for full-sequence processing.
        # However, long prompts (>~13K tokens at 32-head bfloat16) trigger
        # Metal single-buffer OOM (~9.5 GB cap per allocation) because
        # `attention_scores` blows up to (1, heads, seq_len, seq_len) * 2 bytes.
        #
        # Qwen3.5 hybrid (GatedDeltaNet + attention) has an opt-in chunked
        # prefill path for diagnostics and OOM escape hatches. Live decode
        # equivalence is not release-cleared, so native-MTP hybrid text splitting
        # must stay disabled by default. Other hybrid architectures may produce
        # incorrect output when chunked, so we only override the opt-in default
        # when one-shot prefill is *guaranteed* to OOM — a wrong answer beats
        # a hard crash that kills the session.
        _allow_hybrid_chunked = (
            os.environ.get("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL")
            or os.environ.get("VMLINUX_ALLOW_HYBRID_CHUNKED_PREFILL")
        ) in ("1", "true", "True", "yes", "on")
        _hybrid_blocks_chunk = self._is_hybrid and not _allow_hybrid_chunked
        _allow_native_mtp_hybrid_text_split = (
            os.environ.get("VMLINUX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT")
            or os.environ.get("VMLX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT")
        ) in ("1", "true", "True", "yes", "on")
        _native_mtp_hybrid_text_split = (
            _allow_native_mtp_hybrid_text_split
            and
            not has_images
            and self._is_hybrid
            and _native_mtp_model_has_head(self.language_model)
            and _lm_supports_return_logits(self.language_model)
        )

        # mlxstudio#83: auto-force chunking when a one-shot forward would
        # exceed the Metal single-buffer cap (reported by QwenCode/Opencode
        # `/init` sending full-repo context through a hybrid model).
        # Estimate attention_scores = heads * seq_len^2 * 2 bytes. Use a
        # conservative 8 GB threshold (Metal cap is 9.5 GB on 64 GB Macs).
        _OOM_GUARD_BYTES = 8 * 1024 * 1024 * 1024
        _n_heads_guess = _infer_attention_heads_for_hybrid_oom_guard(
            self.language_model
        )
        _predicted_attn_bytes = _n_heads_guess * seq_len * seq_len * 2
        if (
            _hybrid_blocks_chunk
            and not has_images
            and _predicted_attn_bytes > _OOM_GUARD_BYTES
            and os.environ.get("VMLX_DISABLE_HYBRID_AUTO_CHUNK") not in ("1", "true", "True", "yes", "on")
        ):
            # Family-specific safety: Qwen3.5 GatedDeltaNet (qwen3_next /
            # qwen3_5_moe text-config) was verified cache-aware end-to-end.
            # Other hybrid families (Nemotron-Cascade, MiniMax M2, Granite
            # Hybrid, etc.) take the chunked path as OOM-prevention fallback —
            # correctness should be spot-checked by the caller for those.
            _mt = "unknown"
            try:
                cfg_outer = getattr(self.model, "config", None)
                cfg_inner = getattr(cfg_outer, "text_config", None) if cfg_outer is not None else None
                _mt = (
                    getattr(cfg_inner, "model_type", None)
                    or getattr(cfg_outer, "model_type", None)
                    or "unknown"
                )
            except Exception:
                pass
            _verified = _mt in ("qwen3_next", "qwen3_5_moe", "qwen3_5_vl", "qwen3_5")
            logger.info(
                "Hybrid model (family=%s) seq_len=%d: one-shot attention buffer "
                "%.1f GB exceeds Metal single-buffer limit (~9.5 GB). Enabling "
                "chunked prefill — %s. Set VMLX_DISABLE_HYBRID_AUTO_CHUNK=1 to "
                "raise an OOM error instead of chunking.",
                _mt, seq_len, _predicted_attn_bytes / (1024**3),
                (
                    "verified safe on Qwen3.5 GatedDeltaNet" if _verified
                    else "spot-check output for correctness on non-Qwen3.5 hybrid families"
                ),
            )
            _hybrid_blocks_chunk = False

        # TEXT-ONLY FAST PATH: use language_model directly, skip VLM wrapper.
        # The VLM wrapper adds overhead from vision encoder path and some VLM
        # wrappers (e.g. Gemma 4 loaded via smelt) may not accept pixel_values.
        # Using language_model directly avoids this entirely.
        # Hybrid SSM models must go through the full model for correct mask
        # computation UNLESS the opt-in env var is set (vmlx#89).
        # mlxstudio#83: self.language_model already falls back to self.model
        # when the wrapped model has no `.language_model` attr (see __init__).
        # Using `getattr(self.model, 'language_model', None)` here returned
        # None for text-only models routed through MLLM path (e.g., smelt) and
        # silently skipped chunking, falling through to the OOM-prone
        # single-shot `self.model(input_ids, **kwargs)` at the bottom.
        if not has_images and self._model_type == "mimo_v2":
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

        if not has_images and (not _hybrid_blocks_chunk or _native_mtp_hybrid_text_split):
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
                        for capture_boundary in ssm_boundaries:
                            if capture_boundary > processed:
                                _call_lm_prefix_without_logits(
                                    lm,
                                    input_ids[:, processed:capture_boundary],
                                    _lm_kwargs_for(processed, capture_boundary),
                                )
                                _materialize_prefill_cache_state(cache)
                                processed = capture_boundary
                            self._maybe_capture_clean_ssm_boundary(
                                request, cache, all_tokens, capture_boundary
                            )
                        if final_start > processed:
                            _call_lm_prefix_without_logits(
                                lm,
                                input_ids[:, processed:final_start],
                                _lm_kwargs_for(processed, final_start),
                            )
                            _materialize_prefill_cache_state(cache)
                    elif final_start > 0:
                        _call_lm_prefix_without_logits(
                            lm,
                            input_ids[:, :final_start],
                            _lm_kwargs_for(0, final_start),
                        )
                        _materialize_prefill_cache_state(cache)
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
        # Hybrid SSM default: one-shot (safe). Opt-in via
        # VMLX_ALLOW_HYBRID_CHUNKED_PREFILL=1 for hybrid architectures that
        # have been verified cache-aware (Qwen3.5 GatedDeltaNet + attention).
        if (
            not has_images
            and seq_len > self.prefill_step_size * 2
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
                _prefill_keep_alloc = os.environ.get(
                    "VMLX_PREFILL_KEEP_ALLOC", ""
                ).lower() in {"1", "true", "yes", "on"}
                while processed < seq_len - 1:  # -1: keep last token for final logits
                    chunk_size = min(self.prefill_step_size, seq_len - 1 - processed)
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
                    chunk = input_ids[:, processed:processed + chunk_size]
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
                        raise
                    _materialize_prefill_cache_state(cache)
                    processed += chunk_size
                    chunk_num += 1
                    if processed in ssm_boundaries and processed not in ssm_captured_boundaries:
                        if self._maybe_capture_clean_ssm_boundary(
                            request,
                            cache,
                            _hoisted_all_tokens or input_ids[0].tolist(),
                            processed,
                        ):
                            ssm_captured_boundaries.add(processed)
                    if not _prefill_keep_alloc:
                        mx.clear_cache()

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
        if not has_images:
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

        if has_images:
            # Image-expanded prompts must use the one-shot VLM wrapper path.
            # Drop allocator free-list memory and reject impossible requests
            # before Metal executes a command buffer that can kill the server.
            _apply_vlm_image_request_cache_limit()
            mx.clear_cache()
        _raise_if_image_prefill_exceeds_budget(
            has_images=has_images,
            seq_len=seq_len,
            language_model=self.language_model,
        )
        output = self.model(input_ids, **kwargs)
        request.vision_encoded = True

        if hasattr(output, "logits"):
            return output.logits
        return output

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
            req._cache_extra_keys = _mllm_media_cache_extra_keys(req)
            # Save full token list BEFORE cache fetch can mutate req.input_ids.
            # Used later for SSM state cache keying (must be consistent with fetch key).
            _all_tokens = (
                req.input_ids.tolist()
                if req.input_ids is not None and req.input_ids.ndim == 1
                else req.input_ids[0].tolist()
                if req.input_ids is not None
                else []
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
            _media_context = self._request_has_media_cache_context(req)
            _media_cache_allowed = (
                _media_context and self._media_prefix_cache_allowed(req)
            )
            if (
                self._prefix_cache_enabled
                and self.block_aware_cache is not None
                and req.prompt_cache is None
                and (not _media_context or _media_cache_allowed)
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
                        block_table, remaining = self.block_aware_cache.fetch_cache(
                            req.request_id,
                            token_list,
                            cache_extra_keys=_cache_extra_keys,
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
                                        _cache_extra_keys
                                        if _media_cache_allowed
                                        else None
                                    )
                                    _entry = self._ssm_state_cache.fetch(
                                        token_list,
                                        _fetch_num,
                                        cache_extra_keys=_ssm_extra_keys,
                                    ) if _fetch_num > 0 else None
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
                                        _missed_ck = None
                                        _fn = getattr(
                                            self._ssm_state_cache,
                                            "fetch_longest_prefix",
                                            None,
                                        )
                                        if _enable_resume and _fn is not None and _fetch_num > 0:
                                            try:
                                                _missed_ck = _fn(
                                                    token_list,
                                                    _fetch_num,
                                                    cache_extra_keys=_ssm_extra_keys,
                                                )
                                            except Exception:
                                                _missed_ck = None

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
                                                    "request_id": req.request_id,
                                                    "cached_tokens": int(
                                                        getattr(block_table, "num_tokens", 0) or 0
                                                    ),
                                                    "reason": "checkpoint_incomplete",
                                                    "checkpoint_tokens": int(_ck_len or 0),
                                                }
                                                _lookup = getattr(
                                                    self._ssm_state_cache,
                                                    "last_prefix_lookup",
                                                    None,
                                                )
                                                if isinstance(_lookup, dict):
                                                    self._stats.last_hybrid_kv_without_ssm[
                                                        "ssm_prefix_lookup"
                                                    ] = dict(_lookup)
                                                logger.info(
                                                    f"vmlx#91 RESUME skipped for {req.request_id}: "
                                                    f"checkpoint at {_ck_len} has is_complete=False "
                                                    f"(gpl-contaminated) — full prefill"
                                                )
                                                self._mark_required_ssm_checkpoint(
                                                    req,
                                                    int(getattr(block_table, "num_tokens", 0) or 0),
                                                )
                                                self.block_aware_cache.release_cache(req.request_id)
                                                continue
                                            # Trim the block_table down to block-aligned
                                            # <= _ck_len so KV + SSM stay aligned.
                                            _requested_kv_tokens = int(
                                                getattr(block_table, "num_tokens", 0) or 0
                                            )
                                            trimmed = self.block_aware_cache.trim_block_table(
                                                req.request_id, _ck_len
                                            )
                                            if trimmed is not None and trimmed.num_tokens > 0:
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
                                                    "request_id": req.request_id,
                                                    "cached_tokens": int(
                                                        getattr(block_table, "num_tokens", 0) or 0
                                                    ),
                                                    "reason": "checkpoint_below_one_block",
                                                    "checkpoint_tokens": int(_ck_len or 0),
                                                }
                                                _lookup = getattr(
                                                    self._ssm_state_cache,
                                                    "last_prefix_lookup",
                                                    None,
                                                )
                                                if isinstance(_lookup, dict):
                                                    self._stats.last_hybrid_kv_without_ssm[
                                                        "ssm_prefix_lookup"
                                                    ] = dict(_lookup)
                                                logger.info(
                                                    f"vmlx#91 RESUME skipped for {req.request_id}: "
                                                    f"checkpoint at {_ck_len} below one block — "
                                                    f"full prefill required"
                                                )
                                                self._mark_required_ssm_checkpoint(
                                                    req,
                                                    int(getattr(block_table, "num_tokens", 0) or 0),
                                                )
                                                self.block_aware_cache.release_cache(req.request_id)
                                                continue
                                        else:
                                            self._stats.hybrid_kv_without_ssm_hits += 1
                                            self._stats.hybrid_kv_without_ssm_tokens += int(
                                                getattr(block_table, "num_tokens", 0) or 0
                                            )
                                            self._stats.last_hybrid_kv_without_ssm = {
                                                "request_id": req.request_id,
                                                "cached_tokens": int(
                                                    getattr(block_table, "num_tokens", 0) or 0
                                                ),
                                                "reason": (
                                                    "ssm_prefix_resume_disabled"
                                                    if not _enable_resume
                                                    else "no_ssm_companion_state"
                                                ),
                                            }
                                            _lookup = getattr(
                                                self._ssm_state_cache,
                                                "last_prefix_lookup",
                                                None,
                                            )
                                            if isinstance(_lookup, dict):
                                                self._stats.last_hybrid_kv_without_ssm[
                                                    "ssm_prefix_lookup"
                                                ] = dict(_lookup)
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
                                            self.block_aware_cache.release_cache(req.request_id)
                                            continue  # Skip reconstruction

                                # Either non-hybrid OR hybrid with SSM state — reconstruct
                                _cache_execution = {
                                    "request_id": req.request_id,
                                    "cache_detail": None,
                                    "cached_tokens": int(getattr(block_table, "num_tokens", 0) or 0),
                                    "blocks": len(getattr(block_table, "blocks", []) or []),
                                    "selection": "paged",
                                    "disk_hit": bool(_paged_disk_hit),
                                    "reconstructed": False,
                                    "dequantized": False,
                                    "reconstruction_seconds": 0.0,
                                    "dequantization_seconds": 0.0,
                                    "total_worker_cache_seconds": 0.0,
                                }
                                _cache_execution_started = time.perf_counter()
                                _reconstruct_started = time.perf_counter()
                                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                                _cache_execution["reconstruction_seconds"] = round(
                                    max(0.0, time.perf_counter() - _reconstruct_started), 6
                                )
                                _cache_execution["reconstructed"] = reconstructed is not None
                                _cache_execution["reconstruction_ok"] = reconstructed is not None
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
                                if is_hybrid and ssm_states is not None and reconstructed is not None:
                                    # Full hybrid cache reconstruction:
                                    # KV from paged cache + SSM from companion cache
                                    full_cache = _fix_hybrid_cache(
                                        reconstructed, self.language_model,
                                        kv_positions=self._hybrid_kv_positions,
                                        num_model_layers=self._hybrid_num_layers,
                                    )
                                    # Inject stored SSM states at non-KV positions
                                    kv_set = set(self._hybrid_kv_positions or [])
                                    ssm_idx = 0
                                    for layer_idx in range(len(full_cache)):
                                        if layer_idx not in kv_set and ssm_idx < len(ssm_states):
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
                                    )
                                    _cache_execution["cache_detail"] = req._cache_detail
                                    _cache_execution["total_worker_cache_seconds"] = round(
                                        max(0.0, time.perf_counter() - _cache_execution_started),
                                        6,
                                    )
                                    req._cache_execution = dict(_cache_execution)
                                    self._stats.last_cache_execution = dict(_cache_execution)
                                    # Re-attach the gen-prompt suffix: fetch key was
                                    # gpl-stripped but the model MUST see the template
                                    # suffix (<|im_start|>assistant\n<think>\n) to enter
                                    # thinking mode on turn 2.
                                    _full_remaining = (remaining or []) + list(_gpl_suffix)
                                    if _full_remaining:
                                        req.input_ids = mx.array([_full_remaining])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        logger.info(
                                            f"VLM HYBRID cache HIT for {req.request_id}: "
                                            f"{block_table.num_tokens} cached (KV+SSM), "
                                            f"{len(_full_remaining)} remaining "
                                            f"(incl. {len(_gpl_suffix)}-token gen-prompt suffix)"
                                        )
                                    else:
                                        req.input_ids = mx.array([token_list[-1:]])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        logger.info(
                                            f"VLM HYBRID cache FULL HIT for {req.request_id}: "
                                            f"{block_table.num_tokens} cached (KV+SSM)"
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
                                        )
                                    _cache_execution["cache_detail"] = req._cache_detail
                                    _cache_execution["total_worker_cache_seconds"] = round(
                                        max(0.0, time.perf_counter() - _cache_execution_started),
                                        6,
                                    )
                                    req._cache_execution = dict(_cache_execution)
                                    self._stats.last_cache_execution = dict(_cache_execution)
                                    # Re-attach gen-prompt suffix (see hybrid branch above
                                    # for full rationale — same correctness requirement
                                    # for attention-only thinking VLMs).
                                    _full_remaining = (remaining or []) + list(_gpl_suffix)
                                    if _full_remaining:
                                        # Check if remaining tokens contain image placeholders.
                                        # If so, we'd need partial pixel_values which is complex —
                                        # fall back to full prefill instead.
                                        has_images = self._tokens_contain_media_placeholders(
                                            _full_remaining
                                        )
                                        if has_images:
                                            req.prompt_cache = None
                                            req._cached_tokens = 0  # reset — full prefill needed
                                            req._cache_detail = None
                                            logger.info(
                                                f"VLM prefix cache HIT for {req.request_id}: "
                                                f"{block_table.num_tokens} cached tokens, "
                                                f"remaining has images — full prefill"
                                            )
                                        else:
                                            req.input_ids = mx.array([_full_remaining])
                                            req.pixel_values = None
                                            req.attention_mask = None
                                            req.image_grid_thw = None
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
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
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
            # DiskCacheManager.fetch() returns Optional[List[Any]] (exact match only,
            # no partial prefix), NOT a tuple like prefix_cache.fetch_cache().
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
                                        (
                                            _tail,
                                            num_cached,
                                        ) = _prefix_hit_tail_and_cached_tokens(
                                            token_list=token_list,
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

        # Get token sequences and lengths
        input_ids_list = [
            req.input_ids.tolist() if req.input_ids is not None else [0]
            for req in requests
        ]
        lengths = [len(ids) for ids in input_ids_list]

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
                            req.prompt_cache = None
                    else:
                        cache_for_fix = req.prompt_cache
                if req.prompt_cache is not None:
                    if not _validate_prompt_cache(
                        cache_for_fix,
                        source=f"mllm-prefill-cache:{req.request_id}",
                    ):
                        req.prompt_cache = None
                        cache_for_fix = None
                if req.prompt_cache is not None:
                    req_cache = _fix_hybrid_cache(
                        cache_for_fix, self.language_model,
                        kv_positions=self._hybrid_kv_positions,
                        num_model_layers=self._hybrid_num_layers,
                    )
                    # Paged/memory/disk cache reconstruction returns plain
                    # KVCache objects. For JANG/JANGTQ VLMs whose loader
                    # patched make_cache() to TurboQuantKVCache, re-wrap the
                    # fetched KV layers before the prefill tail so the live
                    # decode path keeps the same TQ memory profile as cold
                    # prefill. If the model is not TQ-backed this is a no-op.
                    req_cache = _recompress_to_tq(req_cache, self.language_model)
                else:
                    try:
                        if hasattr(self.language_model, 'make_cache'):
                            req_cache = self.language_model.make_cache()
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
                    logits = self._run_vision_encoding(req, cache=req_cache)
                    if trace is not None:
                        trace.stop("forward")
                        if self._decode_trace:
                            trace.set(
                                cache_after_forward=_cache_layer_debug_summary(req_cache)
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
                        if self.block_aware_cache is not None:
                            try:
                                self.block_aware_cache.release_cache(req.request_id)
                            except Exception:
                                pass
                        req.prompt_cache = None
                        req.input_ids = mx.array([req._original_token_ids])
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
                        try:
                            if hasattr(self.language_model, 'make_cache'):
                                req_cache = self.language_model.make_cache()
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
                        logits = self._run_vision_encoding(req, cache=req_cache)
                        if trace is not None:
                            trace.stop("forward")
                    else:
                        raise
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
                    _ssm_extra_keys = (
                        getattr(req, "_cache_extra_keys", None)
                        if _media_cache_allowed_for_ssm
                        else None
                    )
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
                                    if _media_cache_allowed_for_ssm:
                                        clean_media_cache = (
                                            self._prefill_for_clean_media_prefix_cache(
                                                req, list(all_tokens[:prompt_len])
                                            )
                                        )
                                        if clean_media_cache is not None:
                                            req._media_clean_prefix_cache = clean_media_cache  # type: ignore[attr-defined]
                                            kv_set = set(self._hybrid_kv_positions or [])
                                            clean_ssm_layers: List[Any] = []
                                            for layer_idx, c in enumerate(clean_media_cache):
                                                if layer_idx in kv_set:
                                                    continue
                                                if hasattr(c, "cache") and isinstance(c.cache, list):
                                                    from copy import deepcopy

                                                    cloned = deepcopy(c)
                                                    cloned.cache = [
                                                        mx.contiguous(a) if a is not None else None
                                                        for a in c.cache
                                                    ]
                                                    clean_ssm_layers.append(cloned)
                                                else:
                                                    clean_ssm_layers.append(c)
                                            if clean_ssm_layers:
                                                self._ssm_state_cache.store(
                                                    list(all_tokens[:prompt_len]),
                                                    prompt_len,
                                                    clean_ssm_layers,
                                                    is_complete=True,
                                                    cache_extra_keys=_ssm_extra_keys,
                                                )
                                                logger.info(
                                                    "MLLM media prefix cache: stored clean "
                                                    "media SSM companion for %s "
                                                    "(%d layers, %d-token key)",
                                                    req.request_id,
                                                    len(clean_ssm_layers),
                                                    prompt_len,
                                                )
                                        if clean_media_cache is not None:
                                            logger.info(
                                                "MLLM media prefix cache: clean media "
                                                "prefix cache prepared for %s (%d-token key)",
                                                req.request_id,
                                                prompt_len,
                                            )
                                        # Never queue text-only rederive for media prompts:
                                        # it would rebuild SSM without pixel/video embeddings.
                                        if clean_media_cache is not None:
                                            pass
                                        else:
                                            logger.info(
                                                "MLLM media prefix cache: no clean media "
                                                "prefix cache for %s; media row will remain "
                                                "KV-only/full-prefill on repeat",
                                                req.request_id,
                                            )
                                        continue
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
                        logger.debug(f"SSM state capture failed for {req.request_id}: {e}")
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
                    # Flush stale GPU state before retry
                    mx.clear_cache()
                    try:
                        from mlx_lm.models.cache import KVCache
                        try:
                            req_cache = self.language_model.make_cache()
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
                                self.block_aware_cache.clear()
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
                                if hasattr(self.language_model, 'make_cache'):
                                    req_cache = self.language_model.make_cache()
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

        if logits_processors:
            # Use _original_token_ids (saved before cache fetch trims input_ids)
            # so repetition penalty covers the full prompt, not just uncached tokens.
            prompt_list = getattr(request, '_original_token_ids', None)
            if prompt_list is None:
                ids = request.input_ids
                prompt_list = ids[0].tolist() if ids is not None and ids.ndim > 1 else (
                    ids.tolist() if ids is not None else []
                )

            def sampler_with_processors(logits, _req=request, _prompt_list=prompt_list):
                # Build full token sequence (prompt + generated) so penalty
                # applies to already-generated tokens, not just the prompt,
                # and MiMo can distinguish first-token EOS from natural stop.
                all_tokens = mx.array(_prompt_list + _req.output_tokens)
                processed = logits
                for proc in logits_processors:
                    processed = proc(all_tokens, processed)
                return base_sampler(processed)

            if _native_mtp_sampler_accepts_logits(base_sampler):
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
        if os.environ.get("VMLINUX_NATIVE_MTP", "1") in (
            "0",
            "false",
            "FALSE",
            "no",
            "NO",
            "off",
            "OFF",
        ):
            return "disabled by VMLINUX_NATIVE_MTP=0/--disable-native-mtp"
        if not _native_mtp_model_has_head(self.language_model):
            return "loaded language model has no native MTP head"
        if float(getattr(request, "temperature", 0.0) or 0.0) != 0.0:
            return f"temperature={getattr(request, 'temperature', None)!r} is not deterministic"
        if float(getattr(request, "repetition_penalty", 1.0) or 1.0) != 1.0:
            return f"repetition_penalty={getattr(request, 'repetition_penalty', None)!r} is not 1.0"
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
        return False

    def _step_native_mtp_head(
        self,
        request: MLLMBatchRequest,
        hidden_state: mx.array,
        next_token: mx.array,
        mtp_cache: List[Any],
        *,
        return_hidden: bool = False,
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Run one MTP-head prediction and return sampled token/logprobs/hidden."""
        sampler = self._make_request_sampler(request)
        try:
            mtp_output = self.language_model.mtp_forward(
                hidden_state,
                _native_mtp_ensure_uint32(next_token).reshape(1, 1),
                mtp_cache,
                return_hidden=return_hidden,
            )
        except TypeError:
            mtp_output = self.language_model.mtp_forward(
                hidden_state,
                _native_mtp_ensure_uint32(next_token).reshape(1, 1),
                mtp_cache,
            )
        if isinstance(mtp_output, tuple):
            mtp_logits, mtp_hidden = mtp_output
        elif hasattr(mtp_output, "logits") and hasattr(mtp_output, "hidden_states"):
            mtp_logits, mtp_hidden = mtp_output.logits, mtp_output.hidden_states
        else:
            mtp_logits, mtp_hidden = mtp_output, None
        draft_tok, draft_lp = _native_mtp_sample_one(mtp_logits[:, -1, :], sampler)
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
        drafts: List[mx.array] = []
        draft_lps: List[mx.array] = []
        current_hidden = hidden_state
        current_token = start_token
        for level in range(max(1, int(depth))):
            if stats is not None:
                stats.mtp_forwards += 1
            draft_tok, draft_lp, mtp_hidden = self._step_native_mtp_head(
                request,
                current_hidden,
                current_token,
                mtp_cache,
                return_hidden=level + 1 < depth,
            )
            drafts.append(draft_tok)
            draft_lps.append(draft_lp)
            current_token = draft_tok
            if mtp_hidden is None:
                current_hidden = current_hidden
            else:
                current_hidden = mtp_hidden[:, -1:, :]
        _native_mtp_async_eval(*drafts)
        if stats is not None:
            _native_mtp_trace_stop(stats, "draft_ms", trace_t0)
        return drafts, draft_lps, []

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

        sampler = self._make_request_sampler(request)
        seed_main_forwards = 1
        output = self.language_model(
            first_tok[:, None],
            cache=cache,
            return_hidden=True,
        )
        if isinstance(output, tuple):
            logits, hidden = output
        elif hasattr(output, "logits") and hasattr(output, "hidden_states"):
            logits, hidden = output.logits, output.hidden_states
        else:
            logger.debug("Native MTP seed skipped: model did not return hidden states")
            return False

        next_tok, next_lp = _native_mtp_sample_one(logits[:, -1, :], sampler)
        mtp_cache = self.language_model.make_mtp_cache()
        depth = _native_mtp_depth_for_request(request)
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
        )
        state.stats.seed_main_forwards += seed_main_forwards
        state.stats.mtp_forwards += len(drafts)
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
            "MLLM native MTP path activated for request=%s depth=%d",
            request.request_id,
            depth,
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
            hidden = output.hidden_states
        else:
            raise RuntimeError("native MTP replay did not return hidden states")
        return hidden[:, -1:, :]

    def _run_native_mtp_verify_cycle(
        self,
        request: MLLMBatchRequest,
        cache: List[Any],
        state: MLLMNativeMTPState,
    ) -> None:
        if state.next_main is None or not state.drafts:
            raise RuntimeError("native MTP verify entered without pending drafts")

        sampler = self._make_request_sampler(request)
        verify_inputs = [state.next_main] + list(state.drafts)
        trace_t0 = _native_mtp_trace_start()
        replay_snapshot = _native_mtp_snapshot_replay_cache(cache)
        _native_mtp_trace_stop(state.stats, "snapshot_ms", trace_t0)
        inputs = mx.concatenate(
            [_native_mtp_ensure_uint32(tok) for tok in verify_inputs]
        )
        trace_t0 = _native_mtp_trace_start()
        state.stats.verify_main_forwards += 1
        output = self.language_model(
            inputs[None, :],
            cache=cache,
            return_hidden=True,
        )
        if isinstance(output, tuple):
            logits, hidden = output
        elif hasattr(output, "logits") and hasattr(output, "hidden_states"):
            logits, hidden = output.logits, output.hidden_states
        else:
            raise RuntimeError("native MTP verify did not return hidden states")
        _native_mtp_trace_eval(logits, hidden)
        _native_mtp_trace_stop(state.stats, "verify_ms", trace_t0)

        depth = len(state.drafts)
        trace_t0 = _native_mtp_trace_start()
        target_tokens, target_lps, target_ids = _native_mtp_sample_rows(
            logits[:, -(depth + 1) :, :].reshape(depth + 1, -1),
            sampler,
        )
        _native_mtp_trace_stop(state.stats, "sample_ms", trace_t0)
        trace_t0 = _native_mtp_trace_start()
        _native_mtp_materialize_draft_ids(state)
        _native_mtp_trace_stop(state.stats, "materialize_ms", trace_t0)

        accepted = 0
        for idx, draft_id in enumerate(state.draft_ids):
            if int(target_ids[idx]) != int(draft_id):
                break
            accepted += 1

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
        _native_mtp_maybe_adapt_depth(request.request_id, state)
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
                state.mtp_cache = None
                state.drafts = []
                state.draft_lps = []
                state.draft_ids = []
                return
            state.mtp_cache = state.mtp_cache or self.language_model.make_mtp_cache()
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
        trace_t0 = _native_mtp_trace_start()
        if not _native_mtp_restore_replay_cache(
            cache,
            replay_snapshot,
            depth + 1,
        ):
            raise RuntimeError("native MTP cache rejected rollback")
        _native_mtp_trace_stop(state.stats, "restore_ms", trace_t0)
        accepted_drafts = state.drafts[:accepted]
        for draft_id, draft_lp in zip(state.draft_ids[:accepted], state.draft_lps[:accepted]):
            state.queue.append((draft_id, draft_lp, "draft"))
        correction = target_tokens[accepted]
        correction_id = int(target_ids[accepted])
        state.queue.append((correction_id, target_lps[accepted], "verify"))
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
            state.mtp_cache = None
            state.drafts = []
            state.draft_lps = []
            state.draft_ids = []
            return
        state.mtp_cache = self.language_model.make_mtp_cache()
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
        if not state.queue:
            raise RuntimeError("native MTP verify produced no emit token")
        token, logprobs, source = state.queue.popleft()
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

        # Wrap BatchKVCache with offset-safe proxies for VL models.
        # Several Qwen VL attention layers use cache.offset in slice ops that
        # require int, but BatchKVCache.offset is mx.array. The proxy converts it.
        # Always wrap: a batch that started with N>1 requests can filter down to 1
        # while cache remains BatchKVCache (offset still mx.array). The proxy is a
        # no-op when offset is already int (single-request path with raw KVCache).
        cache = _wrap_batch_caches(cache)

        trace = self._decode_trace
        model_t0 = time.perf_counter() if trace else 0.0

        # Run language model only (not full VLM). Qwen3.5/3.6 mRoPE language
        # models may leave `_rope_deltas` unset when text-only prefill used
        # explicit positions, so keep decode absolute too instead of relying on
        # module-level rope state.
        lm_kwargs: Dict[str, Any] = {"cache": cache}
        if _lm_supports_position_ids(self.language_model):
            position_ids = _absolute_text_position_ids(
                input_tokens,
                cache,
                self.language_model,
            )
            if position_ids is not None:
                lm_kwargs["position_ids"] = position_ids
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
                sampled = shared_sampler(logits)
            else:
                tokens = []
                for i, req in enumerate(batch.requests):
                    req_sampler = self._make_request_sampler(req)
                    tokens.append(req_sampler(logits[i:i+1]))
                sampled = mx.concatenate(tokens, axis=0)
        else:
            sampled = self.sampler(logits)

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
                    "batch=%d",
                    n,
                    (self._decode_trace_model_s / n) * 1000.0,
                    (self._decode_trace_sample_s / n) * 1000.0,
                    model_s * 1000.0,
                    sample_s * 1000.0,
                    int(logits.shape[0]),
                )

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
                    _submit_decode_token_eval(batch.y)
                    _native_mtp_log_stats(
                        batch.requests[0].request_id,
                        mtp_state.stats,
                        "fallback_to_ar",
                    )
                    self._stats.record_native_mtp(
                        request_id=batch.requests[0].request_id,
                        stats=mtp_state.stats,
                        finish_reason="fallback_to_ar",
                        final_depth=mtp_state.depth,
                        fallback_reason=mtp_state.ar_fallback_reason,
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
            _submit_decode_token_eval(batch.y)
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
            num_tok += 1
            batch.num_tokens[i] = num_tok
            req.num_tokens = num_tok
            req.output_tokens.append(token)

            finish_reason = None
            cache_fn = None

            if token in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(i)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(i)
            else:
                keep_idx.append(i)

            if finish_reason is not None:
                mtp_state_for_finish = getattr(req, "_native_mtp_state", None)
                if mtp_state_for_finish is not None:
                    _native_mtp_log_stats(
                        request_id,
                        mtp_state_for_finish.stats,
                        finish_reason,
                    )
                    self._stats.record_native_mtp(
                        request_id=request_id,
                        stats=mtp_state_for_finish.stats,
                        finish_reason=finish_reason,
                        final_depth=mtp_state_for_finish.depth,
                        fallback_reason=mtp_state_for_finish.ar_fallback_reason,
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
                    captured_cache = batch.extract_cache(i)
                cache_fn = lambda c=captured_cache: c

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
                    cache_extra_keys=getattr(req, '_cache_extra_keys', None),
                    gen_prefix_tokens=getattr(req, '_gen_prefix_tokens', None),
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
                if self._tight_memory_prefill_drain:
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

    def _prefill_for_clean_path_dependent_cache(
        self, tokens: List[int]
    ) -> Optional[List[Any]]:
        """Run a clean prompt-only prefill matching a path-dependent cache key.

        Mirrors Scheduler._prefill_for_prompt_only_cache for the MLLM path.
        Returned cache covers exactly `tokens` worth of processing — no
        gen_prompt_len suffix, no generation output. Safe to store with
        is_complete=True.

        SSM re-derive requires contiguous state math across the full prompt.
        Chunking the forward pass broke on the 2nd chunk for fresh
        ``make_cache()`` output because ArraysCache's offset/mask machinery
        (``lengths``/``left_padding``) is only populated when the cache goes
        through ``BatchKVCache`` wrappers. Prefer one-shot when the attention
        buffer fits under the Metal single-buffer cap; skip gracefully
        otherwise (the live prefill's SSM stash still serves as a
        possibly-contaminated companion for thinking-model prompts).
        """
        if not tokens or self.language_model is None:
            return None
        seq_len = len(tokens)
        _OOM_GUARD_BYTES = 8 * 1024 * 1024 * 1024
        _n_heads_guess = _infer_attention_heads_for_hybrid_oom_guard(
            self.language_model
        )
        _predicted_attn_bytes = _n_heads_guess * seq_len * seq_len * 2
        if _predicted_attn_bytes > _OOM_GUARD_BYTES:
            logger.info(
                "MLLM SSM re-derive: skipping clean prefill for %d-token prompt "
                "(predicted attention buffer %.1f GB exceeds Metal single-buffer "
                "limit; re-derive requires contiguous state math that chunking "
                "breaks). Live prefill's SSM stash will be used as the companion.",
                seq_len, _predicted_attn_bytes / (1024**3),
            )
            return None
        try:
            fresh_cache = (
                self.language_model.make_cache()
                if hasattr(self.language_model, "make_cache")
                else None
            )
            if fresh_cache is None:
                from mlx_lm.models.cache import KVCache
                fresh_cache = [KVCache() for _ in range(len(self.language_model.layers))]
            input_ids = mx.array([tokens])
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
            _ = self.language_model(input_ids, cache=fresh_cache)
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
            return fresh_cache
        except Exception as ex:
            logger.warning(f"MLLM clean SSM prefill failed (non-fatal): {ex}")
            return None
        finally:
            for _attr, _value in locals().get("_saved_pos_state", {}).items():
                try:
                    setattr(self.language_model, _attr, _value)
                except Exception:
                    pass

    def _prefill_for_clean_ssm(self, tokens: List[int]) -> Optional[List[Any]]:
        """Compatibility alias for hybrid SSM callers."""
        return self._prefill_for_clean_path_dependent_cache(tokens)

    def _prefill_for_clean_media_prefix_cache(
        self,
        request: "MLLMBatchRequest",
        tokens: List[int],
    ) -> Optional[List[Any]]:
        """Run a media-conditioned clean prefill for a media prefix key.

        Unlike `_prefill_for_clean_path_dependent_cache`, this keeps the VLM
        wrapper and pixel/video tensors in the forward path. It is used only
        for explicit media-prefix-cache experiments because it adds an extra
        prefill on the first request but is the minimal safe way to create SSM
        state at the same media-keyed N-1 boundary as the paged KV blocks.
        """
        if not tokens or self.language_model is None:
            return None
        try:
            fresh_cache = (
                self.language_model.make_cache()
                if hasattr(self.language_model, "make_cache")
                else None
            )
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
            _ = self._run_vision_encoding(clean_req, cache=fresh_cache)
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
            return fresh_cache
        except Exception as ex:
            logger.warning(
                "MLLM clean media prefix prefill failed for %s (non-fatal): %s",
                getattr(request, "request_id", "?"),
                ex,
            )
            return None

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
        logger.info(
            f"MLLM SSM re-derive: clean prefill for {orig_rid} "
            f"({prompt_len} prompt tokens, {len(self._ssm_rederive_queue)} remaining)"
        )
        try:
            clean_cache = self._prefill_for_clean_ssm(list(tokens))
            if clean_cache is None:
                return True
            kv_set = set(self._hybrid_kv_positions or [])
            ssm_layers: List[Any] = []
            for layer_idx, c in enumerate(clean_cache):
                if layer_idx in kv_set:
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
