# SPDX-License-Identifier: Apache-2.0
# Base architecture from waybarrios/vllm-mlx. MLLM VLM scheduling, MLA detection,
# hybrid SSM cache handling, and vision cache added by Jinho Jang (eric@jangq.ai)
# for vMLX (github.com/jjang-ai/vmlx).
"""
MLLM Scheduler -- Multimodal Language Model continuous batching on Apple Metal.

This is the central orchestrator for all multimodal inference in vmlx-engine.
It manages request lifecycle, cache infrastructure, and Metal GPU memory for
Vision Language Models (VLMs) like Qwen3-VL, Qwen3.5-VL, LLaVA, InternVL, etc.

REQUEST LIFECYCLE
-----------------
::

    add_request()       _schedule()          step()          _cleanup_finished()
    ------------> WAITING ----------> RUNNING ------> FINISHED ----------------->
                  (deque)            (dict)    (tokens)    (cache store + cleanup)

1. Requests arrive via add_request() -> waiting deque (FCFS ordering)
2. _schedule() moves requests from waiting -> running (via MLLMBatchGenerator)
3. step() generates one token per step for ALL running requests simultaneously
4. _process_batch_responses() extracts tokens, detokenizes, checks stop
5. _cleanup_finished() stores cache, frees memory, removes request state
6. Async streaming: output_queues deliver RequestOutput per token

CACHE ARCHITECTURE (3-Tier Exclusive Selection)
-----------------------------------------------
The scheduler selects ONE in-memory cache mode at init time, with optional
disk L2 backing. This matches the LLM scheduler's cache design exactly.

**Tier 1 -- In-Memory (mutually exclusive):**

- **PAGED CACHE** (default, recommended):
  PagedCacheManager + BlockAwarePrefixCache.
  Fixed-size blocks (default 64 tokens), O(1) block-level matching.
  Supports BlockDiskStore L2 for persistence.
  Required for hybrid models (auto-switches from memory-aware).
  Config: ``use_paged_cache=True, paged_cache_block_size=64``

- **MEMORY-AWARE CACHE** (good for large models):
  MemoryAwarePrefixCache.
  Auto-sizes to fraction of available RAM with TTL-based expiration
  and LRU eviction. Monitors memory pressure via ``mx.metal.device_info()``.
  Config: ``use_paged_cache=False, use_memory_aware_cache=True``

- **LEGACY PREFIX CACHE** (simple, entry-count based):
  PrefixCacheManager.
  Fixed max_entries, no memory awareness. Trie-based token prefix matching.
  Config: ``use_paged_cache=False, use_memory_aware_cache=False``

**Tier 2 -- Disk (optional, additive to any Tier 1):**

- **DISK CACHE L2** (non-paged paths):
  DiskCacheManager -- serializes KV cache to ``~/.cache/vmlx-engine/``.
  Config: ``enable_disk_cache=True``

- **BLOCK DISK STORE** (paged path only):
  BlockDiskStore -- persists paged blocks to disk.
  Config: ``enable_block_disk_cache=True``

**Cross-Cutting: KV Cache Quantization:**

Storage-boundary quantization for 2-4x memory savings.
Full-precision KVCache during generation -> quantize on store ->
dequantize on fetch. Never modifies model.make_cache().
Config: ``kv_cache_quantization="q4"|"q8", kv_cache_group_size=64``

HYBRID MODEL SUPPORT (SSM + Attention)
---------------------------------------
Models like Qwen3.5-VL have mixed layers: some use KVCache (attention),
others use MambaCache/ArraysCache (SSM/linear attention). This creates a
cache asymmetry problem:

- KVCache layers CAN be prefix-cached (position-independent)
- SSM layers CANNOT -- their state is cumulative and path-dependent

The scheduler handles this via:

1. ``_is_hybrid_model()`` -- detects non-KVCache layers at init
2. Auto-switches to paged cache (memory-aware can't truncate SSM state)
3. ``HybridSSMStateCache`` (companion cache in MLLMBatchGenerator):
   After prefill, captures SSM layer states keyed by prompt tokens.
   On paged cache HIT + SSM companion HIT -> full skip (KV + SSM).
   On paged cache HIT + SSM MISS -> forced full prefill.
4. ``_fix_hybrid_cache()`` -- expands reconstructed KV-only cache back to
   full layer count by inserting fresh ArraysCache at SSM positions
5. ``ensure_mamba_support()`` -- patches mlx-lm's BatchGenerator for
   MambaCache batching (merge, extract, filter)

METAL GPU MEMORY MANAGEMENT
----------------------------
- Metal allocator cache limit: 25% of max working set (floor 512MB).
  Prevents Metal from hoarding freed memory in its allocator free-list,
  leaving more memory available for prefix cache and OS.
- Periodic GC: ``clear_mlx_memory_cache()`` every 60s during sustained traffic
  (via ``_last_metal_gc_time`` timer in ``step()``)
- Idle GC: ``clear_mlx_memory_cache()`` when all requests finish
  (in ``_cleanup_finished`` when ``self.running`` is empty)
- Wired limit: set to ``max_recommended_working_set_size`` at init
- ``mx.async_eval()`` in prefill loop for GPU/CPU overlap

CACHE STORE FLOW (in _cleanup_finished)
-----------------------------------------
When a request finishes, its KV cache is stored for future reuse::

    request._extracted_cache (set by _process_batch_responses)
         |
         +-- Paged: block_aware_cache.store_cache(tokens, states)
         |     +-- Optional: disk L2 store + quantization
         |
         +-- Memory-aware: memory_aware_cache.store(tokens, cache)
         |     +-- Optional: disk L2 store + quantization
         |
         +-- Legacy: prefix_cache.store_cache(tokens, cache)
               +-- Optional: disk L2 store + quantization

Each path uses ``_truncate_hybrid_cache()`` to trim generation tokens,
``_quantize_cache_for_storage()`` if ``kv_cache_bits > 0``, and always
sets ``_extracted_cache = None`` in a finally block to free tensor refs.

CACHE FETCH FLOW (in MLLMBatchGenerator._process_prompts)
----------------------------------------------------------
Before prefill, each request checks for cached KV state::

    1. Paged cache -> block_aware_cache.fetch_cache(request_id, tokens)
       +-- Pure attention model: skip cached prefix tokens
       +-- Hybrid + SSM companion HIT: full skip (KV + SSM)
       +-- Hybrid + SSM MISS: no skip (full prefill needed)
    2. Memory-aware/Legacy -> cache_obj.fetch(tokens)
       +-- Same image-token and hybrid guards
    3. Disk L2 fallback -> disk_cache.fetch(tokens)
       +-- Only for non-hybrid models

KEY CLASSES
-----------
- ``MLLMSchedulerConfig`` -- All cache + scheduling settings (dataclass)
- ``MLLMRequest`` -- Per-request state (prompt, media, tokens, status)
- ``MLLMSchedulerOutput`` -- Output from one step() call
- ``MLLMScheduler`` -- Main scheduler class
"""

from .persistence_outcome import LEDGER as _PERSIST, format_outcome as _format_persistence_outcome


def _record_last_durability(stats, request_id, wait_ms, waited, outcome):
    """Retain the terminal durability fence of the last completed generation
    on the batch stats (published as ``last_durability`` in /health): the
    request id, the wait the client saw, and the persistence ledger's outcome
    (stored / already_durable / skipped / refused / failed / unknown) with its
    retained token count and detail."""
    if stats is None:
        return
    entry = outcome if isinstance(outcome, dict) else {}
    record = {
        "request_id": str(request_id),
        "wait_ms": round(float(wait_ms), 3),
        "waited": bool(waited),
        "cache_outcome": str(entry.get("outcome") or "unknown"),
        "detail": str(entry.get("detail") or "")[:200],
        "retained_tokens": entry.get("retained_tokens"),
        "durable": entry.get("durable"),
        "at": time.time(),
    }
    try:
        stats.last_durability = record
    except Exception:
        pass
from .video_controls import video_controls_from_kwargs as _video_controls_from_kwargs
import asyncio
import hashlib
import logging
import os
import time
import uuid
import threading

import mlx.core as mx
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple


from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

from .utils.ssm_companion_cache import DEFAULT_SSM_COMPANION_ENTRIES

from .mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
    _mllm_media_cache_extra_keys,
    _mllm_media_prefix_cache_family_enabled,
    _model_uses_zaya_cache_contract,
)
from .errors import PromptTooLongError
from .mlx_memory import clear_mlx_memory_cache
from .request import RequestOutput, RequestStatus, SamplingParams
from .utils.head_dim_detection import (
    choose_supported_kv_group_size,
    detect_cache_head_dims,
)
from .utils.hybrid_tq_cache import is_turboquant_make_cache
from .utils.cache_types import (
    detect_dsv4_cache_contract,
    describe_runtime_cache_layout,
    expand_cache_class_names,
)
from .utils.ssm_companion_disk_store import SSMCompanionDiskStore
from .utils.memory_limits import (
    get_effective_metal_working_set_bytes,
    get_metal_ws_guard_threshold,
)
from .prefix_cache import runtime_cache_fingerprint
from .utils.cache_extent import cache_offset, logical_truncate_target

logger = logging.getLogger(__name__)


def _finalize_detokenizer_delta(detokenizer) -> str:
    """Finalize a streaming detokenizer and return its newly readable tail.

    BPE detokenizers may retain a trailing byte sequence until ``finalize()``.
    The complete ``text`` property includes that tail afterwards, but streaming
    consumers only receive ``RequestOutput.new_text``. Reading
    ``last_segment`` after finalization advances the same incremental offset
    used for ordinary token deltas and prevents the final character from being
    present only on the non-streaming surface.
    """
    detokenizer.finalize()
    return detokenizer.last_segment

_PROMOTION_ENABLE_VALUES = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_PROMOTION_DISABLE_VALUES = {"0", "false", "FALSE", "no", "NO", "off", "OFF"}


def _hybrid_clean_store_enabled() -> bool:
    """Let hybrid models extend the chain via the CLEAN re-prefill store.

    Distinct from promotion below, and the difference is the whole point:
    promotion writes back the RESTORED cache (reconstructed attention KV plus
    path-dependent SSM), which was measured to break the model on the first
    extended turn. The clean route instead re-prefills exactly the N-1 cache key
    and stores that typed state — the same treatment ZAYA CCA and Gemma
    mixed-SWA already get for being path-dependent. Hybrid was simply never
    wired into it, so it fell through to a blanket skip and never grew its
    prefix past turn one.

    ON by default. It shipped off only because the route crashed before it could
    store (a rebuilt attention-only base was handed to a linear layer), and the
    resulting half-written state is what made the model collapse on its first
    extended turn — the damage was the broken store, not the restore. With the
    crash fixed, an 8-turn matrix that varies reasoning effort, tools appearing
    and disappearing, and thinking toggled off returned byte-identical text on
    Qwen3.8 across arms whose hit counts differed wildly (0 vs 80, 320 vs 946),
    while reuse climbed to 96% instead of freezing at turn one.

    Set VMLX_HYBRID_CLEAN_STORE=0 to fall back to the old freeze-at-turn-1
    behaviour. Note that on the families where a cache HIT already changes the
    answer (nemotron_h and friends), this makes hits more frequent; it does not
    make them less exact — a cold-vs-warm control with this route disabled on
    both arms drifts by exactly the same 8/9 turns.
    """
    for name in ("VMLX_HYBRID_CLEAN_STORE", "VMLINUX_HYBRID_CLEAN_STORE"):
        value = os.environ.get(name, "")
        if value in _PROMOTION_DISABLE_VALUES:
            return False
        if value in _PROMOTION_ENABLE_VALUES:
            return True
    return True


def _hybrid_prefix_promotion_enabled() -> bool:
    """Opt in to extending a RESTORED hybrid prefix instead of only cold ones.

    Off by default: a hybrid restored prefix pairs attention KV with
    path-dependent SSM state, and promoting it repeatedly is what made
    Bonsai/Qwen3.5 collapse into a token loop. Both spellings are accepted so a
    second name cannot make the switch a silent no-op the way VMLX_NATIVE_MTP
    once did.
    """
    return any(
        os.environ.get(name, "") in _PROMOTION_ENABLE_VALUES
        for name in ("VMLX_HYBRID_PREFIX_PROMOTION", "VMLINUX_HYBRID_PREFIX_PROMOTION")
    )


def _mllm_scheduler_trace_enabled() -> bool:
    return os.environ.get("VMLINUX_MLLM_SCHEDULER_TRACE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Shared with the text scheduler's SchedulerConfig / the --max-cache-blocks CLI
# default. The text path pairs it with a 256-token block; the MLLM path uses 64.
DEFAULT_MAX_CACHE_BLOCKS = 1000
TEXT_REFERENCE_BLOCK_SIZE = 256


def resolve_mllm_index_blocks(
    block_size: int,
    max_cache_blocks: int,
    *,
    default_max_blocks: int = DEFAULT_MAX_CACHE_BLOCKS,
    reference_block_size: int = TEXT_REFERENCE_BLOCK_SIZE,
) -> int:
    """Give the MLLM block index the same TOKEN capacity as the text default.

    ``--max-cache-blocks`` counts BLOCKS, and the MLLM path uses a 64-token
    block where the text path uses 256. The shared default of 1000 therefore
    indexes 256k tokens for text but only 64k for MLLM, silently capping reuse
    far below the context window of models like Gemma 4. Measured: a 77k-token
    Gemma prompt reported 0 cached tokens on an exact repeat and ran SLOWER
    than a cold prefill, while the same probe at 28k reused 28,199 tokens and
    cut TTFT from 9.10s to 0.98s.

    Only the untouched default is rescaled — an operator who passed an explicit
    ``--max-cache-blocks`` keeps exactly what they asked for. This bounds the
    INDEX only; resident RAM stays governed by ``max_resident_bytes``.
    """
    if max_cache_blocks != default_max_blocks:
        return max_cache_blocks
    if block_size <= 0 or block_size >= reference_block_size:
        return max_cache_blocks
    return max_cache_blocks * (reference_block_size // block_size)



def _resolve_prefix_cache_byte_budget(config: Any) -> Optional[int]:
    """Byte ceiling for the legacy prefix cache, derived when not set explicitly.

    An entry-count bound alone does not bound MEMORY: entries on this path are
    whole-KV snapshots whose size grows with context, so 100 entries at 90k
    context is hundreds of GB. An explicit --prefix-cache-max-bytes wins;
    otherwise fall back to the same RAM fraction the memory-aware cache uses
    (--cache-memory-mb / --cache-memory-percent), because that is already the
    user's stated answer to "how much RAM may caching take".

    Returns None only when nothing at all is configured, preserving the old
    unbounded behaviour rather than inventing a limit.
    """
    explicit = getattr(config, "prefix_cache_max_bytes", None)
    if explicit:
        return int(explicit)
    mb = getattr(config, "cache_memory_mb", None)
    if mb:
        return int(mb) * 1024 * 1024
    percent = getattr(config, "cache_memory_percent", None)
    if not percent or percent <= 0:
        return None
    try:
        import psutil

        available = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        return None
    if available <= 0:
        return None
    return max(1024**3, int(available * float(percent)))


@dataclass
class MLLMSchedulerConfig:
    """Configuration for MLLM scheduler.

    All fields mirror the LLM SchedulerConfig for full cache parity.
    The 3-tier cache selection is mutually exclusive:

    - ``use_paged_cache=True`` -> PagedCacheManager + BlockAwarePrefixCache
    - ``use_memory_aware_cache=True`` (and paged off) -> MemoryAwarePrefixCache
    - Both off -> PrefixCacheManager (legacy entry-count based)

    Disk L2 caches are additive on top of any tier.
    KV cache quantization applies at the storage boundary (not during generation).
    """

    # Maximum concurrent MLLM requests in the batch
    max_num_seqs: int = 1
    # Prefill batch size (all queued requests are prefilled together)
    prefill_batch_size: int = 512
    # Completion batch size
    completion_batch_size: int = 512
    # Prefill step size for chunked prefill
    prefill_step_size: int = 2048
    # Processor-output pixel/video tensor LRU. Disabled by default so the
    # SSD-only product contract has no hidden per-media retained MLX arrays.
    enable_vision_cache: bool = False
    # Maximum cache entries
    vision_cache_size: int = 16
    # Default max tokens
    default_max_tokens: int = 256
    # Default video FPS for frame extraction
    default_video_fps: float = 2.0
    # Maximum video frames
    max_video_frames: int = 128

    # Prefix/Paged cache settings
    enable_prefix_cache: bool = True
    # Eric mandatory cache policy: default paged cache OFF for all families.
    # Typed cache families opt into paged only when their native cache requires
    # it (e.g. ZAYA/CCA or hybrid Mamba). MM3 and Gemma mixed-SWA stay paged-off
    # on the current native memory-aware/prompt-L2 path.
    use_paged_cache: bool = False
    paged_cache_block_size: int = 64
    max_cache_blocks: int = DEFAULT_MAX_CACHE_BLOCKS

    # KV cache quantization for prefix cache storage
    kv_cache_quantization: str = "none"  # "none", "q4", "q8"
    kv_cache_group_size: int = 64
    kv_cache_quantization_explicit: bool = False

    # Memory-aware cache settings (L1, recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: Optional[int] = None  # None = auto-detect (cache_memory_percent of RAM)
    cache_memory_percent: float = 0.15  # Fraction of available RAM if auto-detecting (matches CLI + app)
    cache_ttl_minutes: float = 0  # Cache entry TTL in minutes (0 = no expiration)

    # Legacy entry-count prefix cache (fallback when paged+memory-aware both off)
    prefix_cache_size: int = 100
    # An entry COUNT does not bound memory — entries here are whole-KV
    # snapshots that grow with context. _resolve_prefix_cache_byte_budget()
    # already reads this field, but MLLMSchedulerConfig never declared it, so
    # --prefix-cache-max-bytes was silently ignored on every VL/multimodal
    # session and the budget fell back to the RAM-percent default.
    prefix_cache_max_bytes: Optional[int] = None

    # Disk cache L2 (persistent across restarts, non-paged path)
    enable_disk_cache: bool = False
    disk_cache_dir: Optional[str] = None  # None = ~/.cache/vmlx-engine/prompt-cache/<hash>
    disk_cache_max_gb: float = 10.0

    # Block-level disk cache L2 (persistent, paged path)
    enable_block_disk_cache: bool = False
    block_disk_cache_dir: Optional[str] = None  # None = ~/.cache/vmlx-engine/block-cache/<hash>
    block_disk_cache_max_gb: float = 10.0

    # Model path (used to scope disk cache per model)
    model_path: Optional[str] = None

    # Hybrid SSM state cache budget (companion cache for SSM layer states).
    # These entries can be tens or hundreds of MB each on Nemotron/Gemma
    # hybrid models, so bound by both count and bytes.
    ssm_state_cache_size: int = DEFAULT_SSM_COMPANION_ENTRIES
    ssm_state_cache_max_mb: Optional[int] = 0

    # Maximum images per request (guard against Metal OOM from excessive images)
    max_images_per_request: int = 20

    # Optional pre-built single-worker executor for model ops. When the
    # caller (BatchedEngine._start_mllm) loads the model on a dedicated
    # executor, it should pass that same executor here so step()/prefill
    # share one thread with the load. See MLLMScheduler.__init__ for the
    # JANGTQ Metal kernel stream-isolation rationale. None -> scheduler
    # creates its own (load runs on a different thread, JANGTQ-VL
    # bundles will hit the stream issue).
    step_executor: Any = None


@dataclass
class MLLMRequest:
    """
    Extended request for MLLM processing.

    Includes all multimodal data needed for generation.
    """

    request_id: str
    prompt: str
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    audio: Optional[List[Any]] = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.time)

    # Batch generator UID (assigned when scheduled)
    batch_uid: Optional[int] = None

    # Status tracking
    status: RequestStatus = RequestStatus.WAITING
    output_text: str = ""
    output_tokens: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None

    # Token counts
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    # Lifetime generated-token count, NEVER cleared on a recovery retry.
    # `num_output_tokens` is derived from `len(output_tokens)`, and the retry
    # path clears that list, so anything built on it goes BACKWARDS mid-request.
    # `request_progress` is consumed as a liveness signal by a timeout that
    # credits only `progress > last_progress`, so a counter that drops makes a
    # healthy retrying request look wedged. Mirrors `Request.total_output_tokens`
    # in the text scheduler.
    total_output_tokens: int = 0
    # Lifetime total as of the last retry, so the running total can be rebuilt
    # from a cleared `output_tokens` list.
    _retry_output_base: int = 0

    # Video processing parameters (per-request overrides)
    image_token_budget: Optional[int] = None
    # Per-request image controls (ImageControls) and the strict flag
    image_controls: Optional[Any] = None
    media_controls_strict: bool = False
    video_fps: Optional[float] = None
    video_max_frames: Optional[int] = None
    # Normalized per-request video controls (fps, frame cap, pixel budgets,
    # explicit size); video_fps/video_max_frames above stay for callers that
    # only know those two.
    video_controls: Optional[Any] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Error recovery
    _retry_count: int = 0


@dataclass
class MLLMSchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: List[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: Set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: List[RequestOutput] = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False
    # Optional gated diagnostic timings from scheduler.step().
    trace_timings: Dict[str, float] = field(default_factory=dict)


class MLLMScheduler:
    """Scheduler for Vision Language Model requests with continuous batching.

    This is the main entry point for multimodal inference. It owns:

    - **Request queues**: waiting (deque) and running (dict) with thread-safe lock
    - **Cache infrastructure**: one of paged/memory-aware/legacy + optional disk L2
    - **Batch generator**: MLLMBatchGenerator (lazy, recreated on sampler change)
    - **Metal memory**: GC timer, wired limit, cache limit
    - **Streaming**: per-request output queues with detokenizer pool

    Thread safety: _queue_lock (RLock) protects waiting/running mutations.
    step() runs in a background thread; add_request_async() from the event loop.

    Cache init priority (in __init__):
      1. Detect hybrid model -> auto-switch to paged if needed
      2. Paged cache (with optional BlockDiskStore)
      3. Memory-aware cache (elif)
      4. Legacy prefix cache (elif)
      5. Disk cache L2 (additive, non-paged paths)
      6. KV cache quantization setup

    Key methods:
      - ``add_request()`` / ``add_request_async()`` -- enqueue a VLM request
      - ``step()`` -- one generation step (schedule + generate + cleanup)
      - ``stream_outputs()`` -- async generator for streaming tokens
      - ``abort_request()`` -- cancel a running/waiting request
      - ``get_stats()`` -- cache hit rates, throughput, memory usage
      - ``reset()`` -- clear all state and caches

    Example (sync)::

        scheduler = MLLMScheduler(model, processor, config)
        request_id = scheduler.add_request(
            prompt="What's in this image?",
            images=["photo.jpg"]
        )
        while scheduler.has_requests():
            output = scheduler.step()
            for req_output in output.outputs:
                if req_output.finished:
                    print(f"Finished: {req_output.output_text}")

    Example (async streaming)::

        await scheduler.start()
        request_id = await scheduler.add_request_async(...)
        async for output in scheduler.stream_outputs(request_id):
            print(output.new_text, end="")
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: Optional[MLLMSchedulerConfig] = None,
    ):
        """
        Initialize MLLM scheduler with full cache infrastructure.

        Init sequence:
        1. Detect hybrid model (SSM + attention) -> auto-switch to paged
        2. Initialize cache tier (paged > memory-aware > legacy, exclusive)
        3. Initialize disk L2 (additive, if enabled)
        4. Configure KV cache quantization (storage boundary)
        5. Set up request queues, UID mappings, output queues
        6. Start Metal GC timer for sustained-traffic memory management

        Args:
            model: The VLM model
            processor: The VLM processor
            config: Scheduler configuration
        """
        self.model = model
        self.processor = processor
        self.config = config or MLLMSchedulerConfig()

        # Thread-safe lock for wait/run queues since MLLM step() runs in background thread
        self._queue_lock = threading.RLock()
        # Separate lock for batch generator next()/remove() to prevent abort race
        self._batch_lock = threading.RLock()

        # Dedicated single-worker executor for ALL model-touching code
        # (prefill, decode, cache extract). MLX streams are thread-local —
        # asyncio.to_thread dispatches to a multi-worker pool where each
        # call may land on a different thread (asyncio_0, asyncio_1, ...),
        # so Stream(gpu, 1) created during model load on MainThread becomes
        # invisible. Pinning every step() to the same worker thread (and
        # loading the model on that worker) keeps the stream registry
        # consistent and lets JANGTQ Metal kernels resolve their dispatch
        # streams cleanly.
        #
        # The caller (BatchedEngine._start_mllm) may pass a pre-built
        # executor that ALSO ran the model load — using that same executor
        # here ensures load + every subsequent step() share one worker
        # thread. If no executor is provided, a fresh single-worker pool
        # is created (load then runs on a different thread, and JANGTQ-VL
        # bundles will hit the stream issue — pre-existing behavior).
        from concurrent.futures import ThreadPoolExecutor
        _provided = config.step_executor if config is not None else None
        if _provided is not None:
            self._step_executor = _provided
        else:
            self._step_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mllm-worker"
            )

        # Get model config
        self.model_config = getattr(model, "config", None)

        # Token-level Prefix caching for the language model
        self.paged_cache_manager = None
        self.block_aware_cache = None
        self.memory_aware_cache = None
        self.prefix_cache = None
        self.disk_cache = None
        self._block_disk_l2_enabled = False
        self._ssm_companion_disk_store = None
        self._ssm_companion_model_key = ""
        self._kv_cache_bits = 0
        self._kv_cache_group_size = 64
        self._tq_active = False
        self._tq_decoder_warmup_stats: Optional[Dict[str, Any]] = None
        self._hybrid_live_tq_policy = None
        self._hybrid_live_tq_attention_layers = []
        self._hybrid_live_tq_companion_layers = []
        self._hybrid_live_tq_compress_after = 0
        self._hybrid_tq_auto_policy = None
        self._hybrid_tq_default_key_bits = 0
        self._hybrid_tq_default_value_bits = 0

        # Detect hybrid models (mixed KVCache + MambaCache layers)
        lang_model = self.model.language_model if hasattr(self.model, "language_model") else self.model
        # This is an observed runtime contract, not a family/config guess.
        # False proves the DSV4 terminal validator can be skipped; None keeps
        # BlockAwarePrefixCache fail-closed when make_cache() is unavailable.
        self._uses_dsv4_cache = detect_dsv4_cache_contract(lang_model)
        if self._uses_dsv4_cache is None:
            logger.warning(
                "MLLM DSV4 cache contract could not be observed from "
                "language_model.make_cache(); retaining conservative SSD "
                "terminal validation"
            )
        # ZAYA/ZAYA1-VL are hybrid-shaped but have a first-class typed CCA
        # cache contract (KV + conv_state + prev_hs). They must not enter
        # the generic Qwen-style SSM companion path.
        self._uses_zaya_cache = (
            _model_uses_zaya_cache_contract(lang_model)
            or _model_uses_zaya_cache_contract(self.model)
        )
        self._is_hybrid = False
        try:
            self._is_hybrid = self._is_hybrid_model(lang_model)
        except Exception as e:
            logger.warning(f"Failed to detect hybrid cache model: {e}")

        # Detect native TQ before cache-directory construction. Auto and an
        # explicit None setting can otherwise hash to the same ``quant=none``
        # directory (Auto is represented by an omitted q4/q8 flag), allowing
        # one persisted representation to shadow the other after a UI restart.
        self._tq_active = self._detect_turboquant_make_cache()
        self._capture_hybrid_turboquant_policy()

        # Detect mixed-attention models (e.g. Gemma 4 = 25 sliding + 5 full).
        # Detection is diagnostic only. The cache path must preserve
        # RotatingKVCache metadata instead of bypassing all prefix tiers.
        self._mixed_attention_cache_model = False
        try:
            self._mixed_attention_cache_model = self._model_has_mixed_attention(lang_model)
            if self._mixed_attention_cache_model:
                logger.info(
                    "VLM mixed-attention model detected (e.g. Gemma 4 sliding+full). "
                    "Prefix cache remains enabled; RotatingKVCache metadata will be "
                    "preserved during truncation and paged/L2 reconstruction."
                )
        except Exception as e:
            logger.debug(f"Mixed-attention detection failed: {e}")

        if self._uses_zaya_cache and self.config.enable_prefix_cache:
            # In-RAM paged cache is OFF for every family; SSD block-disk L2 is
            # the only tier. ZAYA used to escalate to paged RAM here. It no
            # longer does: the memory-aware lane cannot hold CCA
            # conv_state/prev_hs, so that lane is dropped and ZAYA re-prefills
            # cleanly. The fetch side already refuses a ZAYA chain with no
            # terminal CCA state, so this costs reuse, never correctness.
            if not self.config.use_paged_cache:
                logger.info(
                    "ZAYA/CCA typed cache cannot be served by the VLM "
                    "memory-aware prefix lane. Paged RAM stays OFF (SSD L2 is "
                    "the only tier); disabling the memory-aware lane."
                )
            self.config.use_memory_aware_cache = False

        if self._is_hybrid and not self._uses_zaya_cache:
            try:
                from .utils.mamba_cache import ensure_mamba_support
                ensure_mamba_support()
                logger.info(
                    "VLM hybrid model detected (MambaCache + KVCache layers). "
                    "Mamba batching support enabled."
                )
            except Exception as e:
                logger.warning(f"Failed to enable Mamba batching support: {e}")
                self._is_hybrid = False  # Fall back to standard KV-only handling
            # MambaCache cannot use the legacy memory-aware cache, and paged RAM
            # is OFF for every family. With Block Disk L2 also switched off there
            # is no correct backend for hybrid state, so drop the memory-aware
            # lane and take the reuse loss rather than reuse it incorrectly.
            if (
                self._is_hybrid
                and self.config.enable_prefix_cache
                and not self.config.use_paged_cache
                and not self.config.enable_block_disk_cache
            ):
                logger.info(
                    "Hybrid VLM with Block Disk L2 disabled: paged RAM stays "
                    "OFF (SSD L2 is the only tier); disabling the memory-aware "
                    "lane. Enable --enable-block-disk-cache for hybrid reuse."
                )
                self.config.use_memory_aware_cache = False

        # --- Cache initialization chain (block-aware > memory-aware > legacy) ---
        if self.config.enable_prefix_cache:
            if self.config.use_paged_cache or self.config.enable_block_disk_cache:
                # Paged RAM with optional L2, or authoritative disk-only blocks.
                block_disk_only = bool(
                    self.config.enable_block_disk_cache
                    and not self.config.use_paged_cache
                )
                block_disk_store = None
                if self.config.enable_block_disk_cache:
                    _default_block_cache_root = (
                        self.config.block_disk_cache_dir is None
                    )
                    cache_root = os.path.abspath(
                        os.path.expanduser(
                            self.config.block_disk_cache_dir
                            or os.path.join(
                                "~", ".cache", "vmlx-engine", "block-cache"
                            )
                        )
                    )
                    if self.config.model_path:
                        # Include quant config and paged-cache schema in hash
                        # to prevent cross-config / stale L2 cache poisoning.
                        quant_tag = self.config.kv_cache_quantization or "none"
                        tq_native_tag = (
                            "on"
                            f"-k{self._hybrid_tq_default_key_bits or 0}"
                            f"-v{self._hybrid_tq_default_value_bits or 0}"
                            f"-after{self._hybrid_live_tq_compress_after or 0}"
                            f"-policy{self._hybrid_tq_auto_policy or 'bundle'}"
                            if self._tq_active
                            else "off"
                        )
                        from .prefix_cache import build_block_cache_namespace
                        # ONE recipe, shared with the text scheduler. This used
                        # to be a second, weaker copy: no bundle= weight
                        # fingerprint, no zaya scope, no looped identity — so an
                        # in-place VLM bundle swap kept the same namespace and
                        # replayed the old weights' KV.
                        block_scope_key = build_block_cache_namespace(
                            model=self.model,
                            model_path=self.config.model_path,
                            quant_tag=quant_tag,
                            tq_native_tag=tq_native_tag,
                            tq_enabled=(
                                self._tq_active and not self._uses_zaya_cache
                            ),
                            zaya_scope=(
                                ":zaya_cache_schema=zaya_cca_v1"
                                if self._uses_zaya_cache
                                else ""
                            ),
                        )
                        model_hash = hashlib.sha256(
                            block_scope_key.encode()
                        ).hexdigest()[:12]
                        cache_dir = os.path.join(cache_root, model_hash)
                    else:
                        cache_dir = os.path.join(cache_root, "default")
                    try:
                        from .block_disk_store import BlockDiskStore
                        from .prefix_cache import expected_cache_layer_count
                        # Without expected_num_layers the wrong-model record
                        # validator is DISARMED (block_disk_store treats None as
                        # "skip the check"). The text path has always passed it;
                        # this path did not.
                        _lang = getattr(self.model, "language_model", self.model)
                        block_disk_store = BlockDiskStore(
                            cache_dir=cache_dir,
                            max_size_gb=self.config.block_disk_cache_max_gb,
                            # Idle-time maintenance, the periodic budget rescan,
                            # must not start while a request is in flight.
                            activity_probe=self._block_store_activity_probe,
                            # Admission follows the cache objects actually
                            # instantiated by the loaded model, never a stale
                            # process environment or family-name guess.
                            allow_tq_native=bool(self._tq_active),
                            expected_num_layers=expected_cache_layer_count(
                                _lang, getattr(self, "_hybrid_num_layers", None)
                            ),
                            global_cache_root=cache_root,
                            allow_legacy_hashed_namespaces=(
                                _default_block_cache_root
                            ),
                            allow_legacy_direct_namespace=(
                                not _default_block_cache_root
                            ),
                        )
                        self._block_disk_l2_enabled = True
                        logger.info(
                            f"VLM block disk cache enabled: dir={cache_dir}, "
                            f"max={self.config.block_disk_cache_max_gb}GB"
                        )
                        if self._is_hybrid and not self._uses_zaya_cache:
                            try:
                                self._ssm_companion_disk_store = (
                                    SSMCompanionDiskStore(
                                        directory=os.path.join(
                                            cache_dir, "ssm_companion"
                                        ),
                                        budget_bytes=(
                                            block_disk_store.max_size_bytes
                                        ),
                                        global_budget=(
                                            block_disk_store.global_budget
                                        ),
                                    )
                                )
                                logger.info(
                                    "VLM hybrid SSM companion L2 enabled: dir=%s, "
                                    "shares aggregate root max=%.3gGB",
                                    self._ssm_companion_disk_store.directory,
                                    self.config.block_disk_cache_max_gb,
                                )
                            except Exception as ssm_e:
                                logger.warning(
                                    "VLM hybrid SSM companion L2 init failed; "
                                    "continuing with in-memory SSM companion only: %s",
                                    ssm_e,
                                )
                    except Exception as e:
                        logger.error(
                            f"VLM block disk cache init failed at {cache_dir}: {e}. "
                            "The disk-only route will refuse a RAM fallback."
                        )

                if block_disk_only and block_disk_store is None:
                    raise RuntimeError(
                        "VLM block disk-only cache was requested but its SSD store "
                        "could not be initialized; refusing to substitute a RAM backend"
                    )

                try:
                    from .paged_cache import PagedCacheManager
                    from .prefix_cache import BlockAwarePrefixCache

                    # Parity with the text scheduler (#98): give the MLLM/VL block
                    # pool the SAME RAM-byte ceiling the memory-aware path uses
                    # (default 20% of available RAM, 32 GB hard cap). Without it the
                    # MLLM paged pool was bounded only by max_cache_blocks, so the
                    # in-RAM block KV mirror could ratchet resident memory upward
                    # with distinct prefixes regardless of per-model KV size — the
                    # exact gap the text path closed in Wave-18. enforce_byte_budget()
                    # evicts only free (ref==0) cached blocks, disk-L2 first.
                    from .memory_cache import resolve_paged_resident_policy

                    (
                        _mllm_paged_resident_budget,
                        _mllm_explicit_zero_cache,
                    ) = resolve_paged_resident_policy(self.config, block_disk_only)
                    if _mllm_explicit_zero_cache:
                        logger.info(
                            "cache-memory-mb=0 requested: MLLM paged RAM payloads "
                            "disabled (frugal); blocks restore transiently."
                        )
                    _mllm_index_blocks = resolve_mllm_index_blocks(
                        self.config.paged_cache_block_size,
                        self.config.max_cache_blocks,
                    )
                    if _mllm_index_blocks != self.config.max_cache_blocks:
                        logger.info(
                            "MLLM block index scaled %d -> %d blocks so the default "
                            "indexes the same %d tokens the text path does at its "
                            "256-token block size (this bounds the index only; "
                            "resident RAM stays under the byte ceiling)",
                            self.config.max_cache_blocks,
                            _mllm_index_blocks,
                            _mllm_index_blocks * self.config.paged_cache_block_size,
                        )
                    self.paged_cache_manager = PagedCacheManager(
                        block_size=self.config.paged_cache_block_size,
                        max_blocks=_mllm_index_blocks,
                        disk_store=block_disk_store,
                        max_resident_bytes=_mllm_paged_resident_budget,
                        disk_only=block_disk_only,
                        frugal=_mllm_explicit_zero_cache,
                    )
                    if block_disk_only:
                        logger.info(
                            "MLLM block disk-only prefix backend: paged RAM disabled, "
                            "max_index_blocks=%d; payloads restore transiently from SSD",
                            _mllm_index_blocks,
                        )
                    else:
                        logger.info(
                            "MLLM paged cache RAM ceiling: %.0f MB (%.0f%% of available); "
                            "block pool max_blocks=%d",
                            _mllm_paged_resident_budget / (1024 * 1024),
                            self.config.cache_memory_percent * 100,
                            _mllm_index_blocks,
                        )
                    self.block_aware_cache = BlockAwarePrefixCache(
                        model=lang_model,
                        paged_cache_manager=self.paged_cache_manager,
                        # All three native/path-dependent validators are derived
                        # from the instantiated language-model runtime. Unknown
                        # DSV4 stays conservative; a proven ordinary Qwen/Gemma
                        # cache skips the otherwise redundant full-chain probe.
                        uses_dsv4_cache=self._uses_dsv4_cache,
                        uses_zaya_cache=self._uses_zaya_cache,
                        mixed_attention_cache_model=(
                            self._mixed_attention_cache_model
                        ),
                    )
                    if self._is_hybrid and not self._uses_zaya_cache:
                        effective_kv_bits = (
                            4 if self.config.kv_cache_quantization == "q4"
                            else 8 if self.config.kv_cache_quantization == "q8"
                            else 0
                        )
                        try:
                            from .prefix_cache import compute_model_cache_key

                            model_key = compute_model_cache_key(
                                lang_model,
                                model_path=self.config.model_path,
                                smelt_enabled=False,
                                smelt_pct=0.0,
                                tq_enabled=False,
                                kv_quant_bits=effective_kv_bits,
                            )
                        except Exception:
                            scope = (
                                f"{self.config.model_path or ''}:"
                                f"kv={self.config.kv_cache_quantization}:"
                                f"block={self.config.paged_cache_block_size}"
                            )
                            model_key = hashlib.sha256(scope.encode()).hexdigest()
                        self._ssm_companion_model_key = model_key
                    logger.info(
                        f"VLM {'block disk-only' if block_disk_only else 'paged'} cache enabled: "
                        f"block_size={self.config.paged_cache_block_size}, "
                        f"max_blocks={_mllm_index_blocks}, "
                        f"capacity="
                        f"{_mllm_index_blocks * self.config.paged_cache_block_size} tokens"
                    )
                except Exception as e:
                    self._cleanup_failed_block_cache_initialization(
                        block_disk_store=block_disk_store,
                    )
                    if block_disk_only:
                        raise RuntimeError(
                            "Failed to initialize authoritative VLM block disk-only cache"
                        ) from e
                    logger.warning(f"Failed to initialize VLM paged cache: {e}")

            elif self.config.use_memory_aware_cache:
                # Memory-aware cache (L1, recommended for large models)
                try:
                    from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
                    cache_config = MemoryCacheConfig(
                        max_memory_mb=self.config.cache_memory_mb,
                        max_memory_percent=self.config.cache_memory_percent,
                        ttl_minutes=self.config.cache_ttl_minutes,
                    )
                    self.memory_aware_cache = MemoryAwarePrefixCache(
                        model=lang_model,
                        config=cache_config,
                    )
                    logger.info(
                        f"VLM memory-aware cache enabled: "
                        f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB"
                    )
                except Exception as e:
                    logger.warning(f"VLM memory-aware cache init failed: {e}")

            else:
                # Legacy entry-count prefix cache
                try:
                    from .prefix_cache import PrefixCacheManager
                    # A BYTE budget, not just an entry count.
                    #
                    # This path passed only max_entries, so the cache was bounded
                    # at 100 ENTRIES with no byte bound at all. (An earlier
                    # version of this comment claimed "the LLM path in
                    # scheduler.py has always passed max_bytes" — that was WRONG
                    # and it hid the same hole there for longer: scheduler.py
                    # passes the PARAMETER, but SchedulerConfig defaults
                    # prefix_cache_max_bytes to None, so the value was unbounded
                    # too. Both paths now derive a budget.) During one long
                    # COLD prefill each chunk stores a snapshot holding a full KV
                    # copy — ~5GB at 90k context — so ~50 chunks accumulated ~70GB
                    # and byte eviction never ran because max_bytes was None.
                    #
                    # MEASURED on Qwen3.6-27B with a 101,502-token prompt: with
                    # the prefix cache ON the prefill hit 93.48GB and had to be
                    # declined; with --disable-prefix-cache the identical run
                    # completed in 204.3s at a 63-65GB peak. Same paging, same
                    # everything else.
                    _pc_max_bytes = _resolve_prefix_cache_byte_budget(self.config)
                    self.prefix_cache = PrefixCacheManager(
                        model=lang_model,
                        max_entries=self.config.prefix_cache_size,
                        max_bytes=_pc_max_bytes,
                    )
                    logger.info(
                        "VLM prefix cache enabled: max_entries=%s max_bytes=%s",
                        self.config.prefix_cache_size,
                        (
                            f"{_pc_max_bytes / (1024**3):.1f}GB"
                            if _pc_max_bytes
                            else "unbounded"
                        ),
                    )
                except Exception as e:
                    logger.warning(f"VLM prefix cache init failed: {e}")

        # Disk cache L2 (persistent across restarts, for non-paged paths)
        if self.config.enable_disk_cache and self.config.enable_prefix_cache:
            base_dir = self.config.disk_cache_dir or os.path.expanduser(
                "~/.cache/vmlx-engine/prompt-cache"
            )
            if self.config.model_path:
                quant_tag = self.config.kv_cache_quantization or "none"
                tq_native_tag = (
                    "on"
                    f"-k{self._hybrid_tq_default_key_bits or 0}"
                    f"-v{self._hybrid_tq_default_value_bits or 0}"
                    f"-after{self._hybrid_live_tq_compress_after or 0}"
                    f"-policy{self._hybrid_tq_auto_policy or 'bundle'}"
                    if self._tq_active
                    else "off"
                )
                # Include layer count to invalidate on architecture change
                n_layers = 0
                lm = self.model.language_model if hasattr(self.model, 'language_model') else self.model
                for _attr in ('args', 'config'):
                    _cfg = getattr(lm, _attr, None)
                    if _cfg:
                        n_layers = getattr(_cfg, 'num_hidden_layers', 0)
                        if n_layers:
                            break
                from .prefix_cache import PAGED_CACHE_SCHEMA_VERSION
                scope_key = (
                    f"{self.config.model_path}:quant={quant_tag}:layers={n_layers}"
                    f":tq_native={tq_native_tag}"
                    f":prefix_cache_schema={PAGED_CACHE_SCHEMA_VERSION}"
                    f":{runtime_cache_fingerprint()}"
                )
                model_hash = hashlib.sha256(
                    scope_key.encode()
                ).hexdigest()[:12]
                model_slug = os.path.basename(self.config.model_path.rstrip("/"))
                cache_dir = os.path.join(base_dir, f"{model_slug}_{model_hash}")
            else:
                cache_dir = base_dir
            try:
                from .disk_cache import DiskCacheManager
                self.disk_cache = DiskCacheManager(
                    cache_dir=cache_dir,
                    max_size_gb=self.config.disk_cache_max_gb,
                    allow_tq_native=bool(self._tq_active),
                )
                logger.info(f"VLM disk cache (L2) enabled: dir={cache_dir}")
            except Exception as e:
                logger.warning(f"VLM disk cache init failed: {e}")
        elif self.config.enable_disk_cache and not self.config.enable_prefix_cache:
            logger.warning(
                "VLM disk cache requires prefix cache to be enabled — disk cache disabled"
            )

        # KV cache quantization for prefix cache storage (2-4x memory reduction)
        # MLA models (Mistral 4, DeepSeek V2/V3) store compressed KV latents —
        # quantizing already-compressed representations destroys quality.
        _is_mla = self._detect_mla()
        if self._uses_zaya_cache and self.config.kv_cache_quantization != "none":
            logger.info(
                "ZAYA/CCA typed cache detected — disabling generic KV cache "
                "quantization (was: %s). zaya_cca_v1 must preserve KV plus "
                "CCA conv_state/prev_hs until a typed partial codec proves "
                "numeric parity.",
                self.config.kv_cache_quantization,
            )
            self.config.kv_cache_quantization = "none"
        if _is_mla and self.config.kv_cache_quantization != "none":
            logger.info("KV cache quantization disabled: MLA model (compressed KV latents)")
            self.config.kv_cache_quantization = "none"
        if (
            self._mixed_attention_cache_model
            and getattr(self.config, "kv_cache_quantization_explicit", False) is False
            and self.config.kv_cache_quantization != "none"
        ):
            # Default-ON TurboQuant KV for mixed-SWA (gemma): the RotatingKVCache
            # sliding/full metadata is preserved through prefix/paged/L2, and q4
            # stored-cache quantization is parity-verified coherent (multi-turn +
            # long-context recall). Keep auto q4/q8 active instead of disabling.
            logger.info(
                "Mixed-SWA VLM cache detected — keeping auto %s TurboQuant KV "
                "stored-cache quantization active (RotatingKVCache metadata "
                "preserved; gemma SWA q4 parity verified).",
                self.config.kv_cache_quantization,
            )
        if self.config.kv_cache_quantization != "none":
            if self.config.enable_prefix_cache:
                bits = 4 if self.config.kv_cache_quantization == "q4" else 8
                self._wrap_make_cache_quantized(bits, self.config.kv_cache_group_size)
                logger.info(
                    f"VLM KV cache quantization enabled: {self.config.kv_cache_quantization} "
                    f"(bits={bits}, group_size={self.config.kv_cache_group_size})"
                )
            else:
                logger.warning(
                    f"KV cache quantization '{self.config.kv_cache_quantization}' requested "
                    "but prefix cache is disabled — quantization has no effect without prefix cache"
                )

        self._tq_active = self._detect_turboquant_make_cache()
        self._tq_batch_api = self._detect_turboquant_batch_api()
        self._capture_hybrid_turboquant_policy()
        if (
            self._tq_active
            and not getattr(self.config, "kv_cache_quantization_explicit", False)
            and getattr(self, "_kv_cache_bits", 0)
        ):
            self._kv_cache_bits = 0
            logger.info(
                "VLM KV cache auto mode: native TurboQuant owns stored attention "
                "KV; generic q4/q8 prefix quantization is suppressed"
            )
        self._enforce_turboquant_single_sequence()

        # Get stop tokens from tokenizer
        self.stop_tokens = self._get_stop_tokens()
        self._log_runtime_cache_contract(lang_model)

        # Batch generator (created lazily, recreated on sampler param change)
        self.batch_generator: Optional[MLLMBatchGenerator] = None
        self._current_sampler_params: tuple = ()

        # Request management - following vLLM's design
        self.waiting: deque[MLLMRequest] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, MLLMRequest] = {}  # Running requests by ID
        self.requests: Dict[str, MLLMRequest] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished
        self._pending_aborts: Set[str] = set()  # Deferred aborts (Metal safety)

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # Output queues for async streaming
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self._scheduler_trace_timings: Dict[str, Dict[str, float]] = {}

        # Streaming detokenizer pool for correct multi-byte character handling
        self._detokenizer_pool: Dict[str, Any] = {}

        # Async processing control
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        # See EngineCore's equivalent gate: terminal media/text output is
        # dispatched before cache persistence for low-latency streaming, while
        # new async requests wait until paged/TQ/typed companion state is fully
        # committed.  Sync step-based callers retain synchronous cleanup.
        self._terminal_cleanup_complete = asyncio.Event()
        self._terminal_cleanup_complete.set()

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._cache_hit_requests = 0
        self._cache_hit_tokens = 0
        self._cache_hit_tokens_by_detail: Dict[str, int] = {}

        # Periodic Metal memory cache cleanup timer (matches LLM scheduler).
        # During sustained MLLM traffic, self.running is never empty so
        # _cleanup_finished's clear_mlx_memory_cache() never triggers.
        self._last_metal_gc_time = time.monotonic()
        self._metal_gc_interval = 60.0  # seconds

        # Log cache configuration summary for diagnostics
        cache_mode = "none"
        if self.block_aware_cache is not None:
            cache_mode = "paged"
        elif self.memory_aware_cache is not None:
            cache_mode = "memory-aware"
        elif self.prefix_cache is not None:
            cache_mode = "legacy"
        if self._uses_zaya_cache and self.block_aware_cache is not None:
            logger.info(
                "ZAYA/CCA typed paged prefix cache enabled — VLM cache "
                "records use zaya_cca_v1 state (KV + conv_state + prev_hs); "
                "generic KV quantization remains disabled."
            )
        logger.info(
            f"MLLM Scheduler initialized: cache_mode={cache_mode}, "
            f"hybrid={self._is_hybrid}, "
            f"zaya_cca={self._uses_zaya_cache}, "
            f"kv_quant={self.config.kv_cache_quantization}, "
            f"prompt_l2={self.disk_cache is not None}, "
            f"block_l2={self._block_disk_l2_enabled}, "
            f"max_seqs={self.config.max_num_seqs}"
        )


    def _block_store_activity_probe(self) -> bool:
        """True while any request is waiting or running (block-store maintenance gate)."""

        return bool(getattr(self, "waiting", None) or getattr(self, "running", None))

    def _log_runtime_cache_contract(self, model: Any) -> None:
        """Record and log instantiated per-layer cache classes."""
        if not hasattr(model, "make_cache"):
            return
        try:
            cache = model.make_cache() or []
            runtime_layout = describe_runtime_cache_layout(cache, model=model)
            runtime_layout["source"] = "instantiated_make_cache"
            runtime_layout["factory"] = "model.make_cache"
            runtime_layout["delegated_to_model_make_cache"] = True
            dtype_status = getattr(
                getattr(self, "model", None),
                "_vmlx_quant_metadata_dtype_harmonization",
                None,
            )
            if isinstance(dtype_status, dict):
                runtime_layout["parameter_dtype_harmonization"] = dict(dtype_status)
            # This is observed from the instantiated runtime cache objects,
            # not inferred from a family name or registry default. It remains
            # available while the request-scoped MLLMBatchGenerator is absent,
            # so startup health can report the real hybrid attention layout.
            self._runtime_cache_layout = runtime_layout
            self._runtime_cache_num_layers = runtime_layout["layer_count"]
            self._runtime_cache_kv_positions = (
                None
                if runtime_layout["parallel_layer_indices"]
                else runtime_layout["attention_layer_indices"]
            )
            self._runtime_cache_parallel_hybrid_positions = (
                runtime_layout["parallel_layer_indices"]
            )
            model_type = getattr(self.model_config, "model_type", None) or "unknown"
            logger.info(
                "Runtime cache layout: model_type=%s layers=%d layout=%s",
                model_type,
                runtime_layout["layer_count"],
                ";".join(
                    f"{idx}:{slot_type}"
                    for idx, slot_type in enumerate(runtime_layout["slot_types"])
                ),
            )
        except Exception as exc:
            logger.debug("Runtime cache layout logging skipped: %s", exc)

    def _get_detokenizer(self, request_id: str, tokenizer: Any) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            detok = NaiveStreamingDetokenizer(tokenizer)
            detok.reset()
            self._detokenizer_pool[request_id] = detok
        return self._detokenizer_pool[request_id]

    def _cleanup_detokenizer(self, request_id: str) -> None:
        """Remove the streaming detokenizer for a finished request."""
        self._detokenizer_pool.pop(request_id, None)

    @staticmethod
    def _is_hybrid_model(model: Any) -> bool:
        """Check if VLM language model uses non-standard cache types.

        Returns True for models with mixed KVCache + MambaCache/ArraysCache layers,
        or pure Mamba/SSM models. These need special handling:
        - Auto-switch to paged cache (memory-aware can't truncate SSM states)
        - Enable MambaCache batching support (ensure_mamba_support)
        - HybridSSMStateCache for companion SSM state caching

        Detection: calls model.make_cache() and checks cache type names.
        Pure KV types (KVCache, RotatingKVCache, QuantizedKVCache) -> False.
        Any other type (ArraysCache, MambaCache, CacheList with mixed) -> True.
        """
        if not hasattr(model, "make_cache"):
            return False
        try:
            cache = model.make_cache()
            # Resolve CacheList wrappers to their contents rather than
            # discarding the wrapper name, which assumed every CacheList holds
            # only KV layers. That assumption is false for falcon_h1-style
            # layouts (CacheList(ArraysCache, KVCache)) and misclassified them
            # as plain KV. Shared with the LLM scheduler so the two detectors
            # cannot drift apart again.
            cache_types = expand_cache_class_names(cache)
            # Dynamic detection: any type ending with "KVCache" is a KV type
            kv_types = {t for t in cache_types if t == "KVCache" or t.endswith("KVCache")}
            return bool(cache_types - kv_types)
        except Exception as exc:
            raise RuntimeError(
                "MLLM language-model make_cache() failed during cache-architecture "
                "detection; refusing to classify it as plain KV"
            ) from exc

    def _mllm_request_has_media_cache_context(
        self,
        request: Any,
        token_ids: Optional[List[int]] = None,
    ) -> bool:
        """Return True when a VLM request must not use token-only caches.

        Image/video embeddings are path-dependent. A token-only prefix key is
        safe for text-only follow-up turns, but not for requests that carry
        media inputs or media placeholder tokens. Keep this helper shared by
        every MLLM cache store path so paged, memory-aware, legacy, and disk L2
        cannot drift into different safety policies.
        """
        if request is None:
            return False
        if getattr(request, "images", None) or getattr(request, "videos", None):
            return True
        if not token_ids:
            return False
        try:
            contains_placeholders = (
                self.batch_generator is not None
                and self.batch_generator._tokens_contain_media_placeholders(
                    list(token_ids)
                )
            )
            return bool(contains_placeholders)
        except Exception:
            return False

    def _mllm_media_prefix_cache_allowed(
        self,
        request: Any,
        token_ids: Optional[List[int]] = None,
    ) -> bool:
        """Return True when a media prompt may use media-keyed prefix cache.

        Qwen3.5/3.6 VL, Qwen4Exp, Muse Glimmer, Gemma 4, Step 3.7, and dots3
        have a clean media-conditioned N-1 cache producer and are enabled by
        default. Gemma, Step, and Muse store captured native rotating-SWA plus
        compatible full-attention boundaries under per-placeholder media side
        keys. Other families retain the historical double opt-in until they
        have equivalent typed-cache proof. ZAYA CCA remains excluded because
        its current clean store path is text-only.
        """
        if request is None or getattr(request, "_bypass_prefix_cache", False):
            return False
        if getattr(self, "_uses_zaya_cache", False):
            return False
        model_type = str(
            getattr(getattr(self, "batch_generator", None), "_model_type", "")
            or getattr(self, "_model_type", "")
            or ""
        ).lower()
        if not _mllm_media_prefix_cache_family_enabled(model_type):
            return False
        extra = getattr(request, "_cache_extra_keys", None)
        if not extra:
            return False
        return self._mllm_request_has_media_cache_context(request, token_ids)

    def _detect_mla(self) -> bool:
        """Detect if model uses Multi-head Latent Attention (MLA).

        MLA detection logic and KV cache quantization guard added by Jinho Jang
        (eric@jangq.ai) for vMLX. The base scheduler architecture derives from
        waybarrios/vllm-mlx. The MLA-specific handling (kv_lora_rank detection,
        auto-disabling quantization for compressed latents, prefix cache H=1
        validation) is original vMLX work requiring extensive empirical testing.
        We are leaving this attribution here because we know another inference
        engine is very directly going to be looking at this to attempt to
        rip it off. If you're adapting this code, please credit the original
        author: Jinho Jang (github.com/jjang-ai/vmlx).
        """
        def _cfg_model_type(cfg: Any) -> str:
            if cfg is None:
                return ""
            if isinstance(cfg, dict):
                mt = cfg.get("model_type") or ""
                if not mt and isinstance(cfg.get("text_config"), dict):
                    mt = cfg["text_config"].get("model_type") or ""
                return str(mt).lower()
            mt = getattr(cfg, "model_type", "") or ""
            if not mt:
                tc = getattr(cfg, "text_config", None)
                mt = getattr(tc, "model_type", "") if tc is not None else ""
            return str(mt or "").lower()

        def _cfg_kv_lora_rank(cfg: Any) -> int:
            if cfg is None:
                return 0
            if isinstance(cfg, dict):
                raw = cfg
            else:
                raw = getattr(cfg, "_raw_config", None)
            vals = []
            if isinstance(cfg, dict):
                vals.append(cfg.get("kv_lora_rank", 0))
                tc = cfg.get("text_config")
                if isinstance(tc, dict):
                    vals.append(tc.get("kv_lora_rank", 0))
            else:
                vals.append(getattr(cfg, "kv_lora_rank", 0))
                tc = getattr(cfg, "text_config", None)
                if tc is not None:
                    vals.append(getattr(tc, "kv_lora_rank", 0))
            if isinstance(raw, dict):
                vals.append(raw.get("kv_lora_rank", 0))
                tc = raw.get("text_config")
                if isinstance(tc, dict):
                    vals.append(tc.get("kv_lora_rank", 0))
            for val in vals:
                try:
                    iv = int(val or 0)
                except (TypeError, ValueError):
                    iv = 0
                if iv > 0:
                    return iv
            return 0

        try:
            lm = self.model.language_model if hasattr(self.model, "language_model") else self.model
            candidates = [self.model, lm]
            for obj in list(candidates):
                inner = getattr(obj, "model", None)
                if inner is not None and inner not in candidates:
                    candidates.append(inner)

            cfgs = []
            for obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(obj, attr, None)
                    if cfg is not None:
                        cfgs.append(cfg)
                        raw = getattr(cfg, "_raw_config", None)
                        if isinstance(raw, dict):
                            cfgs.append(raw)
                        tc = getattr(cfg, "text_config", None)
                        if tc is not None:
                            cfgs.append(tc)
                        if isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
                            cfgs.append(cfg["text_config"])

            for cfg in cfgs:
                model_type = _cfg_model_type(cfg)
                if model_type in ("bailing_hybrid", "bailing_moe_v2_5"):
                    # Ling/Bailing's runtime stores full per-head KV, not H=1
                    # compressed MLA latents. Keep normal KV/TQ cache support.
                    continue
                if _cfg_kv_lora_rank(cfg) > 0:
                    return True
                if model_type == "mistral4":
                    return True
        except Exception:
            pass
        return False

    def _detect_turboquant_make_cache(self) -> bool:
        """Detect JANG TurboQuant KV patches on VLM wrappers or language models."""

        def _is_mock_object(obj: Any) -> bool:
            return type(obj).__module__ == "unittest.mock"

        def _safe_attr(obj: Any, name: str) -> Any:
            if obj is None:
                return None
            # unittest.mock fabricates arbitrary attributes on access. A blind
            # wrapper walk over `.model.language_model.model...` never
            # terminates and can hang tests or proxy-heavy integrations. Only
            # use attributes explicitly assigned on mocks.
            if _is_mock_object(obj):
                return getattr(obj, "__dict__", {}).get(name)
            return getattr(obj, name, None)

        def _is_tq_make_cache(obj: Any) -> bool:
            make_cache = _safe_attr(obj, "make_cache")
            return make_cache is not None and is_turboquant_make_cache(make_cache)

        seen = set()
        stack = [self.model]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if _is_tq_make_cache(obj):
                return True
            for attr in ("model", "language_model"):
                nxt = _safe_attr(obj, attr)
                if nxt is not None and nxt is not obj and id(nxt) not in seen:
                    stack.append(nxt)
        return False

    def warm_tq_storage_decoders(self) -> Dict[str, Any]:
        """Materialize VLM TQ storage codecs on the pinned model worker.

        MLLM routes do not use the text ``Scheduler``. Resolve the same patched
        language-model ``make_cache`` owner used by hybrid cache detection,
        then warm only its TurboQuantKVCache slots. ArraysCache/Mamba state and
        rotating-window cache slots remain untouched.
        """
        stats: Dict[str, Any] = {
            "enabled": False,
            "configs": 0,
            "arrays": 0,
            "bytes": 0,
            "probe_tokens": 0,
            "probe_heads": 0,
            "codec_probes": 0,
            "seconds": 0.0,
        }
        if not (self._tq_active and self.config.enable_prefix_cache):
            self._tq_decoder_warmup_stats = stats
            return stats

        def _safe_attr(obj: Any, name: str) -> Any:
            if obj is None:
                return None
            if type(obj).__module__ == "unittest.mock":
                return getattr(obj, "__dict__", {}).get(name)
            return getattr(obj, name, None)

        owner = None
        seen = set()
        stack = [self.model]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            make_cache = _safe_attr(obj, "make_cache")
            if make_cache is not None and is_turboquant_make_cache(make_cache):
                owner = obj
                break
            for attr in ("model", "language_model"):
                child = _safe_attr(obj, attr)
                if child is not None and child is not obj and id(child) not in seen:
                    stack.append(child)

        started = time.perf_counter()
        cache_layers = None
        try:
            if owner is None:
                raise RuntimeError("TurboQuant make_cache owner was not found")
            from .tq_disk_store import warm_tq_decoder_states

            cache_layers = owner.make_cache() or []
            model_args = _safe_attr(owner, "args") or _safe_attr(owner, "config")
            if isinstance(model_args, dict):
                probe_heads = int(model_args.get("num_key_value_heads", 1) or 1)
            else:
                probe_heads = int(
                    _safe_attr(model_args, "num_key_value_heads") or 1
                )
            stats.update(
                warm_tq_decoder_states(cache_layers, probe_heads=probe_heads)
            )
            stats["enabled"] = bool(stats.get("configs"))
            stats["seconds"] = round(time.perf_counter() - started, 6)
            logger.info(
                "MLLM TurboQuant storage decoder startup warmup: configs=%d "
                "arrays=%d bytes=%d codec_probes=%d probe_tokens=%d "
                "probe_heads=%d seconds=%.3f",
                stats["configs"],
                stats["arrays"],
                stats["bytes"],
                stats["codec_probes"],
                stats["probe_tokens"],
                stats["probe_heads"],
                stats["seconds"],
            )
        except Exception as exc:
            stats["seconds"] = round(time.perf_counter() - started, 6)
            stats["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "MLLM TurboQuant storage decoder startup warmup failed: %s",
                exc,
            )
        finally:
            del cache_layers
        self._tq_decoder_warmup_stats = stats
        return stats

    def _detect_turboquant_batch_api(self) -> bool:
        """Return True when all VLM TurboQuant cache slots expose batch API v1."""

        def _is_mock_object(obj: Any) -> bool:
            return type(obj).__module__ == "unittest.mock"

        def _safe_attr(obj: Any, name: str) -> Any:
            if obj is None:
                return None
            if _is_mock_object(obj):
                return getattr(obj, "__dict__", {}).get(name)
            return getattr(obj, name, None)

        seen = set()
        stack = [self.model]
        required_methods = ("extend", "filter", "extract", "prepare", "finalize")
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))

            make_cache = _safe_attr(obj, "make_cache")
            if make_cache is not None and is_turboquant_make_cache(make_cache):
                try:
                    cache = make_cache() or []
                except Exception:
                    return False
                tq_slots = [c for c in cache if type(c).__name__ == "TurboQuantKVCache"]
                if not tq_slots:
                    return False
                return all(
                    getattr(slot, "_vmlx_batch_api", None) == "turboquant_kv_v1"
                    and all(callable(getattr(slot, name, None)) for name in required_methods)
                    for slot in tq_slots
                )

            for attr in ("model", "language_model"):
                nxt = _safe_attr(obj, attr)
                if nxt is not None and nxt is not obj and id(nxt) not in seen:
                    stack.append(nxt)
        return False

    def _capture_hybrid_turboquant_policy(self) -> None:
        """Copy selective hybrid TQ metadata from the patched language model."""

        def _safe_attr(obj: Any, name: str) -> Any:
            if obj is None:
                return None
            if type(obj).__module__ == "unittest.mock":
                return getattr(obj, "__dict__", {}).get(name)
            return getattr(obj, name, None)

        seen = set()
        stack = [self.model]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            make_cache = _safe_attr(obj, "make_cache")
            if make_cache is not None and is_turboquant_make_cache(make_cache):
                self._hybrid_live_tq_policy = getattr(
                    make_cache, "_vmlx_hybrid_tq_policy", None
                )
                self._hybrid_live_tq_attention_layers = list(
                    getattr(make_cache, "_vmlx_hybrid_tq_attention_layers", ()) or []
                )
                self._hybrid_live_tq_companion_layers = list(
                    getattr(make_cache, "_vmlx_hybrid_tq_companion_layers", ()) or []
                )
                self._hybrid_live_tq_compress_after = int(
                    getattr(make_cache, "_vmlx_tq_compress_after", 0) or 0
                )
                self._hybrid_tq_auto_policy = getattr(
                    make_cache, "_vmlx_tq_auto_policy", None
                )
                self._hybrid_tq_default_key_bits = int(
                    getattr(make_cache, "_vmlx_tq_default_key_bits", 0) or 0
                )
                self._hybrid_tq_default_value_bits = int(
                    getattr(make_cache, "_vmlx_tq_default_value_bits", 0) or 0
                )
                return
            for attr in ("model", "language_model"):
                nxt = _safe_attr(obj, attr)
                if nxt is not None and nxt is not obj and id(nxt) not in seen:
                    stack.append(nxt)

    def _enforce_turboquant_single_sequence(self) -> None:
        """Keep MLLM live batching honest for TurboQuantKVCache."""
        if not self._tq_active:
            return

        if self._tq_batch_api:
            logger.info(
                "VLM TurboQuantKVCache live decode preserving configured batching "
                "(batch cache API=turboquant_kv_v1)."
            )
            return

        changed = []
        if self.config.max_num_seqs != 1:
            changed.append(f"max_num_seqs {self.config.max_num_seqs}->1")
            self.config.max_num_seqs = 1
        if self.config.prefill_batch_size != 1:
            changed.append(f"prefill_batch_size {self.config.prefill_batch_size}->1")
            self.config.prefill_batch_size = 1
        if self.config.completion_batch_size != 1:
            changed.append(
                f"completion_batch_size {self.config.completion_batch_size}->1"
            )
            self.config.completion_batch_size = 1
        if changed:
            logger.warning(
                "VLM TurboQuantKVCache live decode is single-sequence only "
                "with this jang_tools build (missing batch API turboquant_kv_v1); "
                "overriding %s.",
                ", ".join(changed),
            )

    def _detect_cache_head_dims(self) -> Tuple[int, ...]:
        """Detect VLM language-model cache trailing dims for KV quant validation."""
        try:
            return detect_cache_head_dims(self.model)
        except Exception as e:
            logger.debug(f"Could not detect VLM cache head dims: {e}")
            return ()

    def _detect_head_dim(self) -> Optional[int]:
        """Detect the primary VLM KV cache trailing dim from config."""
        dims = self._detect_cache_head_dims()
        if dims:
            return dims[0]
        return None

    def _wrap_make_cache_quantized(self, bits: int, group_size: int) -> None:
        """
        Configure KV cache quantization for VLM prefix cache storage.

        Quantization is applied at the storage/retrieval boundary — full-precision
        KVCache during generation, quantized storage in prefix cache for 2-4x
        memory savings. Validates head_dim compatibility and runs round-trip test.
        """
        try:
            from mlx_lm.models.cache import QuantizedKVCache
            import mlx.core as mx
        except ImportError:
            logger.warning(
                "QuantizedKVCache not available. VLM KV cache quantization disabled."
            )
            return

        # Patch QuantizedKVCache.size if needed
        if not hasattr(QuantizedKVCache, '_size_patched'):
            needs_patch = True
            try:
                test_qkv = QuantizedKVCache(group_size=64, bits=bits)
                test_qkv.offset = 42
                if callable(getattr(test_qkv, 'size', None)) and test_qkv.size() == 42:
                    needs_patch = False
            except Exception:
                pass
            if needs_patch:
                def _qkv_size(self):
                    return getattr(self, 'offset', 0)
                QuantizedKVCache.size = _qkv_size
                logger.debug("Patched QuantizedKVCache.size() to return self.offset")
            QuantizedKVCache._size_patched = True

        # Validate cache trailing-dim compatibility
        cache_head_dims = self._detect_cache_head_dims()
        head_dim = cache_head_dims[0] if cache_head_dims else None
        if cache_head_dims:
            adjusted_group_size = choose_supported_kv_group_size(
                cache_head_dims, group_size
            )
            if adjusted_group_size is None:
                logger.error(
                    "VLM KV quant: no supported group_size for "
                    f"cache_head_dims={cache_head_dims}. Disabled."
                )
                return
            if adjusted_group_size != group_size:
                logger.warning(
                    f"VLM KV quant: group_size={group_size} doesn't divide "
                    f"cache_head_dims={cache_head_dims} or is unsupported. "
                    f"Auto-adjusting to {adjusted_group_size}."
                )
                group_size = adjusted_group_size
            logger.info(
                f"VLM KV quant validated: cache_head_dims={cache_head_dims}, "
                f"group_size={group_size}"
            )

        # Round-trip test
        try:
            test_dim = head_dim or 128
            test_tensor = mx.random.normal((1, 4, 8, test_dim))
            quantized = mx.quantize(test_tensor, group_size=group_size, bits=bits)
            dequantized = mx.dequantize(
                quantized[0], quantized[1], quantized[2],
                group_size=group_size, bits=bits,
            )
            mx.eval(dequantized)
            logger.info(f"VLM KV quant round-trip test passed: bits={bits}, group_size={group_size}")
        except Exception as e:
            logger.error(f"VLM KV quant round-trip test FAILED: {e}. Disabling.")
            return

        self._kv_cache_bits = bits
        self._kv_cache_group_size = group_size

    @staticmethod
    def _validate_cache(cache: Any, *, source: str) -> bool:
        """Validate live VLM cache objects before store/reuse."""
        try:
            from .cache_record_validator import reject_live_cache_or_warn
            return reject_live_cache_or_warn(cache, source=source)
        except Exception:
            return cache is not None and (not isinstance(cache, list) or len(cache) > 0)

    def _quantize_cache_for_storage(self, cache: List[Any]) -> List[Any]:
        """
        Quantize KVCache layers for prefix cache storage (2-4x memory reduction).
        Preserves non-KVCache layers (MambaCache, etc.).
        Recurses into CacheList sub-caches for MoE models.
        """
        if not self._kv_cache_bits:
            return cache
        try:
            from mlx_lm.models.cache import KVCache, QuantizedKVCache
            try:
                from mlx_lm.models.cache import CacheList as _CacheList
            except ImportError:
                _CacheList = None
            import mlx.core as mx
        except ImportError:
            return cache

        bits = self._kv_cache_bits
        group_size = self._kv_cache_group_size
        result = []
        for layer_cache in cache:
            if _CacheList is not None and isinstance(layer_cache, _CacheList):
                # MoE: quantize each sub-cache independently
                quantized_subs = []
                for sc in layer_cache.caches:
                    if (
                        isinstance(sc, KVCache)
                        and not isinstance(sc, QuantizedKVCache)
                        and sc.keys is not None
                    ):
                        try:
                            qkv = QuantizedKVCache(group_size=group_size, bits=bits)
                            qkv.keys = tuple(mx.quantize(sc.keys, group_size=group_size, bits=bits))
                            qkv.values = tuple(mx.quantize(sc.values, group_size=group_size, bits=bits))
                            qkv.offset = sc.offset
                            quantized_subs.append(qkv)
                        except Exception as exc:
                            logger.debug("KV quantization failed for CacheList sub-cache: %s", exc)
                            quantized_subs.append(sc)
                    else:
                        quantized_subs.append(sc)
                result.append(_CacheList(*quantized_subs))
            elif (
                isinstance(layer_cache, KVCache)
                and not isinstance(layer_cache, QuantizedKVCache)
                and layer_cache.keys is not None
            ):
                try:
                    qkv = QuantizedKVCache(group_size=group_size, bits=bits)
                    qkv.keys = tuple(mx.quantize(layer_cache.keys, group_size=group_size, bits=bits))
                    qkv.values = tuple(mx.quantize(layer_cache.values, group_size=group_size, bits=bits))
                    qkv.offset = layer_cache.offset
                    result.append(qkv)
                except Exception as exc:
                    logger.debug("KV quantization failed for layer, keeping original: %s", exc)
                    result.append(layer_cache)
            else:
                result.append(layer_cache)
        return result

    def _truncate_hybrid_cache(
        self, raw_cache: List[Any], prompt_len: int
    ) -> Optional[List[Any]]:
        """
        Truncate cache for hybrid models (MambaCache + KVCache layers).

        Unlike the LLM scheduler's _truncate_cache_to_prompt_length, this method
        handles hybrid models by:
        - Truncating KVCache layers normally (to prompt_len - 1)
        - Passing MambaCache layers through unchanged (they're cumulative state)

        MambaCache/ArraysCache layers are included in _extract_cache_states()
        as cumulative entries (stored in last block only, "skip" in others).
        """
        target_len = prompt_len - 1
        if not raw_cache or target_len <= 0:
            return None

        truncated = []
        for layer_cache in raw_cache:
            # Guard: skip dicts (extracted state dicts, not live cache objects).
            # dict.keys is a builtin_function_or_method that matches hasattr
            # but has no .ndim, causing crashes.
            if isinstance(layer_cache, dict):
                truncated.append(layer_cache)
                continue
            if type(layer_cache).__name__ == "MiniMaxM3SparseCache":
                try:
                    from .models.minimax_m3.cache import clone_minimax_m3_sparse
                except Exception:
                    return None
                new_cache = clone_minimax_m3_sparse(
                    layer_cache,
                    target_len,
                    require_idx_keys=True,
                )
                if new_cache is None:
                    return None
                truncated.append(new_cache)
                continue
            if type(layer_cache).__name__ == "TurboQuantKVCache":
                # Keep the architecture-selected TQ type/config through the
                # prompt-boundary truncation.  Demoting this object to KVCache
                # loses the per-layer bit policy and seed immediately before
                # _extract_cache_states(), so paged/L2 storage silently falls
                # back to unencoded float blocks.
                try:
                    from jang_tools.turboquant.cache import TurboQuantKVCache

                    full_k, full_v = layer_cache._get_full_cache()
                    if full_k is None or full_v is None:
                        return None
                    safe_target = logical_truncate_target(layer_cache, target_len, int(full_k.shape[-2]))
                    if safe_target <= 0:
                        return None
                    new_cache = TurboQuantKVCache(
                        key_dim=int(full_k.shape[-1]),
                        value_dim=int(full_v.shape[-1]),
                        key_bits=int(layer_cache.key_bits),
                        value_bits=int(layer_cache.value_bits),
                        seed=int(getattr(layer_cache, "_seed", 42)),
                        compress_after=int(
                            getattr(layer_cache, "compress_after", 0) or 0
                        ),
                        sink_tokens=int(
                            getattr(layer_cache, "sink_tokens", 0) or 0
                        ),
                    )
                    new_cache.keys = full_k[..., :safe_target, :]
                    new_cache.values = full_v[..., :safe_target, :]
                    new_cache.offset = safe_target
                    new_cache.step = int(
                        getattr(layer_cache, "step", safe_target) or safe_target
                    )
                    truncated.append(new_cache)
                except Exception as e:
                    logger.warning(
                        "Failed to preserve TurboQuant cache during hybrid "
                        "prompt truncation: %s",
                        e,
                    )
                    return None
                continue
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                try:
                    k = layer_cache.keys
                    v = layer_cache.values
                    # Guard: k must be a tensor with .ndim (not a method or other object)
                    if not hasattr(k, 'ndim'):
                        truncated.append(layer_cache)
                        continue

                    if isinstance(k, tuple):
                        # QuantizedKVCache
                        try:
                            from mlx_lm.models.cache import QuantizedKVCache
                        except ImportError:
                            return None
                        safe_target = logical_truncate_target(layer_cache, target_len, k[0].shape[-2])
                        if safe_target <= 0:
                            return None
                        new_cache = QuantizedKVCache(
                            group_size=layer_cache.group_size,
                            bits=layer_cache.bits,
                        )
                        new_cache.keys = tuple(t[..., :safe_target, :] for t in k)
                        new_cache.values = tuple(t[..., :safe_target, :] for t in v)
                        new_cache.offset = safe_target
                        truncated.append(new_cache)
                    else:
                        # Standard KVCache / RotatingKVCache.
                        # CRITICAL: preserve the original class for sliding-
                        # window layers (e.g. Gemma 4's 25 sliding_attention
                        # layers). Demoting them to KVCache drops `keep` /
                        # `max_size` / `_idx`, so on the next turn the model
                        # sees a non-rotating buffer with invalid window
                        # state on the next turn.
                        from mlx_lm.models.cache import KVCache
                        cls_name = type(layer_cache).__name__
                        new_cache = None
                        if "Rotating" in cls_name:
                            try:
                                from mlx_lm.models.cache import RotatingKVCache
                                max_size = getattr(layer_cache, "max_size", target_len)
                                keep = getattr(layer_cache, "keep", 0)
                                offset = getattr(layer_cache, "offset", 0)
                                if offset > max_size:
                                    # Wrapped circular buffer — head-aligned
                                    # slice is not in temporal order. Skip.
                                    return None
                                new_cache = RotatingKVCache(
                                    max_size=max_size,
                                    keep=keep,
                                )
                            except ImportError:
                                new_cache = KVCache()
                        if new_cache is None:
                            new_cache = KVCache()
                        ndim = k.ndim
                        if ndim == 4:
                            safe_target = logical_truncate_target(layer_cache, target_len, k.shape[2])
                            new_cache.keys = k[:, :, :safe_target, :]
                            new_cache.values = v[:, :, :safe_target, :]
                        elif ndim == 3:
                            safe_target = logical_truncate_target(layer_cache, target_len, k.shape[1])
                            new_cache.keys = k[:, :safe_target, :]
                            new_cache.values = v[:, :safe_target, :]
                        else:
                            return None
                        new_cache.offset = safe_target
                        if "Rotating" in cls_name and hasattr(new_cache, "_idx"):
                            new_cache._idx = safe_target
                        truncated.append(new_cache)
                except Exception as e:
                    logger.warning(f"Failed to truncate KVCache layer: {e}")
                    return None
            elif hasattr(layer_cache, "caches") and isinstance(
                getattr(layer_cache, "caches", None), (list, tuple)
            ):
                # CacheList (MoE models like DeepSeek V3, Mistral 4): recursively truncate
                # each sub-cache independently, then wrap back in CacheList.
                try:
                    sub_truncated = self._truncate_hybrid_cache(
                        list(layer_cache.caches), prompt_len
                    )
                    if sub_truncated is None:
                        return None
                    from mlx_lm.models.cache import CacheList
                    new_cl = CacheList.__new__(CacheList)
                    new_cl.caches = sub_truncated
                    truncated.append(new_cl)
                except Exception as e:
                    logger.warning(f"Failed to truncate CacheList layer: {e}")
                    truncated.append(layer_cache)
            elif hasattr(layer_cache, "cache") and isinstance(
                getattr(layer_cache, "cache", None), list
            ):
                # MambaCache: pass through unchanged — will be skipped
                # during _extract_cache_states() since it can't be blocked
                truncated.append(layer_cache)
            else:
                # Unknown cache type — skip this layer
                truncated.append(layer_cache)

        return truncated

    def _prepare_tq_cache_for_storage(
        self, raw_cache: List[Any]
    ) -> Optional[List[Any]]:
        """Restore TQ cache identity before a storage-boundary encode.

        MLLMBatch.extract_cache() deliberately returns float KV objects so the
        scheduler can slice the prompt boundary safely.  When the loaded model
        owns an architecture-selected TurboQuant make_cache, re-wrap only those
        attention slots before truncation/extraction.  Otherwise the paged,
        memory-aware, and legacy paths all lose the TQ class/config and silently
        store float blocks.
        """
        if not getattr(self, "_tq_active", False):
            return raw_cache
        language_model = getattr(
            getattr(self, "batch_generator", None), "language_model", None
        )
        if language_model is None:
            logger.warning(
                "Skipping TQ cache store: active TQ model has no language_model"
            )
            return None
        try:
            from .mllm_batch_generator import _recompress_to_tq

            prepared = _recompress_to_tq(raw_cache, language_model)
        except Exception as e:
            logger.warning("Skipping TQ cache store: TQ re-wrap failed: %s", e)
            return None
        if not any(type(layer).__name__ == "TurboQuantKVCache" for layer in prepared):
            logger.warning(
                "Skipping TQ cache store: active TQ model produced no TQ attention slots"
            )
            return None
        return prepared

    def _model_has_mixed_attention(self, lang_model) -> bool:
        """Return True for models that interleave sliding-window and full
        attention layers (Gemma 4 pattern).

        Detection is conservative: we only return True when we can prove
        the config has at least two distinct attention modes. For everything
        else we return False and the normal prefix-cache pipeline runs.
        """
        def _cfg_value(cfg, key):
            if isinstance(cfg, dict):
                return cfg.get(key)
            return getattr(cfg, key, None)

        candidates = []
        for attr in ('args', 'config'):
            cfg = getattr(lang_model, attr, None)
            if cfg is not None:
                candidates.append(cfg)
                tc = _cfg_value(cfg, 'text_config')
                if tc is not None:
                    candidates.append(tc)
        for cfg in candidates:
            cache_subtype = str(_cfg_value(cfg, "cache_subtype") or "").lower()
            model_type = str(_cfg_value(cfg, "model_type") or "").lower()
            text_cfg = _cfg_value(cfg, "text_config")
            text_model_type = str(_cfg_value(text_cfg, "model_type") or "").lower()
            sliding_window = _cfg_value(cfg, "sliding_window")
            if sliding_window is None:
                sliding_window = _cfg_value(text_cfg, "sliding_window")
            if (
                cache_subtype
                in {
                    "mixed_swa_kv",
                    "step3p7_full_sliding_kv",
                    "mimo_v2_asymmetric_swa",
                }
                or (
                    model_type == "step3p7"
                    and text_model_type == "step3p5"
                    and sliding_window is not None
                )
            ):
                return True
            layer_types = _cfg_value(cfg, 'layer_types')
            if layer_types and isinstance(layer_types, (list, tuple)):
                kinds = {str(k).lower() for k in layer_types}
                if len(kinds) >= 2 and any('sliding' in k for k in kinds):
                    return True
        return False

    def _detect_n_kv_heads(self) -> int:
        """Detect number of KV heads from VLM language model config.

        Mirrors Scheduler._detect_n_kv_heads() for GQA head normalization.
        BatchKVCache.merge() inflates H to max across batch; this provides
        the ground-truth KV head count to slice away inflated heads.
        """
        if hasattr(self, '_n_kv_heads_cached'):
            return self._n_kv_heads_cached
        n_kv = 0
        try:
            # VLM wrappers may expose MLA config via language_model.config.text_config
            # (Kimi K2.6 around DeepseekV3, glm_moe_dsa around deepseek_v32).
            # Walk the same model + language_model + inner .model candidates and
            # the same args/config/text_config attrs as Scheduler._detect_mla
            # and PrefixCache._get_n_kv_heads. Carve out Ling/Bailing whose MLA
            # runtime stores expanded per-head KV (do not collapse to H=1).
            candidates = [self.model]
            lm = getattr(self.model, 'language_model', None)
            if lm is not None and lm not in candidates:
                candidates.append(lm)
            for obj in list(candidates):
                inner = getattr(obj, 'model', None)
                if inner is not None and inner not in candidates:
                    candidates.append(inner)

            # PASS 1: MLA detection (H=1 collapse) with bailing carve-out.
            for obj in candidates:
                for attr in ('args', 'config', 'text_config'):
                    cfg = getattr(obj, attr, None)
                    if cfg is None:
                        continue
                    model_type = str(getattr(cfg, 'model_type', '') or '').lower()
                    if not model_type:
                        tc = getattr(cfg, 'text_config', None)
                        if tc is not None:
                            model_type = str(getattr(tc, 'model_type', '') or '').lower()
                    kv_lora_rank = getattr(cfg, 'kv_lora_rank', 0)
                    if not kv_lora_rank:
                        tc = getattr(cfg, 'text_config', None)
                        if tc is not None:
                            kv_lora_rank = getattr(tc, 'kv_lora_rank', 0)
                    if kv_lora_rank and kv_lora_rank > 0:
                        if model_type in ('bailing_hybrid', 'bailing_moe_v2_5'):
                            # Ling/Bailing keeps full per-head KV.
                            n_kv = int(getattr(cfg, 'num_attention_heads', 0) or 0)
                            if not n_kv:
                                tc = getattr(cfg, 'text_config', None)
                                if tc is not None:
                                    n_kv = int(getattr(tc, 'num_attention_heads', 0) or 0)
                        else:
                            n_kv = 1
                        break
                if n_kv:
                    break

            # PASS 2: standard num_key_value_heads / num_kv_heads / num_attention_heads.
            if not n_kv:
                for obj in candidates:
                    for attr in ('args', 'config', 'text_config'):
                        cfg = getattr(obj, attr, None)
                        if cfg is None:
                            continue
                        n_kv = (
                            getattr(cfg, 'num_key_value_heads', 0)
                            or getattr(cfg, 'num_kv_heads', 0)
                        )
                        if not n_kv:
                            tc = getattr(cfg, 'text_config', None)
                            if tc is not None:
                                n_kv = (
                                    getattr(tc, 'num_key_value_heads', 0)
                                    or getattr(tc, 'num_kv_heads', 0)
                                )
                        if n_kv:
                            break
                        n_kv = getattr(cfg, 'num_attention_heads', 0)
                        if not n_kv:
                            tc = getattr(cfg, 'text_config', None)
                            if tc is not None:
                                n_kv = getattr(tc, 'num_attention_heads', 0)
                        if n_kv:
                            break
                    if n_kv:
                        break
        except Exception:
            pass
        if not isinstance(n_kv, int):
            n_kv = 0
        self._n_kv_heads_cached = n_kv
        return n_kv

    def _detect_allowed_n_kv_heads(self) -> Set[int]:
        """Return every config-declared KV head count that can appear by layer.

        MiMo V2 and Gemma-style VLMs can mix full-attention and sliding-window
        attention layers with different KV head counts. Storage-time GQA
        normalization must only trim inflated batch-merge heads, not legitimate
        per-layer SWA heads.
        """
        if hasattr(self, "_allowed_n_kv_heads_cached"):
            return self._allowed_n_kv_heads_cached

        allowed: Set[int] = set()
        primary = self._detect_n_kv_heads()
        if primary > 0:
            allowed.add(primary)

        try:
            candidates = [self.model]
            lm = getattr(self.model, "language_model", None)
            if lm is not None and lm not in candidates:
                candidates.append(lm)
            for obj in list(candidates):
                inner = getattr(obj, "model", None)
                if inner is not None and inner not in candidates:
                    candidates.append(inner)

            for obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(obj, attr, None)
                    if cfg is None:
                        continue
                    for field in (
                        "num_key_value_heads",
                        "num_kv_heads",
                        "num_global_key_value_heads",
                        "global_num_key_value_heads",
                        "swa_num_key_value_heads",
                        "num_swa_key_value_heads",
                        "sliding_num_key_value_heads",
                        "local_num_key_value_heads",
                    ):
                        val = getattr(cfg, field, None)
                        if val is None:
                            tc = getattr(cfg, "text_config", None)
                            if tc is not None:
                                val = getattr(tc, field, None)
                        if isinstance(val, int) and val > 0:
                            allowed.add(val)
        except Exception:
            pass

        self._allowed_n_kv_heads_cached = allowed
        return allowed

    def _normalize_gqa_state(
        self,
        state,
        n_kv: int,
        allowed_n_kv_heads: Optional[Set[int]] = None,
    ):
        """Normalize GQA head inflation in a cache state tuple.

        Returns the state with inflated heads sliced down to n_kv.
        Handles both plain tensors and quantized tuples (data, scales, zeros).
        """
        if not (isinstance(state, tuple) and len(state) == 2 and n_kv > 0):
            return state
        keys, values = state
        if hasattr(keys, 'shape') and len(keys.shape) == 4:
            actual_h = int(keys.shape[1])
            if allowed_n_kv_heads and actual_h in allowed_n_kv_heads:
                return state
            if actual_h > n_kv:
                return (keys[:, :n_kv, :, :], values[:, :n_kv, :, :])
        elif (isinstance(keys, (tuple, list)) and len(keys) >= 1
                and hasattr(keys[0], 'shape') and len(keys[0].shape) == 4
                and keys[0].shape[1] > n_kv):
            actual_h = int(keys[0].shape[1])
            if allowed_n_kv_heads and actual_h in allowed_n_kv_heads:
                return state
            return (
                tuple(t[:, :n_kv, :, :] for t in keys),
                tuple(t[:, :n_kv, :, :] for t in values),
            )
        return state

    def _extract_cache_states(self, raw_cache: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract actual tensor state from each layer cache for paged storage.

        Converts raw KVCache/MambaCache objects into state-dict format that
        BlockAwarePrefixCache.store_cache() expects:
            {"state": (keys, values), "meta_state": (offset,), "class_name": "KVCache"}

        Handles CacheList (MoE models) by extracting each sub-cache independently,
        matching the format expected by _extract_block_tensor_slice().

        This is the VLM equivalent of Scheduler._extract_cache_states().
        """
        if not raw_cache:
            return []

        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None

        n_kv = self._detect_n_kv_heads()
        allowed_n_kv_heads = self._detect_allowed_n_kv_heads()
        extracted = []
        for i, layer_cache in enumerate(raw_cache):
            try:
                # CacheList (MoE models): extract each sub-cache independently.
                # Produces {"class_name": "CacheList", "sub_caches": [...]} format
                # matching what _extract_block_tensor_slice expects.
                if _CacheList is not None and isinstance(layer_cache, _CacheList):
                    sub_caches = []
                    for sc in layer_cache.caches:
                        if hasattr(sc, "cache") and isinstance(
                            getattr(sc, "cache", None), list
                        ):
                            # SSM sub-cache: cumulative state
                            sub_caches.append({
                                "state": sc.state,
                                "meta_state": sc.meta_state,
                                "class_name": type(sc).__name__,
                            })
                        elif hasattr(sc, "state") and hasattr(sc, "meta_state"):
                            sub_state = self._normalize_gqa_state(
                                sc.state,
                                n_kv,
                                allowed_n_kv_heads=allowed_n_kv_heads,
                            )
                            sub_caches.append({
                                "state": sub_state,
                                "meta_state": sc.meta_state,
                                "class_name": type(sc).__name__,
                                **(
                                    {
                                        "tq_config": {
                                            "key_bits": int(sc.key_bits),
                                            "value_bits": int(sc.value_bits),
                                            "seed": int(getattr(sc, "_seed", 42)),
                                        }
                                    }
                                    if type(sc).__name__ == "TurboQuantKVCache"
                                    else {}
                                ),
                            })
                    if sub_caches:
                        extracted.append({
                            "class_name": "CacheList",
                            "state": None,
                            "meta_state": None,
                            "sub_caches": sub_caches,
                        })
                    continue
                # MambaCache/ArraysCache: cumulative state (SSM layers in hybrid models).
                # Include in extraction so _extract_block_tensor_slice() can tag them
                # as ("cumulative", ...) in the last block, enabling prefix cache
                # restore for hybrid SSM models on exact prefix matches.
                if hasattr(layer_cache, "cache") and isinstance(
                    getattr(layer_cache, "cache", None), list
                ):
                    if hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                        cls_name = type(layer_cache).__name__
                        extracted.append({
                            "state": layer_cache.state,
                            "meta_state": layer_cache.meta_state,
                            "class_name": cls_name,
                        })
                    else:
                        cls_name = type(layer_cache).__name__
                        extracted.append({
                            "state": None,
                            "meta_state": None,
                            "class_name": cls_name,
                        })
                    continue
                elif hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                    state = self._normalize_gqa_state(
                        layer_cache.state,
                        n_kv,
                        allowed_n_kv_heads=allowed_n_kv_heads,
                    )
                    meta = layer_cache.meta_state
                    cls_name = type(layer_cache).__name__
                    # Ensure QuantizedKVCache meta includes group_size and bits.
                    if cls_name == "QuantizedKVCache" and isinstance(meta, (tuple, list)) and len(meta) < 3:
                        g = getattr(layer_cache, 'group_size', 64)
                        b = getattr(layer_cache, 'bits', 8)
                        meta = (meta[0] if meta else '0', str(g), str(b))
                    extracted.append({
                        "state": state,
                        "meta_state": meta,
                        "class_name": cls_name,
                        **(
                            {
                                "tq_config": {
                                    "key_bits": int(layer_cache.key_bits),
                                    "value_bits": int(layer_cache.value_bits),
                                    "seed": int(getattr(layer_cache, "_seed", 42)),
                                }
                            }
                            if cls_name == "TurboQuantKVCache"
                            else {}
                        ),
                    })
                else:
                    logger.debug(
                        f"VLM cache layer {i} ({type(layer_cache).__name__}) "
                        f"lacks state/meta_state — skipping"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract VLM cache state from layer {i}: {e}")

        if extracted:
            logger.debug(
                f"VLM cache extraction: {len(extracted)}/{len(raw_cache)} layers"
            )

        return extracted

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer.

        Also checks the model config registry for additional eos_tokens
        (e.g., Gemma 4 uses <turn|> as end-of-turn alongside <eos>).
        """
        stop_tokens = set()
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                stop_tokens.update(tokenizer.eos_token_id)
            else:
                stop_tokens.add(tokenizer.eos_token_id)

        if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids is not None:
            if isinstance(tokenizer.eos_token_ids, (list, set, tuple)):
                stop_tokens.update(tokenizer.eos_token_ids)
            else:
                stop_tokens.add(tokenizer.eos_token_ids)

        # Add extra eos_tokens from model config registry.
        # Includes ALL entries, not just [1:] — for MLLM models the
        # tokenizer is loaded by mlx_vlm without the LLM-path eos override,
        # so index 0 may NOT already be set on the tokenizer. Gemma 3 / 3n
        # specifically: registry puts `<end_of_turn>` at index 0 but the
        # tokenizer's built-in eos_token_id is `<eos>`, so without this
        # the model loops emitting `<end_of_turn>` forever. Set semantics
        # makes duplicates a no-op.
        model_name = getattr(tokenizer, 'name_or_path', None)
        if model_name:
            try:
                from .model_config_registry import get_model_config_registry
                registry = get_model_config_registry()
                model_config = registry.lookup(model_name)
                if model_config.eos_tokens:
                    for eos_str in model_config.eos_tokens:
                        try:
                            ids = tokenizer.encode(eos_str, add_special_tokens=False)
                            if len(ids) == 1:
                                stop_tokens.add(ids[0])
                                logger.debug(f"Added extra stop token: {eos_str!r} → {ids[0]}")
                        except Exception:
                            pass
            except Exception:
                pass

        return stop_tokens

    def _ensure_batch_generator(
        self, sampling_params: Optional[SamplingParams] = None
    ) -> None:
        """Ensure batch generator exists with compatible sampling parameters.

        If sampling_params differ from the current generator's settings,
        the generator is recreated (unless active requests prevent it).

        On recreation, clears ALL 3 cache tiers (paged, memory-aware, legacy)
        because cache objects contain tensor references tied to the old generator.
        Passes all cache objects and quantization settings to the new generator.
        """
        from .sampling import make_sampler

        # If no sampling params provided and generator exists, keep current
        if sampling_params is None and self.batch_generator is not None:
            return

        # Use provided params or sensible defaults
        temp = sampling_params.temperature if sampling_params else 0.7
        top_p = sampling_params.top_p if sampling_params else 0.9
        top_k = sampling_params.top_k if sampling_params else 0
        min_p = sampling_params.min_p if sampling_params else 0.0
        rep_penalty = sampling_params.repetition_penalty if sampling_params else 1.0

        new_params = (temp, top_p, top_k, min_p, rep_penalty)

        if self.batch_generator is not None:
            if self._current_sampler_params != new_params:
                # Sampling params changed — update the generator's default sampler
                # in place instead of recreating. Per-request samplers (via
                # _make_request_sampler) override this anyway, so this only
                # affects requests without explicit sampling params.
                # This avoids clearing all caches on temperature changes.
                self.batch_generator.sampler = make_sampler(
                    temp=temp, top_p=top_p, min_p=min_p, top_k=top_k
                )
                self._current_sampler_params = new_params
                logger.debug(
                    f"Updated MLLM default sampler: temp={temp}, top_p={top_p}, "
                    f"top_k={top_k}, min_p={min_p}"
                )
            return

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)

        self.batch_generator = MLLMBatchGenerator(
            model=self.model,
            processor=self.processor,
            max_tokens=self.config.default_max_tokens,
            stop_tokens=self.stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            enable_vision_cache=self.config.enable_vision_cache,
            vision_cache_size=self.config.vision_cache_size,
            paged_cache_manager=self.paged_cache_manager,
            block_aware_cache=self.block_aware_cache,
            memory_aware_cache=self.memory_aware_cache,
            prefix_cache=self.prefix_cache,
            disk_cache=self.disk_cache,
            kv_cache_bits=self._kv_cache_bits,
            kv_cache_group_size=self._kv_cache_group_size,
            ssm_state_cache_size=self.config.ssm_state_cache_size,
            ssm_state_cache_max_mb=self.config.ssm_state_cache_max_mb,
            ssm_state_disk_store=self._ssm_companion_disk_store,
            ssm_state_cache_model_key=self._ssm_companion_model_key,
            enable_prefix_cache=self.config.enable_prefix_cache,
            uses_zaya_cache=self._uses_zaya_cache,
            mixed_attention_cache_model=self._mixed_attention_cache_model,
        )
        self._current_sampler_params = new_params

    def _prefill_for_prompt_only_cache(self, tokens: List[int]) -> Optional[List[Any]]:
        """Run clean text-only prompt prefill for cache warm endpoint."""
        if not tokens:
            return None
        self._ensure_batch_generator()
        if self.batch_generator is None:
            return None
        return self.batch_generator._prefill_for_clean_path_dependent_cache(tokens)

    # ========== Sync API (step-based) ==========

    def add_request(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[Any]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        request_id: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Add a multimodal request to the scheduler (sync version).

        Args:
            prompt: Text prompt (should be formatted with chat template)
            images: List of image inputs (paths, URLs, base64)
            videos: List of video inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            request_id: Optional custom request ID
            stop: Stop sequences (string patterns)
            **kwargs: Additional generation parameters

        Returns:
            Request ID for tracking
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # H2: Guard against excessive images causing Metal OOM
        if images and len(images) > self.config.max_images_per_request:
            raise ValueError(
                f"Request contains {len(images)} images, exceeding the limit of "
                f"{self.config.max_images_per_request}. Reduce image count or increase "
                f"max_images_per_request in config."
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=kwargs.get("top_k", 0),
            min_p=kwargs.get("min_p", 0.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            logit_bias=kwargs.get("logit_bias"),
            seed=kwargs.get("seed"),
            stop=stop or [],
            stop_token_ids=kwargs.get("stop_token_ids", []),
        )

        request = MLLMRequest(
            request_id=request_id,
            prompt=prompt,
            images=images,
            videos=videos,
            audio=audio,
            sampling_params=sampling_params,
            image_token_budget=kwargs.get("image_token_budget"),
            image_controls=kwargs.get("image_controls"),
            media_controls_strict=bool(kwargs.get("media_controls_strict") or False),
            video_fps=kwargs.get("video_fps"),
            video_max_frames=kwargs.get("video_max_frames"),
            video_controls=_video_controls_from_kwargs(kwargs),
        )
        request.extra_kwargs = {
            key: value
            for key, value in kwargs.items()
            if isinstance(key, str) and key.startswith("_vmlx_")
        }
        if "enable_thinking" in kwargs:
            request.enable_thinking = kwargs.get("enable_thinking")
        cache_extra_keys = kwargs.get("cache_extra_keys")
        if cache_extra_keys is not None:
            request._cache_extra_keys = (
                dict(cache_extra_keys)
                if isinstance(cache_extra_keys, dict)
                else {"request": repr(cache_extra_keys)}
            )
        _max_prompt_tokens = int(kwargs.get("max_prompt_tokens", 0) or 0)
        if _max_prompt_tokens > 0:
            request._max_prompt_tokens = _max_prompt_tokens
            if not images and not videos and not audio:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                try:
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                except TypeError:
                    token_ids = tokenizer.encode(prompt)
                if len(token_ids) > _max_prompt_tokens:
                    raise PromptTooLongError(
                        len(token_ids),
                        _max_prompt_tokens,
                        source="tokenized VLM text prompt",
                        request_id=request_id,
                    )
        # Mark multi-turn requests for cache skip heuristic.
        # num_messages > 2 means at least system + user + assistant history.
        request._has_history = kwargs.get("num_messages", 1) > 2

        # Attach gen_prompt_len for prefix cache key stripping.
        # Strips generation prompt tokens (e.g., <|im_start|>assistant\n<think>\n)
        # from the paged cache block hash to enable multi-turn cache hits.
        _gpl = kwargs.get("gen_prompt_len", 0)
        if _gpl > 0:
            request._gen_prompt_len = _gpl

        # Per-request cache bypass (cache_salt / skip_prefix_cache). When set,
        # the MLLM scheduler will skip EVERY prefix cache layer — paged,
        # memory-aware, legacy, disk L2, block disk, SSM companion, and the
        # multimodal vision / pixel_values caches.
        if kwargs.get("bypass_prefix_cache", False):
            request._bypass_prefix_cache = True

        with self._queue_lock:
            if request_id in self.requests:
                raise ValueError(f"Request {request_id} already exists")
            self.requests[request_id] = request
            self.waiting.append(request)

        logger.debug(
            f"Added MLLM request {request_id}: "
            f"{len(images or [])} images, {len(videos or [])} videos"
        )

        return request_id

    def request_progress(self, request_id: str) -> Optional[int]:
        """Monotonic progress counter for a live request, or None if unknown.

        MLLMRequest has no num_computed_tokens; approximate prefill progress
        with num_prompt_tokens once known, plus the LIFETIME generated count.

        Uses ``total_output_tokens``, not ``num_output_tokens``. The latter is
        derived from ``len(output_tokens)`` and the recovery-retry path clears
        that list, so summing it made this counter DECREASE mid-request. The
        shared consumer in ``server.py`` credits only ``progress >
        last_progress`` as liveness and treats a positive-but-not-greater
        reading as neither progress nor "no reading", so a healthy retrying
        request could be killed as wedged — precisely the defect fixed for the
        text scheduler, which this path had silently kept.
        """
        with self._queue_lock:
            request = self.requests.get(request_id)
            if request is None:
                return None
            # `num_prompt_tokens` is assigned only when the FIRST output token
            # arrives; `_prefill_tokens_done` is advanced per chunk by the
            # generator's prefill loop. max() of the two, not their sum — they
            # describe the SAME tokens, and adding them would double-count the
            # prompt the moment decode starts (the 2x defect fixed earlier
            # today for the text scheduler, reintroduced through a new field).
            prompt_side = max(
                int(getattr(request, "num_prompt_tokens", 0) or 0),
                int(getattr(request, "_prefill_tokens_done", 0) or 0),
            )
            return prompt_side + int(getattr(request, "total_output_tokens", 0) or 0)

    def _cleanup_aborted_paged_request(self, request_id: str) -> None:
        """Release one aborted request without destroying a durable SSD hit.

        ``fetch_cache()`` registers an entry in the block-aware request table
        only after it has found and pinned an existing prefix. Those blocks are
        already durable cache records. Deleting their block table removes the
        promoted block objects from ``allocated_blocks`` and makes the live
        prefix index fail validation, even though the SSD files still exist.

        A cold request has no block-aware entry, so its request table can hold
        partially computed, never-published blocks and must still be deleted.
        """

        paged_entry = None
        if self.block_aware_cache is not None:
            finalize_credit = getattr(
                self.block_aware_cache,
                "finalize_cache_hit_credit",
                None,
            )
            if callable(finalize_credit):
                finalize_credit(request_id)
            paged_entry = self.block_aware_cache._request_tables.pop(
                request_id,
                None,
            )

        if self.paged_cache_manager is None:
            return

        block_table = (
            getattr(paged_entry, "block_table", None)
            if paged_entry is not None
            else None
        )
        if block_table is None:
            self.paged_cache_manager.delete_block_table(request_id)
            return

        self.paged_cache_manager.release_request_refs(block_table)
        self.paged_cache_manager.detach_request(request_id)

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a request.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted
        """
        with self._queue_lock:
            request = self.requests.pop(request_id, None)
            if request is None:
                return False

            # Remove from waiting queue
            if request.status == RequestStatus.WAITING:
                try:
                    self.waiting.remove(request)
                except ValueError:
                    pass

            # DEFER batch generator removal to prevent Metal assertion crash.
            # Client disconnects can happen while Metal command buffers are
            # in-flight. Calling remove() immediately would touch cache
            # tensors mid-computation. Instead, mark for deferred cleanup
            # which runs after the current Metal computation completes.
            if request_id in self.request_id_to_uid:
                if not hasattr(self, '_pending_aborts'):
                    self._pending_aborts = set()
                self._pending_aborts.add(request_id)
                # Don't remove UID mappings here — done in deferred processing

            if request_id in self.running:
                del self.running[request_id]

            # Cold request blocks were never published and must be deleted.
            # Existing SSD-hit blocks are durable and only release their
            # request refs; deleting those mappings poisons same-process reuse.
            self._cleanup_aborted_paged_request(request_id)

            # Clean up streaming detokenizer
            self._cleanup_detokenizer(request_id)

            # Clear extracted cache GC reference
            request._extracted_cache = None

            # Mark as aborted
            request.status = RequestStatus.FINISHED_ABORTED
            self.finished_req_ids.add(request_id)

            # Signal output queue (inside lock to prevent race with queue cleanup)
            if request_id in self.output_queues:
                try:
                    self.output_queues[request_id].put_nowait(None)
                except asyncio.QueueFull:
                    # Force-deliver sentinel to prevent stream_outputs hang
                    try:
                        self.output_queues[request_id].get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self.output_queues[request_id].put_nowait(None)
                    except asyncio.QueueFull:
                        pass

        # Free Metal memory when all requests done
        if not self.running:
            clear_mlx_memory_cache(log=logger)

        logger.debug(f"Aborted request {request_id}")
        return True

    def has_requests(self) -> bool:
        """Check for generation work. Foreground only.

        The queued hybrid SSM idle rederive is surfaced via
        ``has_idle_tasks()`` and drained by ``_process_loop``'s idle branch
        AFTER responses are finalized (vmlx#245) — it must never keep
        ``step()`` on the response path.
        """
        return bool(self.waiting or self.running)

    def has_idle_tasks(self) -> bool:
        """Whether hybrid SSM rederive maintenance is queued (vmlx#245)."""
        generator = getattr(self, "batch_generator", None)
        return bool(
            getattr(self, "_is_hybrid", False)
            and generator is not None
            and getattr(generator, "_ssm_rederive_queue", None)
        )

    def run_one_idle_task(self) -> bool:
        """Run one queued SSM rederive at engine idle. Returns True if ran.

        Must be called on the step executor (Metal stream affinity) and only
        when no foreground work exists — the process loop's idle branch
        guarantees both. The MLLM clean prefill is one-shot by design
        (chunking breaks ArraysCache offset math), so preemption granularity
        is one queue entry, not one chunk.
        """
        if self.waiting or self.running:
            return False
        generator = getattr(self, "batch_generator", None)
        if generator is None or not hasattr(generator, "run_idle_rederive"):
            return False
        try:
            with self._batch_lock:
                return bool(generator.run_idle_rederive())
        except Exception as _rd_err:
            logger.debug(f"MLLM idle rederive tick failed: {_rd_err}")
            return True

    def request_graceful_stop(self, request_id: str) -> bool:
        """Ask an active generator row to finish through normal cleanup.

        Unlike ``abort_request()``, this preserves the request until a real
        finished response carries its prompt-boundary cache into
        ``_cleanup_finished()``.  Used by API parsers after a complete native
        tool call has been recognized but the model has not emitted EOS.
        """
        with self._queue_lock:
            request = self.running.get(request_id)
            uid = self.request_id_to_uid.get(request_id)
            collector_present = request_id in getattr(self, "output_queues", {})
            if (
                request is None
                or uid is None
                or RequestStatus.is_finished(request.status)
            ):
                # The model worker can finish a burst and detach its active row
                # before the async API consumer reaches the already-queued tool
                # bytes.  A live collector means the natural terminal is still
                # owned by stream_outputs(); drain it instead of converting
                # this harmless race into abort_request(), which discards the
                # cache-bearing finish response before scheduler cleanup.
                if collector_present:
                    logger.info(
                        "Graceful parser stop for %s is already terminalizing; "
                        "draining the owned output collector",
                        request_id,
                    )
                    return True
                return False
        generator = self.batch_generator
        stop = getattr(generator, "request_graceful_stop", None)
        if not callable(stop):
            return False
        with self._batch_lock:
            accepted = bool(stop(uid))
        if accepted:
            logger.info(
                "Graceful parser stop requested for %s (uid=%s); "
                "draining to cache-persisted terminal",
                request_id,
                uid,
            )
            return True
        if collector_present:
            logger.info(
                "Graceful parser stop for %s found no active generator row; "
                "draining the owned terminal collector",
                request_id,
            )
            return True
        return False

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _schedule_waiting(self) -> List[MLLMRequest]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        # Use first waiting request's sampling params to configure the generator
        first_params = self.waiting[0].sampling_params if self.waiting else None
        self._ensure_batch_generator(first_params)

        scheduled = []
        batch_requests = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            # Memory-pressure guard: don't admit new requests if GPU memory is critically low
            try:
                active_mem, max_mem = get_effective_metal_working_set_bytes(mx)
                guard_threshold = get_metal_ws_guard_threshold()
                if active_mem > 0 and len(self.running) > 0:
                    if max_mem > 0 and (active_mem / max_mem * 100.0) >= guard_threshold:
                        logger.debug(
                            "Memory pressure (%.1fGB / %.1fGB = %.0f%%), "
                            "deferring new request admission",
                            active_mem / 1e9,
                            max_mem / 1e9,
                            active_mem / max_mem * 100.0,
                        )
                        break
            except Exception:
                pass  # Metal API not available — skip check

            request = self.waiting.popleft()

            # Create batch request
            batch_req = MLLMBatchRequest(
                uid=-1,  # Will be assigned by batch generator
                request_id=request.request_id,
                prompt=request.prompt,
                images=request.images,
                videos=request.videos,
                audio=request.audio,
                max_tokens=request.sampling_params.max_tokens,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                top_k=request.sampling_params.top_k,
                min_p=request.sampling_params.min_p,
                repetition_penalty=request.sampling_params.repetition_penalty,
                frequency_penalty=request.sampling_params.frequency_penalty,
                presence_penalty=request.sampling_params.presence_penalty,
                logit_bias=request.sampling_params.logit_bias,
                seed=request.sampling_params.seed,
                max_prompt_tokens=int(getattr(request, "_max_prompt_tokens", 0) or 0),
                enable_thinking=getattr(request, "enable_thinking", None),
                image_token_budget=request.image_token_budget,
                image_controls=getattr(request, "image_controls", None),
                media_controls_strict=bool(getattr(request, "media_controls_strict", False)),
                video_fps=request.video_fps,
                video_max_frames=request.video_max_frames,
                video_controls=getattr(request, "video_controls", None),
            )
            if request.extra_kwargs:
                batch_req.extra_kwargs.update(request.extra_kwargs)
            # Forward gen_prompt_len so the batch generator can strip it
            # from cache fetch keys to match the store key (which also strips).
            _gpl = getattr(request, '_gen_prompt_len', 0)
            if _gpl > 0:
                batch_req._gen_prompt_len = _gpl
            if getattr(request, '_bypass_prefix_cache', False):
                batch_req._bypass_prefix_cache = True
            if getattr(request, "_cache_extra_keys", None):
                batch_req._cache_extra_keys = dict(request._cache_extra_keys)
            if request.sampling_params.stop:
                batch_req._stop_strings = list(request.sampling_params.stop)
            batch_requests.append(batch_req)

            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            # NOTE: prompt token count is NOT known yet at scheduling time
            # (it comes from batch generator's first response).
            # Tracking is done at request finish time instead.
            # Publish an admission-time miss record (text-scheduler parity:
            # "Publish misses and discarded hits too") so a cold request can
            # never leave a stale/absent last_cache_execution — the finish
            # handler overwrites it with the real per-request record when the
            # generator supplies one.
            _admission_execution = {
                "request_id": request.request_id,
                "cache_detail": None,
                "attempted_cached_tokens": 0,
                "cached_tokens": 0,
                "cache_outcome": "miss",
                "cache_reuse_applied": False,
            }
            request._cache_execution = dict(_admission_execution)
            # The batch generator is created lazily at first prefill — a
            # batch-stats write here silently no-ops on the first request.
            # Hold the record on the scheduler; get_stats falls back to it
            # whenever the generator stats carry none.
            self._admission_cache_execution = dict(_admission_execution)
            logger.info(
                "MLLM admission telemetry record set for %s (scheduler id=%s)",
                request.request_id,
                hex(id(self)),
            )
            _admission_stats = getattr(
                getattr(self, "batch_generator", None), "_stats", None
            )
            if _admission_stats is not None:
                _admission_stats.last_cache_execution = dict(
                    _admission_execution
                )
            scheduled.append(request)

        # Merge per-request stop_token_ids into batch generator stop tokens.
        # Track per-request additions so they can be removed on cleanup
        # (prevents stop tokens leaking across unrelated requests).
        if batch_requests and self.batch_generator is not None:
            # Guarded: upstream mlx_lm BatchGenerator has NO `stop_tokens`
            # attribute — unguarded this raised AttributeError out of
            # scheduling for any request with stop_token_ids on the batched
            # MLLM path (issue #229-#236 review; mirror of the LLM scheduler
            # site). The guard must NOT gate the insert() below.
            if hasattr(self.batch_generator, "stop_tokens"):
                for request in scheduled:
                    if request.sampling_params.stop_token_ids:
                        new_tokens = set(request.sampling_params.stop_token_ids)
                        self.batch_generator.stop_tokens.update(new_tokens)
                        request._added_stop_tokens = new_tokens
            uids = self.batch_generator.insert(batch_requests)

            for uid, request in zip(uids, scheduled):
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid

                logger.debug(f"Scheduled request {request.request_id} (uid={uid})")

        return scheduled

    def _process_batch_responses(
        self, responses: List[MLLMBatchResponse]
    ) -> Tuple[List[RequestOutput], Set[str]]:
        """
        Process responses from batch generator.

        Args:
            responses: List of MLLMBatchResponse objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        def _has_pending_gen_prefix(req: MLLMRequest, resp: MLLMBatchResponse) -> bool:
            if hasattr(req, "_gen_prefix_tokens"):
                return bool(getattr(req, "_gen_prefix_tokens", None))
            return bool(getattr(resp, "gen_prefix_tokens", None) or [])

        def _can_coalesce(req: MLLMRequest, resp: MLLMBatchResponse) -> bool:
            if getattr(req.sampling_params, "stop", None):
                return False
            if _has_pending_gen_prefix(req, resp):
                return False
            if resp.finish_reason == "error" or getattr(resp, "error", None):
                return False
            if getattr(resp, "logprobs", None) is not None:
                return False
            return True

        def _coalesce_until(start: int, req: MLLMRequest, uid: int) -> int:
            end = start
            while end < len(responses):
                candidate = responses[end]
                if candidate.uid != uid:
                    break
                if not _can_coalesce(req, candidate):
                    break
                end += 1
                if candidate.finish_reason is not None:
                    break
            return end

        def _process_coalesced(
            request_id: str,
            request: MLLMRequest,
            burst: List[MLLMBatchResponse],
        ) -> RequestOutput:
            for resp in burst:
                cache_extra_keys = getattr(resp, "cache_extra_keys", None)
                if cache_extra_keys:
                    request._cache_extra_keys = dict(cache_extra_keys)

            if request.num_prompt_tokens == 0:
                for resp in burst:
                    prompt_ids = getattr(resp, "prompt_token_ids", None)
                    if prompt_ids:
                        request.num_prompt_tokens = len(prompt_ids)
                        break

            tokens = [int(resp.token) for resp in burst]
            request.output_tokens.extend(tokens)
            request.num_output_tokens = len(request.output_tokens)
            request.total_output_tokens = (
                request._retry_output_base + request.num_output_tokens
            )

            detok = self._get_detokenizer(request_id, tokenizer)
            for resp in burst:
                if resp.finish_reason != "stop":
                    detok.add_token(resp.token)
            new_text = detok.last_segment

            finish_response = next(
                (resp for resp in reversed(burst) if resp.finish_reason is not None),
                None,
            )
            response_for_usage = finish_response or burst[-1]
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=tokens,
                new_text=new_text,
                output_token_ids=list(request.output_tokens),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=max(
                    int(getattr(resp, "cached_tokens", 0) or 0) for resp in burst
                ),
                cache_detail=(
                    next(
                        (
                            getattr(resp, "cache_detail", "")
                            for resp in reversed(burst)
                            if getattr(resp, "cache_detail", "")
                        ),
                        "",
                    )
                    or ""
                ),
            )

            if finish_response is not None:
                if getattr(finish_response, "prompt_cache", None) is not None:
                    request._extracted_cache = finish_response.prompt_cache
                    # The media clean-store lane keys by a BLOCK-ALIGNED
                    # prefix, which is shorter than prompt_token_ids; using the
                    # full list here would store a payload under a key it does
                    # not cover.
                    request._extracted_tokens = (
                        getattr(finish_response, "clean_store_token_ids", None)
                        or getattr(finish_response, "prompt_token_ids", [])
                    )
                else:
                    logger.info(
                        "VLM finish response for %s had no prompt_cache "
                        "(finish_reason=%s, prompt_tokens=%d); prefix cache "
                        "store will be skipped",
                        request_id,
                        finish_response.finish_reason,
                        len(getattr(finish_response, "prompt_token_ids", []) or []),
                    )

                finish_reason = finish_response.finish_reason
                if finish_reason == "stop":
                    request.status = RequestStatus.FINISHED_STOPPED
                elif finish_reason == "length":
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED

                output.finished = True
                output.finish_reason = finish_reason
                final_text_delta = _finalize_detokenizer_delta(detok)
                if final_text_delta:
                    output.new_text += final_text_delta
                output.output_text = detok.text
                request.output_text = output.output_text
                request.finish_reason = finish_reason

                self.total_prompt_tokens += request.num_prompt_tokens
                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1
                # The finish response is not guaranteed to carry the
                # execution record (proven live on qwen3.8) — scan the whole
                # burst for the newest record-carrying response as fallback.
                _burst_execution = next(
                    (
                        getattr(r, "cache_execution", None)
                        for r in reversed(burst)
                        if isinstance(getattr(r, "cache_execution", None), dict)
                        and getattr(r, "cache_execution", None)
                    ),
                    None,
                )
                self._record_cache_hit(
                    response_for_usage,
                    request,
                    execution_fallback=_burst_execution,
                )

            return output

        idx = 0
        while idx < len(responses):
            response = responses[idx]
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                idx += 1
                continue

            request = self.running.get(request_id)
            if request is None:
                idx += 1
                continue

            coalesced_end = _coalesce_until(idx, request, response.uid)
            if coalesced_end - idx > 1:
                output = _process_coalesced(
                    request_id,
                    request,
                    responses[idx:coalesced_end],
                )
                outputs.append(output)
                if output.finished:
                    finished_ids.add(request_id)
                idx = coalesced_end
                continue

            cache_extra_keys = getattr(response, "cache_extra_keys", None)
            if cache_extra_keys:
                request._cache_extra_keys = dict(cache_extra_keys)

            # Set prompt token count from first response (batch generator knows actual count)
            if request.num_prompt_tokens == 0:
                prompt_ids = getattr(response, "prompt_token_ids", None)
                if prompt_ids:
                    request.num_prompt_tokens = len(prompt_ids)

            # Append token to request
            request.output_tokens.append(response.token)
            request.num_output_tokens = len(request.output_tokens)
            request.total_output_tokens = (
                request._retry_output_base + request.num_output_tokens
            )

            # vmlx#reasoning-leak-2026-04-21: thinking-capable models
            # (Qwen 3.6 / Gemma 4 / MiniMax / Nemotron Cascade) occasionally
            # re-emit the generation prefix (<|im_start|>assistant\n<think>\n
            # or equivalent) as their first output tokens on multi-turn when
            # prior assistant history arrives without a reasoning_content
            # wrapper. The model predicts position L+1..L+gpl as if it were
            # starting a fresh turn from scratch.
            #
            # Detection: compare the first `gen_prompt_len` output tokens
            # against the prompt's trailing `gen_prompt_len` tokens. When they
            # match exactly, the model is echoing its own prompt suffix and we
            # suppress those tokens from the output stream. Once a divergence
            # or `gen_prompt_len` matches consume, pass-through resumes.
            #
            # Skip the check entirely when: no gen-prefix was captured, we've
            # already passed the prefix window, or the request opted out of
            # suppression (e.g. because divergence was detected mid-window).
            # On the first response the gen-prefix arrives from the batch
            # generator via `response.gen_prefix_tokens`; we snapshot it onto
            # the SchedulerRequest so subsequent tokens can consult the same
            # list without re-fetching from the response object.
            if not hasattr(request, "_gen_prefix_tokens"):
                request._gen_prefix_tokens = list(
                    getattr(response, "gen_prefix_tokens", None) or []
                )
            _gen_prefix = request._gen_prefix_tokens
            _skip_this_token = False
            if _gen_prefix:
                _out_idx = request.num_output_tokens - 1  # 0-based index of this token
                if _out_idx < len(_gen_prefix):
                    _expected = _gen_prefix[_out_idx]
                    if response.token == _expected:
                        # Re-emitted prefix — suppress
                        _skip_this_token = True
                        if _out_idx == 0:
                            logger.info(
                                f"Request {request_id}: suppressing re-emitted "
                                f"gen-prefix ({len(_gen_prefix)} tokens, model "
                                f"echoed prompt suffix on multi-turn)"
                            )
                    else:
                        # Divergence inside the window — mark as non-echo so
                        # subsequent tokens are treated normally.
                        request._gen_prefix_tokens = []

            # Use streaming detokenizer for correct multi-byte char handling
            detok = self._get_detokenizer(request_id, tokenizer)

            # Check if this is a stop token BEFORE adding to detokenizer
            # so stop tokens (e.g. <|im_end|>) don't leak into new_text
            is_stop = response.finish_reason == "stop"
            string_stop_truncate = -1  # >=0 when string stop matched

            if _skip_this_token:
                # Token consumed by gen-prefix suppression; no delta to emit
                new_text = ""
            elif not is_stop:
                detok.add_token(response.token)
                new_text = detok.last_segment

                # Post-decode string stop sequence check.
                # MLLMBatchGenerator only handles EOS stop tokens;
                # string stop sequences need decoded-text matching.
                # Skip matching inside <think> blocks — reasoning content
                # should not trigger user-specified stop sequences.
                if request.sampling_params.stop:
                    full_text = detok.text
                    # Skip matching inside unclosed <think> blocks
                    in_think = '<think>' in full_text and '</think>' not in full_text.split('<think>')[-1]
                    if not in_think:
                        max_stop_len = max(len(s) for s in request.sampling_params.stop)
                        search_start = max(0, len(full_text) - len(new_text) - max_stop_len + 1)
                        last_think_end = full_text.rfind('</think>')
                        if last_think_end >= 0:
                            search_start = max(search_start, last_think_end + len('</think>'))
                        for stop_str in request.sampling_params.stop:
                            idx = full_text.find(stop_str, search_start)
                            if idx >= 0:
                                string_stop_truncate = idx
                                new_text = ""
                                break
            else:
                new_text = ""

            # Create output
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=list(request.output_tokens),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=getattr(response, 'cached_tokens', 0),
                cache_detail=getattr(response, 'cache_detail', "") or "",
            )

            # Determine effective finish reason (string stop overrides)
            finish_reason = response.finish_reason
            if string_stop_truncate >= 0:
                finish_reason = "stop"

            # Check if finished
            if finish_reason is not None:
                if getattr(response, "prompt_cache", None) is not None:
                    request._extracted_cache = response.prompt_cache
                    request._extracted_tokens = (
                        getattr(response, "clean_store_token_ids", None)
                        or getattr(response, "prompt_token_ids", [])
                    )
                else:
                    logger.info(
                        "VLM finish response for %s had no prompt_cache "
                        "(finish_reason=%s, prompt_tokens=%d); prefix cache "
                        "store will be skipped",
                        request_id,
                        finish_reason,
                        len(getattr(response, "prompt_token_ids", []) or []),
                    )

                if finish_reason == "stop":
                    request.status = RequestStatus.FINISHED_STOPPED
                elif finish_reason == "length":
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                elif finish_reason == "error":
                    # Issue #56 Bug 1: surface batched prefill failures as a
                    # distinct status. Falls back to FINISHED_STOPPED if the
                    # status enum doesn't have an error variant (older builds).
                    request.status = getattr(
                        RequestStatus, "FINISHED_ERROR", RequestStatus.FINISHED_STOPPED
                    )
                    # Attach error detail on the request so the async output
                    # consumer / server.py can raise it as an HTTPException
                    # instead of a silent 200 with empty content.
                    _err = getattr(response, "error", None)
                    if _err:
                        request._prefill_error = _err
                        output.error = _err
                    _err_code = getattr(response, "error_code", None)
                    if _err_code:
                        output.error_code = _err_code
                        output.error_prompt_tokens = getattr(
                            response, "error_prompt_tokens", None
                        )
                        output.error_max_prompt_tokens = getattr(
                            response, "error_max_prompt_tokens", None
                        )
                        output.error_source = getattr(response, "error_source", None)

                output.finished = True
                output.finish_reason = finish_reason
                finished_ids.add(request_id)

                # Finalize detokenizer and use its complete text
                final_text_delta = _finalize_detokenizer_delta(detok)
                if string_stop_truncate < 0 and final_text_delta:
                    output.new_text += final_text_delta
                if string_stop_truncate >= 0:
                    output.output_text = detok.text[:string_stop_truncate]
                else:
                    output.output_text = detok.text
                request.output_text = output.output_text
                request.finish_reason = finish_reason

                # For string stop: tell batch generator to stop generating
                if string_stop_truncate >= 0 and self.batch_generator is not None:
                    uid = request.batch_uid
                    if uid is not None:
                        try:
                            self.batch_generator.remove([uid])
                        except Exception:
                            pass

                self.total_prompt_tokens += request.num_prompt_tokens
                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1
                self._record_cache_hit(response, request)

                logger.debug(
                    f"Request {request_id} finished: {finish_reason}, "
                    f"prompt={request.num_prompt_tokens}, "
                    f"completion={request.num_output_tokens} tokens"
                )

            outputs.append(output)
            idx += 1

        return outputs, finished_ids

    def _record_cache_hit(
        self,
        response: MLLMBatchResponse,
        request: MLLMRequest,
        execution_fallback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Aggregate per-request cache hits for /v1/cache/stats and /health.

        MLLM responses already expose cached_tokens in per-request API usage.
        Keep scheduler-level telemetry in sync so the panel and release gates
        do not report "0 cached tokens" while the response usage proves a hit.

        Publication of ``last_cache_execution`` runs BEFORE the
        already-recorded short-circuit and accepts a fallback record: live
        gate runs proved that the finish-time response can arrive WITHOUT its
        cache_execution dict (cold requests then publish nothing at all, and
        an earlier record-less call permanently blocked a later
        record-carrying one via the recorded marker). The marker now guards
        only the once-per-request counter increments; the record publication
        is idempotent.
        """
        try:
            cached_tokens = int(getattr(response, "cached_tokens", 0) or 0)
        except Exception:
            cached_tokens = 0
        response_execution = getattr(response, "cache_execution", None)
        if not (isinstance(response_execution, dict) and response_execution):
            if isinstance(execution_fallback, dict) and execution_fallback:
                response_execution = execution_fallback
            else:
                _req_execution = getattr(request, "_cache_execution", None)
                response_execution = (
                    _req_execution
                    if isinstance(_req_execution, dict) and _req_execution
                    else None
                )
        if isinstance(response_execution, dict) and response_execution:
            request._cache_execution = dict(response_execution)
            batch_stats = getattr(
                getattr(self, "batch_generator", None), "_stats", None
            )
            if batch_stats is not None:
                batch_stats.last_cache_execution = dict(response_execution)
        if getattr(request, "_cache_hit_recorded", False):
            return
        request._cached_tokens = cached_tokens
        if cached_tokens <= 0:
            request._cache_hit_recorded = True
            return
        detail = str(
            getattr(response, "cache_detail", "")
            or getattr(request, "_cache_detail", "")
            or "unknown"
        )
        request._cache_detail = detail
        self._cache_hit_requests += 1
        self._cache_hit_tokens += cached_tokens
        self._cache_hit_tokens_by_detail[detail] = (
            self._cache_hit_tokens_by_detail.get(detail, 0) + cached_tokens
        )
        execution = getattr(request, "_cache_execution", None)
        if isinstance(execution, dict):
            execution["cache_detail"] = detail
        batch_stats = getattr(getattr(self, "batch_generator", None), "_stats", None)
        batch_execution = getattr(batch_stats, "last_cache_execution", None)
        if isinstance(batch_execution, dict):
            batch_execution["cache_detail"] = detail
            if batch_stats is not None:
                batch_stats.last_cache_execution = dict(batch_execution)
        request._cache_hit_recorded = True

    def _clean_store_base_from_stored_chain(
        self,
        request_id: str,
        truncated_tokens: List[int],
        cache_extra_keys: Any,
    ) -> Tuple[Optional[List[Any]], int]:
        """Rebuild the longest already-stored prefix of ``truncated_tokens``.

        The clean re-derive that backs a mixed-SWA store used to start from an
        empty cache, so every turn re-prefilled the ENTIRE N-1 prompt — and the
        next request waited on it. Measured on Gemma 4: TTFT 51-92s at 86k-123k
        despite only ~12k fresh tokens, because turn N inherited turn N-1's ~30s
        store. Reconstructing the stored chain first costs far less than
        recomputing it (0.98s at 28k, 5.25s at 77k versus ~30s), and the delta
        is then the only thing forwarded.

        Returns ``(base_cache, covered_tokens)``; ``(None, 0)`` whenever a base
        is unavailable or would not save work, in which case the caller falls
        back to the full clean prefill. Block refs taken by the lookup are
        always released before returning.
        """
        cache = getattr(self, "block_aware_cache", None)
        if cache is None or not truncated_tokens:
            return None, 0
        probe_id = f"{request_id}::clean-store-base"
        try:
            block_table, _remaining = cache.fetch_cache(
                probe_id,
                list(truncated_tokens),
                cache_extra_keys=cache_extra_keys,
            )
            if block_table is None:
                return None, 0
            covered = int(getattr(block_table, "num_tokens", 0) or 0)
            # A base is only worth it when it covers a real majority of the
            # prompt; reconstructing a short chain costs more than forwarding
            # those tokens directly.
            if covered <= 0 or covered >= len(truncated_tokens):
                return None, 0
            if covered * 2 < len(truncated_tokens):
                return None, 0
            _base_started = time.perf_counter()
            base_cache = cache.reconstruct_cache(block_table)
            # Whether this second walk was served from the memo armed by the
            # request's own reconstruction. Logged rather than assumed: the memo
            # is keyed on (block_ids, num_tokens), so it only ever hits when the
            # store's base lands on the SAME table the request reconstructed.
            logger.info(
                "Clean store base for %s: %d tokens in %.3fs (memo_hit=%s)",
                request_id,
                int(getattr(block_table, "num_tokens", 0) or 0),
                max(0.0, time.perf_counter() - _base_started),
                bool(getattr(cache, "_last_reconstruct_memo_hit", False)),
            )
            if base_cache is None:
                return None, 0
            return base_cache, covered
        except Exception as exc:
            logger.debug(
                "Clean-store base reconstruction unavailable for %s (%s); "
                "falling back to full clean prefill",
                request_id,
                exc,
            )
            return None, 0
        finally:
            try:
                cache.release_cache(probe_id)
            except Exception:
                pass

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store KV cache for future prefix reuse.

        For each finished request, this method:
        1. Stores the extracted KV cache via the active cache tier (paged,
           memory-aware, or legacy) with optional disk L2 write and quantization.
        2. Cleans up the streaming detokenizer.
        3. Removes from running dict, UID mappings, paged cache tracking.
        4. Removes from master requests dict to free output_tokens/cache refs.
        5. Triggers Metal GC when all requests done.

        The _extracted_cache and _extracted_tokens attributes are set on the
        MLLMRequest by _process_batch_responses() during step(). They are always
        set to None in a finally block after store to prevent memory leaks.
        """
        trace_enabled = _mllm_scheduler_trace_enabled()
        trace_last = time.perf_counter() if trace_enabled else 0.0
        cleanup_trace: Dict[str, float] = {}

        def _trace_mark(name: str) -> None:
            nonlocal trace_last
            if not trace_enabled:
                return
            now = time.perf_counter()
            cleanup_trace[name] = cleanup_trace.get(name, 0.0) + max(
                now - trace_last, 0.0
            )
            trace_last = now

        # Snapshot stop tokens from requests that will survive this cleanup.
        # This prevents reading from self.running during the loop (it's mutated per iteration).
        _surviving_stops = set()
        for rid, req in self.running.items():
            if rid not in finished_ids:
                _surviving_stops.update(getattr(req, '_added_stop_tokens', set()))
        _trace_mark("snapshot_s")

        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Cacheability is decided by the prompt prefix, not by output
            # length. A first chat turn can produce a one-token answer and
            # still be the exact prefix needed by turn 2; skipping that store
            # makes MLLM routes semantically coherent but forces full
            # re-prefill and reports cached_tokens=0 on multi-turn chats.
            # Benchmarks that legitimately need fresh execution use the
            # explicit cache_salt / skip_prefix_cache bypass below.
            _output_len = getattr(request, 'num_output_tokens', 0) if request else 0
            _skip_cache_store = False
            _skip_cache_store_reason = ""
            # Hard bypass from cache_salt / skip_prefix_cache — overrides
            # normal cache storage and suppresses every store site.
            if request is not None and getattr(request, '_bypass_prefix_cache', False):
                _skip_cache_store = True
                _skip_cache_store_reason = "explicit prefix-cache bypass"
            # A hybrid cache restored for this request combines reconstructed
            # attention KV with path-dependent SSM state.  The restored prefix
            # is safe to consume, but promoting that live, extended cache into
            # a new longer paged entry compounds reconstruction error across
            # turns (Bonsai/Qwen3.5 eventually collapses into a token loop).
            # Keep the cold-prefill blocks reusable and recompute the growing
            # tail instead of recursively storing restored state.
            # The freeze above costs every hybrid conversation its multiturn
            # reuse: only the first cold prefill is ever stored, so turn 5 still
            # replays from turn 1. Opting in is measurable rather than
            # theoretical now that the companion is captured inline ("no
            # re-derive") instead of reconstructed, which is the premise the
            # rationale rests on. Default OFF -- the failure it guards against
            # (token-loop collapse) is worse than slow prefill, so this has to
            # be earned by a byte-exactness A/B over a LONG conversation, not
            # by a TTFT graph.
            if (
                request is not None
                and getattr(self, "_is_hybrid", False)
                and int(getattr(request, "_cached_tokens", 0) or 0) > 0
            ):
                if _hybrid_clean_store_enabled():
                    # Fall through to the clean re-prefill store instead of
                    # skipping: the chain still grows, but from a re-derived
                    # N-1 key rather than from restored state.
                    logger.info(
                        "Hybrid clean store for %s: extending the chain via a "
                        "clean N-1 re-prefill (%d reused)",
                        request_id,
                        int(getattr(request, "_cached_tokens", 0) or 0),
                    )
                elif _hybrid_prefix_promotion_enabled():
                    logger.info(
                        "Hybrid prefix promotion ENABLED for %s: extending a "
                        "restored prefix (%d reused) — experimental, watch for "
                        "drift",
                        request_id,
                        int(getattr(request, "_cached_tokens", 0) or 0),
                    )
                else:
                    _skip_cache_store = True
                    _skip_cache_store_reason = (
                        "hybrid restored-prefix promotion disabled"
                    )
            if _skip_cache_store:
                logger.debug(
                    f"Skipping cache store for {request_id}: "
                    f"output_len={_output_len}, {_skip_cache_store_reason}"
                )
                if request is not None:
                    request._extracted_cache = None
            if trace_enabled:
                logger.info(
                    "VMLINUX_MLLM_CLEANUP_STATE request_id=%s request_present=%s "
                    "block_cache=%s memory_cache=%s prefix_cache=%s disk_cache=%s "
                    "skip_store=%s extracted_cache=%s "
                    "extracted_tokens=%d prompt_tokens=%d completion_tokens=%d",
                    request_id,
                    request is not None,
                    getattr(self, "block_aware_cache", None) is not None,
                    getattr(self, "memory_aware_cache", None) is not None,
                    getattr(self, "prefix_cache", None) is not None,
                    getattr(self, "disk_cache", None) is not None,
                    _skip_cache_store,
                    bool(
                        request is not None
                        and getattr(request, "_extracted_cache", None) is not None
                    ),
                    len(getattr(request, "_extracted_tokens", []) or [])
                    if request is not None
                    else 0,
                    int(getattr(request, "num_prompt_tokens", 0) or 0)
                    if request is not None
                    else 0,
                    int(getattr(request, "num_output_tokens", 0) or 0)
                    if request is not None
                    else 0,
                )

            # --- Cache store: paged path ---
            if self.block_aware_cache is not None and not _skip_cache_store:
                if request is not None and getattr(request, "_extracted_cache", None) is not None:
                    try:
                        token_list = getattr(request, "_extracted_tokens", [])
                        if token_list:
                            media_context = self._mllm_request_has_media_cache_context(
                                request, token_list
                            )
                            media_cache_allowed = (
                                media_context
                                and self._mllm_media_prefix_cache_allowed(
                                    request, token_list
                                )
                            )
                            _media_skip_recorded = False
                            if media_context and not media_cache_allowed:
                                logger.info(
                                    "Skipping VLM prefix cache store for %s: "
                                    "prompt contains media context/placeholders; "
                                    "media embeddings are path-dependent and "
                                    "must not be rebuilt from text-only tokens",
                                    request_id,
                                )
                                request._extracted_cache = None
                                # Record the REAL reason here. Nulling the handle
                                # used to fall through to the generic "resolved
                                # extracted cache is empty" outcome below, which
                                # hid a deliberate family policy behind what read
                                # like a lost cache.
                                _PERSIST.record(
                                    request_id,
                                    "skipped",
                                    "media context: prefix reuse not allowed for this family",
                                )
                                _media_skip_recorded = True
                            prompt_len = len(token_list)
                            truncated_tokens = (
                                token_list[: prompt_len - 1]
                                if prompt_len > 1
                                else list(token_list)
                            )
                            cached_tokens = int(
                                getattr(request, "_cached_tokens", 0) or 0
                            )
                            _uses_zaya_cache = bool(
                                getattr(self, "_uses_zaya_cache", False)
                            )
                            _uses_mixed_attention_cache = bool(
                                getattr(self, "_mixed_attention_cache_model", False)
                            )
                            # A hybrid model's GatedDelta/SSM layers are
                            # path-dependent in exactly the way ZAYA's conv_state
                            # and Gemma's rotating windows are, so it belongs on
                            # the same clean re-prefill route rather than on the
                            # blanket skip that froze its prefix at turn one.
                            # Text turns only. The clean route re-prefills from
                            # token ids, which on a media prompt would rebuild
                            # the prefix WITHOUT the vision embeddings and store
                            # that as if it were the real thing -- the same trap
                            # the captured mixed-SWA branch below exists to
                            # avoid. Media turns keep whatever handling they had
                            # before this route was enabled.
                            _uses_hybrid_clean_store = bool(
                                getattr(self, "_is_hybrid", False)
                                and _hybrid_clean_store_enabled()
                                and not media_context
                            )
                            raw_for_layout = request._extracted_cache
                            if callable(raw_for_layout):
                                try:
                                    raw_for_layout = raw_for_layout()
                                    request._extracted_cache = raw_for_layout
                                except Exception:
                                    raw_for_layout = None
                            if (
                                not _uses_mixed_attention_cache
                                and isinstance(raw_for_layout, list)
                                and any(
                                    type(layer).__name__ == "RotatingKVCache"
                                    for layer in raw_for_layout
                                )
                            ):
                                _uses_mixed_attention_cache = True
                                logger.info(
                                    "Detected mixed-SWA VLM cache layout for %s "
                                    "from extracted RotatingKVCache layers; using "
                                    "clean prompt-boundary store",
                                    request_id,
                                )
                            if truncated_tokens and cached_tokens >= len(truncated_tokens):
                                logger.debug(
                                    "Skipping VLM paged cache store for %s: "
                                    "cached prefix already covers %d/%d cache-key tokens",
                                    request_id,
                                    cached_tokens,
                                    len(truncated_tokens),
                                )
                                _PERSIST.record(request_id, "already_durable", f"cached prefix covers {cached_tokens}/{len(truncated_tokens)} cache-key tokens", retained_tokens=cached_tokens)
                                cache_blocks = None
                                request._extracted_cache = None
                                # A warm turn whose prompt is covered by the
                                # restored boundary has nothing new to store.
                                # The generic "resolved extracted cache is
                                # empty" outcome below must not overwrite this
                                # one: it did, so every such turn read as a
                                # lost cache in the persistence ledger.
                                _media_skip_recorded = True
                            else:
                                raw = request._extracted_cache
                                _mixed_swa_boundary_cache = None
                                if (
                                    _uses_mixed_attention_cache
                                    and isinstance(raw, (list, tuple))
                                ):
                                    _mixed_swa_boundary_cache = (
                                        _assemble_mixed_swa_boundary_cache(
                                            self.batch_generator,
                                            request,
                                            raw,
                                            len(truncated_tokens),
                                        )
                                    )
                                if raw is None:
                                    cache_blocks = None
                                elif _mixed_swa_boundary_cache is not None:
                                    cache_blocks = _mixed_swa_boundary_cache
                                    logger.info(
                                        "Mixed-SWA store for %s: exact prompt-"
                                        "boundary cache assembled (%d layers, "
                                        "boundary=%d) — skipped the second "
                                        "prefill",
                                        request_id,
                                        len(cache_blocks),
                                        len(truncated_tokens),
                                    )
                                elif media_cache_allowed and _uses_mixed_attention_cache:
                                    # A media-conditioned mixed-SWA prefix can
                                    # never be rebuilt from token ids (the
                                    # ordinary path-dependent helper would run
                                    # a second *text-only* prefill and silently
                                    # discard vision-conditioned state), so the
                                    # only valid rotating state is a boundary
                                    # capture taken BEFORE decode advanced the
                                    # rings. `raw` alone does NOT prove that:
                                    # it is the live finish-time cache on warm
                                    # media turns (rings rolled past the
                                    # boundary by the reply length — the
                                    # Gemma-4 cached_tokens plateau), and only
                                    # on the cold aux-prefill path is it
                                    # already an exact prompt-boundary cache.
                                    # Assemble from the end-of-prefill snapshot
                                    # when one exists; otherwise accept `raw`
                                    # only if its rotating offsets PROVE the
                                    # boundary; otherwise store the live cache
                                    # loudly — its rotating layers become
                                    # honest `rotating_kv_pending` markers, and
                                    # fetches walk back to the newest exact
                                    # anchor instead of restoring a rolled
                                    # window as if it were the prompt boundary.
                                    cache_blocks = list(raw)
                                    logger.warning(
                                        "Mixed-SWA media store for %s has "
                                        "NO prompt-boundary capture "
                                        "(boundary=%d, %d layers): storing "
                                        "the live cache — rotating layers "
                                        "will be marked rotating_kv_pending "
                                        "and fetches walk back to the "
                                        "newest exact anchor",
                                        request_id,
                                        len(truncated_tokens),
                                        len(cache_blocks),
                                    )
                                elif (
                                    _uses_zaya_cache
                                    or _uses_mixed_attention_cache
                                    or _uses_hybrid_clean_store
                                ):
                                    # ZAYA CCA and Gemma-style mixed-SWA caches are
                                    # path-dependent. ZAYA conv_state/prev_hs and
                                    # RotatingKVCache windows cannot be recovered
                                    # safely from post-generation state. Re-prefill
                                    # exactly the N-1 cache key and store that clean
                                    # typed state. If the clean prefill is
                                    # unavailable, skip the store instead of writing
                                    # a contaminated hit that later hurts coherence
                                    # or speed.
                                    cache_blocks = None
                                    # Pure-hybrid layouts already hold both
                                    # halves of the boundary state: append-only
                                    # attention KV (sliceable) plus the
                                    # vmlx#109 inline recurrent snapshot taken
                                    # before generation advanced it. Assembling
                                    # those is exact and skips a SECOND full
                                    # prefill — profiled at 40.8% of engine time
                                    # (~28s at 15.4k), running after the
                                    # response and blocking the next request.
                                    if (
                                        _uses_hybrid_clean_store
                                        and not _uses_zaya_cache
                                        and not _uses_mixed_attention_cache
                                    ):
                                        _assembled = _assemble_clean_hybrid_boundary_cache(
                                            self.batch_generator,
                                            request,
                                            raw,
                                            len(truncated_tokens),
                                        )
                                        if _assembled is not None:
                                            cache_blocks = _assembled
                                            logger.info(
                                                "Clean hybrid boundary cache assembled "
                                                "for %s from live KV + inline recurrent "
                                                "snapshot (%d layers, %d tokens) — "
                                                "skipped the second prefill",
                                                request_id,
                                                len(_assembled),
                                                len(truncated_tokens),
                                            )
                                    prefill_fn = getattr(
                                        self.batch_generator,
                                        "_prefill_for_clean_path_dependent_cache",
                                        None,
                                    )
                                    if not callable(prefill_fn):
                                        prefill_fn = getattr(
                                            self.batch_generator,
                                            "_prefill_for_clean_ssm",
                                            None,
                                        )
                                    _tight_memory_drain_active = bool(
                                        getattr(
                                            self.batch_generator,
                                            "_tight_memory_prefill_drain",
                                            False,
                                        )
                                    )
                                    _force_tight_clean_store = os.environ.get(
                                        "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE",
                                        "0",
                                    ).lower() in {"1", "true", "yes", "on"}
                                    _tight_clean_store_max_tokens_raw = os.environ.get(
                                        "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE_MAX_TOKENS"
                                    )
                                    _tight_clean_store_cap_explicit = (
                                        _tight_clean_store_max_tokens_raw is not None
                                    )
                                    try:
                                        _tight_clean_store_max_tokens = int(
                                            _tight_clean_store_max_tokens_raw or "128"
                                        )
                                    except (TypeError, ValueError):
                                        _tight_clean_store_max_tokens = 512
                                    # 768 made the mixed-SWA cache useless in
                                    # practice. This cap sits ON TOP of the
                                    # _safe_headroom_min_gb check below, which
                                    # already refuses the clean re-prefill unless
                                    # 8 GB of Metal working set is free — so the
                                    # token cap was a second, redundant limit,
                                    # and it was the binding one.
                                    #
                                    # MEASURED on Step-3.7-Flash (mixed_swa_kv,
                                    # tight-memory drain active), identical
                                    # prompt sent twice:
                                    #     735 tok -> cached 734    HIT
                                    #     895 tok -> cached None   MISS
                                    # ...every larger size MISS, up to 6820, with
                                    # cached_blocks staying 0 — the store retained
                                    # NOTHING, so every turn re-prefilled in full
                                    # (17.75s / 18.47s / 25.20s TTFT in the app
                                    # where peers got 0.2-0.4s).
                                    #
                                    # Raising ONLY this cap, headroom check
                                    # untouched:
                                    #     before: 6820 tok, cached None, 30.87s
                                    #     after : 6820 tok, cached 6819,  8.09s
                                    # 3.8x on the warm turn, full reuse restored.
                                    #
                                    # The memory guard that matters is still the
                                    # 8 GB free-headroom requirement; this number
                                    # only stops a clean re-prefill so large it
                                    # would dwarf the turn it is meant to speed up.
                                    try:
                                        _safe_headroom_max_tokens = int(
                                            os.environ.get(
                                                "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_SAFE_HEADROOM_MAX_TOKENS",
                                                "32768",
                                            )
                                        )
                                    except (TypeError, ValueError):
                                        _safe_headroom_max_tokens = 32768
                                    try:
                                        _safe_headroom_min_gb = float(
                                            os.environ.get(
                                                "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_SAFE_HEADROOM_MIN_GB",
                                                "8",
                                            )
                                        )
                                    except (TypeError, ValueError):
                                        _safe_headroom_min_gb = 8.0
                                    try:
                                        _active_bytes, _max_ws_bytes = (
                                            get_effective_metal_working_set_bytes(mx)
                                        )
                                    except Exception:
                                        _active_bytes = 0
                                        _max_ws_bytes = 0
                                    _free_bytes = max(
                                        0,
                                        int(_max_ws_bytes or 0) - int(_active_bytes or 0),
                                    )
                                    _bounded_tight_clean_store = (
                                        _tight_clean_store_max_tokens > 0
                                        and prompt_len <= _tight_clean_store_max_tokens
                                    )
                                    _safe_headroom_clean_store = (
                                        _safe_headroom_max_tokens > 0
                                        and prompt_len <= _safe_headroom_max_tokens
                                        and (
                                            not _tight_clean_store_cap_explicit
                                            or _bounded_tight_clean_store
                                        )
                                        and _free_bytes
                                        >= int(max(0.0, _safe_headroom_min_gb) * 1024**3)
                                    )
                                    tight_memory_clean_store_disabled = (
                                        _uses_mixed_attention_cache
                                        and _tight_memory_drain_active
                                        and not _force_tight_clean_store
                                        and not _bounded_tight_clean_store
                                        and not _safe_headroom_clean_store
                                    )
                                    if tight_memory_clean_store_disabled:
                                        logger.info(
                                            "Skipping mixed-SWA VLM paged cache store "
                                            "for %s: tight-memory clean prompt "
                                            "prefill disabled to avoid Metal OOM "
                                            "(prompt_tokens=%d)",
                                            request_id,
                                            prompt_len,
                                        )
                                    elif (
                                        cache_blocks is None
                                        and truncated_tokens
                                        and callable(prefill_fn)
                                    ):
                                        # Only re-prefill when the boundary cache
                                        # could NOT be assembled above.
                                        _base_cache, _base_covered = (
                                            self._clean_store_base_from_stored_chain(
                                                request_id,
                                                truncated_tokens,
                                                getattr(
                                                    request, "_cache_extra_keys", None
                                                ),
                                            )
                                        )
                                        if _base_covered:
                                            logger.info(
                                                "Clean store for %s extends the stored "
                                                "chain: %d/%d tokens reused, forwarding "
                                                "%d new",
                                                request_id,
                                                _base_covered,
                                                len(truncated_tokens),
                                                len(truncated_tokens) - _base_covered,
                                            )
                                        cache_blocks = prefill_fn(
                                            truncated_tokens,
                                            _base_cache,
                                            _base_covered,
                                        )
                                    if cache_blocks is None:
                                        if _uses_zaya_cache:
                                            logger.info(
                                                "Skipping ZAYA VLM paged cache store "
                                                "for %s: clean zaya_cca_v1 prompt "
                                                "prefill unavailable",
                                                request_id,
                                            )
                                        else:
                                            logger.info(
                                                "Skipping mixed-SWA VLM paged cache store "
                                                "for %s: clean mixed_swa_kv_v1 prompt "
                                                "prefill unavailable",
                                                request_id,
                                            )
                                else:
                                    cache_blocks = raw() if callable(raw) else raw
                            if cache_blocks is None and _media_skip_recorded:
                                pass  # outcome already recorded with its real reason above
                            elif cache_blocks is None:
                                _PERSIST.record(request_id, "skipped", "resolved extracted cache is empty")
                                logger.info(
                                    "Skipping VLM paged cache store for %s: "
                                    "resolved extracted cache is empty "
                                    "(mixed_swa=%s, zaya_cca=%s, prompt_tokens=%d, "
                                    "cache_callable=%s, cached_tokens=%d, "
                                    "truncated_tokens=%d, media_context=%s, "
                                    "hybrid_clean_store=%s)",
                                    request_id,
                                    _uses_mixed_attention_cache,
                                    _uses_zaya_cache,
                                    prompt_len,
                                    callable(getattr(request, "_extracted_cache", None)),
                                    cached_tokens,
                                    len(truncated_tokens or []),
                                    bool(media_context),
                                    bool(_uses_hybrid_clean_store),
                                )
                            else:
                                if _uses_zaya_cache or _uses_mixed_attention_cache:
                                    cache_blocks = list(cache_blocks)
                                else:
                                    cache_blocks = self._prepare_tq_cache_for_storage(
                                        cache_blocks
                                    )
                                    if cache_blocks is not None:
                                        cache_blocks = self._truncate_hybrid_cache(
                                            cache_blocks, prompt_len
                                        )
                                if cache_blocks is None:
                                    logger.debug(
                                        f"Cache truncation failed for {request_id}"
                                    )
                                elif not self._validate_cache(
                                    cache_blocks,
                                    source=f"mllm-paged-store:{request_id}",
                                ):
                                    logger.warning(
                                        f"VLM paged cache store rejected unsafe "
                                        f"live cache for {request_id}"
                                    )
                                else:
                                    # NOTE: gen_prompt_len is already stripped from _extracted_tokens
                                    # by the batch generator (_original_token_ids at line 1411-1413).
                                    # Do NOT strip again here — double stripping collapses the
                                    # token list to near-zero length.
                                    # L2: persist to disk before quantization.
                                    # Skip for hybrid models — the prompt-level disk cache
                                    # can't reconstruct SSM state on fetch (mllm_batch_generator
                                    # guards with `if not self._is_hybrid`), so writing is
                                    # wasted I/O. Block-level disk cache (block_disk_store)
                                    # still works for hybrid via the paged path.
                                    if (
                                        self.disk_cache is not None
                                        and not self._is_hybrid
                                        and not media_context
                                    ):
                                        try:
                                            self.disk_cache.store(token_list, cache_blocks)
                                        except Exception as de:
                                            logger.debug(f"VLM disk cache store failed for {request_id}: {de}")

                                    if getattr(self, '_kv_cache_bits', 0):
                                        cache_blocks = self._quantize_cache_for_storage(cache_blocks)
                                        if not self._validate_cache(
                                            cache_blocks,
                                            source=f"mllm-paged-store-quant:{request_id}",
                                        ):
                                            logger.warning(
                                                f"VLM paged cache store rejected unsafe "
                                                f"quantized live cache for {request_id}"
                                            )
                                            cache_blocks = None
                                    if cache_blocks is not None:
                                        cache_states = self._extract_cache_states(cache_blocks)
                                        if cache_states:
                                            _paged_store_kwargs = {
                                                "cache_extra_keys": getattr(
                                                    request,
                                                    "_cache_extra_keys",
                                                    None,
                                                ),
                                            }
                                            if os.environ.get("VMLX_CACHE_HASH_DEBUG") == "1":
                                                _debug_block_size = max(
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
                                                _debug_blocks = [
                                                    "%d:%s"
                                                    % (
                                                        _off,
                                                        hashlib.sha256(
                                                            repr(
                                                                truncated_tokens[
                                                                    _off : _off
                                                                    + _debug_block_size
                                                                ]
                                                            ).encode()
                                                        ).hexdigest()[:12],
                                                    )
                                                    for _off in range(
                                                        0,
                                                        len(truncated_tokens),
                                                        _debug_block_size,
                                                    )
                                                ]
                                                _debug_side_key = hashlib.sha256(
                                                    repr(
                                                        _paged_store_kwargs[
                                                            "cache_extra_keys"
                                                        ]
                                                    ).encode()
                                                ).hexdigest()[:12]
                                                logger.info(
                                                    "mm-store-debug BLOCKS req=%s "
                                                    "n=%d block_size=%d side_key=%s %s",
                                                    request_id,
                                                    len(truncated_tokens),
                                                    _debug_block_size,
                                                    _debug_side_key,
                                                    " ".join(_debug_blocks),
                                                )
                                            if (
                                                self._is_hybrid
                                                and not self._uses_zaya_cache
                                                and not _uses_mixed_attention_cache
                                            ):
                                                _paged_store_kwargs[
                                                    "store_cumulative_state"
                                                ] = False
                                            stored_table = (
                                                self.block_aware_cache.store_cache(
                                                    request_id,
                                                    truncated_tokens,
                                                    cache_states,
                                                    **_paged_store_kwargs,
                                                )
                                            )
                                            side_key_suffix = (
                                                " with cache side-key"
                                                if getattr(
                                                    request,
                                                    "_cache_extra_keys",
                                                    None,
                                                )
                                                else ""
                                            )
                                            raw_retained_tokens = getattr(
                                                stored_table,
                                                "num_tokens",
                                                None,
                                            )
                                            retained_tokens = (
                                                raw_retained_tokens
                                                if isinstance(
                                                    raw_retained_tokens,
                                                    int,
                                                )
                                                and not isinstance(
                                                    raw_retained_tokens,
                                                    bool,
                                                )
                                                else None
                                            )
                                            block_table_ids = getattr(
                                                stored_table,
                                                "block_ids",
                                                None,
                                            )
                                            block_table_blocks = (
                                                len(block_table_ids)
                                                if isinstance(
                                                    block_table_ids,
                                                    (list, tuple),
                                                )
                                                else None
                                            )
                                            if (
                                                retained_tokens is None
                                                or block_table_blocks is None
                                            ):
                                                logger.warning(
                                                    "VLM Scheduler paged Prefix Cache store "
                                                    "returned incomplete retention receipt "
                                                    "for %s: %d layers, retained_tokens=%s, "
                                                    "block_table_blocks=%s, "
                                                    "requested_cache_key_tokens=%d%s",
                                                    request_id,
                                                    len(cache_states),
                                                    retained_tokens,
                                                    block_table_blocks,
                                                    len(truncated_tokens),
                                                    side_key_suffix,
                                                )
                                            elif (
                                                len(truncated_tokens) > 0
                                                and retained_tokens <= 0
                                            ):
                                                logger.warning(
                                                    "VLM Scheduler paged Prefix Cache store "
                                                    "retained no cache-key tokens for %s: "
                                                    "%d layers, block_table_blocks=%s, "
                                                    "requested_cache_key_tokens=%d%s",
                                                    request_id,
                                                    len(cache_states),
                                                    block_table_blocks,
                                                    len(truncated_tokens),
                                                    side_key_suffix,
                                                )
                                            else:
                                                logger.info(
                                                    "VLM Scheduler stored paged Prefix Cache "
                                                    "for %s: %d layers, retained_tokens=%s, "
                                                    "block_table_blocks=%s, "
                                                    "requested_cache_key_tokens=%d%s prefix_key=%s%s",
                                                    request_id,
                                                    len(cache_states),
                                                    retained_tokens,
                                                    block_table_blocks,
                                                    len(truncated_tokens),
                                                    side_key_suffix,
                                                    getattr(
                                                        self.block_aware_cache,
                                                        "prefix_key_for_block_ids",
                                                        lambda _ids: None,
                                                    )(block_table_ids),
                                                    (
                                                        f" block_keys={_bk}"
                                                        if (
                                                            _bk := getattr(
                                                                self.block_aware_cache,
                                                                "block_keys_for_block_ids",
                                                                lambda _ids: None,
                                                            )(block_table_ids)
                                                        )
                                                        else ""
                                                    ),
                                                )
                                            _PERSIST.record(
                                                request_id, "stored",
                                                f"paged {len(cache_states)} layers, blocks={block_table_blocks}",
                                                retained_tokens=int(retained_tokens) if isinstance(retained_tokens, int) else None,
                                            )
                                        else:
                                            _PERSIST.record(request_id, "skipped", "no storable layer states")
                                            logger.info(
                                                "Skipping VLM paged cache store for %s: "
                                                "extracted cache produced no storable layer states "
                                                "(raw_layers=%d, prompt_tokens=%d)",
                                                request_id,
                                                len(cache_blocks)
                                                if isinstance(cache_blocks, list)
                                                else -1,
                                                prompt_len,
                                            )
                        else:
                            _PERSIST.record(request_id, "skipped", "no prompt token ids")
                            logger.info(
                                "Skipping VLM paged cache store for %s: "
                                "finish response had cache but no prompt token ids",
                                request_id,
                            )
                    except Exception as e:
                        _PERSIST.record(request_id, "failed", f"paged: {e}")
                        logger.warning(f"Failed to store VLM paged cache for {request_id}: {e}", exc_info=True)
                    finally:
                        if request is not None:
                            request._extracted_cache = None
                elif request is not None:
                    _PERSIST.record(request_id, "skipped", "no extracted prompt cache on finished request")
                    logger.info(
                        "Skipping VLM paged cache store for %s: no extracted "
                        "prompt cache on finished request (prompt_tokens=%d, "
                        "completion_tokens=%d)",
                        request_id,
                        int(getattr(request, "num_prompt_tokens", 0) or 0),
                        int(getattr(request, "num_output_tokens", 0) or 0),
                    )

            # --- Cache store: memory-aware path ---
            elif self.memory_aware_cache is not None and not _skip_cache_store:
                if (
                    request is not None
                    and getattr(request, "_extracted_cache", None) is not None
                    and getattr(request, "_extracted_tokens", None)
                ):
                    try:
                        prompt_tokens = list(request._extracted_tokens)
                        prompt_len = len(prompt_tokens)
                        if self._mllm_request_has_media_cache_context(
                            request, prompt_tokens
                        ):
                            logger.info(
                                "Skipping VLM memory-aware cache store for %s: "
                                "prompt contains media context/placeholders; "
                                "media embeddings are path-dependent and "
                                "must not be saved under a token-only "
                                "prefix key",
                                request_id,
                            )
                            request._extracted_cache = None
                        raw_cache = request._extracted_cache
                        if callable(raw_cache):
                            raw_cache = raw_cache()
                        if not raw_cache:
                            logger.info(
                                "Skipping VLM memory-aware cache store for %s: "
                                "extracted prompt cache resolved empty "
                                "(raw_type=%s, prompt_tokens=%d)",
                                request_id,
                                type(raw_cache).__name__,
                                prompt_len,
                            )
                        if raw_cache:
                            cache_key_tokens = (
                                prompt_tokens[:prompt_len - 1]
                                if prompt_len > 1
                                else list(prompt_tokens)
                            )
                            _uses_zaya_cache = bool(
                                getattr(self, "_uses_zaya_cache", False)
                            )
                            _uses_mixed_attention_cache = bool(
                                getattr(self, "_mixed_attention_cache_model", False)
                            )
                            if (
                                not _uses_mixed_attention_cache
                                and isinstance(raw_cache, list)
                                and any(
                                    type(layer).__name__ == "RotatingKVCache"
                                    for layer in raw_cache
                                )
                            ):
                                _uses_mixed_attention_cache = True
                                logger.info(
                                    "Detected mixed-SWA VLM memory-aware cache "
                                    "layout for %s from extracted RotatingKVCache "
                                    "layers; using clean prompt-boundary store",
                                    request_id,
                                )
                            if _uses_zaya_cache or _uses_mixed_attention_cache:
                                # ZAYA CCA and Gemma-style mixed-SWA caches are
                                # path-dependent. A post-generation
                                # RotatingKVCache has already advanced beyond
                                # the prompt boundary and may be circularly
                                # wrapped, so generic truncation either returns
                                # None or corrupts temporal order. Re-prefill
                                # exactly the N-1 cache key and store that
                                # typed state, matching the paged-cache path.
                                cache_to_store = None
                                prefill_fn = getattr(
                                    self.batch_generator,
                                    "_prefill_for_clean_path_dependent_cache",
                                    None,
                                )
                                if not callable(prefill_fn):
                                    prefill_fn = getattr(
                                        self.batch_generator,
                                        "_prefill_for_clean_ssm",
                                        None,
                                    )
                                if cache_key_tokens and callable(prefill_fn):
                                    cache_to_store = prefill_fn(cache_key_tokens)
                                if cache_to_store is None:
                                    logger.info(
                                        "Skipping %s VLM memory-aware cache "
                                        "store for %s: clean typed prompt "
                                        "prefill unavailable (prompt_tokens=%d)",
                                        "ZAYA" if _uses_zaya_cache else "mixed-SWA",
                                        request_id,
                                        prompt_len,
                                    )
                            else:
                                raw_cache = self._prepare_tq_cache_for_storage(raw_cache)
                                cache_to_store = (
                                    self._truncate_hybrid_cache(raw_cache, prompt_len)
                                    if raw_cache is not None
                                    else None
                                )
                            if cache_to_store is not None and self._validate_cache(
                                cache_to_store,
                                source=f"mllm-memory-store:{request_id}",
                            ):
                                # L2: persist to disk (skip hybrid — SSM can't be reconstructed on fetch)
                                if self.disk_cache is not None and not self._is_hybrid:
                                    try:
                                        disk_stored = self.disk_cache.store(
                                            cache_key_tokens, cache_to_store
                                        )
                                        logger.info(
                                            "VLM memory-aware disk cache store "
                                            "%s for %s (%d cache-key tokens from "
                                            "%d prompt tokens, %d layers)",
                                            "queued" if disk_stored else "rejected",
                                            request_id,
                                            len(cache_key_tokens),
                                            len(prompt_tokens),
                                            len(cache_to_store)
                                            if isinstance(cache_to_store, list)
                                            else -1,
                                        )
                                    except Exception as de:
                                        logger.warning(
                                            f"VLM disk store failed for {request_id}: {de}"
                                        )
                                if getattr(self, '_kv_cache_bits', 0):
                                    cache_to_store = self._quantize_cache_for_storage(cache_to_store)
                                    if not self._validate_cache(
                                        cache_to_store,
                                        source=f"mllm-memory-store-quant:{request_id}",
                                    ):
                                        logger.warning(
                                            f"VLM memory-aware cache store rejected unsafe "
                                            f"quantized live cache for {request_id}"
                                        )
                                        cache_to_store = None
                                if cache_to_store is not None:
                                    stored = self.memory_aware_cache.store(cache_key_tokens, cache_to_store)
                                    if stored:
                                        logger.info(
                                            f"VLM stored memory-aware cache for {request_id} "
                                            f"({len(cache_key_tokens)} cache-key tokens from "
                                            f"{prompt_len} prompt tokens)"
                                        )
                                        _PERSIST.record(request_id, "stored", "memory-aware", retained_tokens=len(cache_key_tokens))
                                    else:
                                        _PERSIST.record(request_id, "refused", "memory-aware store rejected")
                                        logger.info(
                                            "VLM memory-aware cache store rejected "
                                            "for %s (%d cache-key tokens, %d layers)",
                                            request_id,
                                            len(cache_key_tokens),
                                            len(cache_to_store)
                                            if isinstance(cache_to_store, list)
                                            else -1,
                                        )
                    except Exception as e:
                        _PERSIST.record(request_id, "failed", f"memory-aware: {e}")
                        logger.warning(f"VLM memory-aware cache store failed for {request_id}: {e}")
                    finally:
                        if request is not None:
                            request._extracted_cache = None

            # --- Cache store: legacy prefix cache path ---
            elif self.prefix_cache is not None and not _skip_cache_store:
                if (
                    request is not None
                    and getattr(request, "_extracted_cache", None) is not None
                    and getattr(request, "_extracted_tokens", None)
                ):
                    try:
                        prompt_tokens = list(request._extracted_tokens)
                        prompt_len = len(prompt_tokens)
                        if self._mllm_request_has_media_cache_context(
                            request, prompt_tokens
                        ):
                            logger.info(
                                "Skipping VLM legacy prefix cache store for %s: "
                                "prompt contains media context/placeholders; "
                                "media embeddings are path-dependent and "
                                "must not be saved under a token-only "
                                "prefix key",
                                request_id,
                            )
                            request._extracted_cache = None
                        raw_cache = request._extracted_cache
                        if callable(raw_cache):
                            raw_cache = raw_cache()
                        if raw_cache:
                            raw_cache = self._prepare_tq_cache_for_storage(raw_cache)
                            cache_to_store = (
                                self._truncate_hybrid_cache(raw_cache, prompt_len)
                                if raw_cache is not None
                                else None
                            )
                            if cache_to_store is not None and self._validate_cache(
                                cache_to_store,
                                source=f"mllm-prefix-store:{request_id}",
                            ):
                                cache_key_tokens = (
                                    prompt_tokens[:prompt_len - 1]
                                    if prompt_len > 1
                                    else list(prompt_tokens)
                                )
                                if self.disk_cache is not None and not self._is_hybrid:
                                    try:
                                        self.disk_cache.store(prompt_tokens, cache_to_store)
                                    except Exception as de:
                                        logger.debug(f"VLM disk store failed for {request_id}: {de}")
                                if getattr(self, '_kv_cache_bits', 0):
                                    cache_to_store = self._quantize_cache_for_storage(cache_to_store)
                                    if not self._validate_cache(
                                        cache_to_store,
                                        source=f"mllm-prefix-store-quant:{request_id}",
                                    ):
                                        logger.warning(
                                            f"VLM legacy prefix cache store rejected unsafe "
                                            f"quantized live cache for {request_id}"
                                        )
                                        cache_to_store = None
                                if cache_to_store is not None:
                                    self.prefix_cache.store_cache(cache_key_tokens, cache_to_store)
                                    logger.debug(
                                        f"VLM stored legacy prefix cache for {request_id} "
                                        f"({len(cache_key_tokens)} cache-key tokens from "
                                        f"{prompt_len} prompt tokens)"
                                    )
                                    _PERSIST.record(request_id, "stored", "legacy prefix cache (L1 only)", retained_tokens=len(cache_key_tokens), durable=False)
                    except Exception as e:
                        _PERSIST.record(request_id, "failed", f"legacy: {e}")
                        logger.debug(f"VLM prefix cache store failed for {request_id}: {e}")
                    finally:
                        if request is not None:
                            request._extracted_cache = None
            # The generator and scheduler own distinct request wrappers. A
            # mixed-SWA boundary is handed across by request ID, so every exit
            # path must consume or retire it. Otherwise a skipped store (for
            # example an already-complete hit) can retain a window-sized Metal
            # snapshot after the request is gone, creating the RAM cache this
            # SSD-only mode explicitly forbids.
            if self.batch_generator is not None:
                _swa_snapshots = getattr(
                    self.batch_generator,
                    "_mixed_swa_boundary_snapshots",
                    None,
                )
                if isinstance(_swa_snapshots, dict):
                    _swa_snapshots.pop(str(request_id), None)
            _trace_mark("cache_store_s")

            # Remove per-request stop tokens from batch generator.
            # Use _surviving_stops snapshot (captured before cleanup loop) to avoid
            # reading from self.running which is mutated during the loop.
            if (
                request is not None
                and self.batch_generator is not None
                and getattr(request, '_added_stop_tokens', None)
            ):
                removable = request._added_stop_tokens - _surviving_stops - self.stop_tokens
                self.batch_generator.stop_tokens -= removable

            # Clean up streaming detokenizer
            self._cleanup_detokenizer(request_id)

            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Release the completed request's paged-block refs before detaching
            # its tracking table.  Detaching alone leaves every stored block at
            # ref_count=1, so the free LRU queue stays empty forever once the
            # pool reaches max_cache_blocks.  That prevents both LRU eviction
            # and admission of newer prefixes in long-running VLM sessions.
            paged_entry = None
            if self.block_aware_cache is not None:
                finalize_credit = getattr(
                    self.block_aware_cache, "finalize_cache_hit_credit", None
                )
                if callable(finalize_credit):
                    finalize_credit(request_id)
                paged_entry = self.block_aware_cache._request_tables.pop(
                    request_id, None
                )
            if self.paged_cache_manager is not None:
                block_table = (
                    getattr(paged_entry, "block_table", None)
                    if paged_entry is not None
                    else self.paged_cache_manager.get_block_table(request_id)
                )
                self.paged_cache_manager.release_request_refs(block_table)
                self.paged_cache_manager.detach_request(request_id)

            # Remove from master request dict to free output_tokens and cache refs
            self.requests.pop(request_id, None)

            # Track as finished
            self.finished_req_ids.add(request_id)
            _trace_mark("bookkeeping_s")

        # Clear Metal memory cache when all requests done (vision tensors are large)
        if finished_ids and not self.running:
            clear_mlx_memory_cache(log=logger)
        _trace_mark("clear_memory_s")
        if trace_enabled and finished_ids:
            logger.info(
                "VMLINUX_MLLM_CLEANUP_TRACE finished=%d snapshot_ms=%.3f "
                "cache_store_ms=%.3f bookkeeping_ms=%.3f clear_memory_ms=%.3f",
                len(finished_ids),
                cleanup_trace.get("snapshot_s", 0.0) * 1000.0,
                cleanup_trace.get("cache_store_s", 0.0) * 1000.0,
                cleanup_trace.get("bookkeeping_s", 0.0) * 1000.0,
                cleanup_trace.get("clear_memory_s", 0.0) * 1000.0,
            )

    def _cleanup_finished_after_terminal_dispatch(
        self, finished_ids: Set[str]
    ) -> None:
        """Run deferred terminal cleanup with the normal scheduler lock held."""
        with self._queue_lock:
            self._cleanup_finished(finished_ids)
        # _cleanup_finished() releases every persistent request/cache owner, but
        # its own frame still holds the just-stored ``raw``/``cache_to_store``
        # locals when its in-function clear runs.  The batch generator has the
        # same ordering constraint: its final response closure owns the cache
        # until scheduler cleanup consumes it.  Clearing at either earlier site
        # therefore cannot return those newly freed buffers to Metal.
        #
        # The deferred wrapper is the first point where both cache-owning frames
        # are gone, and it still executes on the model's single step executor.
        # A connected Muse image -> video SSD hit measured a fully settled
        # +105.8 MiB process-RSS ratchet (including +34.5 MiB MLX cache) despite
        # every retained cache tier reporting 0 B.  Reclaim after the frame
        # returns so SSD-only serving does not accumulate one allocator step per
        # completed multimodal turn.
        if finished_ids and not self.running:
            try:
                import gc as _gc

                _gc.collect()
            except Exception as gc_error:  # noqa: BLE001
                logger.debug(
                    "Could not collect released MLLM terminal cache refs: %s",
                    gc_error,
                )
            clear_mlx_memory_cache(log=logger)

    def step(self, *, defer_finished_cleanup: bool = False) -> MLLMSchedulerOutput:
        """Execute one scheduling step -- the core generation loop tick.

        Called repeatedly by _step_and_dispatch() in the background thread.
        Each call:

        1. _schedule_waiting() moves queued requests -> running (under lock)
        2. batch_generator.next() generates one token for all active requests
        3. _process_batch_responses() extracts tokens, detokenizes, checks stops
        4. _cleanup_finished() stores cache, frees memory, removes state unless
           the async streaming loop defers cleanup until after terminal dispatch
        5. Periodic Metal GC every 60s during sustained traffic

        Error recovery: on GPU/cache errors, attempts retry (max 2) per request
        by re-queueing. On persistent failure, finishes request with error.

        Returns:
            MLLMSchedulerOutput with results of this step
        """
        output = MLLMSchedulerOutput()
        trace_enabled = _mllm_scheduler_trace_enabled()
        step_t0 = time.perf_counter() if trace_enabled else 0.0
        trace_last = step_t0

        def _trace_mark(name: str) -> None:
            nonlocal trace_last
            if not trace_enabled:
                return
            now = time.perf_counter()
            output.trace_timings[name] = output.trace_timings.get(name, 0.0) + max(
                now - trace_last, 0.0
            )
            trace_last = now

        # Process deferred aborts — safe now because previous step's
        # Metal computation has completed. Hold _queue_lock to prevent
        # race with abort_request() modifying _pending_aborts/UID maps.
        if self._pending_aborts:
            with self._queue_lock:
                aborts = list(self._pending_aborts)
                self._pending_aborts.clear()
            for rid in aborts:
                with self._queue_lock:
                    uid = self.request_id_to_uid.pop(rid, None)
                if uid is not None:
                    if self.batch_generator is not None:
                        try:
                            with self._batch_lock:
                                self.batch_generator.remove([uid])
                        except Exception as e:
                            logger.warning(f"Deferred abort remove failed for {rid}: {e}")
                    with self._queue_lock:
                        self.uid_to_request_id.pop(uid, None)
                logger.debug(f"Processed deferred abort for {rid}")
        _trace_mark("abort_s")

        # Schedule waiting requests
        with self._queue_lock:
            scheduled = self._schedule_waiting()
            output.scheduled_request_ids = [r.request_id for r in scheduled]
            output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)

            # Identify if we have running requests before releasing lock
            has_running = self.batch_generator is not None and len(self.running) > 0
        _trace_mark("schedule_s")

        # Run generation step if we have running requests (OUTSIDE QUEUE LOCK)
        if has_running:
            try:
                with self._batch_lock:
                    next_fn = (
                        getattr(self.batch_generator, "next_burst", None)
                        if getattr(type(self.batch_generator), "next_burst", None) is not None
                        else None
                    )
                    if callable(next_fn):
                        responses = next_fn()
                    else:
                        responses = self.batch_generator.next()
                _trace_mark("batch_next_s")
            except Exception as step_err:
                # Cache corruption or GPU error — recover by clearing state
                # and rescheduling (matching LLM scheduler pattern).
                # Limit retries to prevent infinite loops on persistent errors.
                import traceback
                logger.error(f"MLLM batch generation error: {step_err}\n{traceback.format_exc()}")
                try:
                    self.batch_generator.close()
                except Exception:
                    pass
                self.batch_generator = None
                self._current_sampler_params = ()

                with self._queue_lock:
                    max_retries = 2
                    retryable = []
                    failed = []
                    for req_id, req in list(self.running.items()):
                        req._retry_count += 1
                        if req._retry_count <= max_retries:
                            req.status = RequestStatus.WAITING
                            # Carry the lifetime total forward BEFORE the list
                            # that derives it is cleared, or progress goes
                            # backwards and the timeout reads a healthy retry as
                            # a wedged request.
                            req._retry_output_base = req.total_output_tokens
                            req.output_tokens.clear()
                            req.num_output_tokens = 0
                            req.output_text = ""
                            req.finish_reason = None
                            self.waiting.appendleft(req)
                            retryable.append(req_id)
                        else:
                            # Permanent failure — send error response
                            req.finish_reason = "stop"
                            req.output_text = f"Generation failed: {step_err}"
                            failed.append(req_id)
                        self._cleanup_detokenizer(req_id)
                    # Preserve already-durable hit blocks across retry while
                    # still deleting cold, never-published request blocks.
                    for req_id in self.running:
                        self._cleanup_aborted_paged_request(req_id)

                    self.running.clear()
                    self.request_id_to_uid.clear()
                    self.uid_to_request_id.clear()

                    if failed:
                        # Push error responses for permanently failed requests
                        output.finished_request_ids = failed
                        for req_id in failed:
                            req = self.requests.get(req_id)
                            if req:
                                output.outputs.append(RequestOutput(
                                    request_id=req_id,
                                    output_text=f"Generation failed: {step_err}",
                                    finished=True,
                                    finish_reason="stop",
                                ))
                        self._cleanup_finished(set(failed))
                        logger.error(
                            f"MLLM scheduler: {len(failed)} requests failed permanently"
                        )

                    if retryable:
                        logger.info(
                            f"MLLM scheduler recovered: "
                            f"{len(retryable)} requests rescheduled"
                        )
                    return output

            output.has_work = True

            if responses:
                with self._queue_lock:
                    outputs, finished_ids = self._process_batch_responses(responses)
                    _trace_mark("process_responses_s")
                    output.outputs = outputs
                    output.finished_request_ids = finished_ids
                    if not defer_finished_cleanup:
                        self._cleanup_finished(finished_ids)
                        _trace_mark("cleanup_finished_s")
                _trace_mark("process_cleanup_s")
            else:
                _trace_mark("process_responses_s")
                _trace_mark("cleanup_finished_s")
                _trace_mark("process_cleanup_s")
        else:
            _trace_mark("batch_next_s")
            _trace_mark("process_responses_s")
            _trace_mark("cleanup_finished_s")
            _trace_mark("process_cleanup_s")

        with self._queue_lock:
            # Clear finished tracking for next step
            self.finished_req_ids = set()
        _trace_mark("finish_tracking_s")

        # Periodic Metal memory cache cleanup during sustained traffic.
        # Vision models hold large pixel_value tensors and cross-attention states
        # that fragment Metal's allocator cache.
        _now = time.monotonic()
        if _now - self._last_metal_gc_time > self._metal_gc_interval:
            self._last_metal_gc_time = _now
            if clear_mlx_memory_cache(log=logger):
                logger.debug("VLM periodic Metal memory cache cleanup")
        _trace_mark("metal_gc_s")

        # Async SSM rederive no longer runs inside step(). It was moved onto
        # the process loop's post-response idle branch (vmlx#245): running it
        # here kept the drain on the response path and let an unpreemptable
        # clean prefill inflate a newly-arrived request's TTFT. See
        # has_idle_tasks() / run_one_idle_task().
        _trace_mark("idle_rederive_s")
        if trace_enabled:
            output.trace_timings["total_s"] = max(time.perf_counter() - step_t0, 0.0)

        return output

    def _dispatch_outputs(self, step_output: "MLLMSchedulerOutput") -> None:
        """Push step outputs to async queues. Must be called on the event loop thread."""
        if step_output.outputs:
            for req_output in step_output.outputs:
                queue = self.output_queues.get(req_output.request_id)
                if queue is not None:
                    try:
                        queue.put_nowait(req_output)
                    except asyncio.QueueFull:
                        if req_output.finished:
                            # Finished token MUST be delivered — drain one to make room
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            try:
                                queue.put_nowait(req_output)
                            except asyncio.QueueFull:
                                pass
                    if req_output.finished:
                        # Sentinel MUST be delivered — without it stream_outputs hangs
                        try:
                            queue.put_nowait(None)
                        except asyncio.QueueFull:
                            # Queue is full but we must deliver sentinel.
                            # This should never happen (8192 buffer), but handle it.
                            logger.warning(f"Output queue full for {req_output.request_id}, forcing sentinel")
                            try:
                                queue.get_nowait()  # Drain one item to make room
                            except asyncio.QueueEmpty:
                                pass
                            try:
                                queue.put_nowait(None)
                            except asyncio.QueueFull:
                                pass

    def _fail_all_requests(self, error_msg: str) -> None:
        """Fail all waiting and running requests with an error so callers don't hang."""
        with self._queue_lock:
            failed_ids = set()
            # Fail running requests
            for req_id, req in list(self.running.items()):
                req.status = RequestStatus.FINISHED_ABORTED
                queue = self.output_queues.get(req_id)
                if queue is not None:
                    try:
                        queue.put_nowait(RequestOutput(
                            request_id=req_id,
                            output_text=f"[Error: {error_msg}]",
                            finished=True,
                            finish_reason="stop",
                        ))
                    except asyncio.QueueFull:
                        pass
                    # Sentinel MUST be delivered separately
                    try:
                        queue.put_nowait(None)
                    except asyncio.QueueFull:
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass
                failed_ids.add(req_id)
            # Fail waiting requests
            for req in list(self.waiting):
                req_id = req.request_id
                req.status = RequestStatus.FINISHED_ABORTED
                queue = self.output_queues.get(req_id)
                if queue is not None:
                    try:
                        queue.put_nowait(RequestOutput(
                            request_id=req_id,
                            output_text=f"[Error: {error_msg}]",
                            finished=True,
                            finish_reason="stop",
                        ))
                    except asyncio.QueueFull:
                        pass
                    # Sentinel MUST be delivered separately
                    try:
                        queue.put_nowait(None)
                    except asyncio.QueueFull:
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass
                failed_ids.add(req_id)
            # Cleanup — do NOT pop output_queues here; the error+sentinel are
            # already enqueued and stream_outputs() will drain them. If we pop
            # the queue now, stream_outputs() will find no queue and return
            # empty, losing the error message (user sees 0 tokens instead of error).
            self.waiting.clear()
            for req_id in failed_ids:
                self.running.pop(req_id, None)
                self.requests.pop(req_id, None)
                # Leave output_queues[req_id] — stream_outputs reads the error then cleans up
                self._cleanup_detokenizer(req_id)
                # Clean up UID mappings
                uid = self.request_id_to_uid.pop(req_id, None)
                if uid is not None:
                    self.uid_to_request_id.pop(uid, None)
            self.finished_req_ids.clear()
            # Reset batch generator so next request starts fresh
            if self.batch_generator is not None:
                try:
                    self.batch_generator.close()
                except Exception:
                    pass
                self.batch_generator = None

    def get_request(self, request_id: str) -> Optional[MLLMRequest]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[MLLMRequest]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def _cleanup_failed_block_cache_initialization(
        self,
        *,
        block_disk_store: Optional[Any],
    ) -> None:
        """Release asynchronous L2 resources after partial cache construction."""
        ssm_disk_store = getattr(self, "_ssm_companion_disk_store", None)
        if ssm_disk_store is not None:
            try:
                ssm_disk_store.shutdown(timeout=None)
            except Exception as exc:
                logger.warning(
                    "Failed to stop VLM SSM companion after cache init failure: %s",
                    exc,
                )
            finally:
                self._ssm_companion_disk_store = None

        manager = getattr(self, "paged_cache_manager", None)
        manager_disk_store = getattr(manager, "_disk_store", None)
        owned_block_store = manager_disk_store or block_disk_store
        if owned_block_store is not None:
            try:
                owned_block_store.shutdown()
            except Exception as exc:
                logger.warning(
                    "Failed to stop VLM block disk cache after cache init failure: %s",
                    exc,
                )
        if manager is not None and manager_disk_store is not None:
            manager._disk_store = None
        self.paged_cache_manager = None
        self.block_aware_cache = None
        self._block_disk_l2_enabled = False

    # ========== Async API (for streaming) ==========

    async def start(self) -> None:
        """Start the async scheduler processing loop."""
        if self._running:
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        logger.info(
            f"MLLM Scheduler started with max_num_seqs={self.config.max_num_seqs}"
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        # Finished media/text output is dispatched before worker-owned cache
        # cleanup.  Preserve that in-flight cleanup when Electron Stop/Start (or
        # process shutdown) follows the visible terminal delta immediately;
        # cancelling here used to lose the just-finished mediaSalt/TQ/L2 state.
        if self._processing_task and not self._terminal_cleanup_complete.is_set():
            logger.info("Waiting for terminal MLLM cache cleanup before stop")
            try:
                await asyncio.wait_for(
                    self._terminal_cleanup_complete.wait(), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Terminal MLLM cache cleanup did not finish within 5s; "
                    "continuing scheduler shutdown"
                )
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self.batch_generator is not None:
            # Drop queued idle re-derives: they are pure cache optimizations,
            # and no forward pass may run after teardown begins (vmlx#245).
            rederive_queue = getattr(
                self.batch_generator, "_ssm_rederive_queue", None
            )
            if rederive_queue:
                rederive_queue.clear()
            self.batch_generator.close()
            self.batch_generator = None

        # The SSM companion publishes through its own asynchronous worker but
        # shares BlockDiskStore's aggregate-budget lease. Drain it before the
        # block store releases that lease.
        ssm_disk_store = getattr(self, "_ssm_companion_disk_store", None)
        if ssm_disk_store is not None:
            ssm_disk_store.shutdown(timeout=None)
            self._ssm_companion_disk_store = None

        # The VLM scheduler owns the block-L2 store even when its model worker
        # executor is provided by BatchedEngine.  Release the background writer
        # and aggregate budget lease before BatchedEngine drops this scheduler;
        # otherwise repeated VLM unload/reload keeps dead same-PID cap owners
        # until process exit.
        disk_store = getattr(
            getattr(self, "paged_cache_manager", None),
            "_disk_store",
            None,
        )
        if disk_store is not None:
            disk_store.shutdown()
            self.paged_cache_manager._disk_store = None
        self._block_disk_l2_enabled = False

        logger.info("MLLM Scheduler stopped")

    async def _process_loop(self) -> None:
        """Main async processing loop."""
        while self._running:
            try:
                if self.has_requests():
                    # Run step on the dedicated single-worker executor
                    # to keep all MLX ops on the SAME thread as model load.
                    # See `_step_executor` field comment for the JANGTQ
                    # Metal kernel stream-isolation rationale.
                    loop = asyncio.get_running_loop()
                    trace_enabled = _mllm_scheduler_trace_enabled()
                    executor_t0 = time.perf_counter() if trace_enabled else 0.0
                    step_output = await loop.run_in_executor(
                        self._step_executor,
                        lambda: self.step(defer_finished_cleanup=True),
                    )
                    executor_s = (
                        time.perf_counter() - executor_t0 if trace_enabled else 0.0
                    )
                    finished_ids = set(step_output.finished_request_ids)
                    if finished_ids:
                        # Close admission before terminal dispatch: the client
                        # may immediately submit another turn from that event.
                        self._terminal_cleanup_complete.clear()
                    # Dispatch outputs on the event loop (asyncio.Queue is not thread-safe)
                    dispatch_t0 = time.perf_counter() if trace_enabled else 0.0
                    self._dispatch_outputs(step_output)
                    dispatch_s = (
                        time.perf_counter() - dispatch_t0 if trace_enabled else 0.0
                    )
                    # Terminal cache persistence can take seconds for a long
                    # paged/TQ/SSM prefix. Dispatch the finished output and its
                    # sentinel first, then yield the event loop so SSE/WebSocket
                    # consumers can flush the final visible delta. Cleanup still
                    # runs on the model's single worker and completes before the
                    # next scheduler step, preserving MLX stream ownership and
                    # cache/bookkeeping ordering.
                    if finished_ids:
                        await asyncio.sleep(0)
                        cleanup_t0 = time.perf_counter() if trace_enabled else 0.0
                        try:
                            await loop.run_in_executor(
                                self._step_executor,
                                self._cleanup_finished_after_terminal_dispatch,
                                finished_ids,
                            )
                        finally:
                            self._terminal_cleanup_complete.set()
                        if trace_enabled:
                            cleanup_s = time.perf_counter() - cleanup_t0
                            step_output.trace_timings["cleanup_finished_s"] = (
                                step_output.trace_timings.get("cleanup_finished_s", 0.0)
                                + cleanup_s
                            )
                            step_output.trace_timings["process_cleanup_s"] = (
                                step_output.trace_timings.get("process_cleanup_s", 0.0)
                                + cleanup_s
                            )
                    if trace_enabled and step_output.outputs:
                        self._record_scheduler_trace(
                            step_output,
                            executor_s=executor_s,
                            dispatch_s=dispatch_s,
                        )
                else:
                    # Idle: drain AT MOST ONE hybrid SSM rederive per
                    # iteration on the step executor (Metal stream affinity).
                    # Only reached after the previous iteration dispatched
                    # outputs and finished terminal cleanup, so maintenance
                    # can never delay a response (vmlx#245). A request
                    # arriving between entries takes the step() branch first.
                    if self.has_idle_tasks():
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            self._step_executor,
                            self.run_one_idle_task,
                        )
                        await asyncio.sleep(0)
                    else:
                        # No work, wait a bit
                        await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                self._terminal_cleanup_complete.set()
                break
            except Exception as e:
                self._terminal_cleanup_complete.set()
                import traceback
                logger.error(f"Error in MLLM process loop: {e}\n{traceback.format_exc()}")
                # Fail all waiting+running requests so callers don't hang forever
                self._fail_all_requests(str(e))
                await asyncio.sleep(0.1)

    def _record_scheduler_trace(
        self,
        step_output: MLLMSchedulerOutput,
        *,
        executor_s: float,
        dispatch_s: float,
    ) -> None:
        for req_output in step_output.outputs:
            request_id = req_output.request_id
            timing = self._scheduler_trace_timings.setdefault(
                request_id,
                {
                    "steps": 0.0,
                    "executor_s": 0.0,
                    "dispatch_s": 0.0,
                    "output_items": 0.0,
                    "output_tokens": 0.0,
                    "step_total_s": 0.0,
                    "schedule_s": 0.0,
                    "batch_next_s": 0.0,
                    "process_cleanup_s": 0.0,
                    "process_responses_s": 0.0,
                    "cleanup_finished_s": 0.0,
                    "metal_gc_s": 0.0,
                },
            )
            timing["steps"] += 1.0
            timing["executor_s"] += max(executor_s, 0.0)
            timing["dispatch_s"] += max(dispatch_s, 0.0)
            timing["output_items"] += 1.0
            timing["output_tokens"] += float(len(req_output.new_token_ids or []))
            trace_timings = step_output.trace_timings or {}
            for key in (
                "total_s",
                "schedule_s",
                "batch_next_s",
                "process_cleanup_s",
                "process_responses_s",
                "cleanup_finished_s",
                "metal_gc_s",
            ):
                target = "step_total_s" if key == "total_s" else key
                timing[target] += max(float(trace_timings.get(key, 0.0) or 0.0), 0.0)
            if req_output.finished:
                logger.info(
                    "VMLINUX_MLLM_SCHEDULER_TRACE request_id=%s steps=%d "
                    "output_items=%d output_tokens=%d executor_ms=%.3f "
                    "dispatch_ms=%.3f step_ms=%.3f schedule_ms=%.3f "
                    "batch_next_ms=%.3f process_cleanup_ms=%.3f "
                    "process_responses_ms=%.3f cleanup_finished_ms=%.3f "
                    "metal_gc_ms=%.3f",
                    request_id,
                    int(timing["steps"]),
                    int(timing["output_items"]),
                    int(timing["output_tokens"]),
                    timing["executor_s"] * 1000.0,
                    timing["dispatch_s"] * 1000.0,
                    timing["step_total_s"] * 1000.0,
                    timing["schedule_s"] * 1000.0,
                    timing["batch_next_s"] * 1000.0,
                    timing["process_cleanup_s"] * 1000.0,
                    timing["process_responses_s"] * 1000.0,
                    timing["cleanup_finished_s"] * 1000.0,
                    timing["metal_gc_s"] * 1000.0,
                )
                self._scheduler_trace_timings.pop(request_id, None)

    async def add_request_async(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[Any]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Add a multimodal request (async version with output queue).

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video inputs
            audio: List of audio inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            **kwargs: Additional parameters

        Returns:
            Request ID for tracking
        """
        # Do not let an immediately-following turn select a prefix while the
        # prior terminal request is still persisting paged/TQ/typed companion
        # cache state.  Event.wait() returns without yielding on the normal
        # open path, so this adds no steady-state scheduling round trip.
        await self._terminal_cleanup_complete.wait()
        request_id = self.add_request(
            prompt=prompt,
            images=images,
            videos=videos,
            audio=audio,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        # Create output queue for streaming
        self.output_queues[request_id] = asyncio.Queue(maxsize=8192)

        return request_id

    async def stream_outputs(
        self,
        request_id: str,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request.

        Args:
            request_id: The request ID to stream

        Yields:
            RequestOutput objects as tokens are generated
        """
        output_queue = self.output_queues.get(request_id)
        if output_queue is None:
            return

        try:
            while True:
                output = await output_queue.get()
                if output is None:
                    break
                if output.finished:
                    # Match EngineCore: image/video/audio and hybrid MLLM tool
                    # turns must not publish their terminal event until typed,
                    # paged, SSM-companion, and disk cleanup is durable. Token
                    # deltas before this object remain fully streaming.
                    _durability_wait_started = time.perf_counter()
                    _durability_was_pending = (
                        not self._terminal_cleanup_complete.is_set()
                    )
                    await self._terminal_cleanup_complete.wait()
                    _outcome = _PERSIST.take(request_id)
                    _durability_wait_ms = (
                        time.perf_counter() - _durability_wait_started
                    ) * 1000.0
                    logger.info(
                        "Terminal durability barrier: request=%s wait_ms=%.3f "
                        "waited=%s %s",
                        request_id,
                        _durability_wait_ms,
                        "true" if _durability_was_pending else "false",
                        _format_persistence_outcome(_outcome),
                    )
                    # Keep the fence as DATA for /health (Cache panel "last
                    # generation durability"): request-exact, so a tool step's
                    # save/fence is visible without reading the log.
                    _record_last_durability(
                        getattr(
                            getattr(self, "batch_generator", None), "_stats", None
                        ),
                        request_id,
                        _durability_wait_ms,
                        _durability_was_pending,
                        _outcome,
                    )
                yield output
        finally:
            # Cleanup queue — runs on normal exit AND GeneratorExit (client disconnect)
            if request_id in self.output_queues:
                del self.output_queues[request_id]
            # If the request is still running (client disconnected mid-stream),
            # abort it so the slot is freed
            request = self.running.get(request_id)
            if request is not None and not RequestStatus.is_finished(request.status):
                self.abort_request(request_id)

    async def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[Any]] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate complete output for a request (non-streaming).

        Args:
            prompt: Text prompt
            images: Image inputs
            videos: Video inputs
            audio: Audio inputs
            **kwargs: Generation parameters

        Returns:
            Final RequestOutput
        """
        request_id = await self.add_request_async(
            prompt=prompt,
            images=images,
            videos=videos,
            audio=audio,
            **kwargs,
        )
        # Hold the request object across terminal dispatch.  Deferred cleanup
        # removes it from ``self.requests`` after queueing the final output,
        # potentially before this coroutine resumes to construct non-streaming
        # usage metadata.
        request = self.requests.get(request_id)

        # Collect all outputs
        final_output = None
        async for output in self.stream_outputs(request_id):
            final_output = output
            if output.finished:
                break

        if final_output is None:
            # Create empty output on error
            final_output = RequestOutput(
                request_id=request_id,
                output_text="",
                finished=True,
                finish_reason="stop",
            )

        # The terminal queued output can predate cleanup-side cache metadata.
        # The scheduler request is authoritative for non-streaming usage, so
        # stamp its accepted cache boundary onto generate()'s final result.
        if request is not None:
            try:
                final_output.cached_tokens = int(
                    getattr(request, "_cached_tokens", 0) or 0
                )
            except (TypeError, ValueError):
                final_output.cached_tokens = 0
            final_output.cache_detail = str(
                getattr(request, "_cache_detail", "")
                or getattr(final_output, "cache_detail", "")
                or ""
            )

        # Cleanup
        if request_id in self.requests:
            del self.requests[request_id]

        return final_output

    # ========== Stats and utilities ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics including all cache mode metrics.

        Returns dict with: queue sizes, token counts, batch generator stats,
        vision cache stats, and active cache tier stats (paged/memory-aware/
        legacy/disk). Used by /v1/stats endpoint and health monitoring.
        """
        with self._queue_lock:
            collector_request_ids = sorted(self.output_queues)
            waiting_request_ids = [
                request.request_id for request in self.waiting
            ]
            running_request_ids = sorted(self.running)
            stats = {
                "num_waiting": len(self.waiting),
                "num_running": len(self.running),
                "waiting_request_ids": waiting_request_ids,
                "running_request_ids": running_request_ids,
                "running_requests": [
                    {
                        "request_id": request_id,
                        "status": getattr(
                            self.running[request_id].status,
                            "name",
                            str(self.running[request_id].status),
                        ),
                    }
                    for request_id in running_request_ids
                ],
                "engine_collector_count": len(collector_request_ids),
                "engine_collector_request_ids": collector_request_ids,
                "terminal_cleanup_pending": (
                    not self._terminal_cleanup_complete.is_set()
                ),
                "num_finished": len(self.finished_req_ids),
                "num_requests_processed": self.num_requests_processed,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "cache_hit_requests": self._cache_hit_requests,
                "cache_hit_tokens": self._cache_hit_tokens,
                "cache_hit_tokens_by_detail": dict(self._cache_hit_tokens_by_detail),
                # Schema parity with the text scheduler: the reuse-skip / partial-
                # downgrade / last-selection telemetry is not tracked on the MLLM
                # path yet, so these report as 0/None. Exposing them here keeps the
                # /health, /v1/cache/stats, Perf-panel and Cache-panel shapes
                # identical across text and VLM/MLLM sessions.
                "cache_reuse_skips": 0,
                "cache_reuse_skip_tokens": 0,
                "last_cache_reuse_skip": None,
                "cache_reuse_partial_downgrades": 0,
                "cache_reuse_partial_tokens": 0,
                "last_cache_reuse_partial": None,
                "last_cache_selection": None,
                "tq_decoder_warmup": self._tq_decoder_warmup_stats,
                # Publish the configured truth before the lazy batch generator
                # exists so the app can always attest that the hidden
                # pixel/video preprocessing LRU is disabled in SSD-only mode.
            "vision_cache": {
                    "enabled": bool(getattr(getattr(self, "config", None), "enable_vision_cache", False)),
                    "max_entries": int(getattr(getattr(self, "config", None), "vision_cache_size", 0) or 0),
                    "pixel_cache_size": 0,
                    "retained_bytes": 0,
                    "retained_bytes_mb": 0.0,
                    "pixel_cache_hits": 0,
                    "pixel_cache_misses": 0,
                },
            }

            if self.batch_generator is not None:
                # A mid-prefill stats read can race the chunked-prefill loop
                # mutating generator structures and THROW; the health
                # assembler's blanket except then swallowed the whole
                # scheduler payload (lce and all counters read as null for
                # the entire prefill — proven live on a 32k chunked cold
                # request). Degrade to the admission fallback below instead
                # of unwinding, and log the first failure per exception type
                # so the degradation is never silent.
                try:
                    batch_stats = self.batch_generator.stats()
                    batch_stats_dict = batch_stats.to_dict()
                    stats["batch_generator"] = batch_stats_dict
                    stats["last_cache_execution"] = batch_stats_dict.get(
                        "last_cache_execution"
                    )
                    stats["vision_cache"] = (
                        self.batch_generator.get_vision_cache_stats()
                    )
                except Exception as exc:
                    if getattr(self, "_gen_stats_err_logged", None) != type(
                        exc
                    ).__name__:
                        self._gen_stats_err_logged = type(exc).__name__
                        logger.info(
                            "MLLM get_stats: generator stats read failed "
                            "mid-flight (%s: %s) — serving admission-record "
                            "fallback",
                            type(exc).__name__,
                            exc,
                        )
            # Admission-record fallback: the generator is created lazily and
            # its stats can lack a record for the newest request (proven live
            # on qwen3.8 — cold requests published nothing). The scheduler's
            # admission-time record keeps request-correlated proofs possible.
            _generator_lce = stats.get("last_cache_execution")
            if not _generator_lce:
                _admission_record = getattr(
                    self, "_admission_cache_execution", None
                )
                if isinstance(_admission_record, dict) and _admission_record:
                    stats["last_cache_execution"] = dict(_admission_record)
                    _fb_id = _admission_record.get("request_id")
                    if getattr(self, "_lce_fb_logged_id", None) != _fb_id:
                        self._lce_fb_logged_id = _fb_id
                        logger.info(
                            "MLLM get_stats: admission-record fallback "
                            "engaged for %s (scheduler id=%s)",
                            _fb_id,
                            hex(id(self)),
                        )
                elif getattr(self, "_lce_gap_logged", 0) < 5:
                    self._lce_gap_logged = getattr(self, "_lce_gap_logged", 0) + 1
                    logger.info(
                        "MLLM get_stats: no generator lce and no admission "
                        "record (generator=%s, admission_attr=%r, "
                        "scheduler id=%s)",
                        type(self.batch_generator).__name__
                        if self.batch_generator is not None
                        else None,
                        getattr(self, "_admission_cache_execution", None),
                        hex(id(self)),
                    )
            else:
                _gen_id = (
                    _generator_lce.get("request_id")
                    if isinstance(_generator_lce, dict)
                    else None
                )
                if getattr(self, "_lce_gen_logged_id", None) != _gen_id:
                    self._lce_gen_logged_id = _gen_id
                    logger.info(
                        "MLLM get_stats: generator lce wins for %s "
                        "(scheduler id=%s)",
                        _gen_id,
                        hex(id(self)),
                    )

            # Cache stats for all cache modes
            if self.block_aware_cache is not None:
                try:
                    paged_stats = self.block_aware_cache.get_stats()
                    stats["paged_cache"] = {
                        "type": "paged",
                        "block_size": self.config.paged_cache_block_size,
                        "max_blocks": self.config.max_cache_blocks,
                        "hits": paged_stats.get("hits", 0),
                        "misses": paged_stats.get("misses", 0),
                        "hit_rate": paged_stats.get("hit_rate", 0.0),
                        "tokens_saved": paged_stats.get("tokens_saved", 0),
                        "entry_count": paged_stats.get("entry_count", 0),
                    }
                    if self.paged_cache_manager is not None:
                        manager = self.paged_cache_manager
                        stats["paged_cache"]["allocated_blocks"] = (
                            len(manager.allocated_blocks)
                            if hasattr(manager, "allocated_blocks")
                            else 0
                        )
                        resident_blocks = [
                            block
                            for block in getattr(manager, "blocks", ())
                            if getattr(block, "cache_data", None) is not None
                            and int(getattr(block, "resident_bytes", 0) or 0) > 0
                        ]
                        resident_bytes = int(
                            getattr(manager, "resident_bytes", 0) or 0
                        )
                        max_resident_bytes = int(
                            getattr(manager, "max_resident_bytes", 0) or 0
                        )
                        stats["paged_cache"].update(
                            {
                                "resident_blocks": len(resident_blocks),
                                "resident_tokens": sum(
                                    int(getattr(block, "token_count", 0) or 0)
                                    for block in resident_blocks
                                ),
                                "resident_bytes": resident_bytes,
                                "resident_bytes_mb": round(
                                    resident_bytes / (1024 * 1024), 2
                                ),
                                "max_resident_bytes": max_resident_bytes,
                                "max_resident_bytes_mb": round(
                                    max_resident_bytes / (1024 * 1024), 2
                                ),
                                "evictions": int(
                                    getattr(manager.stats, "evictions", 0) or 0
                                ),
                            }
                        )
                except Exception:
                    pass
            if self.memory_aware_cache is not None:
                try:
                    stats["memory_aware_cache"] = self.memory_aware_cache.get_stats()
                except Exception:
                    stats["memory_aware_cache"] = {"type": "memory_aware"}
            if self.prefix_cache is not None:
                try:
                    stats["prefix_cache"] = {
                        "type": "legacy",
                        "max_entries": self.config.prefix_cache_size,
                    }
                except Exception:
                    pass
            if self.disk_cache is not None:
                try:
                    # Surface real L2 counters: hits/misses/entries/TQ-native etc.
                    # Previously only reported `{type:disk, enabled:True}` which
                    # left the /v1/cache/stats endpoint showing disk_cache as a
                    # stub even when actual L2 prompt restores worked.
                    disk_stats = self.disk_cache.stats()
                    disk_stats["type"] = "disk"
                    disk_stats["enabled"] = True
                    stats["disk_cache"] = disk_stats
                except Exception:
                    stats["disk_cache"] = {"type": "disk", "enabled": True}

            if self.batch_generator is not None:
                ssm_cache = getattr(self.batch_generator, "_ssm_state_cache", None)
                if ssm_cache is not None:
                    nbytes = int(getattr(ssm_cache, "total_nbytes", 0) or 0)
                    max_bytes = getattr(ssm_cache, "max_bytes", None)
                    stats["ssm_companion_cache"] = {
                        "entries": int(getattr(ssm_cache, "size", 0) or 0),
                        "max_entries": int(getattr(ssm_cache, "max_entries", 0) or 0),
                        "nbytes": nbytes,
                        "nbytes_mb": round(nbytes / (1024 * 1024), 2),
                        "ram_enabled": bool(
                            getattr(ssm_cache, "ram_enabled", False)
                        ),
                        "evictions": int(getattr(ssm_cache, "evictions", 0) or 0),
                        "evicted_bytes": int(
                            getattr(ssm_cache, "evicted_bytes", 0) or 0
                        ),
                        "max_bytes": max_bytes,
                        "max_bytes_mb": (
                            round(max_bytes / (1024 * 1024), 2)
                            if max_bytes is not None
                            else None
                        ),
                        "disk_enabled": bool(getattr(ssm_cache, "disk_enabled", False)),
                        "storage": (
                            "ram_and_ssd"
                            if bool(getattr(ssm_cache, "ram_enabled", False))
                            and bool(getattr(ssm_cache, "disk_enabled", False))
                            else "ram_only"
                            if bool(getattr(ssm_cache, "ram_enabled", False))
                            else "ssd_only"
                            if bool(getattr(ssm_cache, "disk_enabled", False))
                            else "disabled"
                        ),
                    }

            return stats

    def reset(self) -> None:
        """Reset all scheduler state: queues, requests, UIDs, caches, generator.

        Signals all output queues with None sentinel, removes active requests
        from batch generator, clears all collections, destroys batch generator,
        and triggers a final Metal GC. Does NOT clear the cache tiers themselves
        (paged/memory-aware/legacy) -- those persist for reuse.
        """
        with self._queue_lock:
            # Signal all output queues with sentinel to unblock async consumers
            for req_id, queue in self.output_queues.items():
                try:
                    queue.put_nowait(None)
                except (asyncio.QueueFull, Exception):
                    pass

            # Remove from batch generator
            if self.batch_generator is not None:
                uids = list(self.uid_to_request_id.keys())
                if uids:
                    try:
                        self.batch_generator.remove(uids)
                    except Exception:
                        pass

            self.waiting.clear()
            self.running.clear()
            self.requests.clear()
            self.finished_req_ids.clear()
            self.request_id_to_uid.clear()
            self.uid_to_request_id.clear()
            self._detokenizer_pool.clear()

        if self.batch_generator is not None:
            self.batch_generator.close()
            self.batch_generator = None
            self._current_sampler_params = ()

        # Single Metal GC after everything is cleaned up
        clear_mlx_memory_cache(log=logger)

    def deep_reset(self) -> None:
        """Deep reset: clear ALL state including cache tiers.

        Unlike reset(), this also clears prefix/paged/memory-aware caches
        and model-level cache state. Used for sleep/wake transitions.
        """
        self.reset()

        # Clear all cache tiers
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear(force=True)
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

        # Clear model-level cache state
        if hasattr(self.model, 'cache'):
            self.model.cache = None
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'cache'):
                    layer.cache = None
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'cache'):
                    layer.self_attn.cache = None

        clear_mlx_memory_cache(log=logger)


def _assemble_clean_hybrid_boundary_cache(
    batch_generator: Any,
    request: Any,
    live_cache: Any,
    boundary: int,
) -> Optional[List[Any]]:
    """Build the clean prompt-boundary cache WITHOUT a second prefill.

    A path-dependent store needs the cache as it stood at the prompt boundary.
    Recurrent state and rotating windows cannot be recovered from
    post-generation state, so the store re-prefills the whole prompt — a
    second full forward pass. Profiled on Qwen3.8-27B at 15.4k tokens,
    ``_prefill_for_clean_path_dependent_cache`` was 40.8% of the engine
    profile (~28s), running AFTER the response was dispatched and blocking the
    next request: message 2 of a fresh conversation cost ~36s even though it
    was a full cache hit, while message 3 cost 1.8s.

    For the pure-hybrid layout that second pass is unnecessary, because both
    halves of the boundary state are already in hand:

    * attention layers are append-only positional caches — generation only
      appends, so slicing every native positional lane to the boundary recovers
      exactly the prompt-boundary state (the same operation the truncation path
      already performs). QSA/M3-style sparse layers must retain their index-key
      lane and native cache class; they are never demoted to plain ``KVCache``;
    * recurrent layers were deep-copied at the boundary during the live
      prefill by the vmlx#109 inline capture, before generation advanced them.

    Assembling those two is exact, not an approximation. It deliberately does
    NOT apply to ZAYA CCA or mixed-SWA rotating layouts, whose attention state
    genuinely is destroyed by generation — those keep re-prefilling.

    Returns None whenever anything fails to line up, so the caller falls back
    to the existing clean prefill. This never fabricates state: it either has
    the exact boundary cache or it declines.
    """
    if boundary <= 0 or not isinstance(live_cache, (list, tuple)):
        return None
    kv_positions = getattr(batch_generator, "_hybrid_kv_positions", None)
    if not kv_positions:
        logger.info(
            "clean-boundary assembly declined: no _hybrid_kv_positions "
            "(boundary=%d, live_layers=%d)",
            boundary,
            len(live_cache),
        )
        return None
    kv_set = set(kv_positions)

    # The companion store consumes and clears the inline checkpoints, so it
    # hands the boundary snapshot over under this attribute first.
    # The companion store consumes and clears the inline checkpoints, and the
    # scheduler holds a different request wrapper than the generator, so the
    # snapshot is handed over by request_id.
    checkpoints = getattr(request, "_clean_boundary_recurrent", None) or []
    if not checkpoints:
        snaps = getattr(batch_generator, "_clean_boundary_snapshots", None) or {}
        checkpoints = snaps.pop(str(getattr(request, "request_id", "")), None) or []
    if not checkpoints:
        checkpoints = getattr(request, "_inline_ssm_checkpoints", None) or []
    if not checkpoints:
        layers = getattr(request, "_inline_ssm_layers", None)
        bound = int(getattr(request, "_inline_ssm_boundary", 0) or 0)
        toks = getattr(request, "_inline_ssm_tokens", None)
        if layers and bound > 0 and toks:
            checkpoints = [(bound, toks, layers)]
    match = None
    for cp_boundary, _cp_tokens, cp_layers in checkpoints:
        if int(cp_boundary) == int(boundary) and cp_layers:
            match = list(cp_layers)
            break
    if match is None:
        logger.info(
            "clean-boundary assembly declined: no inline checkpoint at %d "
            "(have %s, kv_positions=%d, live_layers=%s)",
            boundary,
            [int(c[0]) for c in checkpoints] or "none",
            len(kv_set),
            len(live_cache),
        )
        return None

    try:
        from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
    except ImportError:
        return None

    assembled: List[Any] = []
    recurrent_iter = iter(match)
    for idx, layer in enumerate(live_cache):
        if idx in kv_set:
            # Rotating/quantized attention is NOT recoverable by slicing.
            if isinstance(layer, (RotatingKVCache, QuantizedKVCache)):
                return None
            if not isinstance(layer, KVCache):
                return None
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                return None
            # `offset` is the only authority on how many tokens the buffer
            # holds; a live KVCache allocates in `step` chunks so the trailing
            # rows are slack, not tokens.
            live_offset = cache_offset(layer)
            if live_offset < boundary or int(keys.shape[2]) < boundary:
                return None
            if type(layer).__name__ == "MiniMaxM3SparseCache":
                try:
                    from .models.minimax_m3.cache import clone_minimax_m3_sparse

                    clone = clone_minimax_m3_sparse(
                        layer,
                        boundary,
                        require_idx_keys=True,
                    )
                except Exception:
                    return None
                if (
                    clone is None
                    or type(clone).__name__ != "MiniMaxM3SparseCache"
                    or int(getattr(clone, "offset", -1)) != int(boundary)
                    or getattr(clone, "idx_keys", None) is None
                    or int(clone.idx_keys.shape[2]) != int(boundary)
                ):
                    return None
                assembled.append(clone)
                continue
            clone = KVCache()
            clone.keys = keys[..., :boundary, :]
            clone.values = values[..., :boundary, :]
            clone.offset = boundary
            assembled.append(clone)
        else:
            try:
                assembled.append(next(recurrent_iter))
            except StopIteration:
                return None
    if next(recurrent_iter, None) is not None:
        return None
    if len(assembled) != len(live_cache):
        return None
    return assembled


def _assemble_mixed_swa_boundary_cache(
    batch_generator: Any,
    request: Any,
    live_cache: Any,
    boundary: int,
) -> Optional[List[Any]]:
    """Assemble the exact N-1 mixed-SWA prompt-boundary cache.

    Provenance-checked, mirroring ``_assemble_clean_hybrid_boundary_cache``:

    * Rotating layers come ONLY from the end-of-prefill snapshot the batch
      generator captured before decode advanced the rings
      (``_maybe_capture_mixed_swa_boundary`` →
      ``batch_generator._mixed_swa_boundary_snapshots[request_id]``). A live
      finish-time RotatingKVCache has rolled past the prompt boundary by the
      reply length and its evicted rows are unrecoverable — substituting it
      is exactly the defect that froze Gemma-4 media conversations at the
      last pre-media anchor.
    * If no snapshot exists, ``live_cache`` is accepted ONLY when its own
      rotating offsets prove it already sits at the boundary (the cold-media
      aux clean prefill produces exactly that). The check is the invariant
      itself, not a tag that could go stale.
    * Full-attention ``KVCache`` layers are append-only, so slicing the live
      buffers to the boundary is exact (same operation the hybrid assembly
      performs). Anything else (TQ wrappers, quantized, unknown classes)
      declines rather than guessing.

    Returns None when the exact boundary cannot be PROVEN; the caller then
    stores the live cache loudly and the store marks rotating layers
    ``rotating_kv_pending`` — visible, never corrupt.
    """
    if boundary <= 0 or not isinstance(live_cache, (list, tuple)):
        return None

    def _rotating_offsets_prove_boundary() -> bool:
        saw_rotating = False
        for layer in live_cache:
            if type(layer).__name__ == "RotatingKVCache":
                saw_rotating = True
                if int(getattr(layer, "offset", -1) or -1) != int(boundary):
                    return False
        return saw_rotating

    # Dots3 assembles its native mixed-SWA boundary in the batch generator
    # before handing the cache to this scheduler.  That list intentionally has
    # no generic RotatingKVCache layers: every attention/recurrent stream is a
    # Dots3LatentCache, and the exact-boundary proof is its logical offset.
    # Do not send an already assembled native cache through the generic
    # RotatingKV-only detector, which would incorrectly downgrade it to
    # rotating_kv_pending and discard the boundary receipt.
    if live_cache and all(
        type(layer).__name__ == "Dots3LatentCache" for layer in live_cache
    ):
        if all(
            int(getattr(layer, "offset", -1) or -1) == int(boundary)
            for layer in live_cache
        ):
            logger.info(
                "dots3 native boundary assembly: cache already sits at "
                "boundary=%d; preserving generator assembly",
                boundary,
            )
            return list(live_cache)
        logger.info(
            "dots3 native boundary assembly declined for %s: live offsets "
            "do not sit at boundary=%d",
            getattr(request, "request_id", "?"),
            boundary,
        )
        return None

    snaps = getattr(batch_generator, "_mixed_swa_boundary_snapshots", None) or {}
    entry = snaps.pop(str(getattr(request, "request_id", "")), None)
    if not entry:
        entry = getattr(request, "_mixed_swa_boundary", None)
    if not entry:
        if _rotating_offsets_prove_boundary():
            logger.info(
                "mixed-SWA boundary assembly: live cache already sits "
                "at boundary=%d (clean aux-prefill provenance proven by "
                "rotating offsets)",
                boundary,
            )
            return list(live_cache)
        logger.info(
            "mixed-SWA boundary assembly declined: no end-of-prefill "
            "snapshot for %s and live rotating offsets do not sit at "
            "boundary=%d",
            getattr(request, "request_id", "?"),
            boundary,
        )
        return None
    try:
        cap_boundary, snap_map = entry
        cap_boundary = int(cap_boundary)
    except (TypeError, ValueError):
        return None
    if cap_boundary != int(boundary) or not isinstance(snap_map, dict):
        logger.info(
            "mixed-SWA boundary assembly declined for %s: snapshot "
            "boundary=%s does not match store boundary=%d (a gen-prompt "
            "suffix or aligned store key moved the goalposts)",
            getattr(request, "request_id", "?"),
            cap_boundary,
            boundary,
        )
        return None

    try:
        from mlx_lm.models.cache import KVCache, QuantizedKVCache
    except ImportError:
        return None

    assembled: List[Any] = []
    for idx, layer in enumerate(live_cache):
        snap = snap_map.get(idx)
        if snap is not None:
            if int(getattr(snap, "offset", -1) or -1) != int(boundary):
                logger.warning(
                    "mixed-SWA boundary assembly declined for %s: "
                    "snapshot layer %d offset=%s != boundary=%d",
                    getattr(request, "request_id", "?"),
                    idx,
                    getattr(snap, "offset", None),
                    boundary,
                )
                return None
            assembled.append(snap)
            continue
        if "Rotating" in type(layer).__name__:
            # A rotating layer the capture did not cover — refusing beats
            # writing a rolled window under an exact anchor.
            logger.warning(
                "mixed-SWA boundary assembly declined for %s: rotating "
                "layer %d (%s) has no boundary snapshot",
                getattr(request, "request_id", "?"),
                idx,
                type(layer).__name__,
            )
            return None
        if (
            isinstance(layer, KVCache)
            and not isinstance(layer, QuantizedKVCache)
            and getattr(layer, "keys", None) is not None
        ):
            keys = layer.keys
            values = layer.values
            live_offset = cache_offset(layer)
            if live_offset < boundary or int(keys.shape[2]) < boundary:
                return None
            clone = KVCache()
            clone.keys = keys[..., :boundary, :]
            clone.values = values[..., :boundary, :]
            clone.offset = boundary
            assembled.append(clone)
            continue
        # Unknown/exotic layer class in a mixed-SWA layout: pass it through
        # untouched. The store's per-block slicing already bounds positional
        # layers to the stored key length, and this is byte-identical to what
        # the pre-fix path fed it for non-rotating layers.
        assembled.append(layer)
    if len(assembled) != len(live_cache):
        return None
    return assembled
