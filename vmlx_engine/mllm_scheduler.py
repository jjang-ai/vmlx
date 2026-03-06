# SPDX-License-Identifier: Apache-2.0
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
- Periodic GC: ``mx.clear_memory_cache()`` every 60s during sustained traffic
  (via ``_last_metal_gc_time`` timer in ``step()``)
- Idle GC: ``mx.clear_memory_cache()`` when all requests finish
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

import asyncio
import logging
import os
import time
import uuid
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple


from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

from .mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
)
from .request import RequestOutput, RequestStatus, SamplingParams

logger = logging.getLogger(__name__)


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
    max_num_seqs: int = 16
    # Prefill batch size (all queued requests are prefilled together)
    prefill_batch_size: int = 16
    # Completion batch size
    completion_batch_size: int = 16
    # Prefill step size for chunked prefill
    prefill_step_size: int = 1024
    # Enable vision embedding cache
    enable_vision_cache: bool = True
    # Maximum cache entries
    vision_cache_size: int = 100
    # Default max tokens
    default_max_tokens: int = 256
    # Default video FPS for frame extraction
    default_video_fps: float = 2.0
    # Maximum video frames
    max_video_frames: int = 128

    # Prefix/Paged cache settings
    enable_prefix_cache: bool = True
    use_paged_cache: bool = True
    paged_cache_block_size: int = 64
    max_cache_blocks: int = 1000

    # KV cache quantization for prefix cache storage
    kv_cache_quantization: str = "none"  # "none", "q4", "q8"
    kv_cache_group_size: int = 64

    # Memory-aware cache settings (L1, recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: Optional[int] = None  # None = auto-detect (cache_memory_percent of RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting
    cache_ttl_minutes: float = 0  # Cache entry TTL in minutes (0 = no expiration)

    # Legacy entry-count prefix cache (fallback when paged+memory-aware both off)
    prefix_cache_size: int = 100

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

    # Hybrid SSM state cache size (companion cache for SSM layer states)
    ssm_state_cache_size: int = 50


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

    # Video processing parameters (per-request overrides)
    video_fps: Optional[float] = None
    video_max_frames: Optional[int] = None

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

        # Get model config
        self.model_config = getattr(model, "config", None)

        # Token-level Prefix caching for the language model
        self.paged_cache_manager = None
        self.block_aware_cache = None
        self.memory_aware_cache = None
        self.prefix_cache = None
        self.disk_cache = None

        # Detect hybrid models (mixed KVCache + MambaCache layers)
        lang_model = self.model.language_model if hasattr(self.model, "language_model") else self.model
        self._is_hybrid = False
        try:
            self._is_hybrid = self._is_hybrid_model(lang_model)
        except Exception as e:
            logger.warning(f"Failed to detect hybrid cache model: {e}")

        if self._is_hybrid:
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
            # Auto-switch to paged cache (MambaCache can't be truncated by memory-aware cache)
            if self._is_hybrid and self.config.enable_prefix_cache and not self.config.use_paged_cache:
                logger.info(
                    "Auto-switching VLM to paged cache for hybrid model "
                    "(memory-aware cache can't truncate MambaCache)."
                )
                self.config.use_paged_cache = True
                self.config.use_memory_aware_cache = False

        # --- Cache initialization chain (paged > memory-aware > legacy) ---
        if self.config.enable_prefix_cache:
            if self.config.use_paged_cache:
                # Paged cache with optional block-level disk store (L2)
                block_disk_store = None
                if self.config.enable_block_disk_cache:
                    cache_dir = self.config.block_disk_cache_dir
                    if cache_dir is None and self.config.model_path:
                        import hashlib
                        model_hash = hashlib.sha256(
                            self.config.model_path.encode()
                        ).hexdigest()[:12]
                        cache_dir = os.path.join(
                            os.path.expanduser("~"),
                            ".cache", "vmlx-engine", "block-cache", model_hash,
                        )
                    elif cache_dir is None:
                        cache_dir = os.path.join(
                            os.path.expanduser("~"),
                            ".cache", "vmlx-engine", "block-cache", "default",
                        )
                    try:
                        from .block_disk_store import BlockDiskStore
                        block_disk_store = BlockDiskStore(
                            cache_dir=cache_dir,
                            max_size_gb=self.config.block_disk_cache_max_gb,
                        )
                        logger.info(
                            f"VLM block disk cache enabled: dir={cache_dir}, "
                            f"max={self.config.block_disk_cache_max_gb}GB"
                        )
                    except Exception as e:
                        logger.error(
                            f"VLM block disk cache init failed at {cache_dir}: {e}. "
                            "Continuing without block disk cache."
                        )

                try:
                    from .paged_cache import PagedCacheManager
                    from .prefix_cache import BlockAwarePrefixCache

                    self.paged_cache_manager = PagedCacheManager(
                        block_size=self.config.paged_cache_block_size,
                        max_blocks=self.config.max_cache_blocks,
                        disk_store=block_disk_store,
                    )
                    self.block_aware_cache = BlockAwarePrefixCache(
                        model=lang_model,
                        paged_cache_manager=self.paged_cache_manager,
                    )
                    logger.info(
                        f"VLM Paged cache enabled: block_size={self.config.paged_cache_block_size}, "
                        f"max_blocks={self.config.max_cache_blocks}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize VLM paged cache: {e}")
                    self.paged_cache_manager = None
                    self.block_aware_cache = None

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
                    self.prefix_cache = PrefixCacheManager(
                        model=lang_model,
                        max_entries=self.config.prefix_cache_size,
                    )
                    logger.info(
                        f"VLM prefix cache enabled: max_entries={self.config.prefix_cache_size}"
                    )
                except Exception as e:
                    logger.warning(f"VLM prefix cache init failed: {e}")

        # Disk cache L2 (persistent across restarts, for non-paged paths)
        if self.config.enable_disk_cache and self.config.enable_prefix_cache:
            import hashlib
            base_dir = self.config.disk_cache_dir or os.path.expanduser(
                "~/.cache/vmlx-engine/prompt-cache"
            )
            if self.config.model_path:
                model_hash = hashlib.sha256(
                    self.config.model_path.encode()
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
                )
                logger.info(f"VLM disk cache (L2) enabled: dir={cache_dir}")
            except Exception as e:
                logger.warning(f"VLM disk cache init failed: {e}")
        elif self.config.enable_disk_cache and not self.config.enable_prefix_cache:
            logger.warning(
                "VLM disk cache requires prefix cache to be enabled — disk cache disabled"
            )

        # KV cache quantization for prefix cache storage (2-4x memory reduction)
        self._kv_cache_bits = 0
        self._kv_cache_group_size = 64
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

        # Get stop tokens from tokenizer
        self.stop_tokens = self._get_stop_tokens()

        # Batch generator (created lazily, recreated on sampler param change)
        self.batch_generator: Optional[MLLMBatchGenerator] = None
        self._current_sampler_params: tuple = ()

        # Request management - following vLLM's design
        self.waiting: deque[MLLMRequest] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, MLLMRequest] = {}  # Running requests by ID
        self.requests: Dict[str, MLLMRequest] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # Output queues for async streaming
        self.output_queues: Dict[str, asyncio.Queue] = {}

        # Streaming detokenizer pool for correct multi-byte character handling
        self._detokenizer_pool: Dict[str, Any] = {}

        # Async processing control
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Periodic Metal memory cache cleanup timer (matches LLM scheduler).
        # During sustained MLLM traffic, self.running is never empty so
        # _cleanup_finished's mx.clear_memory_cache() never triggers.
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
        logger.info(
            f"MLLM Scheduler initialized: cache_mode={cache_mode}, "
            f"hybrid={self._is_hybrid}, "
            f"kv_quant={self.config.kv_cache_quantization}, "
            f"disk_l2={self.disk_cache is not None}, "
            f"max_seqs={self.config.max_num_seqs}"
        )

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
            cache_types = {type(c).__name__ for c in cache}
            kv_only_types = {"KVCache", "RotatingKVCache", "QuantizedKVCache"}
            if cache_types and cache_types.issubset(kv_only_types):
                return False
            return bool(cache_types - kv_only_types)
        except Exception:
            return False

    def _detect_head_dim(self) -> Optional[int]:
        """Detect the VLM language model's KV head dimension."""
        try:
            lm = self.model.language_model if hasattr(self.model, 'language_model') else self.model
            if hasattr(lm, 'args'):
                args = lm.args
                if hasattr(args, 'head_dim') and args.head_dim:
                    return args.head_dim
                if hasattr(args, 'hidden_size') and hasattr(args, 'num_attention_heads'):
                    return args.hidden_size // args.num_attention_heads
            if hasattr(lm, 'config'):
                config = lm.config
                if hasattr(config, 'head_dim') and config.head_dim:
                    return config.head_dim
                if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                    return config.hidden_size // config.num_attention_heads
        except Exception as e:
            logger.debug(f"Could not detect VLM head_dim: {e}")
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
                QuantizedKVCache.size = lambda self: getattr(self, 'offset', 0)
                logger.debug("Patched QuantizedKVCache.size() to return self.offset")
            QuantizedKVCache._size_patched = True

        # Validate head_dim compatibility
        head_dim = self._detect_head_dim()
        if head_dim is not None and head_dim > 0:
            if head_dim % group_size != 0:
                for candidate in [32, 16, 8]:
                    if head_dim % candidate == 0:
                        logger.warning(
                            f"VLM KV quant: group_size={group_size} doesn't divide "
                            f"head_dim={head_dim}. Auto-adjusting to {candidate}."
                        )
                        group_size = candidate
                        break
                else:
                    logger.error(
                        f"VLM KV quant: no valid group_size for head_dim={head_dim}. Disabled."
                    )
                    return
            logger.info(f"VLM KV quant validated: head_dim={head_dim}, group_size={group_size}")

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

    def _quantize_cache_for_storage(self, cache: List[Any]) -> List[Any]:
        """
        Quantize KVCache layers for prefix cache storage (2-4x memory reduction).
        Preserves non-KVCache layers (MambaCache, etc.).
        """
        if not self._kv_cache_bits:
            return cache
        try:
            from mlx_lm.models.cache import QuantizedKVCache
            import mlx.core as mx
        except ImportError:
            return cache

        bits = self._kv_cache_bits
        group_size = self._kv_cache_group_size
        result = []
        for layer_cache in cache:
            if (
                type(layer_cache).__name__ == 'KVCache'
                and layer_cache.keys is not None
            ):
                try:
                    qkv = QuantizedKVCache(group_size=group_size, bits=bits)
                    qkv.keys = mx.quantize(layer_cache.keys, group_size=group_size, bits=bits)
                    qkv.values = mx.quantize(layer_cache.values, group_size=group_size, bits=bits)
                    qkv.offset = layer_cache.offset
                    result.append(qkv)
                except Exception:
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

        MambaCache layers will be skipped during _extract_cache_states() since
        they can't be sliced into blocks. Only KVCache layers are stored.
        """
        target_len = prompt_len - 1
        if not raw_cache or target_len <= 0:
            return None

        truncated = []
        for layer_cache in raw_cache:
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                try:
                    k = layer_cache.keys
                    v = layer_cache.values

                    if isinstance(k, tuple):
                        # QuantizedKVCache
                        try:
                            from mlx_lm.models.cache import QuantizedKVCache
                        except ImportError:
                            return None
                        safe_target = min(target_len, k[0].shape[-2])
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
                        # Standard KVCache
                        from mlx_lm.models.cache import KVCache
                        new_cache = KVCache()
                        ndim = k.ndim
                        if ndim == 4:
                            new_cache.keys = k[:, :, :target_len, :]
                            new_cache.values = v[:, :, :target_len, :]
                        elif ndim == 3:
                            new_cache.keys = k[:, :target_len, :]
                            new_cache.values = v[:, :target_len, :]
                        else:
                            return None
                        new_cache.offset = target_len
                        truncated.append(new_cache)
                except Exception as e:
                    logger.warning(f"Failed to truncate KVCache layer: {e}")
                    return None
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

    def _extract_cache_states(self, raw_cache: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract actual tensor state from each layer cache for paged storage.

        Converts raw KVCache/MambaCache objects into state-dict format that
        BlockAwarePrefixCache.store_cache() expects:
            {"state": (keys, values), "meta_state": (offset,), "class_name": "KVCache"}

        This is the VLM equivalent of Scheduler._extract_cache_states().
        """
        if not raw_cache:
            return []

        extracted = []
        for i, layer_cache in enumerate(raw_cache):
            try:
                # Check MambaCache/ArraysCache FIRST: they also have state/meta_state
                # but their .cache attribute is a list (cumulative state, can't be blocked)
                if hasattr(layer_cache, "cache") and isinstance(
                    getattr(layer_cache, "cache", None), list
                ):
                    # MambaCache/ArraysCache: cumulative state, skip for paged storage
                    continue
                elif hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                    extracted.append({
                        "state": layer_cache.state,
                        "meta_state": layer_cache.meta_state,
                        "class_name": type(layer_cache).__name__,
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
        """Get stop token IDs from tokenizer."""
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
        from mlx_lm.sample_utils import make_sampler

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

        if (
            self.batch_generator is not None
            and self._current_sampler_params == new_params
        ):
            return

        # Can't rebuild if there are active requests
        if self.batch_generator is not None and self.running:
            logger.warning(
                "MLLM sampling parameters changed with active requests. "
                "New requests will use new parameters after current batch completes."
            )
            return

        # Close old generator to restore Metal wired/cache limits
        if self.batch_generator is not None:
            try:
                self.batch_generator.close()
            except Exception:
                pass

        # Clear all caches when BatchGenerator changes — cache objects are
        # tied to their generator instance
        if self.batch_generator is not None:
            if self.block_aware_cache is not None:
                logger.debug(
                    "Clearing paged cache: MLLM BatchGenerator being recreated"
                )
                self.block_aware_cache.clear()
            if self.memory_aware_cache is not None:
                logger.debug(
                    "Clearing memory-aware cache: MLLM BatchGenerator being recreated"
                )
                self.memory_aware_cache.clear()
            if self.prefix_cache is not None:
                logger.debug(
                    "Clearing legacy prefix cache: MLLM BatchGenerator being recreated"
                )
                self.prefix_cache.clear()

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
        )
        self._current_sampler_params = new_params

    # ========== Sync API (step-based) ==========

    def add_request(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
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

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=kwargs.get("top_k", 0),
            min_p=kwargs.get("min_p", 0.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            stop=stop or [],
            stop_token_ids=kwargs.get("stop_token_ids", []),
        )

        request = MLLMRequest(
            request_id=request_id,
            prompt=prompt,
            images=images,
            videos=videos,
            sampling_params=sampling_params,
            video_fps=kwargs.get("video_fps"),
            video_max_frames=kwargs.get("video_max_frames"),
        )

        with self._queue_lock:
            self.requests[request_id] = request
            self.waiting.append(request)

        logger.debug(
            f"Added MLLM request {request_id}: "
            f"{len(images or [])} images, {len(videos or [])} videos"
        )

        return request_id

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
    
            # Remove from batch generator
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if self.batch_generator is not None:
                    self.batch_generator.remove([uid])
                del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]
    
            if request_id in self.running:
                del self.running[request_id]
    
            # Clean up paged cache block tables (prevent leak)
            if self.block_aware_cache is not None:
                self.block_aware_cache._request_tables.pop(request_id, None)
                if self.paged_cache_manager is not None:
                    self.paged_cache_manager.detach_request(request_id)
    
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
            try:
                import mlx.core as mx
                mx.clear_memory_cache()
            except Exception:
                pass

        logger.debug(f"Aborted request {request_id}")
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

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
            request = self.waiting.popleft()

            # Create batch request
            batch_req = MLLMBatchRequest(
                uid=-1,  # Will be assigned by batch generator
                request_id=request.request_id,
                prompt=request.prompt,
                images=request.images,
                videos=request.videos,
                max_tokens=request.sampling_params.max_tokens,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                top_k=request.sampling_params.top_k,
                min_p=request.sampling_params.min_p,
                repetition_penalty=request.sampling_params.repetition_penalty,
                video_fps=request.video_fps,
                video_max_frames=request.video_max_frames,
            )
            batch_requests.append(batch_req)

            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            # NOTE: prompt token count is NOT known yet at scheduling time
            # (it comes from batch generator's first response).
            # Tracking is done at request finish time instead.
            scheduled.append(request)

        # Merge per-request stop_token_ids into batch generator stop tokens.
        # Track per-request additions so they can be removed on cleanup
        # (prevents stop tokens leaking across unrelated requests).
        if batch_requests and self.batch_generator is not None:
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

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Set prompt token count from first response (batch generator knows actual count)
            if request.num_prompt_tokens == 0:
                prompt_ids = getattr(response, "prompt_token_ids", None)
                if prompt_ids:
                    request.num_prompt_tokens = len(prompt_ids)

            # Append token to request
            request.output_tokens.append(response.token)
            request.num_output_tokens = len(request.output_tokens)

            # Use streaming detokenizer for correct multi-byte char handling
            detok = self._get_detokenizer(request_id, tokenizer)

            # Check if this is a stop token BEFORE adding to detokenizer
            # so stop tokens (e.g. <|im_end|>) don't leak into new_text
            is_stop = response.finish_reason == "stop"
            string_stop_truncate = -1  # >=0 when string stop matched

            if not is_stop:
                detok.add_token(response.token)
                new_text = detok.last_segment

                # Post-decode string stop sequence check.
                # MLLMBatchGenerator only handles EOS stop tokens;
                # string stop sequences need decoded-text matching.
                if request.sampling_params.stop:
                    full_text = detok.text
                    for stop_str in request.sampling_params.stop:
                        idx = full_text.find(stop_str)
                        if idx >= 0:
                            string_stop_truncate = idx
                            new_text = ""  # suppress partial output
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
            )

            # Determine effective finish reason (string stop overrides)
            finish_reason = response.finish_reason
            if string_stop_truncate >= 0:
                finish_reason = "stop"

            # Check if finished
            if finish_reason is not None:
                if getattr(response, "prompt_cache", None) is not None:
                    request._extracted_cache = response.prompt_cache
                    request._extracted_tokens = getattr(response, "prompt_token_ids", [])

                if finish_reason == "stop":
                    request.status = RequestStatus.FINISHED_STOPPED
                elif finish_reason == "length":
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED

                output.finished = True
                output.finish_reason = finish_reason
                finished_ids.add(request_id)

                # Finalize detokenizer and use its complete text
                detok.finalize()
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

                logger.debug(
                    f"Request {request_id} finished: {finish_reason}, "
                    f"prompt={request.num_prompt_tokens}, "
                    f"completion={request.num_output_tokens} tokens"
                )

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store KV cache for future prefix reuse.

        For each finished request, this method:
        1. Stores the extracted KV cache via the active cache tier (paged,
           memory-aware, or legacy) with optional disk L2 write and quantization.
        2. Cleans up the streaming detokenizer.
        3. Removes from running dict, UID mappings, paged cache tracking.
        4. Removes from master requests dict to free output_tokens/cache refs.
        5. Triggers Metal GC (mx.clear_memory_cache) when all requests done.

        The _extracted_cache and _extracted_tokens attributes are set on the
        MLLMRequest by _process_batch_responses() during step(). They are always
        set to None in a finally block after store to prevent memory leaks.
        """
        for request_id in finished_ids:
            request = self.running.get(request_id)

            # --- Cache store: paged path ---
            if self.block_aware_cache is not None:
                if request is not None and getattr(request, "_extracted_cache", None) is not None:
                    try:
                        token_list = getattr(request, "_extracted_tokens", [])
                        if token_list:
                            raw = request._extracted_cache
                            cache_blocks = raw() if callable(raw) else raw
                            if cache_blocks is None:
                                logger.debug(f"No cache blocks to store for {request_id}")
                            else:
                                prompt_len = len(token_list)
                                cache_blocks = self._truncate_hybrid_cache(
                                    cache_blocks, prompt_len
                                )
                                if cache_blocks is None:
                                    logger.debug(
                                        f"Cache truncation failed for {request_id}"
                                    )
                                else:
                                    truncated_tokens = token_list[:prompt_len - 1] if prompt_len > 1 else token_list
                                    # L2: persist to disk before quantization
                                    if self.disk_cache is not None:
                                        try:
                                            self.disk_cache.store(truncated_tokens, cache_blocks)
                                        except Exception as de:
                                            logger.debug(f"VLM disk cache store failed for {request_id}: {de}")
                                    if getattr(self, '_kv_cache_bits', 0):
                                        cache_blocks = self._quantize_cache_for_storage(cache_blocks)
                                    cache_states = self._extract_cache_states(cache_blocks)
                                    if cache_states:
                                        self.block_aware_cache.store_cache(
                                            request_id,
                                            truncated_tokens,
                                            cache_states,
                                        )
                                        logger.info(
                                            f"VLM Scheduler stored paged Prefix Cache for "
                                            f"{request_id}: {len(cache_states)} layers, "
                                            f"truncated to {len(truncated_tokens)} tokens"
                                        )
                    except Exception as e:
                        logger.warning(f"Failed to store VLM paged cache for {request_id}: {e}", exc_info=True)
                    finally:
                        if request is not None:
                            request._extracted_cache = None

            # --- Cache store: memory-aware path ---
            elif self.memory_aware_cache is not None:
                if (
                    request is not None
                    and getattr(request, "_extracted_cache", None) is not None
                    and getattr(request, "_extracted_tokens", None)
                ):
                    try:
                        prompt_tokens = list(request._extracted_tokens)
                        prompt_len = len(prompt_tokens)
                        raw_cache = request._extracted_cache
                        if callable(raw_cache):
                            raw_cache = raw_cache()
                        if raw_cache:
                            cache_to_store = self._truncate_hybrid_cache(raw_cache, prompt_len)
                            if cache_to_store is not None:
                                # L2: persist pre-quantization
                                if self.disk_cache is not None:
                                    try:
                                        self.disk_cache.store(prompt_tokens, cache_to_store)
                                    except Exception as de:
                                        logger.debug(f"VLM disk store failed for {request_id}: {de}")
                                if getattr(self, '_kv_cache_bits', 0):
                                    cache_to_store = self._quantize_cache_for_storage(cache_to_store)
                                stored = self.memory_aware_cache.store(prompt_tokens, cache_to_store)
                                if stored:
                                    logger.info(
                                        f"VLM stored memory-aware cache for {request_id} "
                                        f"({prompt_len} prompt tokens)"
                                    )
                    except Exception as e:
                        logger.warning(f"VLM memory-aware cache store failed for {request_id}: {e}")
                    finally:
                        if request is not None:
                            request._extracted_cache = None

            # --- Cache store: legacy prefix cache path ---
            elif self.prefix_cache is not None:
                if (
                    request is not None
                    and getattr(request, "_extracted_cache", None) is not None
                    and getattr(request, "_extracted_tokens", None)
                ):
                    try:
                        prompt_tokens = list(request._extracted_tokens)
                        prompt_len = len(prompt_tokens)
                        raw_cache = request._extracted_cache
                        if callable(raw_cache):
                            raw_cache = raw_cache()
                        if raw_cache:
                            cache_to_store = self._truncate_hybrid_cache(raw_cache, prompt_len)
                            if cache_to_store is not None:
                                if self.disk_cache is not None:
                                    try:
                                        self.disk_cache.store(prompt_tokens, cache_to_store)
                                    except Exception as de:
                                        logger.debug(f"VLM disk store failed for {request_id}: {de}")
                                if getattr(self, '_kv_cache_bits', 0):
                                    cache_to_store = self._quantize_cache_for_storage(cache_to_store)
                                self.prefix_cache.store_cache(prompt_tokens, cache_to_store)
                                logger.debug(
                                    f"VLM stored legacy prefix cache for {request_id} "
                                    f"({prompt_len} prompt tokens)"
                                )
                    except Exception as e:
                        logger.debug(f"VLM prefix cache store failed for {request_id}: {e}")
                    finally:
                        if request is not None:
                            request._extracted_cache = None

            # Remove per-request stop tokens from batch generator
            if (
                request is not None
                and self.batch_generator is not None
                and getattr(request, '_added_stop_tokens', None)
            ):
                # Only remove tokens that no other running request also needs
                other_stops = set()
                for other_id, other_req in self.running.items():
                    if other_id != request_id:
                        other_stops.update(getattr(other_req, '_added_stop_tokens', set()))
                removable = request._added_stop_tokens - other_stops - self.stop_tokens
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

            # Clean up paged cache request tracking to free tensor references
            if self.block_aware_cache is not None:
                self.block_aware_cache._request_tables.pop(request_id, None)
            if self.paged_cache_manager is not None:
                self.paged_cache_manager.detach_request(request_id)

            # Remove from master request dict to free output_tokens and cache refs
            self.requests.pop(request_id, None)

            # Track as finished
            self.finished_req_ids.add(request_id)

        # Clear Metal memory cache when all requests done (vision tensors are large)
        if finished_ids and not self.running:
            try:
                import mlx.core as mx
                mx.clear_memory_cache()
            except Exception:
                pass

    def step(self) -> MLLMSchedulerOutput:
        """Execute one scheduling step -- the core generation loop tick.

        Called repeatedly by _step_and_dispatch() in the background thread.
        Each call:

        1. _schedule_waiting() moves queued requests -> running (under lock)
        2. batch_generator.next() generates one token for all active requests
        3. _process_batch_responses() extracts tokens, detokenizes, checks stops
        4. _cleanup_finished() stores cache, frees memory, removes state
        5. Periodic Metal GC every 60s during sustained traffic

        Error recovery: on GPU/cache errors, attempts retry (max 2) per request
        by re-queueing. On persistent failure, finishes request with error.

        Returns:
            MLLMSchedulerOutput with results of this step
        """
        output = MLLMSchedulerOutput()

        # Schedule waiting requests
        with self._queue_lock:
            scheduled = self._schedule_waiting()
            output.scheduled_request_ids = [r.request_id for r in scheduled]
            output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)
    
            # Identify if we have running requests before releasing lock
            has_running = self.batch_generator is not None and len(self.running) > 0

        # Run generation step if we have running requests (OUTSIDE LOCK)
        if has_running:
            try:
                responses = self.batch_generator.next()
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
                            req.output_tokens.clear()
                            req.num_output_tokens = 0
                            req.output_text = ""
                            req.finish_reason = None
                            self.waiting.appendleft(req)
                            retryable.append(req_id)
                        else:
                            # Permanent failure — send error response
                            req.finish_reason = "error"
                            req.output_text = f"Generation failed: {step_err}"
                            failed.append(req_id)
                        self._cleanup_detokenizer(req_id)
                    # Clean up paged cache block tables for all running requests
                    # to prevent stale entries on retry
                    if self.block_aware_cache is not None:
                        for req_id in self.running:
                            self.block_aware_cache._request_tables.pop(req_id, None)
                    if self.paged_cache_manager is not None:
                        for req_id in self.running:
                            self.paged_cache_manager.detach_request(req_id)

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
                                    finish_reason="error",
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
                    output.outputs = outputs
                    output.finished_request_ids = finished_ids
                    self._cleanup_finished(finished_ids)

        with self._queue_lock:
            # Clear finished tracking for next step
            self.finished_req_ids = set()

        # Periodic Metal memory cache cleanup during sustained traffic.
        # Vision models hold large pixel_value tensors and cross-attention states
        # that fragment Metal's allocator cache.
        _now = time.monotonic()
        if _now - self._last_metal_gc_time > self._metal_gc_interval:
            self._last_metal_gc_time = _now
            try:
                import mlx.core as mx
                mx.clear_memory_cache()
                logger.debug("VLM periodic Metal memory cache cleanup")
            except Exception:
                pass

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
                            finish_reason="error",
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
                            finish_reason="error",
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
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self.batch_generator is not None:
            self.batch_generator.close()
            self.batch_generator = None

        logger.info("MLLM Scheduler stopped")

    async def _process_loop(self) -> None:
        """Main async processing loop."""
        while self._running:
            try:
                if self.has_requests():
                    # Run step in thread to avoid blocking the event loop
                    # (batch_generator.next() does heavy MLX computation)
                    step_output = await asyncio.to_thread(self.step)
                    # Dispatch outputs on the event loop (asyncio.Queue is not thread-safe)
                    self._dispatch_outputs(step_output)
                else:
                    # No work, wait a bit
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                logger.error(f"Error in MLLM process loop: {e}\n{traceback.format_exc()}")
                # Fail all waiting+running requests so callers don't hang forever
                self._fail_all_requests(str(e))
                await asyncio.sleep(0.1)

    async def add_request_async(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
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
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            **kwargs: Additional parameters

        Returns:
            Request ID for tracking
        """
        request_id = self.add_request(
            prompt=prompt,
            images=images,
            videos=videos,
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

        while True:
            output = await output_queue.get()
            if output is None:
                break
            yield output

        # Cleanup queue
        if request_id in self.output_queues:
            del self.output_queues[request_id]

    async def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate complete output for a request (non-streaming).

        Args:
            prompt: Text prompt
            images: Image inputs
            videos: Video inputs
            **kwargs: Generation parameters

        Returns:
            Final RequestOutput
        """
        request_id = await self.add_request_async(
            prompt=prompt,
            images=images,
            videos=videos,
            **kwargs,
        )

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
                finish_reason="error",
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
            stats = {
                "num_waiting": len(self.waiting),
                "num_running": len(self.running),
                "num_finished": len(self.finished_req_ids),
                "num_requests_processed": self.num_requests_processed,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            }

            if self.batch_generator is not None:
                batch_stats = self.batch_generator.stats()
                stats["batch_generator"] = batch_stats.to_dict()
                stats["vision_cache"] = self.batch_generator.get_vision_cache_stats()

            # Cache stats for all cache modes
            if self.block_aware_cache is not None:
                try:
                    stats["paged_cache"] = {
                        "type": "paged",
                        "block_size": self.config.paged_cache_block_size,
                        "max_blocks": self.config.max_cache_blocks,
                    }
                    if self.paged_cache_manager is not None:
                        stats["paged_cache"]["allocated_blocks"] = (
                            len(self.paged_cache_manager.allocated_blocks)
                            if hasattr(self.paged_cache_manager, "allocated_blocks")
                            else 0
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
                    stats["disk_cache"] = {"type": "disk", "enabled": True}
                except Exception:
                    pass

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
        try:
            import mlx.core as mx
            mx.clear_memory_cache()
        except Exception:
            pass
