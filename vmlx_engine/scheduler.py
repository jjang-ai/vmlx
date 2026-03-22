# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for vmlx-engine continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import logging
import os
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from .block_disk_store import BlockDiskStore
from .disk_cache import DiskCacheManager
from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from .paged_cache import PagedCacheManager
from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .utils.mamba_cache import ensure_mamba_support

logger = logging.getLogger(__name__)

# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption (must be specific to avoid
# matching unrelated errors — e.g., "cache" alone would match any error
# mentioning cache files, directories, or variables).
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "BatchKVCache",
    "cache_data",
    "cache corruption",
    "cache mismatch",
    "dimension mismatch",
    "shape mismatch",
    "cannot merge",
    "cannot extract",
    # Metal GPU / OOM errors — recover by clearing cache and rescheduling
    "MTLCommandBuffer",
    "MTLDevice",
    "out of memory",
    "Cannot allocate memory",
    "Allocation failed",
]


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Prefix cache settings
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100  # Max cached entries (legacy, ignored if memory-aware)

    # Memory-aware cache settings (recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: Optional[int] = None  # None = auto-detect (30% of available RAM)
    cache_memory_percent: float = 0.30  # Fraction of available RAM if auto-detecting
    cache_ttl_minutes: float = 0  # Cache entry TTL in minutes (0 = no expiration)

    # Paged cache settings (experimental - for memory efficiency)
    use_paged_cache: bool = (
        False  # Use BlockAwarePrefixCache instead of PrefixCacheManager
    )
    paged_cache_block_size: int = 64  # Tokens per block
    max_cache_blocks: int = 1000  # Maximum number of cache blocks

    # KV cache quantization (reduces GPU memory ~2-4x per cache layer)
    kv_cache_quantization: str = "none"  # "none", "q4", "q8"
    kv_cache_group_size: int = 64

    # Disk cache (L2 persistence for prompt caches)
    enable_disk_cache: bool = False
    disk_cache_dir: Optional[str] = None  # None = ~/.cache/vmlx-engine/prompt-cache/<model_hash>
    disk_cache_max_gb: float = 10.0  # 0 = unlimited
    model_path: Optional[str] = None  # Used to scope disk cache per model

    # Block-level disk cache (L2 for paged cache blocks)
    enable_block_disk_cache: bool = False
    block_disk_cache_dir: Optional[str] = None  # None = ~/.cache/vmlx-engine/block-cache/<model_hash>
    block_disk_cache_max_gb: float = 10.0  # 0 = unlimited


@dataclass
class SchedulerOutput:
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


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SchedulerConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, Request] = {}  # Running requests by ID
        self.requests: Dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: Optional[BatchGenerator] = None
        self._current_sampler_params: Optional[Tuple] = None

        # Base stop tokens (model EOS) — used to prevent over-removal in H1 cleanup
        self.stop_tokens: Set[int] = self._get_stop_tokens()

        # KV cache quantization bits (0 = disabled). Initialized here so all
        # code paths can use self._kv_cache_bits directly without getattr().
        self._kv_cache_bits: int = 0
        self._kv_cache_group_size: int = 64

        # Track if model uses mixed cache types (KVCache + MambaCache)
        self._is_hybrid = self._is_hybrid_model(model)

        # Prefix cache for KV state reuse
        self.prefix_cache: Optional[PrefixCacheManager] = None
        self.memory_aware_cache: Optional[MemoryAwarePrefixCache] = None
        self.paged_cache_manager: Optional[PagedCacheManager] = None
        self.block_aware_cache: Optional[BlockAwarePrefixCache] = None

        # Auto-detect hybrid models (MambaCache + KVCache) and switch to
        # paged cache, since memory-aware cache can't truncate MambaCache.
        if (
            self.config.enable_prefix_cache
            and not self.config.use_paged_cache
            and self.config.use_memory_aware_cache
            and self._is_hybrid
        ):
            logger.info(
                "Non-standard cache model detected (MambaCache/hybrid layers). "
                "Auto-switching to paged cache for correct cache reuse."
            )
            self.config.use_paged_cache = True
            self.config.use_memory_aware_cache = False

        # Active generation KV cache has no explicit memory cap — relies on
        # MLX/Metal's own memory management and macOS memory pressure signals.
        # The prefix cache (L1) has a 32GB hard cap but active KV does not.
        # For large MoE models with many experts, monitor system memory usage.

        # Apply KV cache quantization if requested AND prefix cache is enabled.
        # Quantization only affects prefix cache storage/retrieval — without prefix
        # cache there are no stored KV states to quantize.
        if self.config.kv_cache_quantization != "none":
            if self.config.enable_prefix_cache:
                bits = 4 if self.config.kv_cache_quantization == "q4" else 8
                self._wrap_make_cache_quantized(bits, self.config.kv_cache_group_size)
                logger.info(
                    f"KV cache quantization enabled: {self.config.kv_cache_quantization} "
                    f"(bits={bits}, group_size={self.config.kv_cache_group_size})"
                )
            else:
                logger.warning(
                    f"KV cache quantization '{self.config.kv_cache_quantization}' requested "
                    "but prefix cache is disabled — quantization has no effect without prefix cache"
                )

        if self.config.enable_prefix_cache:
            logger.info(
                "Prefix cache requires continuous batching — enabled automatically"
            )
            if self.config.use_paged_cache:
                # Create optional block-level disk store (L2)
                block_disk_store = None
                if self.config.enable_block_disk_cache:
                    cache_dir = self.config.block_disk_cache_dir
                    if cache_dir is None and self.config.model_path:
                        import hashlib
                        # Include quant config in hash to prevent cross-config cache poisoning
                        # (same fix as prompt disk cache — C3)
                        quant_tag = self.config.kv_cache_quantization or "none"
                        block_scope_key = f"{self.config.model_path}:quant={quant_tag}"
                        model_hash = hashlib.sha256(
                            block_scope_key.encode()
                        ).hexdigest()[:12]
                        cache_dir = os.path.join(
                            os.path.expanduser("~"),
                            ".cache", "vmlx-engine", "block-cache", model_hash,
                        )
                    elif cache_dir is None:
                        logger.warning(
                            "Block disk cache: model_path not set, using shared 'default' dir. "
                            "Different models will share cache — this may cause issues."
                        )
                        cache_dir = os.path.join(
                            os.path.expanduser("~"),
                            ".cache", "vmlx-engine", "block-cache", "default",
                        )
                    try:
                        block_disk_store = BlockDiskStore(
                            cache_dir=cache_dir,
                            max_size_gb=self.config.block_disk_cache_max_gb,
                        )
                        logger.info(
                            f"Block disk cache enabled: dir={cache_dir}, "
                            f"max={self.config.block_disk_cache_max_gb}GB"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to initialize block disk cache at {cache_dir}: {e}. "
                            "Continuing without disk cache."
                        )
                        block_disk_store = None

                # Use paged cache for memory efficiency
                self.paged_cache_manager = PagedCacheManager(
                    block_size=self.config.paged_cache_block_size,
                    max_blocks=self.config.max_cache_blocks,
                    disk_store=block_disk_store,
                )
                self.block_aware_cache = BlockAwarePrefixCache(
                    model=model,
                    paged_cache_manager=self.paged_cache_manager,
                )
                logger.info(
                    f"Paged cache enabled: block_size={self.config.paged_cache_block_size}, "
                    f"max_blocks={self.config.max_cache_blocks}"
                )
            elif self.config.use_memory_aware_cache:
                # Use memory-aware cache (recommended for large models)
                cache_config = MemoryCacheConfig(
                    max_memory_mb=self.config.cache_memory_mb,
                    max_memory_percent=self.config.cache_memory_percent,
                    ttl_minutes=self.config.cache_ttl_minutes,
                )
                self.memory_aware_cache = MemoryAwarePrefixCache(
                    model=model,
                    config=cache_config,
                )
                logger.info(
                    f"Memory-aware cache enabled: "
                    f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB"
                )
            else:
                # Use legacy entry-count based prefix cache
                self.prefix_cache = PrefixCacheManager(
                    model=model,
                    max_entries=self.config.prefix_cache_size,
                )
                logger.info(
                    f"Prefix cache enabled with max_entries={self.config.prefix_cache_size}"
                )

        # Disk cache (L2) for persistent prompt cache across restarts.
        # Disk cache entries are loaded lazily on cache miss — no L2-to-L1
        # warmup at startup. This avoids loading GBs of cache into RAM but
        # means first request pays full prefill cost.
        self.disk_cache: Optional[DiskCacheManager] = None
        if self.config.enable_disk_cache and self.config.enable_prefix_cache:
            import hashlib
            base_dir = self.config.disk_cache_dir or os.path.expanduser(
                "~/.cache/vmlx-engine/prompt-cache"
            )
            # Scope disk cache per model AND quantization to prevent stale cross-config hits.
            # Uses a short hash of model path + quant config as a subdirectory.
            # Without this, restarting with different --kv-cache-quantization could load
            # quantized tensors as unquantized data (or vice versa), producing garbage output.
            if self.config.model_path:
                quant_tag = self.config.kv_cache_quantization or "none"
                scope_key = f"{self.config.model_path}:quant={quant_tag}"
                model_hash = hashlib.sha256(
                    scope_key.encode()
                ).hexdigest()[:12]
                model_slug = os.path.basename(self.config.model_path.rstrip("/"))
                cache_dir = os.path.join(base_dir, f"{model_slug}_{model_hash}")
            else:
                cache_dir = base_dir
            self.disk_cache = DiskCacheManager(
                cache_dir=cache_dir,
                max_size_gb=self.config.disk_cache_max_gb,
            )
        elif self.config.enable_disk_cache and not self.config.enable_prefix_cache:
            logger.warning(
                "Disk cache requires prefix cache to be enabled — disk cache disabled"
            )

        # Log disk cache + paged cache backend status
        if self.disk_cache is not None and self.block_aware_cache is not None:
            logger.info(
                "Disk cache enabled with paged cache backend — "
                "L2 writes happen during cache extraction (pre-quantization)"
            )

        # Streaming detokenizer pool for correct multi-byte character handling.
        # Single-token decode breaks emoji and other multi-byte UTF-8 chars.
        self._detokenizer_pool: Dict[str, Any] = {}

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Periodic Metal memory cache cleanup timer.
        # During sustained multi-request traffic, self.running is never empty
        # so _cleanup_finished's mx.clear_memory_cache() never triggers.
        # This timer ensures Metal's internal allocator cache gets flushed
        # periodically (every 60s) even during continuous load.
        self._last_metal_gc_time = time.monotonic()
        self._metal_gc_interval = 60.0  # seconds

    @staticmethod
    def _is_hybrid_model(model: Any) -> bool:
        """Check if model uses non-standard cache types requiring paged cache.

        Returns True for:
        - Hybrid models (mixed KVCache + MambaCache layers)
        - Pure Mamba/SSM models (all MambaCache/ArraysCache layers)

        These models cannot use memory-aware cache (which needs truncatable KVCache)
        and must be routed to paged cache for correct prefix caching.
        """
        if not hasattr(model, "make_cache"):
            return False
        try:
            cache = model.make_cache()
            cache_types = {type(c).__name__ for c in cache}
            # Standard KV-only models don't need special handling.
            # Match any class name ending with "KVCache" (e.g., KVCache,
            # RotatingKVCache, QuantizedKVCache, ChunkedKVCache) so future
            # KV cache variants are handled automatically without hardcoding.
            kv_types = {t for t in cache_types if t == "KVCache" or t.endswith("KVCache")}
            if cache_types and cache_types == kv_types:
                return False
            # Any non-KV cache type (MambaCache, ArraysCache, etc.) needs paged cache.
            # Discard CacheList — it's a wrapper (used by MoE models) that contains
            # KVCache layers, not a hybrid SSM cache type.
            non_kv = cache_types - kv_types
            non_kv.discard("CacheList")
            return bool(non_kv)
        except Exception as e:
            logger.warning(f"make_cache() failed during hybrid detection: {e}")
            return False

    def _detect_head_dim(self) -> Optional[int]:
        """Detect the model's KV head dimension from model config."""
        try:
            # Inspect model config for head_dim
            if hasattr(self.model, 'args'):
                args = self.model.args
                if hasattr(args, 'head_dim') and args.head_dim:
                    return args.head_dim
                if hasattr(args, 'hidden_size') and hasattr(args, 'num_attention_heads'):
                    return args.hidden_size // args.num_attention_heads
            # Try model.config
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'head_dim') and config.head_dim:
                    return config.head_dim
                if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                    return config.hidden_size // config.num_attention_heads
        except Exception as e:
            logger.debug(f"Could not detect head_dim: {e}")
        return None

    def _detect_n_kv_heads(self) -> int:
        """Detect number of KV heads from model config (for GQA head normalization).

        BatchKVCache.merge() inflates the H dimension to the maximum across
        all caches in the batch. When the cache is extracted and stored in
        paged blocks, the inflated head count persists. On the next turn,
        reconstruct_cache() builds a cache with wrong H, causing a broadcast
        error. This method provides the ground-truth KV head count so
        extraction can slice away the inflated heads.
        """
        if hasattr(self, '_n_kv_heads_cached'):
            return self._n_kv_heads_cached
        n_kv = 0
        try:
            for cfg_source in ('args', 'config'):
                cfg = getattr(self.model, cfg_source, None)
                if cfg is None:
                    continue
                # Try model-specific KV head count attributes
                n_kv = (
                    getattr(cfg, 'num_key_value_heads', 0)
                    or getattr(cfg, 'num_kv_heads', 0)
                )
                if n_kv:
                    break
                # Fall back to num_attention_heads (MHA — all heads are KV heads)
                n_kv = getattr(cfg, 'num_attention_heads', 0)
                if n_kv:
                    break
        except Exception:
            pass
        # Ensure the result is always a plain int (guards against MagicMock
        # or other non-int returns from getattr on unusual model configs)
        if not isinstance(n_kv, int):
            n_kv = 0
        self._n_kv_heads_cached = n_kv
        return n_kv

    def _wrap_make_cache_quantized(self, bits: int, group_size: int) -> None:
        """
        Configure KV cache quantization for prefix cache storage.

        Quantization is applied at the storage/retrieval boundary of the prefix
        cache, NOT at model.make_cache() level. This preserves full compatibility
        with BatchGenerator (which requires KVCache/BatchKVCache) while reducing
        prefix cache memory footprint by 2-4x.

        During generation: full-precision KVCache (no quality loss).
        In prefix cache: quantized QuantizedKVCache (memory savings).

        Performs init-time validation:
        1. Verifies QuantizedKVCache is available
        2. Checks model head_dim compatibility with group_size
        3. Runs a quantize/dequantize round-trip test
        4. Auto-adjusts group_size or disables if incompatible
        """
        try:
            from mlx_lm.models.cache import QuantizedKVCache
            import mlx.core as mx
        except ImportError:
            logger.warning(
                "QuantizedKVCache not available in this mlx-lm version. "
                "KV cache quantization disabled."
            )
            return

        # Patch QuantizedKVCache.size to return self.offset (upstream bug: returns 0).
        # Use a regular method (not property) to match KVCache.size() interface.
        if not hasattr(QuantizedKVCache, '_size_patched'):
            needs_patch = True
            try:
                test_qkv = QuantizedKVCache(group_size=64, bits=bits)
                test_qkv.offset = 42
                # size() is a method on _BaseCache, so call with parens
                if callable(getattr(test_qkv, 'size', None)) and test_qkv.size() == 42:
                    needs_patch = False
                    logger.info("QuantizedKVCache.size() already returns offset — upstream fix detected")
            except Exception:
                pass

            if needs_patch:
                def _qkv_size(self):
                    return getattr(self, 'offset', 0)
                QuantizedKVCache.size = _qkv_size
                logger.debug("Patched QuantizedKVCache.size() to return self.offset")
            QuantizedKVCache._size_patched = True

        # Validate head_dim compatibility with group_size.
        # mx.quantize() requires group_size to divide the last dimension.
        head_dim = self._detect_head_dim()
        if head_dim is not None and head_dim > 0:
            if head_dim % group_size != 0:
                # Try common group sizes that divide head_dim
                for candidate in [32, 16, 8]:
                    if head_dim % candidate == 0:
                        logger.warning(
                            f"KV cache quantization: group_size={group_size} does not divide "
                            f"head_dim={head_dim}. Auto-adjusting to group_size={candidate}."
                        )
                        group_size = candidate
                        break
                else:
                    logger.error(
                        f"KV cache quantization: no valid group_size found for head_dim={head_dim}. "
                        f"Disabling KV cache quantization."
                    )
                    return
            logger.info(f"KV cache quantization validated: head_dim={head_dim}, group_size={group_size}")

        # Run a quantize/dequantize round-trip test with realistic tensor shapes.
        try:
            test_dim = head_dim or 128
            test_shape = (1, 4, 8, test_dim)  # (batch, heads, seq, head_dim)
            test_tensor = mx.random.normal(test_shape)
            quantized = mx.quantize(test_tensor, group_size=group_size, bits=bits)
            dequantized = mx.dequantize(
                quantized[0], quantized[1], quantized[2],
                group_size=group_size, bits=bits,
            )
            # Force evaluation to catch lazy computation errors
            mx.eval(dequantized)
            if dequantized.shape != test_tensor.shape:
                raise ValueError(
                    f"Shape mismatch: input {test_tensor.shape} vs output {dequantized.shape}"
                )
            logger.info(
                f"KV cache quantization round-trip test passed: "
                f"bits={bits}, group_size={group_size}, test_shape={test_shape}"
            )
        except Exception as e:
            logger.error(
                f"KV cache quantization round-trip test FAILED: {e}. "
                f"Disabling KV cache quantization to prevent generation failures."
            )
            return

        self._kv_cache_bits = bits
        self._kv_cache_group_size = group_size
        # Persist adjusted group_size to config so diagnostics/stats are accurate
        if hasattr(self.config, 'kv_cache_group_size'):
            self.config.kv_cache_group_size = group_size

    def _quantize_cache_for_storage(self, cache: List[Any]) -> List[Any]:
        """
        Convert KVCache layers to QuantizedKVCache for prefix cache storage.

        Quantizes keys/values using mx.quantize() to reduce memory by 2-4x.
        Preserves non-KVCache layers (MambaCache, RotatingKVCache, etc.).
        Recurses into CacheList sub-caches for MoE models.
        Falls back to unquantized storage on any error.
        """
        if not getattr(self, '_kv_cache_bits', 0):
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
        quantized_count = 0
        for i, layer_cache in enumerate(cache):
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
                            qkv.keys = tuple(mx.quantize(
                                sc.keys, group_size=group_size, bits=bits
                            ))
                            qkv.values = tuple(mx.quantize(
                                sc.values, group_size=group_size, bits=bits
                            ))
                            qkv.offset = sc.offset
                            quantized_subs.append(qkv)
                            quantized_count += 1
                        except Exception as e:
                            logger.warning(
                                f"KV quantization failed for CacheList sub-cache in layer {i}: {e}. "
                                f"Storing unquantized."
                            )
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
                    qkv.keys = tuple(mx.quantize(
                        layer_cache.keys, group_size=group_size, bits=bits
                    ))
                    qkv.values = tuple(mx.quantize(
                        layer_cache.values, group_size=group_size, bits=bits
                    ))
                    qkv.offset = layer_cache.offset
                    result.append(qkv)
                    quantized_count += 1
                except Exception as e:
                    # Quantization failed for this layer — store unquantized
                    logger.warning(
                        f"KV cache quantization failed for layer {i} "
                        f"(keys shape={layer_cache.keys.shape}): {e}. "
                        f"Storing unquantized."
                    )
                    result.append(layer_cache)
            else:
                result.append(layer_cache)
        if quantized_count > 0:
            logger.debug(
                f"Quantized {quantized_count}/{len(cache)} cache layers "
                f"(bits={bits}, group_size={group_size})"
            )
        return result

    def _dequantize_cache_for_use(self, cache: List[Any]) -> Optional[List[Any]]:
        """
        Convert QuantizedKVCache layers to KVCache for BatchGenerator.

        Dequantizes stored quantized keys/values back to full precision.
        BatchGenerator requires KVCache (not QuantizedKVCache) for its batch
        operations (merge, extract, filter).
        Recurses into CacheList sub-caches for MoE models.

        Returns None if dequantization fails (caller should treat as cache miss).
        """
        try:
            from mlx_lm.models.cache import KVCache, QuantizedKVCache
            try:
                from mlx_lm.models.cache import CacheList as _CacheList
            except ImportError:
                _CacheList = None
            import mlx.core as mx
        except ImportError:
            return cache

        result = []
        for i, layer_cache in enumerate(cache):
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
                                logger.warning(
                                    f"KV dequantization failed in CacheList layer {i}: {e}. "
                                    f"Treating as cache miss."
                                )
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
                        logger.warning(
                            f"KV cache dequantization failed for layer {i}: {e}. "
                            f"Treating as cache miss."
                        )
                        return None
                else:
                    # QuantizedKVCache with keys=None — empty layer, use fresh KVCache
                    # (BatchGenerator cannot handle QuantizedKVCache objects)
                    result.append(KVCache())
            else:
                result.append(layer_cache)
        return result

    def _prefill_for_prompt_only_cache(
        self, prompt_tokens: List[int]
    ) -> Optional[List[Any]]:
        """
        Run a prefill-only forward pass to get cache state for the given tokens.

        For hybrid models (MambaCache + KVCache), MambaCache is cumulative
        and can't be truncated from post-generation state. This method runs
        a separate prefill pass to capture cache state with exactly the given
        tokens, without output token contamination.

        Args:
            prompt_tokens: Token IDs to prefill (typically prompt[:-1])

        Returns:
            List of cache objects with state for exactly the given tokens,
            or None on failure
        """
        if not prompt_tokens:
            return None
        try:
            import mlx.core as mx

            fresh_cache = self.model.make_cache()

            # Process in chunks to avoid Metal GPU timeout on long prompts
            chunk_size = 2048
            for start in range(0, len(prompt_tokens), chunk_size):
                chunk = prompt_tokens[start : start + chunk_size]
                input_ids = mx.array([chunk])
                _ = self.model(input_ids, cache=fresh_cache)
                # Materialize after each chunk to prevent massive lazy graph
                eval_args = []
                for c in fresh_cache:
                    if hasattr(c, "keys") and c.keys is not None:
                        # QuantizedKVCache: keys/values are tuples of arrays
                        if isinstance(c.keys, tuple):
                            eval_args.extend(c.keys)
                            eval_args.extend(c.values)
                        else:
                            eval_args.extend([c.keys, c.values])
                    elif hasattr(c, "cache") and isinstance(c.cache, list):
                        for arr in c.cache:
                            if hasattr(arr, "shape"):
                                eval_args.append(arr)
                if eval_args:
                    mx.eval(*eval_args)

            return fresh_cache
        except Exception as e:
            logger.warning(f"Prefill-only pass failed: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _get_actual_tokenizer(self, tokenizer: Any) -> Any:
        """
        Get the actual tokenizer from a processor or tokenizer.

        MLLM models use processors (e.g., Qwen3VLProcessor) which wrap
        the tokenizer. This method extracts the actual tokenizer.
        """
        # If it has encode method, it's already a tokenizer
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            return tokenizer
        # If it's a processor, get the wrapped tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        # Fallback to the original
        return tokenizer

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text, handling both tokenizers and processors.
        """
        return self._actual_tokenizer.decode(token_ids)

    def _get_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
            detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
            detok.reset()
            self._detokenizer_pool[request_id] = detok
        return self._detokenizer_pool[request_id]

    def _cleanup_detokenizer(self, request_id: str) -> None:
        """Remove the streaming detokenizer for a finished request."""
        self._detokenizer_pool.pop(request_id, None)

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer or processor."""
        stop_tokens = set()
        # Check both the processor/tokenizer and the actual tokenizer
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                if isinstance(tok.eos_token_id, list):
                    stop_tokens.update(tok.eos_token_id)
                else:
                    stop_tokens.add(tok.eos_token_id)
            if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
                if isinstance(tok.eos_token_ids, (list, set, tuple)):
                    stop_tokens.update(tok.eos_token_ids)
                else:
                    # Handle case where eos_token_ids is a single int
                    stop_tokens.add(tok.eos_token_ids)
        return stop_tokens

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        )

        # Build logits processors (e.g., repetition penalty)
        logits_processors = None
        if sampling_params.repetition_penalty and sampling_params.repetition_penalty != 1.0:
            from mlx_lm.sample_utils import make_logits_processors
            logits_processors = make_logits_processors(
                repetition_penalty=sampling_params.repetition_penalty,
            )

        stop_tokens = self._get_stop_tokens()
        # Add custom stop token IDs
        if sampling_params.stop_token_ids:
            stop_tokens.update(sampling_params.stop_token_ids)

        return BatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
        )

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> None:
        """Ensure BatchGenerator exists with compatible settings."""
        sampler_params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
            sampling_params.top_k,
            sampling_params.repetition_penalty,
        )

        # Create new generator if needed or if sampling params changed
        if (
            self.batch_generator is None
            or self._current_sampler_params != sampler_params
        ):
            # If we have an existing generator with requests, we need to drain it first
            if self.batch_generator is not None and self.running:
                logger.warning(
                    "Sampling parameters changed with active requests. "
                    "New requests will use new parameters after current batch completes."
                )
                return

            # Clear all prefix caches when BatchGenerator changes —
            # BatchKVCache objects and block tables are tied to their generator instance
            if self.batch_generator is not None:
                if self.block_aware_cache is not None:
                    logger.debug(
                        "Clearing paged cache: BatchGenerator being recreated"
                    )
                    self.block_aware_cache.clear()
                if self.memory_aware_cache is not None:
                    logger.debug(
                        "Clearing memory-aware cache: BatchGenerator being recreated"
                    )
                    self.memory_aware_cache.clear()
                elif self.prefix_cache is not None:
                    logger.debug(
                        "Clearing prefix cache: BatchGenerator being recreated"
                    )
                    self.prefix_cache.clear()

            self.batch_generator = self._create_batch_generator(sampling_params)
            self._current_sampler_params = sampler_params

    def _validate_cache(self, cache: Any) -> bool:
        """
        Validate that a cache object is usable.

        Supports all mlx-lm cache types: KVCache, RotatingKVCache,
        QuantizedKVCache, MambaCache, ArraysCache, and CacheList.

        Args:
            cache: The cache object to validate

        Returns:
            True if cache is valid and usable
        """
        if cache is None:
            return False

        # Check if it's a list of cache layers
        if isinstance(cache, list):
            if len(cache) == 0:
                return False
            for layer_cache in cache:
                if layer_cache is None:
                    return False
                if not self._validate_single_cache(layer_cache):
                    return False
            return True

        # Check CacheList structure
        if hasattr(cache, "caches"):
            if cache.caches is None:
                return False
            for c in cache.caches:
                if c is None:
                    return False
                if not self._validate_single_cache(c):
                    return False

        return True

    @staticmethod
    def _validate_single_cache(layer_cache: Any) -> bool:
        """Validate a single cache layer object."""
        if layer_cache is None:
            return False

        # KVCache / RotatingKVCache / QuantizedKVCache: check keys/values
        if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            if layer_cache.keys is None or layer_cache.values is None:
                return False
            return True

        # MambaCache / ArraysCache: check .cache list
        if hasattr(layer_cache, "cache") and isinstance(
            getattr(layer_cache, "cache", None), list
        ):
            return True

        # CacheList: validate sub-caches recursively
        if hasattr(layer_cache, "caches") and isinstance(
            getattr(layer_cache, "caches", None), (list, tuple)
        ):
            return all(
                Scheduler._validate_single_cache(c) for c in layer_cache.caches
            )

        # Extracted state dicts (from _extract_cache_states)
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            return layer_cache["state"] is not None

        # Unknown type but not None - allow it through
        return True

    @staticmethod
    def _truncate_cache_to_prompt_length(
        raw_cache: List[Any], prompt_len: int
    ) -> Optional[List[Any]]:
        """
        Truncate extracted cache objects to prompt_len - 1 tokens.

        After generation, KVCache objects contain state for prompt+output.
        We truncate to prompt_len - 1 (not prompt_len) because on cache hit
        the scheduler feeds the LAST prompt token for generation kickoff.
        If the cache already contains that last token's KV state, the model
        would see it twice, producing wrong output.

        By storing prompt_len - 1 tokens of KV state:
        - On exact match: remaining=[], scheduler feeds last token,
          model processes it against the N-1 cached KV states → correct
        - On forward prefix match: remaining has extra tokens including
          the Nth token → model processes them normally → correct

        MambaCache/ArraysCache layers are cumulative and cannot be
        truncated to an exact token boundary.  When encountered, they are
        passed through unchanged so that KV layers can still be truncated
        and stored in the block cache — avoiding a full re-prefill that
        would otherwise dominate post-generation latency on hybrid models.

        Args:
            raw_cache: List of cache layer objects from BatchGenerator
            prompt_len: Number of prompt tokens

        Returns:
            Truncated cache list, or None if truncation not possible.
            For hybrid models, SSM layers are included unchanged (callers
            should skip L2 disk writes when self._is_hybrid is True).
        """
        # We store N-1 tokens so the last token can be re-fed on cache hit
        target_len = prompt_len - 1
        if not raw_cache or target_len <= 0:
            return None

        truncated = []
        for layer_cache in raw_cache:
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                # Positional cache: truncate to target length
                try:
                    k = layer_cache.keys
                    v = layer_cache.values

                    if isinstance(k, tuple):
                        # QuantizedKVCache: keys/values are tuples of 3 arrays
                        # (data_uint32, scales, zeros) each with seq axis at dim -2
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
                        new_cache.keys = tuple(
                            t[..., :safe_target, :] for t in k
                        )
                        new_cache.values = tuple(
                            t[..., :safe_target, :] for t in v
                        )
                        new_cache.offset = safe_target
                        truncated.append(new_cache)
                    else:
                        # Standard KVCache / RotatingKVCache: keys/values are tensors
                        from mlx_lm.models.cache import KVCache
                        cls_name = type(layer_cache).__name__
                        if "Rotating" in cls_name:
                            try:
                                from mlx_lm.models.cache import RotatingKVCache
                                max_size = getattr(layer_cache, 'max_size', target_len)
                                keep = getattr(layer_cache, 'keep', 0)
                                offset = getattr(layer_cache, 'offset', 0)
                                _idx = getattr(layer_cache, '_idx', 0)

                                if offset > max_size:
                                    # Circular buffer has wrapped — slots are NOT in
                                    # chronological order. Naive slice gives wrong tokens.
                                    # Skip caching for this layer rather than corrupt.
                                    return None

                                new_cache = RotatingKVCache(
                                    max_size=max_size,
                                    keep=keep,
                                )
                            except ImportError:
                                new_cache = KVCache()
                        else:
                            new_cache = KVCache()
                        ndim = k.ndim
                        if ndim == 4:
                            safe_target = min(target_len, k.shape[2])
                            new_cache.keys = k[:, :, :safe_target, :]
                            new_cache.values = v[:, :, :safe_target, :]
                        elif ndim == 3:
                            safe_target = min(target_len, k.shape[1])
                            new_cache.keys = k[:, :safe_target, :]
                            new_cache.values = v[:, :safe_target, :]
                        else:
                            return None
                        new_cache.offset = min(target_len, safe_target)
                        # Restore _idx for RotatingKVCache — use original _idx clamped to truncated length
                        if "Rotating" in cls_name and hasattr(new_cache, '_idx'):
                            new_cache._idx = min(_idx, safe_target)
                        truncated.append(new_cache)
                except ImportError:
                    return None
            elif hasattr(layer_cache, "caches") and isinstance(
                getattr(layer_cache, "caches", None), (list, tuple)
            ):
                # CacheList (DeepSeek V3.2, Falcon H1): contains sub-caches.
                # Recursively truncate each sub-cache.
                sub_result = Scheduler._truncate_cache_to_prompt_length(
                    layer_cache.caches, prompt_len
                )
                if sub_result is None:
                    return None
                try:
                    from mlx_lm.models.cache import CacheList
                    new_cache_list = CacheList.__new__(CacheList)
                    new_cache_list.caches = tuple(sub_result)
                    truncated.append(new_cache_list)
                except ImportError:
                    return None
            elif hasattr(layer_cache, "cache") and isinstance(
                getattr(layer_cache, "cache", None), list
            ):
                # MambaCache/ArraysCache: cumulative state — cannot truncate
                # to an exact token boundary.  Pass through unchanged so KV
                # layers are still truncated and stored in the block cache.
                # The SSM state includes output-token effects, so it should
                # NOT be persisted to the L2 disk cache.
                truncated.append(layer_cache)
            else:
                # Unknown cache type
                return None

        return truncated

    def _extract_cache_states(self, raw_cache: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract actual tensor state from each layer cache.

        This extracts the real KV data using mlx-lm's cache.state property,
        allowing the data to be stored and reconstructed later even after
        the BatchGenerator is recreated.

        Args:
            raw_cache: List of KVCache objects from mlx-lm

        Returns:
            List of dicts with {state: (keys, values), meta_state: (offset,), class_name: str}
        """
        if not raw_cache:
            return []

        extracted = []
        failed = 0
        class_counts: Dict[str, int] = {}
        for i, layer_cache in enumerate(raw_cache):
            try:
                # CacheList (MoE models like DeepSeek V3.2, Falcon H1):
                # wrapper with .caches attribute containing sub-caches.
                # Extract each sub-cache's state and store as a list.
                if hasattr(layer_cache, "caches") and isinstance(
                    getattr(layer_cache, "caches", None), (list, tuple)
                ):
                    sub_states = []
                    all_ok = True
                    n_kv = self._detect_n_kv_heads()
                    for j, sub_cache in enumerate(layer_cache.caches):
                        if hasattr(sub_cache, "state") and hasattr(sub_cache, "meta_state"):
                            sub_state = sub_cache.state
                            # Normalize GQA head inflation in sub-caches too
                            # (handles both plain tensors and quantized tuples)
                            if (isinstance(sub_state, tuple) and len(sub_state) == 2
                                    and n_kv > 0):
                                sk, sv = sub_state
                                if (hasattr(sk, 'shape') and len(sk.shape) == 4
                                        and sk.shape[1] > n_kv):
                                    sub_state = (sk[:, :n_kv, :, :],
                                                 sv[:, :n_kv, :, :])
                                elif (isinstance(sk, (tuple, list)) and len(sk) >= 1
                                        and hasattr(sk[0], 'shape')
                                        and len(sk[0].shape) == 4
                                        and sk[0].shape[1] > n_kv):
                                    sub_state = (
                                        tuple(t[:, :n_kv, :, :] for t in sk),
                                        tuple(t[:, :n_kv, :, :] for t in sv),
                                    )
                            sub_states.append({
                                "state": sub_state,
                                "meta_state": sub_cache.meta_state,
                                "class_name": type(sub_cache).__name__,
                            })
                        else:
                            logger.debug(
                                f"Layer {i} CacheList sub-cache {j} "
                                f"({type(sub_cache).__name__}) lacks state/meta_state"
                            )
                            all_ok = False
                            break
                    if all_ok and sub_states:
                        cls_name = "CacheList"
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        extracted.append({
                            "state": None,
                            "meta_state": None,
                            "class_name": cls_name,
                            "sub_caches": sub_states,
                        })
                    else:
                        failed += 1
                elif hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                    state = layer_cache.state  # (keys, values) MLX arrays
                    meta = layer_cache.meta_state  # (offset,) as strings
                    cls_name = type(layer_cache).__name__

                    # Normalize GQA head inflation from BatchKVCache.merge().
                    # merge() broadcasts H to max across all caches in the batch,
                    # but the true KV head count is smaller for GQA/MQA models.
                    # Slice away the inflated heads before storing.
                    # Handles both standard KVCache (plain tensors) and
                    # QuantizedKVCache (tuple-of-tuples: (data, scales, zeros)).
                    if isinstance(state, tuple) and len(state) == 2:
                        keys, values = state
                        n_kv = self._detect_n_kv_heads()
                        if n_kv > 0:
                            if hasattr(keys, 'shape') and len(keys.shape) == 4:
                                # Standard KVCache: keys/values are 4D tensors
                                if keys.shape[1] > n_kv:
                                    orig_h = keys.shape[1]
                                    keys = keys[:, :n_kv, :, :]
                                    values = values[:, :n_kv, :, :]
                                    state = (keys, values)
                                    if i == 0:
                                        logger.debug(
                                            f"GQA head normalization: sliced H "
                                            f"{orig_h} → {n_kv}"
                                        )
                            elif isinstance(keys, (tuple, list)) and len(keys) >= 1:
                                # QuantizedKVCache: keys/values are tuples of
                                # (data, scales, zeros) — check first component
                                first_k = keys[0]
                                if (hasattr(first_k, 'shape')
                                        and len(first_k.shape) == 4
                                        and first_k.shape[1] > n_kv):
                                    orig_h = first_k.shape[1]
                                    keys = tuple(
                                        t[:, :n_kv, :, :] for t in keys
                                    )
                                    values = tuple(
                                        t[:, :n_kv, :, :] for t in values
                                    )
                                    state = (keys, values)
                                    if i == 0:
                                        logger.debug(
                                            f"GQA head normalization (quantized): "
                                            f"sliced H {orig_h} → {n_kv}"
                                        )

                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    extracted.append(
                        {
                            "state": state,
                            "meta_state": meta,
                            "class_name": cls_name,
                        }
                    )
                else:
                    logger.debug(
                        f"Layer {i} ({type(layer_cache).__name__}) lacks state/meta_state"
                    )
                    failed += 1
            except Exception as e:
                logger.warning(f"Failed to extract state from layer {i}: {e}")
                failed += 1

        if failed > 0:
            logger.warning(
                f"Cache extraction: {len(extracted)}/{len(raw_cache)} layers succeeded, "
                f"{failed} failed"
            )

        # Log extraction summary for debugging hybrid model issues
        if extracted:
            counts_str = ", ".join(f"{k}={v}" for k, v in class_counts.items())
            logger.debug(
                f"Cache extraction: {len(extracted)}/{len(raw_cache)} layers "
                f"({counts_str})"
            )

        # Return what we got - partial extraction is better than nothing
        # The reconstruction logic handles missing layers gracefully
        return extracted

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                # Handle both tokenizers and processors (for MLLM models)
                if hasattr(self.tokenizer, "encode"):
                    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
                elif hasattr(self.tokenizer, "tokenizer") and hasattr(
                    self.tokenizer.tokenizer, "encode"
                ):
                    # Processor wraps tokenizer (e.g., Qwen3VLProcessor)
                    request.prompt_token_ids = self.tokenizer.tokenizer.encode(
                        request.prompt
                    )
                else:
                    raise AttributeError(
                        f"Tokenizer {type(self.tokenizer)} has no 'encode' method. "
                        "Continuous batching requires a tokenizer with encode support."
                    )
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Reject empty prompts — they would spin forever in the scheduler
        # since there are no tokens to prefill and the request never finishes
        if not request.prompt_token_ids or len(request.prompt_token_ids) == 0:
            raise ValueError(
                f"Request {request.request_id} has empty prompt tokens. "
                "Cannot schedule a request with no input tokens."
            )

        # Check prefix cache for cached KV state
        if self.block_aware_cache is not None:
            # Use paged cache
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                request.prompt_token_ids,
            )
            if block_table and block_table.num_tokens > 0:
                # Reconstruct actual KVCache objects from stored tensor data
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                if reconstructed:
                    # Dequantize for BatchGenerator compatibility
                    if getattr(self, '_kv_cache_bits', 0):
                        reconstructed = self._dequantize_cache_for_use(reconstructed)
                    if reconstructed is None:
                        # Dequantization failed — treat as cache miss
                        request.remaining_tokens = request.prompt_token_ids
                        logger.info(
                            f"Request {request.request_id}: dequantization failed, "
                            f"treating as cache miss"
                        )
                    else:
                        # Fix hybrid cache structure: prefix cache stores only
                        # KVCache (attention) layers. For hybrid SSM models
                        # (KVCache + MambaCache/ArraysCache), the reconstructed
                        # cache may have fewer entries than model layers. Expand
                        # it by inserting fresh SSM caches at non-KV positions.
                        if self._is_hybrid and hasattr(self.model, 'make_cache'):
                            try:
                                from .mllm_batch_generator import _fix_hybrid_cache
                                num_before = len(reconstructed)
                                reconstructed = _fix_hybrid_cache(
                                    reconstructed, self.model
                                )
                                num_after = len(reconstructed) if reconstructed else 0
                                # LLM scheduler has no SSM companion cache.
                                # If _fix_hybrid_cache expanded the cache (added
                                # zeroed SSM layers), or returned a fresh cache,
                                # the SSM state is wrong — treat as cache miss.
                                if reconstructed is not None and num_after > num_before:
                                    logger.info(
                                        f"Request {request.request_id}: "
                                        f"hybrid reconstruction expanded {num_before}->"
                                        f"{num_after} layers (SSM layers zeroed, no "
                                        f"companion state in LLM scheduler) — "
                                        f"treating as cache miss for correctness"
                                    )
                                    reconstructed = None
                                    request.remaining_tokens = request.prompt_token_ids
                                    self.block_aware_cache.release_cache(request.request_id)
                                elif reconstructed is not None:
                                    from mlx_lm.models.cache import KVCache as _KVC
                                    kv_layers = [c for c in reconstructed
                                                 if isinstance(c, _KVC)]
                                    if kv_layers and all(
                                        getattr(c, 'offset', 1) == 0
                                        for c in kv_layers
                                    ):
                                        logger.warning(
                                            f"Request {request.request_id}: "
                                            f"hybrid cache fix returned fresh cache, "
                                            f"treating as cache miss"
                                        )
                                        reconstructed = None
                                        request.remaining_tokens = request.prompt_token_ids
                            except ImportError:
                                pass
                            except Exception as e:
                                logger.warning(
                                    f"Request {request.request_id}: "
                                    f"hybrid cache fix failed: {e}, "
                                    f"treating as cache miss"
                                )
                                reconstructed = None
                                request.remaining_tokens = request.prompt_token_ids
                        if reconstructed is not None:
                            request.prompt_cache = reconstructed
                            request.block_table = block_table
                            request.cached_tokens = block_table.num_tokens
                            request.shared_prefix_blocks = len(block_table.block_ids)
                            request.remaining_tokens = remaining
                            logger.info(
                                f"Request {request.request_id}: paged cache hit, "
                                f"{request.cached_tokens} tokens in "
                                f"{request.shared_prefix_blocks} blocks, "
                                f"{len(remaining)} remaining to process"
                            )
                else:
                    # Reconstruction failed, treat as cache miss
                    request.remaining_tokens = request.prompt_token_ids
                    request.cached_tokens = 0
                    logger.info(
                        f"Request {request.request_id}: paged cache reconstruction failed"
                    )
            else:
                request.remaining_tokens = request.prompt_token_ids
                logger.info(
                    f"Request {request.request_id}: paged cache miss, "
                    f"processing all {len(request.prompt_token_ids)} tokens"
                )
        elif self.memory_aware_cache is not None:
            # Use memory-aware prefix cache
            cache, remaining = self.memory_aware_cache.fetch(request.prompt_token_ids)
            if cache:
                # Dequantize for BatchGenerator compatibility
                if getattr(self, '_kv_cache_bits', 0):
                    cache = self._dequantize_cache_for_use(cache)
                if cache is None:
                    # Dequantization failed — treat as cache miss
                    request.remaining_tokens = request.prompt_token_ids
                    logger.info(
                        f"Request {request.request_id}: dequantization failed, "
                        f"treating as cache miss"
                    )
                else:
                    request.prompt_cache = cache
                    request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                    request.remaining_tokens = remaining
                    logger.info(
                        f"Request {request.request_id}: cache hit, "
                        f"{request.cached_tokens} tokens cached, "
                        f"{len(remaining)} remaining to process"
                    )
            else:
                request.remaining_tokens = request.prompt_token_ids
                logger.info(
                    f"Request {request.request_id}: cache miss, "
                    f"processing all {len(request.prompt_token_ids)} tokens"
                )
        elif self.prefix_cache is not None:
            # Use legacy prefix cache
            cache, remaining = self.prefix_cache.fetch_cache(request.prompt_token_ids)
            if cache:
                # Dequantize for BatchGenerator compatibility
                if getattr(self, '_kv_cache_bits', 0):
                    cache = self._dequantize_cache_for_use(cache)
                if cache is None:
                    # Dequantization failed — treat as cache miss
                    request.remaining_tokens = request.prompt_token_ids
                    logger.info(
                        f"Request {request.request_id}: dequantization failed, "
                        f"treating as cache miss"
                    )
                else:
                    request.prompt_cache = cache
                    request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                    request.remaining_tokens = remaining
                    logger.debug(
                        f"Request {request.request_id}: cache hit, "
                        f"{request.cached_tokens} tokens cached, "
                        f"{len(remaining)} tokens remaining"
                    )
            else:
                request.remaining_tokens = request.prompt_token_ids
        else:
            request.remaining_tokens = request.prompt_token_ids

        # L2: Disk cache fallback when in-memory cache missed
        if request.prompt_cache is None and self.disk_cache is not None:
            disk_cache = self.disk_cache.fetch(request.prompt_token_ids)
            if disk_cache is not None:
                # Disk cache stores full-precision N-1 tokens (last prompt token re-fed on hit)
                # Dequantize if KV cache quantization is active (disk stores full precision
                # but may have been quantized before storage in some paths)
                if getattr(self, '_kv_cache_bits', 0):
                    disk_cache = self._dequantize_cache_for_use(disk_cache)
                if disk_cache is None:
                    # Dequantization failed — treat as full cache miss
                    logger.info(
                        f"Request {request.request_id}: disk cache dequantization "
                        f"failed, treating as cache miss"
                    )
                else:
                    request.prompt_cache = disk_cache
                    request.cached_tokens = len(request.prompt_token_ids) - 1
                    request.remaining_tokens = request.prompt_token_ids[-1:]
                    # Also populate L1 memory cache for faster subsequent hits.
                    # Quantize for L1 if KV quant is enabled (disk stores full precision).
                    l1_data = disk_cache
                    if getattr(self, '_kv_cache_bits', 0):
                        try:
                            l1_data = self._quantize_cache_for_storage(disk_cache)
                        except Exception:
                            pass  # Store full-precision on quant failure
                    if self.block_aware_cache is not None:
                        try:
                            extracted = self._extract_cache_states(l1_data)
                            if extracted:
                                self.block_aware_cache.store_cache(
                                    request.request_id,
                                    list(request.prompt_token_ids),
                                    extracted,
                                )
                                # Clean up request table entry — blocks persist via LRU
                                self.block_aware_cache._request_tables.pop(
                                    request.request_id, None
                                )
                                self.block_aware_cache.paged_cache.detach_request(
                                    request.request_id
                                )
                        except Exception:
                            pass
                    elif self.memory_aware_cache is not None:
                        try:
                            self.memory_aware_cache.store(
                                request.prompt_token_ids, l1_data
                            )
                        except Exception:
                            pass
                    elif self.prefix_cache is not None:
                        try:
                            self.prefix_cache.store_cache(
                                list(request.prompt_token_ids), l1_data
                            )
                        except Exception:
                            pass
                    logger.info(
                        f"Request {request.request_id}: disk cache hit (L2), "
                        f"{request.cached_tokens} tokens restored from disk"
                    )

        # Add to tracking
        self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a request, cleaning up all associated resources.

        This is the primary cleanup method for ALL request lifecycle paths:
        normal completion, client disconnect, engine errors, and explicit
        cancellation. It cleans up: waiting queue, running dict, BatchGenerator
        UIDs, paged cache tracking, extracted KV cache refs, detokenizer state,
        Metal memory cache, and the master requests registry.

        Safe to call multiple times (idempotent) — returns False on repeat calls.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted, False otherwise
        """
        request = self.requests.pop(request_id, None)
        if request is None:
            return False

        # Remove from waiting queue
        if request.status == RequestStatus.WAITING:
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Clean up per-request stop tokens from shared BatchGenerator
        # Must happen BEFORE removing from running, so we can still check
        # which tokens are still needed by surviving requests.
        added_stops = getattr(request, '_added_stop_tokens', None)
        if added_stops and self.batch_generator is not None:
            # Only remove tokens not needed by other running requests
            surviving_stops = set()
            for rid, req in self.running.items():
                if rid != request.request_id:
                    surviving_stops.update(getattr(req, '_added_stop_tokens', set()))
            removable = added_stops - surviving_stops
            if removable:
                self.batch_generator.stop_tokens -= removable

        # Remove from running (BatchGenerator)
        if request.request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request.request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request.request_id]

        # Clean up paged cache tracking (prevent block table leaks)
        # Use delete_block_table (not detach_request) so ref_counts are
        # decremented — aborted requests don't store blocks in prefix cache,
        # so detach would orphan them with permanently elevated ref_count.
        if self.block_aware_cache is not None:
            self.block_aware_cache._request_tables.pop(request_id, None)
            self.block_aware_cache.paged_cache.delete_block_table(request_id)

        # Clear extracted cache reference to help GC
        if hasattr(request, '_extracted_cache'):
            request._extracted_cache = None

        # Clean up streaming detokenizer
        self._cleanup_detokenizer(request_id)

        if request_id in self.running:
            del self.running[request_id]

        # Mark as aborted
        request.set_finished(RequestStatus.FINISHED_ABORTED)
        self.finished_req_ids.add(request_id)

        # Clear Metal memory cache if no other requests are running
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

    def shutdown(self) -> None:
        """Shutdown the scheduler and flush disk caches. Idempotent."""
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True

        # Flush prompt-level disk cache (DiskCacheManager)
        if getattr(self, 'disk_cache', None) is not None:
            logger.info("Shutting down prompt disk cache...")
            self.disk_cache.shutdown()
            logger.info("Prompt disk cache shutdown complete")

        # Flush block-level disk cache (BlockDiskStore)
        if hasattr(self, 'paged_cache_manager') and self.paged_cache_manager:
            disk_store = getattr(self.paged_cache_manager, '_disk_store', None)
            if disk_store is not None:
                logger.info("Shutting down block disk cache...")
                disk_store.shutdown()
                logger.info("Block disk cache shutdown complete")

    def _schedule_waiting(self) -> List[Request]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            # Determine tokens to process and cache to use
            # Note: Don't use `remaining_tokens or prompt_token_ids` because empty list
            # is falsy in Python. For exact cache match, remaining_tokens=[] but we should
            # pass just the last token so BatchGenerator can start generation.
            if (
                request.remaining_tokens is not None
                and len(request.remaining_tokens) == 0
            ):
                # Exact cache match - pass only last token for generation kickoff
                tokens_to_process = request.prompt_token_ids[-1:]
            elif request.remaining_tokens:
                tokens_to_process = request.remaining_tokens
            else:
                tokens_to_process = request.prompt_token_ids
            cache_to_use = request.prompt_cache  # May be None

            # Validate cache before using it
            if cache_to_use is not None:
                if not self._validate_cache(cache_to_use):
                    logger.warning(
                        f"Request {request.request_id}: invalid cache, "
                        f"proceeding without cache"
                    )
                    cache_to_use = None
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                else:
                    # Check memory: _merge_caches doubles cache memory temporarily
                    # Skip cache if available memory is tight
                    try:
                        from .memory_cache import estimate_kv_cache_memory
                        cache_bytes = estimate_kv_cache_memory(cache_to_use)
                        import psutil
                        avail = psutil.virtual_memory().available
                        # Memory amplification during dequantize + merge:
                        # - q4: quantized + full precision coexist = ~5x quantized size
                        # - q8: quantized + full precision coexist = ~3x quantized size
                        # - No quant: merge overhead only = ~2x
                        kv_bits = getattr(self, '_kv_cache_bits', 0)
                        if kv_bits and kv_bits <= 4:
                            multiplier = 5.0
                        elif kv_bits and kv_bits <= 8:
                            multiplier = 3.0
                        else:
                            multiplier = 2.0
                        needed = cache_bytes * multiplier
                        if needed > avail:
                            logger.warning(
                                f"Request {request.request_id}: skipping cache reuse "
                                f"(need {needed / 1048576:.0f}MB, "
                                f"available {avail / 1048576:.0f}MB)"
                            )
                            cache_to_use = None
                            request.prompt_cache = None
                            request.cached_tokens = 0
                            request.remaining_tokens = request.prompt_token_ids
                            tokens_to_process = request.prompt_token_ids
                    except ImportError:
                        pass  # psutil is a required dep but handle gracefully
                    except Exception as e:
                        logger.debug(f"Memory check failed, skipping: {e}")

            # Insert into BatchGenerator with optional cache.
            # Wrapped in try/except to prevent lost requests — if insert fails
            # completely, put the request back in the waiting queue.
            try:
                try:
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        max_tokens=[request.sampling_params.max_tokens],
                        caches=[cache_to_use] if cache_to_use else None,
                    )
                except Exception as e:
                    # Cache-related insertion failure - retry without cache
                    if cache_to_use is not None:
                        logger.warning(
                            f"Request {request.request_id}: cache insertion failed "
                            f"({type(e).__name__}: {e}), retrying without cache"
                        )
                        cache_to_use = None
                        request.prompt_cache = None
                        request.cached_tokens = 0
                        request.remaining_tokens = request.prompt_token_ids
                        tokens_to_process = request.prompt_token_ids
                        uids = self.batch_generator.insert(
                            [tokens_to_process],
                            max_tokens=[request.sampling_params.max_tokens],
                            caches=None,
                        )
                    else:
                        raise
            except Exception as e:
                # Both insert attempts failed — put request back to avoid permanent loss
                logger.error(
                    f"Request {request.request_id}: insert failed completely "
                    f"({type(e).__name__}: {e}), returning to waiting queue"
                )
                self.waiting.appendleft(request)
                break

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                self.running[request.request_id] = request
                scheduled.append(request)

                # H1 parity: Add per-request stop tokens to shared batch generator
                # Track additions so they can be removed on cleanup
                if request.sampling_params.stop_token_ids and self.batch_generator is not None:
                    new_tokens = set(request.sampling_params.stop_token_ids)
                    self.batch_generator.stop_tokens.update(new_tokens)
                    request._added_stop_tokens = new_tokens

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = (
                    f", {request.cached_tokens} cached"
                    if request.cached_tokens > 0
                    else ""
                )
                logger.debug(
                    f"Scheduled request {request.request_id} (uid={uid}) "
                    f"with {request.num_prompt_tokens} tokens{cache_info}"
                )

        return scheduled

    def _process_batch_responses(
        self, responses: List[Any]
    ) -> Tuple[List[RequestOutput], Set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Append token to request
            request.append_output_token(response.token)

            # Use streaming detokenizer for correct multi-byte char handling
            detok = self._get_detokenizer(request_id)

            # Check if finished BEFORE adding token to detokenizer
            # so stop tokens (e.g. <|im_end|>) don't leak into new_text
            is_stop = response.finish_reason == "stop"
            string_stop_truncate = -1  # >=0 when string stop matched

            if not is_stop:
                detok.add_token(response.token)
                new_text = detok.last_segment

                # Post-decode string stop sequence check.
                # BatchGenerator only handles integer stop_token_ids;
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
                # Stop token: don't decode it, just flush any buffered text
                new_text = ""

            # Create output
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=request.cached_tokens,
            )

            # Determine effective finish reason (string stop overrides)
            finish_reason = response.finish_reason
            if string_stop_truncate >= 0:
                finish_reason = "stop"

            # Check if finished
            if finish_reason is not None:
                if finish_reason == "stop":
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif finish_reason == "length":
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

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

                # For string stop: tell BatchGenerator to stop generating
                if string_stop_truncate >= 0 and self.batch_generator is not None:
                    uid = self.request_id_to_uid.get(request_id)
                    if uid is not None:
                        try:
                            self.batch_generator.remove([uid])
                        except Exception:
                            pass

                # Extract cache for future reuse
                if hasattr(response, "prompt_cache"):
                    try:
                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache:
                            # For paged cache, extract actual tensor states
                            # This allows cache to survive BatchGenerator recreation
                            if self.block_aware_cache is not None:
                                # Skip re-extraction for full cache-hit requests.
                                # Blocks already exist from the original cold store.
                                if (
                                    hasattr(request, "cached_tokens")
                                    and request.cached_tokens
                                    >= len(request.prompt_token_ids)
                                ):
                                    pass  # Already cached, nothing to do
                                else:
                                    # Paged cache: truncate to N-1 tokens so the
                                    # last prompt token can be re-fed on cache hit.
                                    # Without this, the last token's KV would be
                                    # duplicated with wrong positional encoding.
                                    prompt_len = len(request.prompt_token_ids)
                                    cache_for_extract = (
                                        self._truncate_cache_to_prompt_length(
                                            raw_cache, prompt_len
                                        )
                                    )

                                    if cache_for_extract is not None:
                                        # L2: Persist to disk (full-precision,
                                        # pre-quantization).  Skip for hybrid models
                                        # because SSM layers retain output-token
                                        # state that can't be cleanly truncated.
                                        if (
                                            self.disk_cache is not None
                                            and not self._is_hybrid
                                        ):
                                            try:
                                                self.disk_cache.store(
                                                    list(request.prompt_token_ids),
                                                    cache_for_extract,
                                                )
                                            except Exception as de:
                                                logger.debug(
                                                    f"Disk cache store failed for "
                                                    f"{request_id}: {de}"
                                                )
                                        # Quantize for storage-efficient extraction
                                        if getattr(self, '_kv_cache_bits', 0):
                                            cache_for_extract = (
                                                self._quantize_cache_for_storage(
                                                    cache_for_extract
                                                )
                                            )
                                        extracted_cache = (
                                            self._extract_cache_states(
                                                cache_for_extract
                                            )
                                        )
                                        if extracted_cache:
                                            request._extracted_cache = extracted_cache
                                            logger.info(
                                                f"Extracted {len(extracted_cache)} "
                                                f"layer states for request "
                                                f"{request_id}"
                                            )
                                        else:
                                            logger.warning(
                                                f"Cache extraction returned empty "
                                                f"for {request_id}"
                                            )
                                    else:
                                        logger.warning(
                                            f"Cannot produce prompt-only cache for "
                                            f"{request_id}, skipping paged cache store"
                                        )
                            else:
                                # Standard cache stores object references
                                request._extracted_cache = raw_cache
                        else:
                            logger.info(
                                f"No cache returned from BatchGenerator for {request_id}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to extract cache for {request_id}: {e}")

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        # H1 parity: Snapshot stop tokens from requests that will SURVIVE this cleanup.
        # This prevents removing tokens still needed by other running requests.
        _surviving_stops = set()
        for rid, req in self.running.items():
            if rid not in finished_ids:
                _surviving_stops.update(getattr(req, '_added_stop_tokens', set()))

        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Store cache for future reuse
            if request is not None and request.prompt_token_ids:
                if self.block_aware_cache is not None:
                    # Store in paged cache
                    # IMPORTANT: Use ONLY prompt tokens for block hashing/indexing.
                    # Using prompt+output would misalign block boundaries since the
                    # next request with the same prompt would search for prompt-only
                    # token hashes, which wouldn't match blocks that span the
                    # prompt/output boundary.
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            prompt_tokens = list(request.prompt_token_ids)
                            self.block_aware_cache.store_cache(
                                request_id,
                                prompt_tokens,
                                request._extracted_cache,
                            )
                            logger.info(
                                f"Stored paged cache for request {request_id} "
                                f"({len(prompt_tokens)} prompt tokens, "
                                f"{len(request._extracted_cache)} layers, "
                                f"cache truncated to {len(prompt_tokens) - 1} tokens)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to store paged cache for {request_id}: {e}"
                            )
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None
                    # NOTE: Do NOT call release_cache() here - that would delete
                    # block tables from the paged cache. Blocks should persist for
                    # future requests to share; LRU eviction handles cleanup.
                    # But DO remove the per-request tracking entries to prevent
                    # unbounded memory growth.
                    self.block_aware_cache._request_tables.pop(request_id, None)
                    self.block_aware_cache.paged_cache.detach_request(request_id)

                elif self.memory_aware_cache is not None:
                    # Store in memory-aware prefix cache
                    # Key is prompt tokens only. Cache is truncated to prompt_len-1
                    # so the last token can be re-fed on cache hit for generation.
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            prompt_tokens = list(request.prompt_token_ids)
                            prompt_len = len(prompt_tokens)
                            cache_to_store = self._truncate_cache_to_prompt_length(
                                request._extracted_cache, prompt_len
                            )
                            if cache_to_store is None:
                                logger.debug(
                                    f"Request {request_id}: cannot truncate cache "
                                    f"to prompt length (hybrid model), skipping store"
                                )
                            else:
                                # Quantize for storage efficiency
                                if getattr(self, '_kv_cache_bits', 0):
                                    cache_to_store = self._quantize_cache_for_storage(
                                        cache_to_store
                                    )
                                stored = self.memory_aware_cache.store(
                                    prompt_tokens,
                                    cache_to_store,
                                )
                                if stored:
                                    logger.info(
                                        f"Stored cache for request {request_id} "
                                        f"({prompt_len} prompt tokens, "
                                        f"KV truncated to {prompt_len - 1})"
                                    )
                                else:
                                    logger.warning(
                                        f"Cache store rejected for request {request_id} "
                                        f"({prompt_len} tokens) — entry too large for budget"
                                    )
                                # L2: Also persist to disk (full-precision, before GC).
                                # Skip for hybrid models — SSM layers retain
                                # output-token state that can't be truncated.
                                if (
                                    self.disk_cache is not None
                                    and not self._is_hybrid
                                ):
                                    try:
                                        disk_data = self._truncate_cache_to_prompt_length(
                                            request._extracted_cache, prompt_len
                                        )
                                        if disk_data is not None:
                                            self.disk_cache.store(prompt_tokens, disk_data)
                                    except Exception as de:
                                        logger.debug(f"Disk cache store failed for {request_id}: {de}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to store memory-aware cache for {request_id}: {e}"
                            )
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None

                elif self.prefix_cache is not None:
                    # Store in legacy prefix cache (same truncation as memory-aware)
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            prompt_tokens = list(request.prompt_token_ids)
                            prompt_len = len(prompt_tokens)
                            cache_to_store = self._truncate_cache_to_prompt_length(
                                request._extracted_cache, prompt_len
                            )
                            if cache_to_store is not None:
                                # Quantize for storage efficiency
                                if getattr(self, '_kv_cache_bits', 0):
                                    cache_to_store = self._quantize_cache_for_storage(
                                        cache_to_store
                                    )
                                self.prefix_cache.store_cache(
                                    prompt_tokens,
                                    cache_to_store,
                                )
                                logger.debug(
                                    f"Stored cache for request {request_id} "
                                    f"({prompt_len} prompt tokens, "
                                    f"truncated from {prompt_len + len(request.output_token_ids)})"
                                )
                                # L2: Also persist to disk (full-precision).
                                # Skip for hybrid models — SSM layers retain
                                # output-token state that can't be truncated.
                                if (
                                    self.disk_cache is not None
                                    and not self._is_hybrid
                                ):
                                    try:
                                        disk_data = self._truncate_cache_to_prompt_length(
                                            request._extracted_cache, prompt_len
                                        )
                                        if disk_data is not None:
                                            self.disk_cache.store(prompt_tokens, disk_data)
                                    except Exception as de:
                                        logger.debug(f"Disk cache store failed for {request_id}: {de}")
                        except Exception as e:
                            logger.debug(f"Failed to store cache for {request_id}: {e}")
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None

            # H1 parity: Remove per-request stop tokens from batch generator
            if (
                request is not None
                and self.batch_generator is not None
                and getattr(request, '_added_stop_tokens', None)
            ):
                removable = request._added_stop_tokens - _surviving_stops - self.stop_tokens
                if removable:
                    self.batch_generator.stop_tokens -= removable

            # Clean up streaming detokenizer
            self._cleanup_detokenizer(request_id)

            # Remove from running and requests dict (prevents memory leak)
            if request_id in self.running:
                del self.running[request_id]
            self.requests.pop(request_id, None)

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Track as finished
            self.finished_req_ids.add(request_id)

        # Only clear Metal memory cache when no other requests are actively
        # running. Calling mx.clear_memory_cache() during an active prefill
        # can interfere with in-flight GPU operations and cause crashes.
        if finished_ids and not self.running:
            try:
                import mlx.core as mx
                mx.clear_memory_cache()
            except Exception:
                pass

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        error_str = str(error)
        return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Clear batch generator (this is the source of the corruption)
        self.batch_generator = None
        self._current_sampler_params = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        logger.info("Cache recovery completed")

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # Reset request state — must clear ALL generation state so the
            # retried request starts from scratch with correct token budget
            request.status = RequestStatus.WAITING
            request.batch_uid = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            request.output_token_ids = []
            request.output_text = ""
            request.num_computed_tokens = 0

            # Clear extracted cache to prevent poisoning paged cache with stale
            # data from the destroyed BatchGenerator context
            if hasattr(request, '_extracted_cache'):
                request._extracted_cache = None

            # Clear stale detokenizer — request will restart from scratch
            self._cleanup_detokenizer(request_id)

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def step(self, max_retries: int = 2) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. Automatically recovers from cache/batch errors

        Cache error recovery only applies to BatchGenerator.next() and
        response processing — scheduling errors propagate immediately.

        Args:
            max_retries: Number of times to retry on cache errors (default 2)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Schedule waiting requests (errors here propagate immediately —
        # these are logic errors, not cache corruption)
        scheduled = self._schedule_waiting()
        output.scheduled_request_ids = [r.request_id for r in scheduled]
        output.num_scheduled_tokens = sum(
            r.num_prompt_tokens for r in scheduled
        )

        # Run generation step with cache error recovery
        if self.batch_generator is not None and self.running:
            for attempt in range(max_retries + 1):
                try:
                    responses = self.batch_generator.next()
                    output.has_work = True

                    if responses:
                        outputs, finished_ids = self._process_batch_responses(responses)
                        output.outputs = outputs
                        output.finished_request_ids = finished_ids
                        self._cleanup_finished(finished_ids)

                    # Success - break out of retry loop
                    break

                except Exception as e:
                    # Recover from cache/batch corruption or GPU errors.
                    # Pattern matching checks error message content.
                    # IndexError/TypeError during generation are *likely* cache-related
                    # (stale offsets, type mismatches from dequantized data) — treat as
                    # recoverable but log the full traceback for debugging.
                    is_pattern_match = self._is_cache_corruption_error(e)
                    is_gen_type_error = isinstance(e, (IndexError, TypeError))
                    is_cache_error = is_pattern_match or is_gen_type_error
                    if is_gen_type_error and not is_pattern_match:
                        logger.warning(
                            f"Treating {type(e).__name__} as potential cache error "
                            f"(may indicate a real bug): {e}",
                            exc_info=True,
                        )
                    if is_cache_error and attempt < max_retries:
                        logger.warning(
                            f"Batch generation error (attempt {attempt + 1}/{max_retries + 1}): "
                            f"{type(e).__name__}: {e} — recovering with cache clear"
                        )
                        self._recover_from_cache_error()
                        self._reschedule_running_requests()
                        # Re-schedule after recovery
                        self._schedule_waiting()
                    else:
                        logger.error(f"Error in batch generation step: {e}")
                        raise

        # Clear finished tracking for next step
        self.finished_req_ids.clear()

        # Periodic Metal memory cache cleanup during sustained traffic.
        # When requests are always running, _cleanup_finished never calls
        # mx.clear_memory_cache(). This timer ensures periodic cleanup
        # to prevent Metal's internal allocator cache from growing unbounded.
        now = time.monotonic()
        if now - self._last_metal_gc_time > self._metal_gc_interval:
            self._last_metal_gc_time = now
            try:
                import mlx.core as mx
                mx.clear_memory_cache()
                logger.debug("Periodic Metal memory cache cleanup")
            except Exception:
                pass

        return output

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[Request]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        # Include cache stats
        if self.block_aware_cache is not None:
            stats["paged_cache"] = self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            stats["memory_aware_cache"] = self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            stats["prefix_cache"] = self.prefix_cache.get_stats()
        return stats

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.block_aware_cache is not None:
            return self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            return self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            return self.prefix_cache.get_stats()
        return None

    def reset(self) -> None:
        """Reset the scheduler state."""
        # Abort all requests
        for request_id in list(self.requests.keys()):
            self.abort_request(request_id)

        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self.batch_generator = None
        self._current_sampler_params = None
        self._detokenizer_pool.clear()

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def deep_reset(self) -> None:
        """
        Deep reset that clears ALL cache state including model-level caches.

        This is more aggressive than reset() and should be used when
        switching engines or recovering from errors.
        """
        # Standard reset first
        self.reset()

        # Invalidate cached model config values so they are re-detected
        # if the scheduler is ever reused with a different model
        if hasattr(self, '_n_kv_heads_cached'):
            del self._n_kv_heads_cached

        # Clear any model-level cache state
        # MLX models may have internal cache references
        if hasattr(self.model, "cache"):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "cache"):
                    layer.cache = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                    layer.self_attn.cache = None

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")
