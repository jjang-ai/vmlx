# SPDX-License-Identifier: Apache-2.0
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
  - Keyed by hash(tuple(token_ids[:prompt_len]))
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

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .vision_embedding_cache import VisionEmbeddingCache

logger = logging.getLogger(__name__)


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
                if not isinstance(tmpl, KVCache) and isinstance(cached, KVCache):
                    result[i] = tmpl
                    fixed = True
            if fixed:
                logger.debug("Fixed hybrid cache: replaced KVCache at SSM positions")
            return result

        # Cache shorter than model — expand using template
        positions = kv_positions if kv_positions is not None else [
            i for i, t in enumerate(template) if isinstance(t, KVCache)
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


class HybridSSMStateCache:
    """Companion cache for SSM layer states in hybrid models.

    Hybrid models (SSM + attention, e.g. Qwen3.5-VL) store KVCache in the
    prefix cache but lose MambaCache/ArraysCache state. This companion cache
    stores SSM state captured at the prompt boundary during prefill, keyed
    by the prompt token prefix hash.

    On a prefix cache HIT for a hybrid model, if the companion SSM state also
    hits, we can reconstruct the FULL cache (KV + SSM) and skip the prefix
    entirely — saving all compute on prefix tokens.

    Without this, hybrid cache hits are wasted: the model must do a full
    prefill through all layers because SSM state is cumulative.
    """

    def __init__(self, max_entries: int = 50):
        self._store: OrderedDict[str, List[Any]] = OrderedDict()
        self._max_entries = max_entries

    def _key(self, token_ids: List[int], num_tokens: int) -> str:
        """Deterministic hash key from token prefix using SHA-256."""
        import hashlib
        import json
        data = json.dumps(token_ids[:num_tokens], separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()

    def store(
        self,
        token_ids: List[int],
        num_tokens: int,
        ssm_states: List[Any],
    ) -> None:
        """Store SSM layer states for a prompt prefix."""
        key = self._key(token_ids, num_tokens)
        # Remove existing entry to update position
        if key in self._store:
            del self._store[key]
        self._store[key] = ssm_states
        # Evict oldest if over limit
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def fetch(
        self, token_ids: List[int], num_tokens: int
    ) -> Optional[List[Any]]:
        """Fetch SSM states for a matching prompt prefix."""
        key = self._key(token_ids, num_tokens)
        states = self._store.get(key)
        if states is not None:
            # Move to end (most recently used)
            self._store.move_to_end(key)
        return states

    def clear(self) -> None:
        self._store.clear()


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
    logprobs: mx.array  # Log probabilities
    finish_reason: Optional[str] = None  # "stop", "length", or None
    prompt_cache: Optional[Callable[[], List[Any]]] = None  # Cache extraction function
    prompt_token_ids: Optional[List[int]] = None  # Original tokenized prompt for prefix key
    cached_tokens: int = 0  # Number of prompt tokens served from cache


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
    logprobs: List[mx.array]  # Log probs for each request
    max_tokens: List[int]  # Max tokens per request
    num_tokens: List[int]  # Tokens generated per request
    cache: List[Any]  # BatchKVCache for language model
    requests: List[MLLMBatchRequest]  # Full request data

    def __len__(self) -> int:
        return len(self.uids)

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
            elif isinstance(layer_cache, KVCache):
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
            logger.warning(f"Layer {i} merge failed ({type(layer_cache).__name__}), using empty BatchKVCache: {e}")
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
        elif isinstance(c, KVCache):
            converted.append(BatchKVCache.merge([c]))
        elif RotatingKVCache is not None and isinstance(c, RotatingKVCache):
            if BatchRotatingKVCache is not None:
                converted.append(BatchRotatingKVCache.merge([c]))
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
        vision_cache_size: int = 100,
        paged_cache_manager: Optional[Any] = None,
        block_aware_cache: Optional[Any] = None,
        memory_aware_cache: Optional[Any] = None,
        prefix_cache: Optional[Any] = None,
        disk_cache: Optional[Any] = None,
        kv_cache_bits: int = 0,
        kv_cache_group_size: int = 64,
        ssm_state_cache_size: int = 50,
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

        # Companion SSM state cache for hybrid models (MambaCache + KVCache).
        # Stores SSM layer states at prompt boundary so hybrid cache HITs can
        # skip the full prefix instead of wasting the KV cache hit.
        self._ssm_state_cache = HybridSSMStateCache(max_entries=ssm_state_cache_size)

        # Get language model for text generation
        self.language_model = getattr(model, "language_model", model)

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

        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.prefill_step_size = prefill_step_size

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
                self._hybrid_kv_positions = [i for i, t in enumerate(template) if isinstance(t, KVCache)]
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
            max_encoding_entries=vision_cache_size // 2,
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
        if mx.metal.is_available():
            # Use non-deprecated API when available (MLX ≥ 0.25)
            _device_info = getattr(mx, 'device_info', None) or mx.metal.device_info
            _set_cache = getattr(mx, 'set_cache_limit', None) or mx.metal.set_cache_limit
            # Check disk-streaming mode — server.py manages wired/cache limits
            try:
                from . import server as _server_module
                _is_streaming = getattr(_server_module, '_stream_from_disk', False)
            except Exception:
                _is_streaming = False
            # In streaming mode: do NOT override wired limit (server.py set a
            # reduced limit to allow SSD paging) or cache limit (set to 0).
            if not _is_streaming:
                self._old_wired_limit = mx.set_wired_limit(
                    _device_info()["max_recommended_working_set_size"]
                )
                # Set Metal allocator cache limit to 25% of max working set
                # (floor 512MB). Bounds the Metal allocator's free-list,
                # preventing it from hoarding memory that prefix cache / OS needs.
                try:
                    max_ws = _device_info()["max_recommended_working_set_size"]
                    cache_limit = max(512 * 1024 * 1024, int(max_ws * 0.25))
                    self._old_cache_limit = _set_cache(cache_limit)
                    logger.info(
                        f"Metal cache limit set to {cache_limit / (1024**3):.2f}GB "
                        f"(25% of {max_ws / (1024**3):.1f}GB max working set)"
                    )
                except Exception as e:
                    logger.debug(f"Metal cache limit not available: {e}")
            else:
                logger.info("Disk-streaming mode: skipping wired limit + cache limit override")

    def close(self) -> None:
        """Release resources and reset wired/cache limits."""
        if self._old_wired_limit is not None:
            mx.synchronize(MLLMBatchGenerator._stream)
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

        Args:
            request: Request to preprocess
        """
        from mlx_vlm.utils import prepare_inputs

        tic = time.perf_counter()

        # Collect all images (including video frames)
        all_images = []

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
                extract_video_frames_smart,
                save_frames_to_temp,
                DEFAULT_FPS,
                MAX_FRAMES,
            )

            fps = request.video_fps or DEFAULT_FPS
            max_frames = request.video_max_frames or MAX_FRAMES

            for video in request.videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path,
                        fps=fps,
                        max_frames=max_frames,
                    )
                    frame_paths = save_frames_to_temp(frames)
                    all_images.extend(frame_paths)
                except Exception as e:
                    logger.warning(f"Failed to process video: {e}")

        # Check pixel cache first
        cached_pixels = self.vision_cache.get_pixel_cache(all_images, request.prompt)
        if cached_pixels is not None:
            # Cache hit - use cached pixel values
            request.input_ids = cached_pixels.input_ids
            request.pixel_values = cached_pixels.pixel_values
            request.attention_mask = cached_pixels.attention_mask
            request.image_grid_thw = cached_pixels.image_grid_thw
            request.extra_kwargs = dict(cached_pixels.extra_kwargs)

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

        # Prepare inputs using mlx_vlm
        inputs = prepare_inputs(
            self.processor,
            images=all_images if all_images else None,
            prompts=request.prompt,
            image_token_index=image_token_index,
        )

        request.input_ids = inputs.get("input_ids")
        request.pixel_values = inputs.get("pixel_values")
        request.attention_mask = inputs.get("attention_mask")

        # Extract extra kwargs
        request.extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        request.image_grid_thw = request.extra_kwargs.pop("image_grid_thw", None)

        processing_time = time.perf_counter() - tic

        # Store in pixel cache for future reuse
        if all_images and request.pixel_values is not None:
            self.vision_cache.set_pixel_cache(
                images=all_images,
                prompt=request.prompt,
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                image_grid_thw=request.image_grid_thw,
                extra_kwargs=request.extra_kwargs,
                processing_time=processing_time,
            )

        self._stats.num_images_processed += len(all_images)
        self._stats.vision_encoding_time += processing_time

        logger.debug(
            f"Preprocessed request {request.request_id}: "
            f"{len(all_images)} images, {request.input_ids.size if request.input_ids is not None else 0} tokens "
            f"({processing_time:.2f}s)"
        )

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
        kwargs = dict(request.extra_kwargs)
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

        # Chunked prefill for text-only VLM requests with long prompts.
        # Image requests must run in one shot (vision encoder needs full sequence).
        # Short prompts (< 2x prefill_step_size) also run in one shot (overhead not worth it).
        # Hybrid SSM models (GatedDeltaNet/Mamba + attention) must run in one shot
        # because the language_model's forward pass computes masks from specific
        # cache positions (fa_idx/ssm_idx) that assume full-sequence processing.
        if not has_images and seq_len > self.prefill_step_size * 2 and not self._is_hybrid:
            # Use language_model directly for chunked text prefill
            lm = getattr(self.model, 'language_model', None)
            if lm is not None and cache is not None:
                processed = 0
                chunk_num = 0
                while processed < seq_len - 1:  # -1: keep last token for final logits
                    chunk_size = min(self.prefill_step_size, seq_len - 1 - processed)
                    chunk = input_ids[:, processed:processed + chunk_size]
                    try:
                        lm(chunk, cache=cache)
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
                    mx.eval([c.state for c in cache if hasattr(c, 'state')])
                    processed += chunk_size
                    chunk_num += 1
                    mx.clear_cache()

                # Final chunk: get logits from last token
                last_chunk = input_ids[:, processed:]
                output = lm(last_chunk, cache=cache)
                request.vision_encoded = True
                if hasattr(output, "logits"):
                    return output.logits
                return output

        # Standard single-shot VLM forward (image requests or short text)
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

        for req in requests:
            self._preprocess_request(req)
            # Save full token list BEFORE cache fetch can mutate req.input_ids.
            # Used later for SSM state cache keying (must be consistent with fetch key).
            req._original_token_ids = (
                req.input_ids.tolist()
                if req.input_ids is not None and req.input_ids.ndim == 1
                else req.input_ids[0].tolist()
                if req.input_ids is not None
                else []
            )
            # Track how many prompt tokens were served from cache (for usage reporting)
            req._cached_tokens = 0
            # After preprocessing, the prompt is fully tokenized including image patches.
            # Query the BlockAwarePrefixCache for reusable KV blocks.
            # fetch_cache returns (block_table, remaining_tokens) — NOT cache objects!
            # IMPORTANT: Skip prefix cache for requests WITH images — image placeholder
            # tokens are identical for same-sized images regardless of content, so a
            # cache hit would serve KV states from a different image's vision encoding.
            # Text-only follow-up requests (no new images) can safely use prefix cache.
            has_images = req.pixel_values is not None
            if self.block_aware_cache is not None and req.prompt_cache is None and not has_images:
                if req.input_ids is not None:
                    try:
                        token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()
                        block_table, remaining = self.block_aware_cache.fetch_cache(req.request_id, token_list)
                        if block_table is not None:
                                # Hybrid models (SSM + attention, e.g. Qwen3.5-VL):
                                # Prefix cache stores only KVCache (attention) layers.
                                # SSM layers are cumulative state that must process ALL tokens.
                                # For hybrid models without companion SSM state, the cached
                                # KV blocks are useless — skip reconstruction entirely to
                                # avoid allocating huge tensors that will be thrown away.
                                is_hybrid = self._is_hybrid

                                if is_hybrid:
                                    # Check companion SSM state cache BEFORE reconstruction.
                                    # Block-align num_tokens to match the store key alignment
                                    # (store uses (prompt_len // bs) * bs).
                                    _fetch_num = block_table.num_tokens
                                    if self.block_aware_cache is not None:
                                        _bs = getattr(self.block_aware_cache, 'block_size', 64)
                                        _fetch_num = (_fetch_num // _bs) * _bs
                                    ssm_states = self._ssm_state_cache.fetch(
                                        token_list, _fetch_num
                                    ) if _fetch_num > 0 else None
                                    if ssm_states is None:
                                        # No SSM state — can't use cached KV, must do full prefill.
                                        # Release the block refs that fetch_cache incremented
                                        # to prevent ref_count leak → OOM on subsequent requests.
                                        logger.info(
                                            f"VLM prefix cache MISS for {req.request_id}: "
                                            f"{block_table.num_tokens} KV blocks found but "
                                            f"no SSM companion state — full prefill required"
                                        )
                                        self.block_aware_cache.release_cache(req.request_id)
                                        continue  # Skip reconstruction

                                # Either non-hybrid OR hybrid with SSM state — reconstruct
                                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                                if reconstructed is not None:
                                    reconstructed = _dequantize_cache(reconstructed)
                                    if reconstructed is None:
                                        # Dequantize failed — release block refs to prevent leak
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

                                    req.prompt_cache = full_cache
                                    req._cached_tokens = block_table.num_tokens
                                    if remaining:
                                        req.input_ids = mx.array([remaining])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        logger.info(
                                            f"VLM HYBRID cache HIT for {req.request_id}: "
                                            f"{block_table.num_tokens} cached (KV+SSM), "
                                            f"{len(remaining)} remaining"
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
                                    # Pure attention VLM: can safely skip cached prefix tokens.
                                    # Trim input_ids to only remaining (uncached) tokens.
                                    req.prompt_cache = reconstructed
                                    req._cached_tokens = block_table.num_tokens
                                    if remaining:
                                        # Check if remaining tokens contain image placeholders.
                                        # If so, we'd need partial pixel_values which is complex —
                                        # fall back to full prefill instead.
                                        model_config = getattr(self.model, "config", None)
                                        img_token_id = (
                                            getattr(model_config, "image_token_index", None)
                                            if model_config else None
                                        )
                                        has_images = img_token_id is not None and img_token_id in remaining
                                        if has_images:
                                            req.prompt_cache = None
                                            req._cached_tokens = 0  # reset — full prefill needed
                                            logger.info(
                                                f"VLM prefix cache HIT for {req.request_id}: "
                                                f"{block_table.num_tokens} cached tokens, "
                                                f"remaining has images — full prefill"
                                            )
                                        else:
                                            req.input_ids = mx.array([remaining])
                                            req.pixel_values = None
                                            req.attention_mask = None
                                            req.image_grid_thw = None
                                            logger.info(
                                                f"VLM prefix cache HIT for {req.request_id}: "
                                                f"{block_table.num_tokens} cached, "
                                                f"{len(remaining)} remaining (text-only)"
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
            elif (self.memory_aware_cache is not None or self.prefix_cache is not None) and req.prompt_cache is None:
                if req.input_ids is not None:
                    try:
                        token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()

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

                                # Hybrid model check (same logic as paged path)
                                is_hybrid = self._is_hybrid
                                if is_hybrid:
                                    logger.info(
                                        f"VLM memory/legacy cache HIT for {req.request_id}: "
                                        f"{len(token_list) - len(remaining)} cached tokens "
                                        f"(hybrid model — full prefill required)"
                                    )
                                else:
                                    req.prompt_cache = cache
                                    num_cached = len(token_list) - len(remaining)
                                    if remaining:
                                        model_config = getattr(self.model, "config", None)
                                        img_token_id = (
                                            getattr(model_config, "image_token_index", None)
                                            if model_config else None
                                        )
                                        has_images = img_token_id is not None and img_token_id in remaining
                                        if has_images:
                                            req.prompt_cache = None
                                            logger.info(
                                                f"VLM cache HIT for {req.request_id}: "
                                                f"remaining has images — full prefill"
                                            )
                                        else:
                                            req._cached_tokens = num_cached
                                            req.input_ids = mx.array([remaining])
                                            req.pixel_values = None
                                            req.attention_mask = None
                                            req.image_grid_thw = None
                                            logger.info(
                                                f"VLM cache HIT for {req.request_id}: "
                                                f"{num_cached} cached, "
                                                f"{len(remaining)} remaining"
                                            )
                                    else:
                                        req._cached_tokens = len(token_list)
                                        req.input_ids = mx.array([token_list[-1:]])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        logger.info(
                                            f"VLM cache FULL HIT for {req.request_id}: "
                                            f"{len(token_list)} cached tokens"
                                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch VLM cache for {req.request_id}: {e}")

            # L2: Disk cache fallback when in-memory cache missed.
            # DiskCacheManager.fetch() returns Optional[List[Any]] (exact match only,
            # no partial prefix), NOT a tuple like prefix_cache.fetch_cache().
            if req.prompt_cache is None and self.disk_cache is not None:
                if req.input_ids is not None:
                    try:
                        token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()
                        disk_result = self.disk_cache.fetch(token_list)
                        if disk_result is not None:
                            if not self._is_hybrid:
                                # Check for image tokens in remaining suffix
                                model_config = getattr(self.model, "config", None)
                                img_token_id = (
                                    getattr(model_config, "image_token_index", None)
                                    if model_config else None
                                )
                                has_images = img_token_id is not None and img_token_id in token_list
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
                                    else:
                                        req.prompt_cache = disk_result
                                        req._cached_tokens = len(token_list)
                                        # Disk cache is exact-match (hash-based), all tokens cached.
                                        # Set input to last token only for decode phase.
                                        req.input_ids = mx.array([token_list[-1:]])
                                        req.pixel_values = None
                                        req.attention_mask = None
                                        req.image_grid_thw = None
                                        logger.info(
                                            f"VLM disk cache (L2) HIT for {req.request_id}: "
                                            f"{len(token_list)} cached tokens"
                                        )
                    except Exception as e:
                        logger.debug(f"VLM disk cache fetch failed for {req.request_id}: {e}")

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
                    req_cache = _fix_hybrid_cache(
                        cache_for_fix, self.language_model,
                        kv_positions=self._hybrid_kv_positions,
                        num_model_layers=self._hybrid_num_layers,
                    )
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

                try:
                    logits = self._run_vision_encoding(req, cache=req_cache)
                except ValueError as ve:
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
                        logits = self._run_vision_encoding(req, cache=req_cache)
                    else:
                        raise
                per_request_caches.append(req_cache)

                last_logits = logits[:, -1, :]
                logprobs = last_logits - mx.logsumexp(
                    last_logits, axis=-1, keepdims=True
                )
                req_sampler = self._make_request_sampler(req)
                sampled = req_sampler(logprobs)

                # Async submit cache states to GPU for CPU/GPU overlap
                try:
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
                    mx.async_eval(sampled, logprobs, *cache_states)
                except Exception as e:
                    logger.warning(f"Cache state submission error (non-fatal): {e}")
                    mx.async_eval(sampled, logprobs)

                first_tokens.append(sampled.item())
                all_logprobs.append(logprobs.squeeze(0))
                succeeded_requests.append(req)

                # Capture SSM state at prompt boundary for hybrid models
                if self._is_hybrid and req.prompt_cache is None:
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
                                        mx.contiguous(mx.array(a)) if a is not None else None
                                        for a in c.cache
                                    ]
                                    ssm_layers.append(cloned)
                                else:
                                    ssm_layers.append(c)
                        if ssm_layers:
                            all_tokens = getattr(req, '_original_token_ids', None)
                            if all_tokens is None:
                                all_tokens = input_ids_list[i]
                            prompt_len = len(all_tokens)
                            if self.block_aware_cache is not None:
                                bs = getattr(self.block_aware_cache, 'block_size', 64)
                                prompt_len = (prompt_len // bs) * bs
                            if prompt_len > 0:
                                self._ssm_state_cache.store(
                                    all_tokens, prompt_len, ssm_layers
                                )
                                logger.debug(
                                    f"Captured SSM state at prompt boundary for "
                                    f"{req.request_id}: {len(ssm_layers)} layers, "
                                    f"{prompt_len} tokens (block-aligned)"
                                )
                    except Exception as e:
                        logger.debug(f"SSM state capture failed for {req.request_id}: {e}")
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
                        last_logits = logits[:, -1, :]
                        logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
                        req_sampler = self._make_request_sampler(req)
                        sampled = req_sampler(logprobs)
                        mx.async_eval(sampled, logprobs)
                        first_tokens.append(sampled.item())
                        all_logprobs.append(logprobs.squeeze(0))
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
                            # Also clear SSM state cache — stale SSM state can cause
                            # shape mismatches in hybrid models
                            self._ssm_state_cache.clear()
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
                                last_logits = logits[:, -1, :]
                                logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
                                req_sampler = self._make_request_sampler(req)
                                sampled = req_sampler(logprobs)
                                mx.async_eval(sampled, logprobs)
                                first_tokens.append(sampled.item())
                                all_logprobs.append(logprobs.squeeze(0))
                                succeeded_requests.append(req)
                                continue
                            except Exception as nuclear_err:
                                logger.error(f"Nuclear retry also failed for {req.request_id}: {nuclear_err}")
                        else:
                            logger.error(f"Retry also failed for {req.request_id}: {retry_err}")
                # Per-request prefill failure (bad image, OOM, etc.)
                # Queue an immediate error response instead of killing the entire batch.
                logger.error(
                    f"Prefill failed for {req.request_id}: {prefill_err} — "
                    f"other requests in batch will continue"
                )
                self._prefill_errors.append(MLLMBatchResponse(
                    uid=req.uid,
                    request_id=req.request_id,
                    token=0,
                    logprobs=mx.zeros((1,)),
                    finish_reason="stop",
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
            if len(per_request_caches) == 1 and not force_batch_cache:
                # Single request with no active batch: keep raw KVCache/ArraysCache
                # to preserve integer offsets (Qwen3.5 needs cache.offset as int).
                # If force_batch_cache is True, we're extending into an active batch
                # and MUST produce batch-aware caches for extend() compatibility.
                batch_cache = per_request_caches[0]
            else:
                batch_cache = _merge_caches(per_request_caches)
        except Exception as e:
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

        return MLLMBatch(
            uids=[req.uid for req in requests],
            request_ids=[req.request_id for req in requests],
            y=y,
            logprobs=all_logprobs,
            max_tokens=[req.max_tokens for req in requests],
            num_tokens=[0] * len(requests),
            cache=batch_cache,
            requests=requests,
        )

    def _make_request_sampler(self, request: MLLMBatchRequest) -> Callable[[mx.array], mx.array]:
        """Create a sampler for a specific request's sampling parameters.

        Each request can have different temperature/top_p/top_k/min_p.
        Repetition penalty is applied via logits_processors when set.
        Samplers are cached on the request to avoid per-step reconstruction.
        """
        cached = getattr(request, '_cached_sampler', None)
        if cached is not None:
            return cached

        from mlx_lm.sample_utils import make_sampler
        base_sampler = make_sampler(
            temp=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else 0,
            min_p=request.min_p if request.min_p > 0 else 0.0,
        )

        # Apply repetition penalty if set for this request
        rep_penalty = getattr(request, "repetition_penalty", 1.0)
        if rep_penalty is not None and rep_penalty != 1.0:
            from mlx_lm.sample_utils import make_logits_processors
            logits_procs = make_logits_processors(repetition_penalty=rep_penalty)
            # Use _original_token_ids (saved before cache fetch trims input_ids)
            # so repetition penalty covers the full prompt, not just uncached tokens.
            prompt_list = getattr(request, '_original_token_ids', None)
            if prompt_list is None:
                ids = request.input_ids
                prompt_list = ids[0].tolist() if ids is not None and ids.ndim > 1 else (
                    ids.tolist() if ids is not None else []
                )
            def sampler_with_penalty(logits, _req=request, _prompt_list=prompt_list):
                # Build full token sequence (prompt + generated) so penalty
                # applies to already-generated tokens, not just the prompt.
                all_tokens = mx.array(_prompt_list + _req.output_tokens)
                processed = logits
                for proc in logits_procs:
                    processed = proc(all_tokens, processed)
                return base_sampler(processed)
            request._cached_sampler = sampler_with_penalty
            return sampler_with_penalty

        request._cached_sampler = base_sampler
        return base_sampler

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

        # Run language model only (not full VLM)
        output = self.language_model(input_tokens, cache=cache)

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        logits = logits[:, -1, :]

        # Per-request sampling using each request's sampling parameters
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        batch = self.active_batch
        if batch and len(batch.requests) == logprobs.shape[0]:
            tokens = []
            for i, req in enumerate(batch.requests):
                req_sampler = self._make_request_sampler(req)
                tokens.append(req_sampler(logprobs[i:i+1]))
            sampled = mx.concatenate(tokens, axis=0)
        else:
            sampled = self.sampler(logprobs)

        return sampled, list(logprobs)

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
                        isinstance(c, KVCache) and not isinstance(c, BatchKVCache)
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

        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)

        y = y.tolist()
        toc = time.perf_counter()

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
                # Extract cache NOW before batch.filter() invalidates indices
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
                )
            )

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

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
