# SPDX-License-Identifier: Apache-2.0
"""
MLLM Batch Generator for multimodal continuous batching.

This module implements continuous batching for Multimodal Language Models (MLLMs)
like Qwen3-VL, following the same architecture as LLM continuous batching but
adapted for vision models.

Key insight: VLM models have a `model.language_model` which is a standard LLM.
After the initial forward pass with vision encoding, text generation uses only
the language model - which CAN be batched using the same BatchKVCache pattern.

Architecture:
1. Vision inputs are processed per-request (not batched)
2. Initial VLM forward pass extracts cross-attention states / encoder outputs
3. Language model generation is batched using BatchKVCache (like LLM batching)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .multimodal_processor import MultimodalProcessor
from .vision_embedding_cache import VisionEmbeddingCache

logger = logging.getLogger(__name__)


def _dequantize_cache(cache: List[Any]) -> List[Any]:
    """Dequantize QuantizedKVCache layers to KVCache for batch generation.

    BatchGenerator requires full-precision KVCache objects for merge/extract.
    Returns original cache unmodified if no quantized layers found.
    """
    try:
        from mlx_lm.models.cache import KVCache, QuantizedKVCache
    except ImportError:
        return cache

    has_quantized = any(isinstance(c, QuantizedKVCache) for c in cache)
    if not has_quantized:
        return cache

    result = []
    for layer_cache in cache:
        if isinstance(layer_cache, QuantizedKVCache) and layer_cache.keys is not None:
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
                logger.warning(f"KV dequantization failed: {e}, using original")
                result.append(layer_cache)
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
            # Cache already correct length — no expansion needed
            if len(cache) == num_model_layers:
                return cache
            # Cache length doesn't match expected KV layer count — can't fix
            if len(cache) != len(kv_positions):
                logger.warning(
                    f"Cache length mismatch: {len(cache)} reconstructed vs "
                    f"{len(kv_positions)} KV positions in {num_model_layers}-layer model"
                )
                return cache

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
                f"{len(positions)} KV positions in {n_layers}-layer model"
            )
            return cache

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
        logger.warning(f"_fix_hybrid_cache failed: {e}, using original cache")
        return cache


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
        for c in self.cache:
            if hasattr(c, "filter"):
                c.filter(keep_idx_array)

    def extract_cache(self, idx: int) -> List[Any]:
        """
        Extract cache for a single request (for caching).

        Args:
            idx: Index of request in batch

        Returns:
            Cache state for that request
        """
        extracted = []
        for c in self.cache:
            if hasattr(c, "extract"):
                # Batched cache (BatchKVCache, BatchMambaCache) — extract single request
                extracted.append(c.extract(idx))
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


class MLLMBatchGenerator:
    """
    Batch generator for Vision Language Models.

    This class manages continuous batching for MLLM requests:

    1. Vision Encoding Phase:
       - Process images/videos through vision encoder (per-request)
       - Extract vision features and merge with text embeddings
       - Store cross-attention states for language model

    2. Language Generation Phase:
       - Use language model with BatchKVCache for batched generation
       - Generate tokens for all requests simultaneously
       - Same pattern as LLM BatchGenerator

    Example:
        >>> generator = MLLMBatchGenerator(model, processor)
        >>> uids = generator.insert([request1, request2])
        >>> while responses := generator.next():
        ...     for resp in responses:
        ...         print(f"Request {resp.request_id}: token={resp.token}")
    """

    # Generation stream for async eval
    _stream = None

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        mm_processor: Optional[MultimodalProcessor] = None,
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
    ):
        """
        Initialize MLLM batch generator.

        Args:
            model: The VLM model (must have model.language_model)
            processor: The VLM processor for tokenization and image processing
            mm_processor: Optional MultimodalProcessor for input preparation
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
        """
        self.model = model
        self.processor = processor
        self.mm_processor = mm_processor
        self.paged_cache_manager = paged_cache_manager
        self.block_aware_cache = block_aware_cache

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
            except Exception:
                pass

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
        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.metal.device_info()["max_recommended_working_set_size"]
            )

    def close(self) -> None:
        """Release resources and reset wired limit."""
        if self._old_wired_limit is not None:
            mx.synchronize(MLLMBatchGenerator._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

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

        This runs the full VLM model (vision + language) on the prompt,
        which encodes the images and prepares the language model cache.

        Args:
            request: Preprocessed request with input_ids and pixel_values
            cache: Optional pre-initialized BatchKVCache list

        Returns:
            Logits from the forward pass
        """
        kwargs = dict(request.extra_kwargs)
        if request.pixel_values is not None:
            kwargs["pixel_values"] = request.pixel_values
        if request.attention_mask is not None:
            # VLM models use "mask" parameter (not "attention_mask")
            kwargs["mask"] = request.attention_mask
        if request.image_grid_thw is not None:
            kwargs["image_grid_thw"] = request.image_grid_thw
        if cache is not None:
            kwargs["cache"] = cache

        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        output = self.model(input_ids, **kwargs)
        request.vision_encoded = True

        if hasattr(output, "logits"):
            return output.logits
        return output

    def _process_prompts(self, requests: List[MLLMBatchRequest]) -> MLLMBatch:
        tic = time.perf_counter()

        for req in requests:
            self._preprocess_request(req)
            # After preprocessing, the prompt is fully tokenized including image patches.
            # Query the BlockAwarePrefixCache for reusable KV blocks.
            # fetch_cache returns (block_table, remaining_tokens) — NOT cache objects!
            if self.block_aware_cache is not None and req.prompt_cache is None:
                if req.input_ids is not None:
                    try:
                        token_list = req.input_ids.tolist() if req.input_ids.ndim == 1 else req.input_ids[0].tolist()
                        block_table, remaining = self.block_aware_cache.fetch_cache(req.request_id, token_list)
                        if block_table is not None:
                            # Cache hit! Reconstruct actual KVCache objects from stored block data.
                            reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                            if reconstructed is not None:
                                # Dequantize if KV quant was applied during storage.
                                # BatchGenerator needs full-precision KVCache, not QuantizedKVCache.
                                reconstructed = _dequantize_cache(reconstructed)
                                req.prompt_cache = reconstructed
                                logger.info(
                                    f"VLM prefix cache HIT for {req.request_id}: "
                                    f"{block_table.num_tokens} cached tokens, "
                                    f"{len(remaining)} remaining"
                                )
                    except Exception as e:
                        logger.warning(f"Failed to fetch paged cache for {req.request_id}: {e}")

        # Get token sequences and lengths
        input_ids_list = [
            req.input_ids.tolist() if req.input_ids is not None else [0]
            for req in requests
        ]
        lengths = [len(ids) for ids in input_ids_list]
        max_length = max(lengths)
        padding = [max_length - seq_len for seq_len in lengths]

        self._stats.prompt_tokens += sum(lengths)

        per_request_caches = []
        first_tokens = []
        all_logprobs = []

        for i, req in enumerate(requests):
            with mx.stream(MLLMBatchGenerator._stream):
                # Per-request prefill uses standard KVCache (integer offsets)
                # so that mlx_vlm model layers can use cache.offset as a Python int.
                # After all prefills, we merge into a single BatchKVCache for decode.
                if req.prompt_cache is not None:
                    # Pre-filled cache from prefix cache — these are KVCache objects.
                    # For hybrid models (e.g. Qwen3.5 with SSM + attention layers),
                    # replace KVCache at SSM positions with fresh ArraysCache.
                    # SSM layers need ArraysCache (cumulative state, not block-cached),
                    # and using KVCache at SSM positions causes make_mask signature errors.
                    req_cache = _fix_hybrid_cache(
                        req.prompt_cache, self.language_model,
                        kv_positions=self._hybrid_kv_positions,
                        num_model_layers=self._hybrid_num_layers,
                    )
                else:
                    # Create fresh per-request cache using model.make_cache()
                    # This correctly creates KVCache for attention layers and
                    # MambaCache/ArraysCache for SSM layers in hybrid models.
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

                logits = self._run_vision_encoding(req, cache=req_cache)
                per_request_caches.append(req_cache)

                last_logits = logits[:, -1, :]
                logprobs = last_logits - mx.logsumexp(
                    last_logits, axis=-1, keepdims=True
                )
                # Use per-request sampler based on temperature/top_p
                req_sampler = self._make_request_sampler(req)
                sampled = req_sampler(logprobs)

                # Eval all cache states: KVCache has .state, MambaCache has .cache
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
                    mx.eval(sampled, logprobs, *cache_states)
                except Exception as e:
                    logger.warning(f"Cache state eval error (non-fatal): {e}")
                    mx.eval(sampled, logprobs)

                first_tokens.append(sampled.item())
                all_logprobs.append(logprobs.squeeze(0))

        y = mx.array(first_tokens)

        # Merge per-request caches into batch-aware caches for batched decode.
        # Handles KVCache→BatchKVCache, MambaCache→BatchMambaCache, etc.
        try:
            if len(per_request_caches) == 1:
                # Single request: keep original per-request caches (KVCache, ArraysCache)
                # instead of converting to BatchKVCache/BatchMambaCache.
                # This preserves integer offsets that models like Qwen3.5 require
                # (Qwen3_5Attention uses cache.offset as Python int for mask slicing).
                # No filter/extend is needed for single-request batches.
                batch_cache = per_request_caches[0]
            else:
                batch_cache = _merge_caches(per_request_caches)
        except Exception as e:
            logger.error(f"Cache merge failed, creating fresh batch cache: {e}")
            from mlx_lm.models.cache import BatchKVCache
            num_layers = len(per_request_caches[0]) if per_request_caches else len(self.language_model.layers)
            batch_cache = [BatchKVCache([0] * len(per_request_caches)) for _ in range(num_layers)]

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
            # Flatten prompt to 1D list — input_ids is 2D [1, seq_len] from prepare_inputs
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
        Internal next() implementation.

        Returns:
            List of MLLMBatchResponse for this step
        """
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0

        # Only start a new batch when there is no active batch generating.
        # MLLM vision encoding produces per-request KV caches that cannot be
        # safely extended into an active batch's cache (shape mismatch in
        # attention layers). Instead, queued requests wait until the current
        # batch finishes, then all get processed together in one prefill.
        if num_active == 0:
            requests = self.unprocessed_requests[: self.completion_batch_size]

            if len(requests) == 0:
                self.active_batch = None
                return []

            new_batch = self._process_prompts(requests)
            self.unprocessed_requests = self.unprocessed_requests[len(requests) :]
            self.active_batch = new_batch
            prompt_processing = True

        # Generate next token for active batch
        batch = self.active_batch
        if batch is None:
            return []

        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

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
                        req.input_ids[0].tolist() if req.input_ids is not None and req.input_ids.ndim > 1
                        else req.input_ids.tolist() if req.input_ids is not None
                        else []
                    ),
                )
            )

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

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
