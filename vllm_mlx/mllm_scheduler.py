# SPDX-License-Identifier: Apache-2.0
"""
MLLM Scheduler for multimodal continuous batching.

This scheduler handles Multimodal Language Model requests with continuous
batching support, following the same architecture as the LLM scheduler.

Key features:
- Batch processing of multiple MLLM requests
- Vision embedding caching for repeated images
- Step-based generation loop (like LLM scheduler)
- Support for both streaming and non-streaming generation

Architecture:
1. Requests arrive via add_request() -> waiting queue
2. Scheduler moves requests from waiting to running (via MLLMBatchGenerator)
3. step() method generates one token for ALL running requests
4. Finished requests are removed and outputs returned
"""

import asyncio
import logging
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
from .multimodal_processor import MultimodalProcessor
from .request import RequestOutput, RequestStatus, SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class MLLMSchedulerConfig:
    """Configuration for MLLM scheduler."""

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
    """
    Scheduler for Vision Language Model requests with continuous batching.

    This scheduler manages the lifecycle of MLLM requests using the
    MLLMBatchGenerator for efficient batch processing:

    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via batch generator)
    3. step() generates one token for ALL running requests simultaneously
    4. Finished requests are removed and outputs returned

    Example:
        >>> scheduler = MLLMScheduler(model, processor, config)
        >>> # Add requests
        >>> request_id = scheduler.add_request(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> # Run generation loop
        >>> while scheduler.has_requests():
        ...     output = scheduler.step()
        ...     for req_output in output.outputs:
        ...         if req_output.finished:
        ...             print(f"Finished: {req_output.output_text}")

    For async usage with streaming:
        >>> await scheduler.start()
        >>> request_id = await scheduler.add_request_async(...)
        >>> async for output in scheduler.stream_outputs(request_id):
        ...     print(output.new_text, end="")
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: Optional[MLLMSchedulerConfig] = None,
    ):
        """
        Initialize MLLM scheduler.

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

        # Multimodal processor for input preparation
        self.mm_processor = MultimodalProcessor(
            model=model,
            processor=processor,
            config=self.model_config,
        )

        # Token-level Prefix caching for the language model
        self.paged_cache_manager = None
        self.block_aware_cache = None

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

        if self.config.enable_prefix_cache and self.config.use_paged_cache:
            try:
                from .paged_cache import PagedCacheManager
                from .prefix_cache import BlockAwarePrefixCache
                
                self.paged_cache_manager = PagedCacheManager(
                    block_size=self.config.paged_cache_block_size,
                    max_blocks=self.config.max_cache_blocks,
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

    def _get_detokenizer(self, request_id: str, tokenizer: Any) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
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
        or pure Mamba/SSM models. These need paged cache (memory-aware cache can't
        truncate cumulative SSM states).
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

        # Clear paged cache when BatchGenerator changes — cache objects are
        # tied to their generator instance
        if self.batch_generator is not None and self.block_aware_cache is not None:
            logger.debug(
                "Clearing paged cache: MLLM BatchGenerator being recreated"
            )
            self.block_aware_cache.clear()

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)

        self.batch_generator = MLLMBatchGenerator(
            model=self.model,
            processor=self.processor,
            mm_processor=self.mm_processor,
            max_tokens=self.config.default_max_tokens,
            stop_tokens=self.stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            paged_cache_manager=self.paged_cache_manager,
            block_aware_cache=self.block_aware_cache,
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

        # Signal output queue
        if request_id in self.output_queues:
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

        # Merge per-request stop_token_ids into batch generator stop tokens
        if batch_requests and self.batch_generator is not None:
            for request in scheduled:
                if request.sampling_params.stop_token_ids:
                    self.batch_generator.stop_tokens.update(
                        request.sampling_params.stop_token_ids
                    )
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
        """Clean up finished requests."""
        for request_id in finished_ids:
            if self.block_aware_cache is not None:
                request = self.running.get(request_id)
                if request is not None and getattr(request, "_extracted_cache", None) is not None:
                    try:
                        token_list = getattr(request, "_extracted_tokens", [])
                        if token_list:
                            cache_blocks = request._extracted_cache()
                            if cache_blocks is None:
                                logger.debug(f"No cache blocks to store for {request_id}")
                            else:
                                prompt_len = len(token_list)
                                # For hybrid models (MambaCache + KVCache), the standard
                                # truncation fails because MambaCache is cumulative.
                                # Use MLLM-specific truncation that skips Mamba layers
                                # and only truncates KVCache layers.
                                cache_blocks = self._truncate_hybrid_cache(
                                    cache_blocks, prompt_len
                                )
                                if cache_blocks is None:
                                    logger.debug(
                                        f"Cache truncation failed for {request_id}"
                                    )
                                else:
                                    truncated_tokens = token_list[:prompt_len - 1] if prompt_len > 1 else token_list
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

            # Track as finished
            self.finished_req_ids.add(request_id)

    def step(self) -> MLLMSchedulerOutput:
        """
        Execute one scheduling step (no queue pushing — see _step_and_dispatch).

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via MLLMBatchGenerator
        3. Processes outputs and handles finished requests

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

        return output

    def _dispatch_outputs(self, step_output: "MLLMSchedulerOutput") -> None:
        """Push step outputs to async queues. Must be called on the event loop thread."""
        if step_output.outputs:
            for req_output in step_output.outputs:
                queue = self.output_queues.get(req_output.request_id)
                if queue is not None:
                    try:
                        queue.put_nowait(req_output)
                        if req_output.finished:
                            queue.put_nowait(None)  # Signal end
                    except asyncio.QueueFull:
                        pass

    def _fail_all_requests(self, error_msg: str) -> None:
        """Fail all waiting and running requests with an error so callers don't hang."""
        with self._queue_lock:
            failed_ids = set()
            # Fail running requests
            for req_id, req in list(self.running.items()):
                req.status = RequestStatus.FINISHED_STOPPED
                queue = self.output_queues.get(req_id)
                if queue is not None:
                    try:
                        queue.put_nowait(RequestOutput(
                            request_id=req_id,
                            output_text=f"[Error: {error_msg}]",
                            finished=True,
                            finish_reason="error",
                        ))
                        queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
                failed_ids.add(req_id)
            # Fail waiting requests
            for req in list(self.waiting):
                req_id = req.request_id
                req.status = RequestStatus.FINISHED_STOPPED
                queue = self.output_queues.get(req_id)
                if queue is not None:
                    try:
                        queue.put_nowait(RequestOutput(
                            request_id=req_id,
                            output_text=f"[Error: {error_msg}]",
                            finished=True,
                            finish_reason="error",
                        ))
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
        """Get scheduler statistics."""
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
                # Add vision embedding cache stats from batch generator
                stats["vision_embedding_cache"] = (
                    self.batch_generator.get_vision_cache_stats()
                )

            if self.batch_generator:
                stats["vision_cache"] = self.batch_generator.get_vision_cache_stats()

            return stats

    def reset(self) -> None:
        """Reset the scheduler state."""
        with self._queue_lock:
            # Abort all requests
            for request_id in list(self.requests.keys()):
                self.abort_request(request_id)
    
            self.waiting.clear()
            self.running.clear()
            self.requests.clear()
            self.finished_req_ids.clear()
            self.request_id_to_uid.clear()
            self.uid_to_request_id.clear()

        if self.batch_generator is not None:
            self.batch_generator.close()
            self.batch_generator = None

        # Vision cache is inside batch_generator, already cleaned up above
