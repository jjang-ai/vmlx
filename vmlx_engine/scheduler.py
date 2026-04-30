# SPDX-License-Identifier: Apache-2.0
# Base architecture from waybarrios/vllm-mlx. MLA cache guards, gen_prompt_len
# prefix cache fix, hybrid SSM handling, and MoE CacheList support added by
# Jinho Jang (eric@jangq.ai) for vMLX (github.com/jjang-ai/vmlx).
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
import random
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from mlx_lm.generate import BatchGenerator, generation_stream
from mlx_lm.sample_utils import make_sampler
from .block_disk_store import BlockDiskStore
from .disk_cache import DiskCacheManager
from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from .paged_cache import PagedCacheManager
from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
from .prompt_lookup import NgramIndex, find_draft_tokens, pld_stats
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .mllm_batch_generator import HybridSSMStateCache, _fix_hybrid_cache
from .state_machine import SequenceStateMachine, make_state_machine
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


def _rebuild_meta_state_after_truncation(
    cls_name: str,
    orig_meta: tuple,
    safe_len: int,
) -> Optional[tuple]:
    """Rebuild a cache layer's meta_state after slicing its KV tensors to
    ``safe_len`` tokens. Returns ``None`` to signal "cannot safely truncate —
    skip this store" (used for RotatingKVCache when the circular buffer has
    already wrapped).

    Why this exists: different mlx-lm cache classes pack different fields
    into ``meta_state``, and blindly overwriting slot 0 with the new length
    silently corrupted RotatingKVCache's ``keep`` field, producing word-loop
    generations after the first cache hit on Gemma 4 (25 sliding + 5 full
    attention layers).

    meta_state layouts (from mlx_lm/models/cache.py):
      - ``KVCache``          → ``(offset,)``
      - ``QuantizedKVCache`` → ``(offset, group_size, bits)``
      - ``RotatingKVCache``  → ``(keep, max_size, offset, _idx)``
    """
    if "Rotating" in cls_name:
        if not orig_meta or len(orig_meta) < 4:
            return (str(0), str(safe_len), str(safe_len), str(safe_len))
        try:
            keep = int(orig_meta[0])
            max_size = int(orig_meta[1])
            offset = int(orig_meta[2])
        except (ValueError, TypeError):
            return (str(0), str(safe_len), str(safe_len), str(safe_len))
        if offset > max_size:
            # Circular buffer has wrapped — slot order != token order, so
            # a head-aligned slice is meaningless. Refuse to store.
            return None
        return (
            str(keep),
            str(max_size),
            str(safe_len),
            str(safe_len),
        )
    # KVCache / QuantizedKVCache: slot 0 IS the offset. Preserve tail
    # (group_size, bits, …) unchanged.
    if orig_meta:
        return (str(safe_len),) + tuple(orig_meta[1:])
    return (str(safe_len),)


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
    # Optional global byte budget for the prefix cache. None = unlimited (eviction
    # by entry count only). When set, eviction also fires when total cached bytes
    # exceed this. Mirrors mlx-lm 0.31.2 LRUPromptCache(--prompt-cache-bytes).
    prefix_cache_max_bytes: Optional[int] = None
    # Default cache_type for entries stored at request completion. Segment
    # boundaries (Agent 2) override this with "system" / "user" as appropriate.
    prefix_cache_default_type: str = "assistant"

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
    disk_cache_dir: Optional[str] = (
        None  # None = ~/.cache/vmlx-engine/prompt-cache/<model_hash>
    )
    disk_cache_max_gb: float = 10.0  # 0 = unlimited
    model_path: Optional[str] = None  # Used to scope disk cache per model

    # Loader fingerprint inputs (F6 + A4 Concern #1). Mixed into the trie
    # cache key so two sessions on the same model with divergent loader
    # configs (smelt %, JANG quant bits) never share K/V entries — divergent
    # tensors otherwise produce silent corruption on cross-session fetch.
    smelt_enabled: bool = False
    smelt_pct: Optional[float] = None  # Smelt expert percentage when enabled

    # Block-level disk cache (L2 for paged cache blocks)
    enable_block_disk_cache: bool = False
    block_disk_cache_dir: Optional[str] = (
        None  # None = ~/.cache/vmlx-engine/block-cache/<model_hash>
    )
    block_disk_cache_max_gb: float = 10.0  # 0 = unlimited

    # Prompt Lookup Decoding (PLD) speculative acceleration
    pld_enabled: bool = (
        False  # Enable PLD (opt-in; best for long structured/repetitive output)
    )
    pld_summary_interval: int = (
        487  # Log effectiveness summary every N spec-decode tokens
    )

    # SequenceStateMachine for token-level reasoning/stop detection (Phase 3c).
    # When True (default) the per-token loop builds a state machine from the
    # active reasoning parser's tag tokens and uses O(1) per-token state
    # transitions instead of O(L) substring scans. Falls back to substring
    # scan automatically when no parser is registered or the parser provides
    # no tag tokens. Set False to force the legacy substring path for
    # debugging or rollback. See `vmlx_engine/state_machine.py` and
    # `agentprogress/2/decisions.md` D-A2-005.
    use_state_machine_stops: bool = True

    # SSM companion cache size for hybrid (Mamba/GatedDelta) models. Mirrors
    # `MLLMSchedulerConfig.ssm_state_cache_size`. Default 50 matches the prior
    # hardcoded value. A3→A2-002 audit fix (2026-04-08): LLM scheduler was
    # ignoring this entirely.
    ssm_state_cache_size: int = 50

    # Dedicated single-worker ThreadPoolExecutor that loaded the model and
    # must run every step()/BatchGenerator call. MLX streams are
    # thread-local — if step() runs on a different thread than load, JANGTQ
    # Metal kernels fail with `RuntimeError: There is no Stream(gpu, N) in
    # current thread`. The MLLM scheduler has had this since 2026-04-25
    # (mlxstudio JANGTQ-VL thread fix); the LLM path was missing it,
    # causing uvicorn workers to crash on every JANGTQ chat request.
    # `BatchedEngine._start_llm` now constructs a `llm-worker` executor,
    # loads on it, then forwards it here so `EngineCore._engine_loop`
    # dispatches `scheduler.step()` to the same worker thread.
    step_executor: Any = None


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
        self._pending_aborts: Set[str] = set()  # Deferred aborts (processed in step())

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

        # TTFT EWMA tracking (alpha = 0.1 gives ~10-sample effective window)
        self._ewma_ttft: float = 0.0
        self._ttft_alpha: float = 0.1

        # Track if model uses mixed cache types (KVCache + MambaCache)
        self._is_hybrid = self._is_hybrid_model(model)
        self._tq_active = getattr(model, "make_cache", None) and getattr(
            model.make_cache, "__name__", ""
        ) in ("_tq_make_cache", "_turboquant_make_cache")

        # Mixed-attention models (Gemma 4 = sliding + full) currently drift
        # under the prefix-cache reconstruct path and produce word loops on
        # the 3rd multi-turn request. Transparently bypass the prefix cache
        # on every request for these models until the root cause is found.
        # Cache_salt / skip_prefix_cache remain the per-request escape hatch
        # for every other class of benchmark.
        self._force_bypass_prefix_cache = False
        try:
            self._force_bypass_prefix_cache = self._model_has_mixed_attention(model)
            if self._force_bypass_prefix_cache:
                logger.warning(
                    "LLM mixed-attention model detected (e.g. Gemma 4 sliding+full). "
                    "Prefix cache is auto-bypassed on every request — multi-turn "
                    "reconstructed KV state causes generation loops at the current "
                    "rep_penalty floor."
                )
        except Exception as e:
            logger.debug(f"Mixed-attention detection failed: {e}")

        # Per-model SequenceStateMachine for token-level reasoning/stop detection.
        # Lazy-built on first request because we need the reasoning parser instance
        # which lives in server.py as a module-level singleton (would be a circular
        # import if eager). When None, the per-token loop falls back to the legacy
        # substring `<think>` scan in `_should_skip_string_stop_for_reasoning`.
        # See `agentprogress/2/decisions.md` D-A2-005 / D-A2-006 for the design.
        self._reasoning_sm: Optional[SequenceStateMachine] = None
        self._reasoning_sm_resolved: bool = False
        # Rollback flag — if Phase 3c integration regresses, set this to False
        # to fall back to the legacy substring path without rebuilding.
        self._use_sm_stops: bool = getattr(self.config, "use_state_machine_stops", True)

        # Pre-compute hybrid cache layout for SSM companion.
        # _hybrid_kv_positions: layer indices that are KVCache (attention).
        # _hybrid_num_layers: total layer count in model cache.
        self._hybrid_kv_positions: Optional[List[int]] = None
        self._hybrid_num_layers: Optional[int] = None
        self._ssm_state_cache: Optional[HybridSSMStateCache] = None
        if self._is_hybrid and hasattr(model, "make_cache"):
            try:
                from mlx_lm.models.cache import KVCache as _KVC

                _template = model.make_cache()
                self._hybrid_num_layers = len(_template)
                self._hybrid_kv_positions = [
                    i
                    for i, t in enumerate(_template)
                    if type(t).__name__
                    in (
                        "KVCache",
                        "RotatingKVCache",
                        "QuantizedKVCache",
                        "TurboQuantKVCache",
                    )
                ]
                # Honour SchedulerConfig.ssm_state_cache_size when set; default 50
                # mirrors the prior hardcoded value. Fixes A3→A2-002 (config drift):
                # MLLM scheduler honoured `MLLMSchedulerConfig.ssm_state_cache_size`
                # but the LLM scheduler hardcoded 50, ignoring user config.
                _ssm_cache_size = getattr(self.config, "ssm_state_cache_size", 50) or 50
                self._ssm_state_cache = HybridSSMStateCache(max_entries=_ssm_cache_size)
                logger.info(
                    f"Hybrid SSM cache: {len(self._hybrid_kv_positions)}/"
                    f"{self._hybrid_num_layers} KV layers, "
                    f"SSM companion enabled"
                )
            except Exception as _e:
                logger.warning(f"Failed to init hybrid SSM cache layout: {_e}")

        # Prompt lookup decoding — measurement state (Phase 1)
        # Maps request_id -> (draft_tokens, expected_start_output_idx, hit_count)
        self._pld_pending: Dict[str, Tuple[List[int], int, int]] = {}
        # Per-request n-gram hash index for O(1) draft lookup
        self._pld_ngram_indices: Dict[str, NgramIndex] = {}

        # Prompt lookup decoding — Phase 2/3 (actual batched verification)
        # Phase 2 (temp≈0): greedy acceptance, argmax bonus.
        # Phase 3 (temp>0): probabilistic acceptance, sampled correction/bonus.
        self._pld_spec_enabled: bool = self.config.pld_enabled
        self._pld_spec_max_temp: float = float(os.getenv("VMLX_PLD_MAX_TEMP", "1.0"))
        # Adaptive K: hybrid SSM/attention models process verify tokens
        # sequentially in SSM layers.  K=2 balances verify cost (1.75x)
        # against per-cycle overhead (remove/insert ≈ 15-30ms).  K=1 has
        # lower verify cost (1.0x) but pays the same fixed overhead per
        # cycle with fewer tokens to amortize it.  K=2 wins when the
        # fixed overhead exceeds ~15ms, which remove/insert clearly does.
        self._pld_num_drafts: int = 2 if self._is_hybrid else 5
        self._pld_spec_attempts: int = 0
        self._pld_spec_accepted: int = 0  # total accepted draft tokens
        self._pld_spec_wasted: int = 0  # total rejected draft tokens
        # Per-window counters for periodic summary (reset after each log)
        self._pld_win_attempts: int = 0
        self._pld_win_accepted: int = 0
        self._pld_win_full: int = 0  # rounds where all K drafts accepted
        self._pld_win_zero: int = 0  # rounds where 0 drafts accepted
        self._pld_win_tokens: int = 0  # tokens emitted while PLD active
        self._pld_win_d0_skip: int = 0  # d0 pre-check skips (wasted cycles avoided)
        # Auto-tune: TCP slow-start inspired wall-clock throughput control.
        # Window starts at 10 tokens, doubles each positive window (exponential
        # growth), caps at _pld_summary_interval.  On congestion (PLD hurting),
        # disables and resets window to 10.  Probes after 5× interval tokens.
        self._pld_auto_enabled: bool = True
        self._pld_at_window: int = 1  # current auto-tune window (TCP cwnd)
        self._pld_at_probe_tokens: int = 0  # tokens counted while disabled
        self._pld_win_cycle_wall_s: float = 0.0
        self._pld_win_step_wall_s: float = 0.0
        self._pld_win_total_tokens: int = 0
        self._pld_summary_interval: int = self.config.pld_summary_interval
        if self._pld_spec_enabled:
            logger.info(
                "[PLD] enabled — K=%d (%s model), d0 pre-check active, "
                "auto-tune on (slow-start window=1→%d)",
                self._pld_num_drafts,
                "hybrid" if self._is_hybrid else "pure-attention",
                self._pld_summary_interval,
            )
        self._pld_summary_next: int = 1  # first window is 1 token (slow start)

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
        # MLA models (DeepSeek V3, Mistral 4) store compressed KV latents — quantizing
        # these destroys quality. Auto-disable for MLA, same as MLLM scheduler.
        # Original MLA cache integration by Jinho Jang (eric@jangq.ai) — vMLX/mlxstudio.
        _is_mla = False
        try:
            _model_args = getattr(self.model, "args", None)
            if _model_args and getattr(_model_args, "kv_lora_rank", 0) > 0:
                _is_mla = True
            elif _model_args and getattr(_model_args, "model_type", "") == "mistral4":
                _is_mla = True
            # DeepSeek V4-Flash / V4-Pro: MLA with head_dim=512 (single latent
            # KV head, broadcast to all 64 q heads). Stored KV is already a
            # compressed latent — quantizing again destroys decode quality
            # and doesn't save much (1 KV head × head_dim=512). Force-off
            # same as DeepSeek V3 / Mistral 4.
            elif _model_args and getattr(_model_args, "model_type", "") == "deepseek_v4":
                _is_mla = True
        except Exception:
            pass
        # User opt-in override for the MLA auto-disable. DSV4 / DeepSeek V3 /
        # Mistral 4 stash compressed latents in KV — quantizing them again
        # is double-lossy and harms quality. We force-off by default. But
        # for long-context decode where users care about RAM more than
        # marginal quality loss (e.g. "DSV4 with 128k context, accept some
        # drift to fit on 128GB"), expose `VMLX_ALLOW_MLA_KV_QUANT=1` so the
        # user can opt into TurboQuant KV-quant on MLA at their own risk.
        _allow_mla_kvq = os.environ.get(
            "VMLX_ALLOW_MLA_KV_QUANT"
        ) in ("1", "true", "True", "yes", "on")
        if self.config.kv_cache_quantization != "none" and _is_mla and not _allow_mla_kvq:
            logger.info(
                f"MLA model detected (kv_lora_rank > 0) — disabling KV cache quantization "
                f"(was: {self.config.kv_cache_quantization}). MLA stores compressed latents "
                f"that should not be further quantized. Set VMLX_ALLOW_MLA_KV_QUANT=1 to "
                f"override if you accept the quality risk."
            )
            self.config.kv_cache_quantization = "none"
        elif self.config.kv_cache_quantization != "none" and _is_mla and _allow_mla_kvq:
            logger.warning(
                f"MLA model + KV cache quantization='{self.config.kv_cache_quantization}' "
                f"requested via VMLX_ALLOW_MLA_KV_QUANT=1 — running double-lossy KV quant "
                f"on compressed latents. Expect some output drift; turn off if quality matters."
            )
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
                            ".cache",
                            "vmlx-engine",
                            "block-cache",
                            model_hash,
                        )
                    elif cache_dir is None:
                        logger.warning(
                            "Block disk cache: model_path not set, using shared 'default' dir. "
                            "Different models will share cache — this may cause issues."
                        )
                        cache_dir = os.path.join(
                            os.path.expanduser("~"),
                            ".cache",
                            "vmlx-engine",
                            "block-cache",
                            "default",
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
                    model_path=self.config.model_path,
                    smelt_enabled=self.config.smelt_enabled,
                    smelt_pct=self.config.smelt_pct,
                    tq_enabled=self._tq_active,
                    kv_quant_bits=self._kv_cache_bits,
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
                    model_path=self.config.model_path,
                )
                logger.info(
                    f"Memory-aware cache enabled: "
                    f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB"
                )
            else:
                # Use legacy entry-count based prefix cache (now with optional
                # global byte budget + cache-type LRU priority — system entries
                # are pinned and evicted last so shared system prompts persist
                # across users/sessions).
                self.prefix_cache = PrefixCacheManager(
                    model=model,
                    max_entries=self.config.prefix_cache_size,
                    max_bytes=self.config.prefix_cache_max_bytes,
                    model_path=self.config.model_path,
                    smelt_enabled=self.config.smelt_enabled,
                    smelt_pct=self.config.smelt_pct,
                    tq_enabled=self._tq_active,
                    kv_quant_bits=self._kv_cache_bits,
                )
                _bytes_msg = (
                    f", max_bytes={self.config.prefix_cache_max_bytes}"
                    if self.config.prefix_cache_max_bytes is not None
                    else ""
                )
                logger.info(
                    f"Prefix cache enabled with max_entries={self.config.prefix_cache_size}"
                    f"{_bytes_msg}, type-priority=(assistant→user→system)"
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
            # Scope disk cache per model, quantization, AND layer count to prevent
            # stale cross-config hits. Without this, restarting with a different model
            # at the same path could load tensors with wrong layer count or head dims.
            if self.config.model_path:
                quant_tag = self.config.kv_cache_quantization or "none"
                # Include layer count to invalidate on architecture change
                n_layers = 0
                for _attr in ("args", "config"):
                    _cfg = getattr(self.model, _attr, None)
                    if _cfg:
                        n_layers = getattr(_cfg, "num_hidden_layers", 0)
                        if n_layers:
                            break
                scope_key = (
                    f"{self.config.model_path}:quant={quant_tag}:layers={n_layers}"
                )
                model_hash = hashlib.sha256(scope_key.encode()).hexdigest()[:12]
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
    def _model_has_mixed_attention(model: Any) -> bool:
        """Return True for models that interleave sliding-window and full
        attention layers (Gemma 4 pattern). Detection is conservative: only
        returns True when the config clearly lists at least two distinct
        attention modes including at least one sliding variant.
        """
        candidates = []
        for attr in ('args', 'config'):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                candidates.append(cfg)
                tc = getattr(cfg, 'text_config', None)
                if tc is not None:
                    candidates.append(tc)
        for cfg in candidates:
            layer_types = getattr(cfg, 'layer_types', None)
            if layer_types and isinstance(layer_types, (list, tuple)):
                kinds = set(layer_types)
                if len(kinds) >= 2 and any('sliding' in str(k) for k in kinds):
                    return True
        return False

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
            kv_types = {
                t for t in cache_types if t == "KVCache" or t.endswith("KVCache")
            }
            if cache_types and cache_types == kv_types:
                return False
            # DeepSeek V4 `DeepseekV4Cache` wraps a RotatingKVCache internally
            # and exposes the standard KV surface (update_and_fetch / make_mask
            # / offset). The compressor + indexer state buffers it carries are
            # cumulative-but-recomputable (unlike true SSM state) — treat as
            # plain KV so the paged / memory-aware / disk L2 cache paths work
            # identically to DeepSeek V3.
            non_kv = cache_types - kv_types
            non_kv.discard("CacheList")
            non_kv.discard("DeepseekV4Cache")
            return bool(non_kv)
        except Exception as e:
            logger.warning(f"make_cache() failed during hybrid detection: {e}")
            return False

    def _detect_head_dim(self) -> Optional[int]:
        """Detect the model's KV head dimension from model config."""
        try:
            # Inspect model config for head_dim
            if hasattr(self.model, "args"):
                args = self.model.args
                if hasattr(args, "head_dim") and args.head_dim:
                    return args.head_dim
                if hasattr(args, "hidden_size") and hasattr(
                    args, "num_attention_heads"
                ):
                    return args.hidden_size // args.num_attention_heads
            # Try model.config
            if hasattr(self.model, "config"):
                config = self.model.config
                if hasattr(config, "head_dim") and config.head_dim:
                    return config.head_dim
                if hasattr(config, "hidden_size") and hasattr(
                    config, "num_attention_heads"
                ):
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
        if hasattr(self, "_n_kv_heads_cached"):
            return self._n_kv_heads_cached
        n_kv = 0
        try:
            # Build candidate list: model + VLM wrapper inner models
            candidates = [self.model]
            lm = getattr(self.model, "language_model", None)
            if lm is not None:
                candidates.append(lm)
            mm = getattr(self.model, "model", None)
            if mm is not None and mm is not self.model:
                candidates.append(mm)
            # TWO-PASS: MLA detection first (kv_lora_rank → H=1),
            # then num_key_value_heads. VLM wrappers may have
            # num_key_value_heads=32 on args but kv_lora_rank only
            # on nested text_config — single-pass breaks too early.
            for model_obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    kv_lora_rank = getattr(cfg, "kv_lora_rank", 0)
                    if not kv_lora_rank:
                        tc = getattr(cfg, "text_config", None)
                        if tc is not None:
                            kv_lora_rank = getattr(tc, "kv_lora_rank", 0)
                    if kv_lora_rank and kv_lora_rank > 0:
                        n_kv = 1
                        break
                if n_kv:
                    break
            if not n_kv:
                for model_obj in candidates:
                    for attr in ("args", "config", "text_config"):
                        cfg = getattr(model_obj, attr, None)
                        if cfg is None:
                            continue
                        n_kv = getattr(cfg, "num_key_value_heads", 0) or getattr(
                            cfg, "num_kv_heads", 0
                        )
                        if n_kv:
                            break
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
        if not hasattr(QuantizedKVCache, "_size_patched"):
            needs_patch = True
            try:
                test_qkv = QuantizedKVCache(group_size=64, bits=bits)
                test_qkv.offset = 42
                # size() is a method on _BaseCache, so call with parens
                if callable(getattr(test_qkv, "size", None)) and test_qkv.size() == 42:
                    needs_patch = False
                    logger.info(
                        "QuantizedKVCache.size() already returns offset — upstream fix detected"
                    )
            except Exception:
                pass

            if needs_patch:

                def _qkv_size(self):
                    return getattr(self, "offset", 0)

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
            logger.info(
                f"KV cache quantization validated: head_dim={head_dim}, group_size={group_size}"
            )

        # Run a quantize/dequantize round-trip test with realistic tensor shapes.
        try:
            test_dim = head_dim or 128
            test_shape = (1, 4, 8, test_dim)  # (batch, heads, seq, head_dim)
            test_tensor = mx.random.normal(test_shape)
            quantized = mx.quantize(test_tensor, group_size=group_size, bits=bits)
            dequantized = mx.dequantize(
                quantized[0],
                quantized[1],
                quantized[2],
                group_size=group_size,
                bits=bits,
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
        if hasattr(self.config, "kv_cache_group_size"):
            self.config.kv_cache_group_size = group_size

    def _quantize_cache_for_storage(self, cache: List[Any]) -> List[Any]:
        """
        Convert KVCache layers to QuantizedKVCache for prefix cache storage.

        Quantizes keys/values using mx.quantize() to reduce memory by 2-4x.
        Preserves non-KVCache layers (MambaCache, RotatingKVCache, etc.).
        Recurses into CacheList sub-caches for MoE models.
        Falls back to unquantized storage on any error.
        """
        if not getattr(self, "_kv_cache_bits", 0):
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
                            qkv.keys = tuple(
                                mx.quantize(sc.keys, group_size=group_size, bits=bits)
                            )
                            qkv.values = tuple(
                                mx.quantize(sc.values, group_size=group_size, bits=bits)
                            )
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
                    qkv.keys = tuple(
                        mx.quantize(layer_cache.keys, group_size=group_size, bits=bits)
                    )
                    qkv.values = tuple(
                        mx.quantize(
                            layer_cache.values, group_size=group_size, bits=bits
                        )
                    )
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
                                    sc.keys[0],
                                    sc.keys[1],
                                    sc.keys[2],
                                    sc.group_size,
                                    sc.bits,
                                )
                                kv.values = mx.dequantize(
                                    sc.values[0],
                                    sc.values[1],
                                    sc.values[2],
                                    sc.group_size,
                                    sc.bits,
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
                            layer_cache.keys[0],
                            layer_cache.keys[1],
                            layer_cache.keys[2],
                            layer_cache.group_size,
                            layer_cache.bits,
                        )
                        kv.values = mx.dequantize(
                            layer_cache.values[0],
                            layer_cache.values[1],
                            layer_cache.values[2],
                            layer_cache.group_size,
                            layer_cache.bits,
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
        """Get stop token IDs from tokenizer or processor.

        Also checks the model config registry for additional eos_tokens
        (e.g., Gemma 4 uses <turn|> as end-of-turn alongside <eos>).
        """
        stop_tokens = set()
        # Check both the processor/tokenizer and the actual tokenizer
        tok_for_encode = None
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            if tok_for_encode is None:
                tok_for_encode = tok
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

        # Add extra eos_tokens from model config registry (e.g., <turn|> for Gemma 4)
        if tok_for_encode is not None:
            try:
                from .model_config_registry import get_model_config_registry

                registry = get_model_config_registry()
                # Try to find model name from tokenizer's name_or_path
                model_name = getattr(tok_for_encode, "name_or_path", None)
                if model_name:
                    model_config = registry.lookup(model_name)
                    if model_config.eos_tokens and len(model_config.eos_tokens) > 1:
                        for eos_str in model_config.eos_tokens[1:]:
                            try:
                                ids = tok_for_encode.encode(
                                    eos_str, add_special_tokens=False
                                )
                                if len(ids) == 1:
                                    stop_tokens.add(ids[0])
                                    logger.debug(
                                        f"Added extra stop token: {eos_str!r} → {ids[0]}"
                                    )
                            except Exception:
                                pass
            except Exception:
                pass

        return stop_tokens

    def _resolve_reasoning_state_machine(self) -> Optional[SequenceStateMachine]:
        """Lazily build a `SequenceStateMachine` for the current model.

        Pulls the active reasoning parser from `server._reasoning_parser`
        (lazy import to avoid the scheduler→server circular dependency) and
        asks it for `reasoning_tag_token_seqs(tokenizer)`. If the parser
        provides start/end token sequences, builds a state machine with
        a `normal → reasoning → normal` transition table; otherwise returns
        None and the per-token loop falls back to the legacy substring path.

        Result is cached on `self._reasoning_sm` after the first call —
        the resolved state is `_reasoning_sm_resolved=True` even when the
        result is None, so we don't retry the resolution every token.
        """
        if self._reasoning_sm_resolved:
            return self._reasoning_sm
        self._reasoning_sm_resolved = True
        if not self._use_sm_stops:
            return None
        try:
            from . import server as _server  # lazy to dodge circular import

            parser = getattr(_server, "_reasoning_parser", None)
            if parser is None:
                return None
            tags = parser.reasoning_tag_token_seqs(
                self._actual_tokenizer or self.tokenizer
            )
            if not (tags.get("start") or tags.get("end")):
                return None
            stop_token_seqs = (
                [[t] for t in self.stop_tokens] if self.stop_tokens else []
            )
            self._reasoning_sm = make_state_machine(
                model_key=getattr(self.model, "__class__", type(self.model)).__name__,
                reasoning_parser_id=type(parser).__name__,
                reasoning_start_tokens=tags.get("start") or (),
                reasoning_end_tokens=tags.get("end") or (),
                stop_token_sequences=stop_token_seqs,
            )
            return self._reasoning_sm
        except Exception as e:
            logger.debug(
                f"_resolve_reasoning_state_machine: fell back to legacy path ({e!r})"
            )
            return None

    def _advance_request_state_machine(
        self, request: Request, tokens: List[int]
    ) -> None:
        """Advance the per-request state machine across the given tokens.

        No-op when the resolver returned None (no reasoning parser configured)
        or when `_use_sm_stops` is disabled. Lazy-creates `request._sm_state`
        on first call. Called from the per-token loop on every emitted token
        (regular sampled token + any speculative tokens).
        """
        sm = self._resolve_reasoning_state_machine()
        if sm is None:
            return
        state = getattr(request, "_sm_state", None)
        if state is None:
            state = sm.make_state()
            # Phase 4 prep (Agent 1 directive 2026-04-08): on first init,
            # advance the state machine across the cached prefix tokens
            # returned by `PrefixCacheManager.fetch_cache`. The cached tokens
            # are known-clean by the trie's contract, so we use the
            # `advance_from` skip-walk (single O(L) pass, no halt detection)
            # rather than calling `match` per token. This puts the matcher
            # in the correct state (normal/reasoning) before the first
            # newly-emitted token without re-scanning the prefix on every
            # subsequent token.
            #
            # AUDIT FIX 2026-04-08 (Agent 2 self-finding): originally read
            # `request._cached_prefix_len` which is NEVER WRITTEN anywhere
            # in the codebase — it was a stub field added in Phase 3d in
            # anticipation of Agent 1 wiring it from `PrefixCacheManager.
            # fetch_cache` which never happened. The `advance_from` skip
            # was therefore dead code (always reading 0). The canonical
            # field for "tokens recovered from prefix cache" is
            # `request.cached_tokens`, set in 6+ places throughout
            # `scheduler.add_request` for both paged and legacy paths.
            cached_prefix_len = (
                getattr(request, "cached_tokens", 0)
                or getattr(request, "_cached_prefix_len", 0)
                or 0
            )
            if cached_prefix_len > 0:
                cached_tokens_slice = (request.prompt_token_ids or [])[
                    :cached_prefix_len
                ]
                if cached_tokens_slice:
                    try:
                        state = sm.advance_from(state, cached_tokens_slice)
                    except Exception as e:
                        logger.debug(
                            f"_advance_request_state_machine: advance_from failed "
                            f"on cached prefix ({e!r}) — falling back to fresh state"
                        )
                        state = sm.make_state()
        for tok in tokens:
            state, _seq, current = SequenceStateMachine.match(state, tok)
            if current is None:
                # State machine signalled halt — record but don't terminate
                # here; the per-token loop's existing finish_reason flow owns
                # termination so we don't double-fire.
                break
        request._sm_state = state

    def _store_cache_with_segments(
        self,
        request: Request,
        prompt_tokens: List[int],
        prompt_cache: List[Any],
    ) -> None:
        """Store completed prompt cache, segmented by chat-role boundaries.

        Phase 3d (Agent 2): When the request carries `_segment_boundaries`
        (populated by an API gateway during chat-template rendering), iterate
        over each boundary and call `prefix_cache.store_cache(prefix_tokens,
        prefix_cache, cache_type=role)`. This drives Agent 1's
        `PrefixCacheManager` cache_type-priority LRU so system prefixes are
        pinned and shared across users/sessions.

        When `_segment_boundaries` is None or empty (legacy callers), falls
        back to a single store with the default `cache_type="assistant"` —
        identical to the pre-Phase-3d behaviour.

        The cache trim per segment uses `_truncate_cache_to_prompt_length`
        when available so the per-segment cache reflects the per-segment
        prefix length, not the full prompt. If the truncation helper is
        unavailable on this scheduler instance, the segment falls back to
        storing the full prompt cache under the boundary's role (still
        useful for cache_type LRU priority even without per-segment trim).
        """
        # F1 backport: dispatch to whichever cache layer is active. The
        # production default is memory-aware; paged is the hybrid auto-switch
        # target; legacy entry-count is opt-in. All three now accept
        # cache_type for cross-session sharing breakthrough activation.
        active_layer = "none"
        if self.prefix_cache is not None:
            active_layer = "prefix"
        elif self.memory_aware_cache is not None:
            active_layer = "memory"
        elif self.block_aware_cache is not None:
            active_layer = "block"
        else:
            return

        def _do_store(tokens_seg: List[int], cache_seg: List[Any], role: str) -> None:
            """Single-layer store dispatcher honouring cache_type."""
            if active_layer == "prefix":
                self.prefix_cache.store_cache(tokens_seg, cache_seg, cache_type=role)
            elif active_layer == "memory":
                self.memory_aware_cache.store(tokens_seg, cache_seg, cache_type=role)
            elif active_layer == "block":
                # Block cache stores per-request; segment-prefix storage isn't
                # block-friendly, so we only tag the full-prompt store.
                # Segment prefixes still update LRU priority via the role tag.
                self.block_aware_cache.store_cache(
                    request.request_id,
                    list(tokens_seg),
                    self._extract_cache_states(cache_seg),
                    cache_type=role,
                )

        boundaries = getattr(request, "_segment_boundaries", None) or []
        if not boundaries:
            # Legacy single-store path — preserves pre-Phase-3d behaviour.
            try:
                _do_store(prompt_tokens, prompt_cache, "assistant")
            except Exception as e:
                logger.debug(f"_store_cache_with_segments legacy path: {e}")
            return

        # Iterate boundaries in increasing order. Each boundary stores a
        # PREFIX of length `idx` under its role. The full prompt is also
        # stored at the END under the final boundary's role (or under
        # "assistant" if the last boundary doesn't already cover all tokens).
        try:
            sorted_bounds = sorted(boundaries, key=lambda b: b[0])
            seen_full = False
            for idx, role in sorted_bounds:
                if idx <= 0 or idx > len(prompt_tokens):
                    continue
                prefix = prompt_tokens[:idx]
                # Trim the cache to the prefix length where possible.
                # Block cache can't trim individual layers — skip trimming and
                # rely on the role tag to drive priority eviction only.
                trimmed = prompt_cache
                if active_layer != "block":
                    trim_helper = getattr(
                        self, "_truncate_cache_to_prompt_length", None
                    )
                    if trim_helper is not None and idx < len(prompt_tokens):
                        try:
                            trimmed = trim_helper(prompt_cache, idx) or prompt_cache
                        except Exception:
                            trimmed = prompt_cache
                if active_layer == "block" and idx < len(prompt_tokens):
                    # For block cache, only the full-prompt store carries
                    # actual data — skip per-segment to avoid duplicate blocks.
                    continue
                _do_store(prefix, trimmed, role)
                if idx == len(prompt_tokens):
                    seen_full = True
            # Always make sure the full prompt is stored too — under the
            # default "assistant" type if no boundary covered the tail.
            if not seen_full:
                _do_store(prompt_tokens, prompt_cache, "assistant")
        except Exception as e:
            logger.debug(f"_store_cache_with_segments: {e}")
            # Fall back to legacy single-store on any segment-store error so
            # we never silently lose the cache entry.
            try:
                _do_store(prompt_tokens, prompt_cache, "assistant")
            except Exception:
                pass

    @staticmethod
    def _pick_cache_type_for_request(request: Request) -> str:
        """Choose the cache_type to tag a full-prompt store with, based on
        the request's segment boundaries (Phase 3d / F11). Picks the
        highest-priority role present (system > user > assistant). When no
        boundaries exist, returns the safe default "assistant".

        This lets the memory-aware and block-aware paths participate in F1
        cache_type LRU priority eviction without needing to invoke the full
        segments helper.
        """
        try:
            boundaries = getattr(request, "_segment_boundaries", None) or []
            roles = {
                role
                for _, role in boundaries
                if role in ("system", "user", "assistant")
            }
            for r in ("system", "user", "assistant"):
                if r in roles:
                    return r
        except Exception:
            pass
        return "assistant"

    def _is_request_in_reasoning(self, request: Request) -> bool:
        """Return True iff the per-request state machine is currently in
        reasoning state. Used to skip user-supplied string-stop matching
        inside `<think>` blocks (or whatever tag tokens the active parser
        defines). Falls back to substring scan via `request_text_in_think()`
        when the state machine is unavailable.
        """
        state = getattr(request, "_sm_state", None)
        if state is None:
            return False
        return SequenceStateMachine.current_state(state) == "reasoning"

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
        if (
            sampling_params.repetition_penalty
            and sampling_params.repetition_penalty != 1.0
        ):
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
            # If we have an existing generator with in-flight requests, we
            # can't swap it out mid-batch — the new request would run under
            # the old sampler. Previously this silently returned, which
            # meant the new request's repetition_penalty / temperature /
            # top_p was dropped and it inherited whatever the currently
            # running batch had configured. That produced extremely
            # confusing "2nd request ignores my repetition_penalty" behavior
            # and masked actual bugs during testing.
            #
            # Fix: still log the warning, but ALSO log the specific param
            # delta and explicitly note that the new request is going to
            # use the old params until the batch drains. The alternative
            # (forcing new requests to wait for a drain) trades correctness
            # for latency — out of scope for this fix.
            if self.batch_generator is not None and self.running:
                _old = self._current_sampler_params
                logger.warning(
                    "Sampling parameters changed with active requests. "
                    "New request will TEMPORARILY run with old params "
                    "(temp=%s top_p=%s min_p=%s top_k=%s rep_pen=%s) "
                    "until the current batch drains. New target was "
                    "(temp=%s top_p=%s min_p=%s top_k=%s rep_pen=%s).",
                    _old[0] if _old else None,
                    _old[1] if _old else None,
                    _old[2] if _old else None,
                    _old[3] if _old else None,
                    _old[4] if _old else None,
                    sampler_params[0],
                    sampler_params[1],
                    sampler_params[2],
                    sampler_params[3],
                    sampler_params[4],
                )
                return

            # Clear all prefix caches when BatchGenerator changes —
            # BatchKVCache objects and block tables are tied to their generator instance
            if self.batch_generator is not None:
                if self.block_aware_cache is not None:
                    logger.debug("Clearing paged cache: BatchGenerator being recreated")
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

        # KVCache / RotatingKVCache / QuantizedKVCache: check keys/values.
        # TurboQuantKVCache after compress() has keys=None, values=None
        # but is valid — compressed data is in _compressed_keys/_compressed_values
        # and decoded buffers in _joined_k/_joined_v.
        if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            if layer_cache.keys is None or layer_cache.values is None:
                # Check for TQ compressed state (keys cleared after compress)
                if (
                    getattr(layer_cache, "_compressed_keys", None) is not None
                    or getattr(layer_cache, "_joined_k", None) is not None
                ):
                    return True  # TQ layer — valid despite keys=None
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
            return all(Scheduler._validate_single_cache(c) for c in layer_cache.caches)

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

        # Metal safety: sync all pending ops before slicing post-generation arrays.
        # Lazy MLX slices can corrupt Metal command buffers when later evaluated.
        # Convert through numpy to fully decouple from Metal.
        try:
            import mlx.core as mx
            import numpy as np

            mx.synchronize()
        except ImportError:
            pass

        def _to_numpy(arr):
            """Convert an evaluated MLX array to numpy (safe CPU memcpy).

            bf16 goes through float32 (not float16): fp16 has only 5 exponent
            bits vs bf16's 8, so the downcast silently clips attention KV
            values and corrupts cached state enough to produce word loops
            on sensitive models (e.g. Gemma 4 JANG multi-turn rep_pen basin).
            """
            try:
                if arr.dtype == mx.bfloat16:
                    return np.array(arr.astype(mx.float32))
                return np.array(arr)
            except Exception:
                return arr

        def _from_numpy(arr, orig_dtype=None):
            """Convert numpy back to MLX, restoring original dtype if needed."""
            try:
                result = mx.array(arr)
                if orig_dtype is not None and orig_dtype == mx.bfloat16:
                    result = result.astype(mx.bfloat16)
                return result
            except Exception:
                return arr

        truncated = []
        for layer_cache in raw_cache:
            # Guard: skip dicts (extracted state dicts, not live cache objects).
            # dict.keys is a builtin_function_or_method that matches hasattr
            # but has no .ndim, causing "'builtin_function_or_method' object
            # has no attribute 'ndim'" crashes.
            if isinstance(layer_cache, dict):
                truncated.append(layer_cache)
                continue
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                # Positional cache: truncate to target length
                try:
                    k = layer_cache.keys
                    v = layer_cache.values
                    # Guard: k must be a tensor with .ndim (not a method or other object)
                    if not hasattr(k, "ndim"):
                        truncated.append(layer_cache)
                        continue

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
                            _from_numpy(_to_numpy(t)[..., :safe_target, :], t.dtype)
                            for t in k
                        )
                        new_cache.values = tuple(
                            _from_numpy(_to_numpy(t)[..., :safe_target, :], t.dtype)
                            for t in v
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

                                max_size = getattr(layer_cache, "max_size", target_len)
                                keep = getattr(layer_cache, "keep", 0)
                                offset = getattr(layer_cache, "offset", 0)
                                _idx = getattr(layer_cache, "_idx", 0)

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
                        # Convert PARENT arrays to numpy BEFORE slicing.
                        # Slicing in numpy avoids the Metal command buffer
                        # bug entirely — no lazy MLX ops, no GPU involvement.
                        if ndim == 4:
                            safe_target = min(target_len, k.shape[2])
                            np_k, np_v = _to_numpy(k), _to_numpy(v)
                            new_cache.keys = _from_numpy(
                                np_k[:, :, :safe_target, :], k.dtype
                            )
                            new_cache.values = _from_numpy(
                                np_v[:, :, :safe_target, :], v.dtype
                            )
                        elif ndim == 3:
                            safe_target = min(target_len, k.shape[1])
                            np_k, np_v = _to_numpy(k), _to_numpy(v)
                            new_cache.keys = _from_numpy(
                                np_k[:, :safe_target, :], k.dtype
                            )
                            new_cache.values = _from_numpy(
                                np_v[:, :safe_target, :], v.dtype
                            )
                        else:
                            return None
                        new_cache.offset = min(target_len, safe_target)
                        # Restore _idx for RotatingKVCache — use original _idx clamped to truncated length
                        if "Rotating" in cls_name and hasattr(new_cache, "_idx"):
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
                        if hasattr(sub_cache, "state") and hasattr(
                            sub_cache, "meta_state"
                        ):
                            sub_state = sub_cache.state
                            # Normalize GQA head inflation in sub-caches too
                            # (handles both plain tensors and quantized tuples)
                            if (
                                isinstance(sub_state, tuple)
                                and len(sub_state) == 2
                                and n_kv > 0
                            ):
                                sk, sv = sub_state
                                if (
                                    hasattr(sk, "shape")
                                    and len(sk.shape) == 4
                                    and sk.shape[1] > n_kv
                                ):
                                    sub_state = (sk[:, :n_kv, :, :], sv[:, :n_kv, :, :])
                                elif (
                                    isinstance(sk, (tuple, list))
                                    and len(sk) >= 1
                                    and hasattr(sk[0], "shape")
                                    and len(sk[0].shape) == 4
                                    and sk[0].shape[1] > n_kv
                                ):
                                    sub_state = (
                                        tuple(t[:, :n_kv, :, :] for t in sk),
                                        tuple(t[:, :n_kv, :, :] for t in sv),
                                    )
                            sub_states.append(
                                {
                                    "state": sub_state,
                                    "meta_state": sub_cache.meta_state,
                                    "class_name": type(sub_cache).__name__,
                                }
                            )
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
                        extracted.append(
                            {
                                "state": None,
                                "meta_state": None,
                                "class_name": cls_name,
                                "sub_caches": sub_states,
                            }
                        )
                    else:
                        failed += 1
                # MambaCache/ArraysCache: cumulative state (SSM layers in hybrid models).
                # Cannot be sliced by token position, but CAN be stored as cumulative
                # state in the last block.  _extract_block_tensor_slice() tags these as
                # ("cumulative", ...) for last blocks and ("skip",) for earlier blocks.
                # This allows prefix cache to restore the full hybrid cache (KV + SSM)
                # on exact prefix matches, avoiding the forced miss that previously
                # disabled prefix caching for all hybrid SSM models (Nemotron, etc.).
                elif hasattr(layer_cache, "cache") and isinstance(
                    getattr(layer_cache, "cache", None), list
                ):
                    if hasattr(layer_cache, "state") and hasattr(
                        layer_cache, "meta_state"
                    ):
                        cls_name = type(layer_cache).__name__
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        extracted.append(
                            {
                                "state": layer_cache.state,
                                "meta_state": layer_cache.meta_state,
                                "class_name": cls_name,
                            }
                        )
                    else:
                        # SSM layer without state/meta_state — include placeholder
                        # so layer indices stay aligned with model layers
                        cls_name = type(layer_cache).__name__
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        extracted.append(
                            {
                                "state": None,
                                "meta_state": None,
                                "class_name": cls_name,
                            }
                        )
                    continue
                elif hasattr(layer_cache, "state") and hasattr(
                    layer_cache, "meta_state"
                ):
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
                            if hasattr(keys, "shape") and len(keys.shape) == 4:
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
                                if (
                                    hasattr(first_k, "shape")
                                    and len(first_k.shape) == 4
                                    and first_k.shape[1] > n_kv
                                ):
                                    orig_h = first_k.shape[1]
                                    keys = tuple(t[:, :n_kv, :, :] for t in keys)
                                    values = tuple(t[:, :n_kv, :, :] for t in values)
                                    state = (keys, values)
                                    if i == 0:
                                        logger.debug(
                                            f"GQA head normalization (quantized): "
                                            f"sliced H {orig_h} → {n_kv}"
                                        )

                    # Ensure QuantizedKVCache meta includes group_size and bits.
                    # meta_state from QuantizedKVCache is ('offset', 'group_size', 'bits')
                    # but if the cache was quantized post-extraction, meta may only have
                    # ('offset',). Pad with actual values to prevent wrong defaults on reconstruct.
                    if (
                        cls_name == "QuantizedKVCache"
                        and isinstance(meta, (tuple, list))
                        and len(meta) < 3
                    ):
                        g = getattr(layer_cache, "group_size", 64)
                        b = getattr(layer_cache, "bits", 8)
                        meta = (meta[0] if meta else "0", str(g), str(b))

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

        # Model-level forced bypass (e.g. Gemma 4 mixed sliding+full
        # attention). See __init__ for rationale. Applies on top of any
        # per-request cache_salt / skip_prefix_cache bypass.
        if getattr(self, "_force_bypass_prefix_cache", False):
            request._bypass_prefix_cache = True

        # Reset PLD auto-tune window on each new request — each generation
        # is a different workload, so cwnd from the previous request is
        # irrelevant.  But only reset if PLD was actively running; if
        # auto-tune disabled it, respect that decision (the probe will
        # re-enable periodically to check if conditions changed).
        if self._pld_spec_enabled and self._pld_auto_enabled:
            self._pld_at_window = 1
            self._pld_summary_next = 1

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

        # Per-request cache bypass (from cache_salt / skip_prefix_cache on the
        # API request). When set, skip EVERY prefix cache lookup below AND
        # ensure no store happens at the end. This is the hard guarantee that
        # benchmark runs need to avoid pollution from prior requests.
        _bypass = bool(getattr(request, "_bypass_prefix_cache", False))
        if _bypass:
            logger.debug(
                f"Request {request.request_id}: _bypass_prefix_cache=True, "
                "skipping paged / memory-aware / legacy prefix / SSM companion cache lookups"
            )
            request.remaining_tokens = request.prompt_token_ids

        # Check prefix cache for cached KV state.
        # Strip gen_prompt_len from fetch key to match store (which also strips).
        # The suffix is re-attached to `remaining` so the model still sees the
        # template trailer (e.g. `<|im_start|>assistant\n<think>\n`) during prefill.
        _full_tokens_list = list(request.prompt_token_ids)
        _gpl_fetch = getattr(request, "_gen_prompt_len", 0) or 0
        if 0 < _gpl_fetch < len(_full_tokens_list):
            _fetch_tokens = _full_tokens_list[:-_gpl_fetch]
            _gpl_suffix_tokens = _full_tokens_list[-_gpl_fetch:]
        else:
            _fetch_tokens = _full_tokens_list
            _gpl_suffix_tokens = []

        if self.block_aware_cache is not None and not _bypass:
            # Use paged cache
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                _fetch_tokens,
            )
            # Re-append gpl suffix to remaining so model sees template trailer.
            if _gpl_suffix_tokens:
                remaining = list(remaining or []) + list(_gpl_suffix_tokens)
            if block_table and block_table.num_tokens > 0:
                # Reconstruct actual KVCache objects from stored tensor data
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                if reconstructed:
                    # Dequantize for BatchGenerator compatibility
                    if getattr(self, "_kv_cache_bits", 0):
                        reconstructed = self._dequantize_cache_for_use(reconstructed)
                    if reconstructed is None:
                        # Dequantization failed — treat as cache miss
                        request.remaining_tokens = request.prompt_token_ids
                        logger.info(
                            f"Request {request.request_id}: dequantization failed, "
                            f"treating as cache miss"
                        )
                    else:
                        # Hybrid SSM models: combine reconstructed KV blocks
                        # with SSM companion state for a full cache hit.
                        if self._is_hybrid:
                            try:
                                # Fetch SSM companion state.
                                # MUST use block_table.num_tokens (the KV block
                                # boundary) — NOT the full prompt length.
                                # KV blocks are stored with gen_prompt stripped,
                                # so if SSM companion was stored at a different
                                # position (e.g., full tokens with gen_prompt),
                                # it won't match → correct fallback to full prefill.
                                # Using full token count caused position mismatch:
                                # SSM at 17 tokens + KV at 11 tokens → double-
                                # processing gen_prompt through SSM → garbage output.
                                # REQ-A3-001: fetch now returns Optional[Tuple[List[Any], bool]].
                                # is_complete is unused on this LLM hybrid fast path; Agent 1's
                                # trie consumes it directly via PrefixCacheManager.
                                _ssm_tokens = list(request.prompt_token_ids)
                                _fetch_num = block_table.num_tokens
                                # P0-1 diagnostic: log every fetch attempt with
                                # tokens-tail so we can correlate hash+key with
                                # store events.
                                logger.info(
                                    "SSM companion attempt fetch: req=%s "
                                    "fetch_num=%d cache_size=%d tokens_tail=%s",
                                    request.request_id,
                                    _fetch_num,
                                    len(self._ssm_state_cache._store) if self._ssm_state_cache is not None else -1,
                                    _ssm_tokens[max(0, _fetch_num-8):_fetch_num] if _fetch_num > 0 else [],
                                )
                                _entry = (
                                    self._ssm_state_cache.fetch(
                                        _ssm_tokens,
                                        _fetch_num,
                                    )
                                    if self._ssm_state_cache is not None
                                    and _fetch_num > 0
                                    else None
                                )
                                if _entry is None:
                                    ssm_states = None
                                else:
                                    ssm_states, _is_complete = _entry
                                    if not _is_complete:
                                        # Captured post-gpl-prefill — state reflects
                                        # more tokens than the stored key. Reuse would
                                        # double-apply gen-prompt tokens on re-feed and
                                        # cause <think></think> loops. Reject.
                                        logger.info(
                                            f"SSM companion for {request.request_id}: "
                                            f"is_complete=False (gpl-contaminated), "
                                            f"rejecting hit — full prefill"
                                        )
                                        ssm_states = None

                                # vmlx#91 RESUME: exact SSM miss, but a stored
                                # checkpoint at a shorter position may still be
                                # a valid prefix. Trim KV block_table to match
                                # the checkpoint and re-reconstruct — prefills
                                # only the tail delta instead of full ~58K.
                                # Infrastructure: SSMCompanionCache.fetch_longest_prefix
                                # + BlockAwarePrefixCache.trim_block_table. Default ON;
                                # set VMLX_DISABLE_SSM_PREFIX_RESUME=1 to force
                                # legacy full-prefill fallback.
                                if not ssm_states and self._ssm_state_cache is not None and _fetch_num > 0:
                                    import os as _os
                                    _resume_disabled = _os.environ.get(
                                        "VMLX_DISABLE_SSM_PREFIX_RESUME"
                                    ) in ("1", "true", "True", "yes", "on")
                                    _fn_lp = getattr(
                                        self._ssm_state_cache,
                                        "fetch_longest_prefix",
                                        None,
                                    )
                                    _missed_ck = None
                                    if not _resume_disabled and _fn_lp is not None:
                                        try:
                                            _missed_ck = _fn_lp(_ssm_tokens, _fetch_num)
                                        except Exception:
                                            _missed_ck = None
                                    if _missed_ck is not None:
                                        _ck_len, _ck_states, _ck_complete = _missed_ck
                                        if not _ck_complete:
                                            logger.info(
                                                f"Request {request.request_id}: vmlx#91 RESUME "
                                                f"skipped — checkpoint at {_ck_len} has "
                                                f"is_complete=False (gpl-contaminated), full prefill"
                                            )
                                            _missed_ck = None
                                            _ck_states = None
                                    if _missed_ck is not None:
                                        _trimmed = self.block_aware_cache.trim_block_table(
                                            request.request_id, _ck_len
                                        )
                                        if _trimmed is not None and _trimmed.num_tokens > 0:
                                            _re_reconstructed = self.block_aware_cache.reconstruct_cache(
                                                _trimmed
                                            )
                                            if _re_reconstructed is not None:
                                                if getattr(self, "_kv_cache_bits", 0):
                                                    _re_reconstructed = self._dequantize_cache_for_use(
                                                        _re_reconstructed
                                                    )
                                            if _re_reconstructed is not None:
                                                block_table = _trimmed
                                                reconstructed = _re_reconstructed
                                                ssm_states = _ck_states
                                                remaining = list(
                                                    request.prompt_token_ids[block_table.num_tokens:]
                                                )
                                                logger.info(
                                                    f"Request {request.request_id}: "
                                                    f"vmlx#91 RESUME — trimmed KV to "
                                                    f"{block_table.num_tokens} tokens "
                                                    f"(block-aligned from SSM "
                                                    f"checkpoint at {_ck_len}), reusing "
                                                    f"SSM companion state. Prefill tail: "
                                                    f"{len(remaining)} tokens"
                                                )

                                if not ssm_states:
                                    # No SSM companion (None or empty) — fall back to full
                                    # prefill for THIS request. Keep KV blocks cache-resident
                                    # so cross-session reuse stays possible. Previously this
                                    # called release_cache → delete_block_table → free_block,
                                    # which poisoned the cache: every SSM miss recycled the
                                    # KV blocks, and the next session found neither KV nor SSM.
                                    # Empirical Pass 1/Pass 2 wallclock noise (-2%) confirmed
                                    # the warm-pass was effectively write-only across sessions.
                                    # Use detach_request to drop only the per-request refs.
                                    logger.info(
                                        f"Request {request.request_id}: "
                                        f"hybrid paged MISS — "
                                        f"{block_table.num_tokens} KV tokens cached "
                                        f"but no SSM companion, full prefill "
                                        f"(blocks kept cached for future sessions)"
                                    )
                                    reconstructed = None
                                    request.remaining_tokens = request.prompt_token_ids
                                    self.block_aware_cache.detach_request(
                                        request.request_id
                                    )
                                else:
                                    # Expand KV-only cache to full layer count
                                    # and inject SSM states at non-KV positions
                                    full_cache = _fix_hybrid_cache(
                                        reconstructed,
                                        self.model,
                                        kv_positions=self._hybrid_kv_positions,
                                        num_model_layers=self._hybrid_num_layers,
                                    )
                                    if full_cache is not None:
                                        kv_set = set(self._hybrid_kv_positions or [])
                                        ssm_idx = 0
                                        for layer_i in range(len(full_cache)):
                                            if layer_i not in kv_set and ssm_idx < len(
                                                ssm_states
                                            ):
                                                full_cache[layer_i] = ssm_states[
                                                    ssm_idx
                                                ]
                                                ssm_idx += 1
                                        reconstructed = full_cache
                                        request._cache_detail = f"paged+ssm({ssm_idx})"
                                        logger.info(
                                            f"Request {request.request_id}: "
                                            f"hybrid paged HIT — "
                                            f"{block_table.num_tokens} tokens "
                                            f"(KV + {ssm_idx} SSM layers)"
                                        )
                                    else:
                                        # _fix_hybrid_cache failed
                                        logger.warning(
                                            f"Request {request.request_id}: "
                                            f"hybrid cache fix returned None"
                                        )
                                        reconstructed = None
                                        request.remaining_tokens = (
                                            request.prompt_token_ids
                                        )
                                        self.block_aware_cache.release_cache(
                                            request.request_id
                                        )
                            except Exception as e:
                                logger.warning(
                                    f"Request {request.request_id}: "
                                    f"hybrid cache reconstruction failed: {e}"
                                )
                                reconstructed = None
                                request.cached_tokens = 0
                                request.remaining_tokens = request.prompt_token_ids
                        if reconstructed is not None:
                            # Re-compress to TQ for memory efficiency during
                            # decode. Safe because blocks now store ORIGINAL
                            # float16 (extracted before TQ recompress on store).
                            # Single round of TQ lossy = same as first inference.
                            request.prompt_cache = reconstructed
                            request.block_table = block_table
                            request.cached_tokens = block_table.num_tokens
                            request.shared_prefix_blocks = len(block_table.block_ids)
                            request.remaining_tokens = remaining
                            if not getattr(request, "_cache_detail", ""):
                                request._cache_detail = "paged"
                            logger.info(
                                f"Request {request.request_id}: paged cache hit, "
                                f"{request.cached_tokens} tokens in "
                                f"{request.shared_prefix_blocks} blocks, "
                                f"{len(remaining)} remaining to process"
                            )
                else:
                    # Reconstruction failed, treat as cache miss.
                    # mlxstudio#73: fetch_cache incremented block refs; if we
                    # don't release them here, a long-running session leaks
                    # refs on every reconstruction failure (e.g. MLX API
                    # drift, quantization mismatch, TQ dequant error). Over
                    # thousands of turns this exhausts the block pool and
                    # makes the scheduler appear frozen (requests wait for
                    # allocations that will never free). Release now.
                    try:
                        self.block_aware_cache.release_cache(request.request_id)
                    except Exception as _rel_err:
                        logger.debug(
                            f"release_cache after reconstruct failure "
                            f"threw ({type(_rel_err).__name__}): {_rel_err}"
                        )
                    request.remaining_tokens = request.prompt_token_ids
                    request.cached_tokens = 0
                    logger.info(
                        f"Request {request.request_id}: paged cache reconstruction "
                        f"failed — block refs released, full prefill"
                    )
            else:
                request.remaining_tokens = request.prompt_token_ids
                logger.info(
                    f"Request {request.request_id}: paged cache miss, "
                    f"processing all {len(request.prompt_token_ids)} tokens"
                )
        elif self.memory_aware_cache is not None and not _bypass:
            # Use memory-aware prefix cache (gpl-stripped fetch; suffix re-attached)
            cache, remaining = self.memory_aware_cache.fetch(_fetch_tokens)
            if _gpl_suffix_tokens:
                remaining = list(remaining or []) + list(_gpl_suffix_tokens)
            if cache:
                # Dequantize for BatchGenerator compatibility
                if getattr(self, "_kv_cache_bits", 0):
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
                    request.cached_tokens = len(request.prompt_token_ids) - len(
                        remaining
                    )
                    request.remaining_tokens = remaining
                    request._cache_detail = "memory"
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
        elif self.prefix_cache is not None and not _bypass:
            # Use legacy prefix cache (gpl-stripped fetch; suffix re-attached)
            cache, remaining = self.prefix_cache.fetch_cache(_fetch_tokens)
            if _gpl_suffix_tokens:
                remaining = list(remaining or []) + list(_gpl_suffix_tokens)
            if cache:
                # Dequantize for BatchGenerator compatibility
                if getattr(self, "_kv_cache_bits", 0):
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
                    request.cached_tokens = len(request.prompt_token_ids) - len(
                        remaining
                    )
                    request.remaining_tokens = remaining
                    request._cache_detail = "prefix"
                    logger.debug(
                        f"Request {request.request_id}: cache hit, "
                        f"{request.cached_tokens} tokens cached, "
                        f"{len(remaining)} tokens remaining"
                    )
            else:
                request.remaining_tokens = request.prompt_token_ids
        else:
            request.remaining_tokens = request.prompt_token_ids

        # L2: Disk cache fallback when in-memory cache missed.
        # Strip gen_prompt_len from the fetch key to match the store key.
        # Thinking models append generation-prompt tokens that change between
        # turns — the cache key must exclude them for consistent SHA-256 matching.
        # Bypass: if the request set _bypass_prefix_cache, skip the disk L2
        # fallback too (otherwise we'd service the request with stale state).
        if request.prompt_cache is None and self.disk_cache is not None and not _bypass:
            _disk_fetch_tokens = list(request.prompt_token_ids)
            _gpl = getattr(request, "_gen_prompt_len", 0) or 0
            if _gpl > 0 and _gpl < len(_disk_fetch_tokens):
                _disk_fetch_tokens = _disk_fetch_tokens[:-_gpl]
            disk_cache = self.disk_cache.fetch(_disk_fetch_tokens)
            if disk_cache is not None:
                # Disk cache stores full-precision N-1 tokens (last prompt token re-fed on hit)
                # Dequantize if KV cache quantization is active (disk stores full precision
                # but may have been quantized before storage in some paths)
                if getattr(self, "_kv_cache_bits", 0):
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
                    # Annotate cache_detail: "disk+tq" for TQ-native 26x-compressed
                    # files, "disk" for standard float16 format.
                    _tq_disk = (
                        hasattr(self.disk_cache, "_last_fetch_tq_native")
                        and self.disk_cache._last_fetch_tq_native
                    )
                    request._cache_detail = "disk+tq" if _tq_disk else "disk"
                    # Recover the original cache_type so the L1 backfill keeps
                    # the same priority (system entries stay pinned across the
                    # disk roundtrip — F3).
                    _l1_type = getattr(
                        self.disk_cache, "_last_fetch_cache_type", "assistant"
                    )
                    # Also populate L1 memory cache for faster subsequent hits.
                    # Quantize for L1 if KV quant is enabled (disk stores full precision).
                    l1_data = disk_cache
                    if getattr(self, "_kv_cache_bits", 0):
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
                                    cache_type=_l1_type,
                                )
                                # Clean up request table entry. Release request refs so
                                # blocks become "cached but free" (still hash-resident,
                                # now reclaimable via LRU).
                                _entry = self.block_aware_cache._request_tables.pop(
                                    request.request_id, None
                                )
                                self.block_aware_cache.paged_cache.release_request_refs(
                                    _entry.block_table if _entry else None
                                )
                                self.block_aware_cache.paged_cache.detach_request(
                                    request.request_id
                                )
                        except Exception:
                            pass
                    elif self.memory_aware_cache is not None:
                        try:
                            self.memory_aware_cache.store(
                                request.prompt_token_ids,
                                l1_data,
                                cache_type=_l1_type,
                            )
                        except Exception:
                            pass
                    elif self.prefix_cache is not None:
                        try:
                            self.prefix_cache.store_cache(
                                list(request.prompt_token_ids),
                                l1_data,
                                cache_type=_l1_type,
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

        IMPORTANT: BatchGenerator removal is DEFERRED to the next step() call.
        Client disconnects can happen while Metal command buffers are in-flight.
        Calling batch_generator.remove() immediately would touch cache tensors
        mid-computation, triggering Metal assertion failures that crash the
        process. The deferred approach lets the current Metal computation
        complete before cleanup.

        Safe to call multiple times (idempotent) — returns False on repeat calls.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted, False otherwise
        """
        request = self.requests.pop(request_id, None)
        if request is None:
            return False

        # Remove from waiting queue (safe — no Metal involvement)
        if request.status == RequestStatus.WAITING:
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # DEFER BatchGenerator removal to step() — see docstring above.
        # Only defer if the request is actually in the batch generator.
        if request.request_id in self.request_id_to_uid:
            self._pending_aborts.add(request.request_id)

        # Clean up per-request stop tokens from shared BatchGenerator
        # Must happen BEFORE removing from running, so we can still check
        # which tokens are still needed by surviving requests.
        added_stops = getattr(request, "_added_stop_tokens", None)
        if added_stops and self.batch_generator is not None:
            # Only remove tokens not needed by other running requests
            surviving_stops = set()
            for rid, req in self.running.items():
                if rid != request.request_id:
                    surviving_stops.update(getattr(req, "_added_stop_tokens", set()))
            removable = added_stops - surviving_stops
            if removable:
                self.batch_generator.stop_tokens -= removable

        # Remove from running (BatchGenerator) — DEFERRED.
        # The actual batch_generator.remove() happens in _process_pending_aborts()
        # which runs at the start of step() after Metal has synchronized.
        # UID cleanup also deferred — done in _process_pending_aborts().

        # Clean up paged cache tracking (prevent block table leaks)
        # Use delete_block_table (not detach_request) so ref_counts are
        # decremented — aborted requests don't store blocks in prefix cache,
        # so detach would orphan them with permanently elevated ref_count.
        if self.block_aware_cache is not None:
            self.block_aware_cache._request_tables.pop(request_id, None)
            self.block_aware_cache.paged_cache.delete_block_table(request_id)

        # Clear extracted cache reference to help GC
        if hasattr(request, "_extracted_cache"):
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

    def _process_pending_aborts(self) -> None:
        """Process deferred abort requests.

        Called at the start of step() after the previous batch_generator.next()
        has completed. At this point Metal has synchronized and it's safe to
        call batch_generator.remove() without risking assertion failures on
        in-flight command buffers.
        """
        aborts = list(self._pending_aborts)
        self._pending_aborts.clear()
        for request_id in aborts:
            uid = self.request_id_to_uid.pop(request_id, None)
            if uid is not None:
                if self.batch_generator is not None:
                    try:
                        self.batch_generator.remove([uid])
                    except Exception as e:
                        logger.warning(
                            f"Deferred abort remove failed for {request_id}: {e}"
                        )
                self.uid_to_request_id.pop(uid, None)
            logger.debug(f"Processed deferred abort for {request_id}")

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
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        # Flush prompt-level disk cache (DiskCacheManager)
        if getattr(self, "disk_cache", None) is not None:
            logger.info("Shutting down prompt disk cache...")
            self.disk_cache.shutdown()
            logger.info("Prompt disk cache shutdown complete")

        # Flush block-level disk cache (BlockDiskStore)
        if hasattr(self, "paged_cache_manager") and self.paged_cache_manager:
            disk_store = getattr(self.paged_cache_manager, "_disk_store", None)
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
            # Memory-pressure guard: don't admit new requests if GPU memory is critically low
            try:
                import mlx.core as mx

                # vmlx#94: prefer mx.* top-level APIs, fall back to mx.metal.*
                # for pre-0.31 MLX builds. Same pattern used in server.py.
                _get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
                _device_info = getattr(mx, "device_info", None) or mx.metal.device_info

                active_mem = _get_active()
                if active_mem > 0 and len(self.running) > 0:
                    max_mem = _device_info().get(
                        "max_recommended_working_set_size", 0
                    )
                    if max_mem > 0 and active_mem / max_mem > 0.85:
                        logger.debug(
                            f"Memory pressure ({active_mem / 1e9:.1f}GB / {max_mem / 1e9:.1f}GB = "
                            f"{active_mem / max_mem:.0%}), deferring new request admission"
                        )
                        break
            except Exception:
                pass  # Metal API not available — skip check

            request = self.waiting.popleft()

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            # Track first-schedule time for TTFT (only set once per request)
            if not hasattr(request, "_schedule_time"):
                request._schedule_time = time.perf_counter()

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
                        kv_bits = getattr(self, "_kv_cache_bits", 0)
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
                if (
                    request.sampling_params.stop_token_ids
                    and self.batch_generator is not None
                ):
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
            if hasattr(response, "token"):
                is_first_token = request.num_computed_tokens == 0
                request.append_output_token(response.token)
                if is_first_token and hasattr(request, "_schedule_time"):
                    ttft = time.perf_counter() - request._schedule_time
                    self._ewma_ttft = (
                        self._ttft_alpha * ttft
                        + (1 - self._ttft_alpha) * self._ewma_ttft
                    )
            else:
                continue

            # ── Prompt Lookup Decoding — measurement (Phase 1) ──────────────
            # Retrospectively check whether the previous draft prediction was
            # correct, then generate a new draft for the next position.
            # Zero effect on generation output; pure stat collection.
            # Gated on _pld_spec_enabled to avoid find_draft_tokens overhead
            # on servers not using PLD.
            if self._pld_spec_enabled:
                try:
                    current_idx = (
                        request.num_output_tokens - 1
                    )  # 0-based, just appended
                    pending = self._pld_pending.get(request_id)
                    if pending is not None:
                        draft_tokens, start_idx, hit_count = pending
                        pos = current_idx - start_idx
                        if 0 <= pos < len(draft_tokens):
                            if response.token == draft_tokens[pos]:
                                hit_count += 1
                                if pos == 0:
                                    pld_stats.first_hit += 1
                                pld_stats.total_hit_depth += 1
                                self._pld_pending[request_id] = (
                                    draft_tokens,
                                    start_idx,
                                    hit_count,
                                )
                            else:
                                # Miss — record completed sequence and clear
                                pld_stats.completed_seqs += 1
                                del self._pld_pending[request_id]
                        else:
                            pld_stats.completed_seqs += 1
                            del self._pld_pending[request_id]

                    if request_id not in self._pld_pending:
                        full_tokens = list(request.prompt_token_ids) + list(
                            request.output_token_ids
                        )
                        drafts = find_draft_tokens(full_tokens)
                        if drafts:
                            pld_stats.draft_found += 1
                            self._pld_pending[request_id] = (
                                drafts,
                                request.num_output_tokens,
                                0,
                            )

                    pld_stats.total_tokens += 1
                    if pld_stats.total_tokens % 200 == 0:
                        pld_stats.log_summary()
                except Exception:
                    pass  # Never let measurement break generation
            # ── end PLD measurement ──────────────────────────────────────────

            # ── PLD Phase 2: speculative extension ───────────────────────────
            # Attempt to accept K draft tokens + bonus in one forward pass.
            # spec_tokens = [d1, ..., d_M, bonus_token] if any accepted; else []
            # Any stop token in spec_tokens truncates the list at that point.
            spec_tokens: List[int] = []
            spec_hit_stop = False
            _step_t0 = time.perf_counter()
            if (
                self._pld_spec_enabled
                and self._pld_auto_enabled
                and response.finish_reason is None
                and self.batch_generator is not None
            ):
                try:
                    _pld_t0 = time.perf_counter()
                    raw_spec = self._try_speculative_decode(
                        request_id, request, response.token
                    )
                    self._pld_win_cycle_wall_s += time.perf_counter() - _pld_t0
                    _spec_stops = self.stop_tokens | set(
                        request.sampling_params.stop_token_ids or []
                    )
                    for tok in raw_spec:
                        if tok in _spec_stops:
                            spec_hit_stop = True
                            break
                        spec_tokens.append(tok)
                        request.append_output_token(tok)
                except Exception:
                    spec_tokens = []  # never let speculative break generation
            # ── end PLD Phase 2 ───────────────────────────────────────────────

            # Use streaming detokenizer for correct multi-byte char handling
            detok = self._get_detokenizer(request_id)

            # Check if finished BEFORE adding token to detokenizer
            # so stop tokens (e.g. <|im_end|>) don't leak into new_text
            is_stop = response.finish_reason == "stop" or spec_hit_stop
            string_stop_truncate = -1  # >=0 when string stop matched

            if not is_stop:
                # Capture text start so we can diff after adding all tokens
                text_before = detok.text
                detok.add_token(response.token)
                for tok in spec_tokens:
                    detok.add_token(tok)
                new_text = detok.text[len(text_before) :]

                # Advance the per-request reasoning state machine on every
                # emitted token. No-op when no reasoning parser is registered
                # or `_use_sm_stops` is disabled. See decisions.md D-A2-005,
                # D-A2-006.
                self._advance_request_state_machine(
                    request, [response.token, *spec_tokens]
                )

                # Post-decode string stop sequence check.
                # BatchGenerator only handles integer stop_token_ids;
                # string stop sequences need decoded-text matching.
                # Skip matching inside reasoning blocks — reasoning content
                # should not trigger user-specified stop sequences.
                if request.sampling_params.stop:
                    full_text = detok.text
                    # Prefer the token-level state machine when the parser
                    # provided tag tokens. Fall back to the legacy substring
                    # scan for parsers without tag-token support (e.g.
                    # `Gemma4ReasoningParser`, `GptOssReasoningParser`)
                    # so behaviour matches the pre-Phase-3c baseline for
                    # those models.
                    if self._reasoning_sm is not None:
                        in_think = self._is_request_in_reasoning(request)
                    else:
                        in_think = (
                            "<think>" in full_text
                            and "</think>" not in full_text.split("<think>")[-1]
                        )
                    if not in_think:
                        max_stop_len = max(len(s) for s in request.sampling_params.stop)
                        search_start = max(
                            0, len(full_text) - len(new_text) - max_stop_len + 1
                        )
                        last_think_end = full_text.rfind("</think>")
                        if last_think_end >= 0:
                            search_start = max(
                                search_start, last_think_end + len("</think>")
                            )
                        for stop_str in request.sampling_params.stop:
                            idx = full_text.find(stop_str, search_start)
                            if idx >= 0:
                                string_stop_truncate = idx
                                new_text = ""
                                break
            else:
                # Stop token: don't decode it, just flush any buffered text
                new_text = ""

            # Create output — include cache_detail and base token + any accepted spec tokens
            _detail = getattr(request, "_cache_detail", "")
            # Annotate TQ if active (skip if already annotated, e.g. "disk+tq")
            if _detail and getattr(self, "_tq_active", False) and "+tq" not in _detail:
                _detail += "+tq"
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token] + spec_tokens,
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=request.cached_tokens,
                cache_detail=_detail,
            )

            # Determine effective finish reason (string stop or spec stop override)
            finish_reason = response.finish_reason
            if spec_hit_stop:
                finish_reason = "stop"
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
                                if hasattr(
                                    request, "cached_tokens"
                                ) and request.cached_tokens >= len(
                                    request.prompt_token_ids
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
                                        # TQ re-wrap: BatchKVCache.extract() always
                                        # returns plain KVCache objects even for TQ
                                        # Extract FIRST from original float16 cache.
                                        # This ensures blocks store original quality
                                        # data (not TQ-decoded lossy float16).
                                        # On fetch, TQ recompress is safe because
                                        # it's a single round of lossy (same as
                                        # original inference).
                                        if getattr(self, "_kv_cache_bits", 0):
                                            cache_for_extract = (
                                                self._quantize_cache_for_storage(
                                                    cache_for_extract
                                                )
                                            )
                                        extracted_cache = self._extract_cache_states(
                                            cache_for_extract
                                        )
                                        # L2 disk: TQ recompress a COPY for 26x
                                        # smaller disk files. The original cache
                                        # objects are unchanged (extract already ran).
                                        if (
                                            self.disk_cache is not None
                                            and not self._is_hybrid
                                        ):
                                            try:
                                                from .mllm_batch_generator import (
                                                    _recompress_to_tq,
                                                )

                                                tq_for_disk = _recompress_to_tq(
                                                    cache_for_extract, self.model
                                                )
                                                _disk_store_tokens = list(
                                                    request.prompt_token_ids
                                                )
                                                _gpl_s = (
                                                    getattr(
                                                        request, "_gen_prompt_len", 0
                                                    )
                                                    or 0
                                                )
                                                if _gpl_s > 0 and _gpl_s < len(
                                                    _disk_store_tokens
                                                ):
                                                    _disk_store_tokens = (
                                                        _disk_store_tokens[:-_gpl_s]
                                                    )
                                                self.disk_cache.store(
                                                    _disk_store_tokens,
                                                    tq_for_disk,
                                                    cache_type=self._pick_cache_type_for_request(
                                                        request
                                                    ),
                                                )
                                            except Exception as de:
                                                logger.debug(
                                                    f"Disk cache store failed for "
                                                    f"{request_id}: {de}"
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

            # Auto-tune timing: track wall time per step and tokens produced
            if self._pld_spec_enabled:
                self._pld_win_step_wall_s += time.perf_counter() - _step_t0
                self._pld_win_total_tokens += 1 + len(spec_tokens)
                # Trigger summary/probe based on total tokens, not just PLD
                # tokens — otherwise auto-disabled PLD never gets probed.
                if self._pld_win_total_tokens >= self._pld_summary_next:
                    self._pld_maybe_log_summary()

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        # H1 parity: Snapshot stop tokens from requests that will SURVIVE this cleanup.
        # This prevents removing tokens still needed by other running requests.
        _surviving_stops = set()
        for rid, req in self.running.items():
            if rid not in finished_ids:
                _surviving_stops.update(getattr(req, "_added_stop_tokens", set()))

        for request_id in finished_ids:
            # Clean up PLD state
            self._pld_pending.pop(request_id, None)
            self._pld_ngram_indices.pop(request_id, None)

            request = self.running.get(request_id)

            # PHASE 2 OPTIMIZATION: Skip cache storage for short-output requests.
            # For benchmark workloads (MMLU: 14k requests, 1-token answers), cache
            # extraction + truncation + storage dominates per-request overhead.
            # Single-turn requests with very short outputs will never benefit from
            # prefix cache reuse, so skip entirely.
            _output_len = getattr(request, "num_output_tokens", 0) if request else 0
            _skip_cache_store = (
                _output_len > 0
                and _output_len <= 3
                and not getattr(request, "_has_history", False)
            )
            # Hard cache bypass from the API request (cache_salt / skip_prefix_cache).
            # This overrides all heuristics — the benchmark client asked for
            # guaranteed fresh execution, so nothing gets stored either.
            if request is not None and getattr(request, "_bypass_prefix_cache", False):
                _skip_cache_store = True
            if _skip_cache_store and request is not None:
                logger.debug(
                    f"Skipping cache store for {request_id}: "
                    f"short output ({_output_len} tokens), likely single-turn"
                )
                if hasattr(request, "_extracted_cache"):
                    request._extracted_cache = None

            # Always clean up paged cache tracking entries regardless of
            # cache skip, to prevent unbounded memory growth on benchmarks.
            # Release request refs here so completed-request blocks can enter
            # the free LRU queue while remaining cache-resident.
            if self.block_aware_cache is not None:
                _entry = self.block_aware_cache._request_tables.pop(request_id, None)
                self.block_aware_cache.paged_cache.release_request_refs(
                    _entry.block_table if _entry else None
                )
                self.block_aware_cache.paged_cache.detach_request(request_id)

            # Hybrid SSM companion state capture.
            # MUST run BEFORE the paged cache store below, because
            # the paged store clears request._extracted_cache in its
            # finally block. If we run after, _extracted_cache is None.
            # Store SSM layer states keyed by block-aligned prompt tokens
            # so future prefix cache hits can reconstruct full KV+SSM cache.
            #
            # LIMITATION: For thinking models (gen_prompt_len > 0), SSM
            # companion is SKIPPED. The extracted SSM state includes
            # gen_prompt + output tokens, placing it at position
            # P+gpl+output instead of P (the KV block boundary).
            # Injecting this contaminated state causes garbled output.
            # Future fix: async re-derive or capture-during-prefill.
            if (
                self._is_hybrid
                and self._ssm_state_cache is not None
                and request is not None
                and request.prompt_token_ids
                and not _skip_cache_store
            ):
                try:
                    logger.info(
                        f"SSM companion: entering store for {request_id} "
                        f"(hybrid={self._is_hybrid}, has_cache={hasattr(request, '_extracted_cache') and request._extracted_cache is not None})"
                    )
                    _gpl = getattr(request, "_gen_prompt_len", 0) or 0
                    all_tokens = list(request.prompt_token_ids)
                    if _gpl > 0 and _gpl < len(all_tokens):
                        all_tokens = all_tokens[:-_gpl]
                    prompt_len = len(all_tokens)

                    # SSM state from _extracted_cache is post-generation:
                    # it includes gen_prompt + output tokens processing.
                    # For thinking models (gpl > 0), the extracted SSM
                    # state is contaminated — storing it causes position
                    # mismatch on fetch → garbled output. Skip storage;
                    # KV blocks still provide partial TTFT benefit.
                    if _gpl > 0:
                        # Queue deferred SSM re-derive instead of skipping.
                        # The post-gen SSM state is contaminated by thinking
                        # tokens, so we can't store it directly. But we CAN
                        # queue a re-derive that runs during idle time (no
                        # active requests) — a separate prefill pass on just
                        # the prompt tokens to capture clean SSM state. This
                        # doesn't help the CURRENT conversation but ensures
                        # the NEXT request with the same prompt prefix gets
                        # a full KV+SSM cache hit instead of re-prefilling.
                        if not hasattr(self, "_ssm_rederive_queue"):
                            self._ssm_rederive_queue = []
                        # Cap queue at 20 entries to prevent unbounded growth
                        # under sustained load. Oldest entries are the least
                        # useful (newer prompts more likely to be re-requested).
                        if len(self._ssm_rederive_queue) >= 20:
                            self._ssm_rederive_queue.pop(0)
                        self._ssm_rederive_queue.append(
                            (list(all_tokens), prompt_len, request_id)
                        )
                        logger.info(
                            f"SSM companion: queued deferred re-derive for "
                            f"{request_id} (gpl={_gpl}, {prompt_len} prompt "
                            f"tokens, will run during next idle period)"
                        )
                    elif prompt_len > 0:
                        _ssm_key_tokens = all_tokens
                        _ssm_key_len = prompt_len

                        # ── Extract SSM layers from generated cache ──
                        ssm_layers = []
                        if (
                            hasattr(request, "_extracted_cache")
                            and request._extracted_cache
                        ):
                            _KV_NAMES = {
                                "KVCache",
                                "QuantizedKVCache",
                                "RotatingKVCache",
                                "CacheList",
                                "TurboQuantKVCache",
                            }
                            _ssm_candidates = 0
                            for layer_sd in request._extracted_cache:
                                if not isinstance(layer_sd, dict):
                                    continue
                                cls = layer_sd.get("class_name", "")
                                if cls in _KV_NAMES:
                                    continue
                                _ssm_candidates += 1
                                state = layer_sd.get("state")
                                meta = layer_sd.get("meta_state")
                                if state is not None:
                                    try:
                                        import mlx_lm.models.cache as _cm

                                        cc = getattr(_cm, cls, None)
                                        if cc and hasattr(cc, "from_state"):
                                            ssm_layers.append(
                                                cc.from_state(state, meta)
                                            )
                                        else:
                                            logger.debug(
                                                f"SSM layer {cls}: no "
                                                f"from_state in mlx_lm"
                                            )
                                    except Exception as _se:
                                        logger.debug(f"SSM {cls} restore: {_se}")
                            if _ssm_candidates > 0 and not ssm_layers:
                                logger.warning(
                                    f"SSM companion: {_ssm_candidates} "
                                    f"candidate layers but 0 restored "
                                    f"for {request_id}"
                                )
                        if ssm_layers:
                            self._ssm_state_cache.store(
                                _ssm_key_tokens, _ssm_key_len, ssm_layers
                            )
                            logger.info(
                                f"Stored SSM companion for "
                                f"{request_id}: {len(ssm_layers)} "
                                f"layers, {_ssm_key_len}-tok key"
                            )
                except Exception as _ssm_e:
                    logger.warning(
                        f"SSM companion store failed for {request_id}: {_ssm_e}",
                        exc_info=True,
                    )

            # Store cache for future reuse
            if (
                request is not None
                and request.prompt_token_ids
                and not _skip_cache_store
            ):
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
                            # Strip generation prompt tokens from cache key.
                            # Original gen_prompt_len prefix cache fix by Jinho Jang
                            # (eric@jangq.ai) — vMLX. This solved 100% cache miss for
                            # all thinking models (Nemotron, Qwen3, DeepSeek-R1, Mistral 4).
                            # Chat templates append assistant role tokens at the end
                            # (e.g., <|im_start|>assistant\n<think>\n) which always
                            # differ on subsequent turns. Including them in the block
                            # hash causes 100% cache misses in multi-turn conversations.
                            gen_prompt_len = getattr(request, "_gen_prompt_len", 0)
                            cache_data = request._extracted_cache
                            if gen_prompt_len > 0 and gen_prompt_len < len(
                                prompt_tokens
                            ):
                                prompt_tokens = prompt_tokens[:-gen_prompt_len]
                                # Also truncate KV cache data to match shortened key.
                                # Without this, KV has more tokens than the key,
                                # causing duplicate KV entries on cache hit → <unk> flood.
                                # cache_data is extracted state dicts (not raw objects),
                                # so truncate the tensors within each dict directly.
                                target = len(prompt_tokens) - 1  # N-1 for re-feed
                                if target > 0:
                                    truncated_dicts = []
                                    trunc_ok = True
                                    for sd in cache_data:
                                        if not isinstance(sd, dict):
                                            trunc_ok = False
                                            break
                                        state = sd.get("state")
                                        cls_name = sd.get("class_name", "")
                                        if (
                                            state is None
                                            or cls_name == "CacheList"
                                        ):
                                            # CacheList/skip: pass through
                                            truncated_dicts.append(sd)
                                            continue
                                        if isinstance(state, tuple) and len(state) == 2:
                                            keys, values = state
                                            if (
                                                hasattr(keys, "shape")
                                                and len(keys.shape) >= 3
                                            ):
                                                seq_dim = (
                                                    2 if len(keys.shape) == 4 else 1
                                                )
                                                safe = min(target, keys.shape[seq_dim])
                                                if safe > 0:
                                                    if len(keys.shape) == 4:
                                                        keys = keys[:, :, :safe, :]
                                                        values = values[:, :, :safe, :]
                                                    else:
                                                        keys = keys[:, :safe, :]
                                                        values = values[:, :safe, :]
                                                    new_meta = _rebuild_meta_state_after_truncation(
                                                        cls_name,
                                                        sd.get("meta_state", ()),
                                                        safe,
                                                    )
                                                    if new_meta is None:
                                                        # RotatingKVCache with wrapped
                                                        # buffer: cannot safely truncate.
                                                        # Skip this store entirely.
                                                        trunc_ok = False
                                                        break
                                                    truncated_dicts.append(
                                                        {
                                                            **sd,
                                                            "state": (keys, values),
                                                            "meta_state": new_meta,
                                                        }
                                                    )
                                                    continue
                                            elif (
                                                isinstance(keys, (tuple, list))
                                                and len(keys) >= 1
                                            ):
                                                # QuantizedKVCache: tuple of (data, scales, zeros)
                                                first_k = keys[0]
                                                if hasattr(first_k, "shape"):
                                                    safe = min(
                                                        target, first_k.shape[-2]
                                                    )
                                                    if safe > 0:
                                                        keys = tuple(
                                                            t[..., :safe, :]
                                                            for t in keys
                                                        )
                                                        values = tuple(
                                                            t[..., :safe, :]
                                                            for t in values
                                                        )
                                                        new_meta = _rebuild_meta_state_after_truncation(
                                                            cls_name,
                                                            sd.get("meta_state", ()),
                                                            safe,
                                                        )
                                                        if new_meta is None:
                                                            trunc_ok = False
                                                            break
                                                        truncated_dicts.append(
                                                            {
                                                                **sd,
                                                                "state": (keys, values),
                                                                "meta_state": new_meta,
                                                            }
                                                        )
                                                        continue
                                        # Unknown format: pass through
                                        truncated_dicts.append(sd)
                                    if trunc_ok and truncated_dicts:
                                        cache_data = truncated_dicts
                            self.block_aware_cache.store_cache(
                                request_id,
                                prompt_tokens,
                                cache_data,
                                cache_type=self._pick_cache_type_for_request(request),
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
                    # NOTE: Tracking cleanup (pop + detach) moved above the
                    # _skip_cache_store guard so it runs unconditionally.

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
                            # Strip gen_prompt_len from store key — symmetric with
                            # the fetch path (which also strips). Thinking templates
                            # append role trailer tokens that differ on every turn;
                            # without this strip, fetches on later turns miss 100%.
                            _gpl_store = getattr(request, "_gen_prompt_len", 0) or 0
                            if 0 < _gpl_store < len(prompt_tokens):
                                prompt_tokens = prompt_tokens[:-_gpl_store]
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
                                if getattr(self, "_kv_cache_bits", 0):
                                    cache_to_store = self._quantize_cache_for_storage(
                                        cache_to_store
                                    )
                                stored = self.memory_aware_cache.store(
                                    prompt_tokens,
                                    cache_to_store,
                                    cache_type=self._pick_cache_type_for_request(
                                        request
                                    ),
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
                                # NOTE: Disk L2 store is handled by the paged
                                # cache path (Task 1 above). Memory-aware cache
                                # is mutually exclusive with paged — if we're here,
                                # paged cache is disabled and disk L2 is not
                                # available for this configuration.
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
                            # Strip gen_prompt_len from store key (symmetric with fetch).
                            _gpl_store = getattr(request, "_gen_prompt_len", 0) or 0
                            if 0 < _gpl_store < len(prompt_tokens):
                                prompt_tokens = prompt_tokens[:-_gpl_store]
                            prompt_len = len(prompt_tokens)
                            cache_to_store = self._truncate_cache_to_prompt_length(
                                request._extracted_cache, prompt_len
                            )
                            if cache_to_store is not None:
                                # Quantize for storage efficiency
                                if getattr(self, "_kv_cache_bits", 0):
                                    cache_to_store = self._quantize_cache_for_storage(
                                        cache_to_store
                                    )
                                # Phase 3d: store with chat-segment cache_type
                                # awareness when the request carries
                                # `_segment_boundaries` (populated by API
                                # gateways during chat-template rendering).
                                # Falls back to legacy single-store with
                                # cache_type="assistant" when boundaries are
                                # absent — zero regression for existing callers.
                                self._store_cache_with_segments(
                                    request,
                                    prompt_tokens,
                                    cache_to_store,
                                )
                                logger.debug(
                                    f"Stored cache for request {request_id} "
                                    f"({prompt_len} prompt tokens, "
                                    f"truncated from {prompt_len + len(request.output_token_ids)})"
                                )
                                # NOTE: Disk L2 store is handled by the paged
                                # cache path. Legacy prefix cache is L1-only.
                        except Exception as e:
                            logger.debug(f"Failed to store cache for {request_id}: {e}")
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None

            # H1 parity: Remove per-request stop tokens from batch generator
            if (
                request is not None
                and self.batch_generator is not None
                and getattr(request, "_added_stop_tokens", None)
            ):
                removable = (
                    request._added_stop_tokens - _surviving_stops - self.stop_tokens
                )
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
        try:
            self.batch_generator.close()
        except Exception:
            pass
        self.batch_generator = None
        self._current_sampler_params = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
        if self._ssm_state_cache is not None:
            self._ssm_state_cache.clear()

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
            if hasattr(request, "_extracted_cache"):
                request._extracted_cache = None

            # Clear stale detokenizer — request will restart from scratch
            self._cleanup_detokenizer(request_id)

            # Clear PLD state for this request
            self._pld_pending.pop(request_id, None)
            self._pld_ngram_indices.pop(request_id, None)

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def _pld_maybe_log_summary(self) -> None:
        """Emit a PLD effectiveness summary at INFO level every N tokens.

        Logged when cumulative tokens emitted via spec decode rounds crosses
        the next summary threshold. Resets window counters after logging so
        each summary reflects the most recent interval only.

        Metrics:
          rounds   — spec decode attempts in this window
          accept   — mean draft tokens accepted per round (max=K=5)
          full     — % of rounds where all K drafts accepted (best case)
          zero     — % of rounds where 0 drafts accepted (overhead, no gain)
          eff      — effective tokens per forward pass: (accepted+rounds) /
                     (2*rounds). >1.0 means PLD is helping; <1.0 means overhead
                     exceeds benefit. Baseline (no PLD) = 1.0.
        """
        # Trigger on either PLD tokens or total tokens (for disabled-state probe)
        if (
            self._pld_win_tokens < self._pld_summary_next
            and self._pld_win_total_tokens < self._pld_summary_next
        ):
            return

        n = self._pld_win_attempts
        _win_size = self._pld_at_window

        if n == 0:
            if self._pld_auto_enabled:
                # PLD was enabled but d0 pre-check filtered every cycle —
                # no opportunities to help. Disable to avoid per-token
                # overhead from find_draft_tokens + d0 check.
                self._pld_auto_enabled = False
                self._pld_at_window = 1
                self._pld_at_probe_tokens = 0
                logger.info(
                    "[PLD:3b1f] auto-tune — disabled (0 rounds in %d tokens, "
                    "d0 pre-check filtered all)",
                    self._pld_win_total_tokens,
                )
            else:
                # Already disabled.  Count toward probe interval.
                self._pld_at_probe_tokens += self._pld_win_total_tokens
                if self._pld_at_probe_tokens >= self._pld_summary_interval * 5:
                    self._pld_auto_enabled = True
                    self._pld_at_window = 1
                    self._pld_at_probe_tokens = 0
                    logger.info(
                        "[PLD:3b1f] auto-tune probe — re-enabling with window=%d",
                        self._pld_at_window,
                    )
            self._pld_win_total_tokens = 0
            self._pld_win_step_wall_s = 0.0
            self._pld_win_cycle_wall_s = 0.0
            self._pld_summary_next = self._pld_at_window
            return

        accepted = self._pld_win_accepted
        full_pct = 100 * self._pld_win_full / n
        zero_pct = 100 * self._pld_win_zero / n
        avg_accept = accepted / n
        eff = (accepted + n) / (2 * n)

        # Auto-tune: compare PLD window throughput vs estimated baseline.
        _autotune_msg = ""
        base_time = self._pld_win_step_wall_s - self._pld_win_cycle_wall_s
        base_tok = self._pld_win_total_tokens - self._pld_win_tokens
        if base_time > 0 and base_tok >= 1 and self._pld_win_step_wall_s > 0:
            baseline_tok_s = base_tok / base_time
            window_tok_s = self._pld_win_total_tokens / self._pld_win_step_wall_s
            ratio = window_tok_s / baseline_tok_s if baseline_tok_s > 0 else 1.0

            if ratio < 0.95:
                # Congestion: PLD is hurting. Disable and reset window.
                self._pld_auto_enabled = False
                self._pld_at_window = 1  # reset cwnd for next probe
                self._pld_at_probe_tokens = 0
                _autotune_msg = (
                    f" — AUTO-DISABLED cwnd={_win_size} "
                    f"(window {window_tok_s:.0f} tok/s "
                    f"< baseline {baseline_tok_s:.0f} tok/s × 0.95)"
                )
            else:
                # No congestion: grow window (TCP slow start).
                old_window = self._pld_at_window
                self._pld_at_window = min(
                    self._pld_at_window * 2, self._pld_summary_interval
                )
                _autotune_msg = (
                    f"  cwnd={old_window}→{self._pld_at_window} "
                    f"wallclock={window_tok_s:.0f}/{baseline_tok_s:.0f} tok/s"
                )

        logger.info(
            "[PLD:3b1f] summary over last %d tokens — "
            "rounds=%d  accept=%.1f/%d  full=%.0f%%  zero=%.0f%%  "
            "d0_skip=%d  eff=%.2f tok/pass%s",
            self._pld_win_tokens,
            n,
            avg_accept,
            self._pld_num_drafts,
            full_pct,
            zero_pct,
            self._pld_win_d0_skip,
            eff,
            _autotune_msg,
        )

        # Reset window counters
        self._pld_win_attempts = 0
        self._pld_win_accepted = 0
        self._pld_win_full = 0
        self._pld_win_zero = 0
        self._pld_win_tokens = 0
        self._pld_win_d0_skip = 0
        self._pld_win_cycle_wall_s = 0.0
        self._pld_win_step_wall_s = 0.0
        self._pld_win_total_tokens = 0
        self._pld_summary_next = self._pld_at_window

    def _try_speculative_decode(
        self,
        request_id: str,
        request: Request,
        last_token: int,
    ) -> List[int]:
        """
        Prompt Lookup Decoding — Phase 2/3: batched speculative verification.

        After BatchGenerator emits token `last_token`, this method:
          1. Peeks at active_batch.logprobs[e] BEFORE remove() to get
             forward_logprobs — the model's prediction for what comes after
             last_token.  (response.logprobs is the distribution that GENERATED
             last_token, one step behind what we need.)
          2. Extracts the KV cache by removing the request from BatchGenerator.
             Cache is at offset N — last_token is already in it.
          3. Finds K draft tokens via n-gram lookup in the full token sequence.
          4. Runs ONE forward pass: model([d0, ..., d_{K-1}], cache).
             forward_logprobs is used for d0 acceptance; no last_token prefix
             in verify_input means no pre-trim and no SSM offset accumulation.
          5. Accepts the prefix up to the first mismatch (M <= K tokens).
          6. Trims the cache to the accepted prefix.
          7. Re-inserts the request into BatchGenerator with the bonus token.
          8. Updates uid maps so subsequent step() calls find the request.

        Returns extra tokens to append: [d0, ..., d_M, bonus_token].
        On any failure returns [] — generation continues normally.
        A guaranteed finally block ensures the request is never orphaned.

        Phase 2 (temp≈0): greedy acceptance, argmax bonus.
        Phase 3 (temp>0): probabilistic acceptance — accept d_i with probability
        p(d_i | context); on rejection sample a correction token with d_i
        excluded. Bonus is always sampled (not argmax). This provably preserves
        the original sampling distribution (Leviathan et al., 2023).

        NOTE — concurrent request interaction (open question #5):
        The remove/re-insert cycle pulls this request out of the active decode
        batch for the duration of the verify forward pass. Under concurrent
        load (batch_size > 1) this reduces decode-phase batch occupancy for
        one step per spec decode round. Throughput impact at batch_size > 1
        is unmeasured. PLD is safe and correct at any concurrency level but
        may hurt aggregate throughput when multiple requests are in flight.
        """
        import mlx.core as mx
        import numpy as _np

        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None

        temp = request.sampling_params.temperature
        if temp > self._pld_spec_max_temp:
            return []

        # vmlx#92: PLD verify-and-reinsert only works on MLLMBatchGenerator
        # which exposes `.active_batch` (for forward-logprobs peek) and
        # `remove(..., return_prompt_caches=True)` returning trimmable
        # caches. On pure text / non-MLLM paths the generator is mlx-lm's
        # plain BatchGenerator, which has neither.
        #
        # Before this guard the attribute access at `active_batch` raised
        # AttributeError; the try/except + finally path re-inserted a
        # malformed cache; and the next step() crashed with `<class 'list'>
        # does not yet support batching with history`, forcing the scheduler
        # to clear the entire paged cache. Every PLD-enabled server hitting
        # a text model corrupted itself within the first few tokens.
        #
        # Short-circuit cleanly here — the retrospective n-gram analyzer in
        # prompt_lookup.py still runs inline on every decode step, so
        # PLD telemetry / theoretical-speedup stats stay accurate; only
        # the batched verify-and-reinsert cycle is gated off.
        if not hasattr(self.batch_generator, "active_batch"):
            return []

        full_tokens = list(request.prompt_token_ids) + list(request.output_token_ids)
        ngram_idx = self._pld_ngram_indices.get(request_id)
        if ngram_idx is None:
            ngram_idx = NgramIndex()
            self._pld_ngram_indices[request_id] = ngram_idx
        drafts = ngram_idx.find_drafts(
            full_tokens, num_draft_tokens=5, max_ngram_size=3
        )
        if not drafts:
            return []

        remaining = request.sampling_params.max_tokens - request.num_output_tokens
        if remaining <= 1:
            return []
        drafts = drafts[: min(len(drafts), remaining - 1)]

        uid = self.request_id_to_uid.get(request_id)
        if uid is None:
            return []

        self._pld_spec_attempts += 1
        old_uid = uid
        kv_cache = None
        removed = False

        try:
            # 1. Peek at forward logprobs BEFORE removing from BatchGenerator.
            #
            #    In BatchGenerator._next():
            #      y, logprobs = batch.y, batch.logprobs  ← OLD tokens/logprobs
            #      batch.y, batch.logprobs = _step(y)     ← NEW (forward) logprobs
            #      response = Response(uid, y[e], logprobs[e], ...)
            #
            #    So response.logprobs = OLD logprobs (the distribution that
            #    generated last_token, argmax==last_token at T=0).
            #    active_batch.logprobs[e] = NEW logprobs = prediction AFTER
            #    last_token — exactly what we need for d0 acceptance.
            ab = self.batch_generator.active_batch
            if ab is None or uid not in ab.uids:
                raise RuntimeError(
                    "uid not in active_batch — cannot get forward logprobs"
                )
            ab_idx = ab.uids.index(uid)
            forward_logprobs = ab.logprobs[ab_idx]

            # 1b. d0 pre-check: avoid the expensive remove/verify/insert cycle
            #     when the first draft token has negligible acceptance probability.
            #     forward_logprobs are already materialized by BatchGenerator.
            if temp <= 1.67e-6:
                # Greedy: exact check — if argmax != d0, acceptance is zero.
                d0_check = int(mx.argmax(forward_logprobs).item())
                if d0_check != drafts[0]:
                    self._pld_win_d0_skip += 1
                    return []
            else:
                # T>0: skip if d0 logprob below threshold.  forward_logprobs
                # are already log(softmax(logits)) — a single index lookup is
                # far cheaper than recomputing softmax over the full vocab.
                # Threshold -2.0 ≈ p>13% at T=1; at T=0.3 the effective
                # probability is higher, so this is conservative.
                _lp_d0 = forward_logprobs[drafts[0]].item()
                if _lp_d0 < -2.0:
                    self._pld_win_d0_skip += 1
                    return []

            # Trim to configured K (dynamic K=3 was tested and regressed —
            # on hybrid models, p(full_accept) drops with K and any miss
            # forces full rewind, so K=2 remains the sweet spot).
            drafts = drafts[: self._pld_num_drafts]

            # 2. Extract KV cache — removes request from BatchGenerator
            cache_dict = self.batch_generator.remove([uid], return_prompt_caches=True)
            removed = True
            kv_cache = cache_dict.get(uid)
            if kv_cache is None:
                raise RuntimeError("remove() returned no cache")

            # 2b. Save ArraysCache state before verification so we can restore
            #     it on partial rejection (hybrid models only).
            #     The slice arrays from extract_cache() are kept alive by
            #     Python's reference counting regardless of batch.filter().
            saved_array_caches: dict = {}
            for i, c in enumerate(kv_cache):
                if not c.is_trimmable():
                    saved_array_caches[i] = list(c.cache)

            # 2. Single batched verification forward pass
            # Input:   [d0, ..., d_{K-1}]  — K tokens (no last_token prefix)
            # Cache:   holds t0...t_N  (last_token already in cache at offset N)
            # forward_logprobs: model's prediction after last_token (from batch gen)
            # logits[i] predicts what comes after d_i:
            #   forward_logprobs  → should equal d0   (prediction after last_token)
            #   logits[0]      → should equal d1   (prediction after d0)
            #   logits[i]      → should equal d_{i+1}
            #   logits[K-1]    → bonus token (free prediction after d_{K-1})
            num_drafts = len(drafts)
            verify_input = mx.array([drafts])  # (1, K)

            with mx.stream(generation_stream):
                logits = self.model(verify_input, cache=kv_cache)
                if temp <= 1.67e-6:
                    # Greedy: argmax forces the full forward pass
                    predicted = mx.argmax(logits[0], axis=-1)  # (K,)
                    mx.eval(predicted)
                else:
                    # Phase 3: evaluate full logits to force the forward pass
                    mx.eval(logits)

            # 3. Accept prefix — greedy (temp≈0) or probabilistic (Phase 3)
            if temp <= 1.67e-6:
                predicted = predicted.tolist()  # length K
                # d0: check forward_logprobs (prediction after last_token)
                d0_predicted = int(mx.argmax(forward_logprobs).item())
                num_accept = 0
                if d0_predicted == drafts[0]:
                    num_accept = 1
                    # d1..d_{K-1}: check logits[0, i-1] (prediction after d_{i-1})
                    for i in range(1, num_drafts):
                        if predicted[i - 1] == drafts[i]:
                            num_accept += 1
                        else:
                            break
                # bonus: prediction at position num_accept
                if num_accept == 0:
                    bonus_token = (
                        d0_predicted  # correction: model's actual pred at pos N
                    )
                else:
                    bonus_token = predicted[
                        num_accept - 1
                    ]  # pred after d_{num_accept-1}

            else:
                # Phase 3: accept d_i with prob p(d_i | context).
                # Scalar log-probability instead of full-vocab softmax:
                # log p_T(d) = logprobs[d]/T - logsumexp(logprobs/T)
                # logsumexp is O(V) but produces a scalar — avoids
                # materializing the full 150K probability vector.
                import math

                _lp_scaled = forward_logprobs / temp
                _log_p_d0 = _lp_scaled[drafts[0]] - mx.logsumexp(_lp_scaled)
                mx.eval(_log_p_d0)

                num_accept = 0
                if random.random() < math.exp(_log_p_d0.item()):
                    num_accept = 1
                    # d1..d_{K-1}: accept from logits[0, i-1], lazy per-position
                    for i in range(1, num_drafts):
                        _lp_i = logits[0, i - 1] / temp
                        _log_p_di = _lp_i[drafts[i]] - mx.logsumexp(_lp_i)
                        if random.random() < math.exp(_log_p_di.item()):
                            num_accept += 1
                        else:
                            break

                # Correction/bonus: sample at the first un-accepted position.
                # On rejection exclude the rejected token so it cannot be
                # re-drawn (preserves the conditional distribution).
                #
                # make_sampler expects log-probabilities (not raw logits) —
                # apply_top_p calls mx.exp() internally.
                sampler = make_sampler(
                    temp=temp,
                    top_p=request.sampling_params.top_p,
                    min_p=request.sampling_params.min_p,
                    top_k=request.sampling_params.top_k,
                )
                if num_accept == 0:
                    # d0 rejected: correction from forward_logprobs, excluding d0
                    bonus_logprobs = mx.where(
                        mx.arange(forward_logprobs.shape[-1]) == drafts[0],
                        mx.full(
                            forward_logprobs.shape,
                            float("-inf"),
                            dtype=forward_logprobs.dtype,
                        ),
                        forward_logprobs,
                    )
                else:
                    # bonus/correction: prediction after d_{num_accept-1}
                    bonus_raw = logits[0, num_accept - 1]
                    bonus_logprobs = bonus_raw - mx.logsumexp(
                        bonus_raw, axis=-1, keepdims=True
                    )
                    if num_accept < num_drafts:
                        rejected_tok = drafts[num_accept]
                        bonus_logprobs = mx.where(
                            mx.arange(bonus_logprobs.shape[-1]) == rejected_tok,
                            mx.full(
                                bonus_logprobs.shape,
                                float("-inf"),
                                dtype=bonus_logprobs.dtype,
                            ),
                            bonus_logprobs,
                        )
                bonus_token = sampler(bonus_logprobs).item()

            # 4. Roll back cache to the accepted prefix.
            #
            #    After the forward pass every layer is advanced by K positions.
            #    Three cases:
            #
            #    a) All K drafts accepted (num_to_trim == 0):
            #       KVCache and ArraysCache are both correctly at N+K.
            #       Nothing to do.
            #
            #    b) Partial/full rejection AND model has ArraysCache layers:
            #       Trimming KVCache works but ArraysCache cannot be rewound.
            #       Restoring ArraysCache to its pre-verification state (N) and
            #       rewinding KVCache by K (back to offset N) keeps both caches
            #       consistent at zero offset.  A correction token is computed
            #       from forward_logprobs and returned to the client.
            #
            #    c) Partial/full rejection, pure KV-cache model:
            #       Trim KVCache by (K - num_accept) positions.  The existing
            #       partial-trim logic applies; ArraysCache restore is skipped.
            #
            #    Standard KVCache grows by concatenation — trim() only adjusts
            #    offset, which update_and_fetch() immediately overwrites with
            #    keys.shape[-2].  We must slice the arrays directly.
            #    QuantizedKVCache uses offset as a write pointer, so setting
            #    offset alone is sufficient.
            num_to_trim = num_drafts - num_accept

            if num_to_trim == 0:
                # Case (a): full accept — both caches consistent, nothing to do.
                pass

            elif saved_array_caches:
                # Case (b): rejection on hybrid model — restore ArraysCache,
                # rewind KVCache to pre-verify offset (N), emit correction token.
                #
                # verify_input = [d0..d_{K-1}] advanced both caches by K steps.
                # Restoring both to N keeps SSM/KV offset at zero.  Accepted
                # drafts (if any) are discarded — we cannot advance KVCache to
                # N+j while rewinding ArraysCache to N.
                #
                # Compute correction at position N (after last_token in cache):
                if temp <= 1.67e-6:
                    correction_token = d0_predicted
                else:
                    cb_logprobs = forward_logprobs
                    if num_accept == 0:
                        cb_logprobs = mx.where(
                            mx.arange(forward_logprobs.shape[-1]) == drafts[0],
                            mx.full(
                                forward_logprobs.shape,
                                float("-inf"),
                                dtype=forward_logprobs.dtype,
                            ),
                            forward_logprobs,
                        )
                    cb_sampler = make_sampler(
                        temp=temp,
                        top_p=request.sampling_params.top_p,
                        min_p=request.sampling_params.min_p,
                        top_k=request.sampling_params.top_k,
                    )
                    correction_token = cb_sampler(cb_logprobs).item()

                for i, c in enumerate(kv_cache):
                    if i in saved_array_caches:
                        c.cache = saved_array_caches[i]
                for c in kv_cache:
                    if not c.is_trimmable() or c.offset == 0:
                        continue
                    pre_verify_offset = max(0, c.offset - num_drafts)  # N+K - K = N
                    if _CacheList is not None and isinstance(c, _CacheList):
                        c.trim(num_drafts)
                        continue
                    if isinstance(c.keys, mx.array):
                        # Numpy roundtrip: materialize before slicing to avoid
                        # Metal command buffer corruption from lazy MLX ops.
                        # bfloat16 → float16 for numpy (no native bf16 support).
                        _kd, _vd = c.keys.dtype, c.values.dtype
                        _ka = (
                            c.keys.astype(mx.float16)
                            if "bfloat16" in str(_kd)
                            else c.keys
                        )
                        _va = (
                            c.values.astype(mx.float16)
                            if "bfloat16" in str(_vd)
                            else c.values
                        )
                        _k, _v = _np.array(_ka), _np.array(_va)
                        c.keys = mx.array(_k[..., :pre_verify_offset, :]).astype(_kd)
                        c.values = mx.array(_v[..., :pre_verify_offset, :]).astype(_vd)
                    c.offset = pre_verify_offset
                    if hasattr(c, "_idx"):  # RotatingKVCache: sync write pointer
                        c._idx = pre_verify_offset
                new_uids = self.batch_generator.insert(
                    [[correction_token]],
                    max_tokens=[max(1, remaining - 1)],
                    caches=[kv_cache],
                )
                removed = False
                new_uid = new_uids[0]
                del self.uid_to_request_id[old_uid]
                self.request_id_to_uid[request_id] = new_uid
                self.uid_to_request_id[new_uid] = request_id
                self._pld_spec_wasted += num_drafts
                self._pld_win_attempts += 1
                self._pld_win_accepted += num_accept
                if num_accept == 0:
                    self._pld_win_zero += 1
                self._pld_win_tokens += 1  # only correction token emitted
                self._pld_maybe_log_summary()
                logger.debug(
                    "[PLD-spec] hybrid partial reject: rewound %d/%d, correction=%d",
                    num_accept,
                    num_drafts,
                    correction_token,
                )
                return [correction_token]

            else:
                # Case (c): rejection on pure KV-cache model — partial trim.
                for c in kv_cache:
                    if not c.is_trimmable() or c.offset == 0:
                        continue
                    accepted_offset = max(0, c.offset - num_to_trim)
                    if _CacheList is not None and isinstance(c, _CacheList):
                        c.trim(num_to_trim)
                        continue
                    if isinstance(c.keys, mx.array):
                        _kd, _vd = c.keys.dtype, c.values.dtype
                        _ka = (
                            c.keys.astype(mx.float16)
                            if "bfloat16" in str(_kd)
                            else c.keys
                        )
                        _va = (
                            c.values.astype(mx.float16)
                            if "bfloat16" in str(_vd)
                            else c.values
                        )
                        _k, _v = _np.array(_ka), _np.array(_va)
                        c.keys = mx.array(_k[..., :accepted_offset, :]).astype(_kd)
                        c.values = mx.array(_v[..., :accepted_offset, :]).astype(_vd)
                    c.offset = accepted_offset
                    if hasattr(c, "_idx"):  # RotatingKVCache: sync write pointer
                        c._idx = accepted_offset

            # 5. Re-insert with bonus token (next to-be-processed token)
            new_remaining = max(1, remaining - num_accept - 1)
            new_uids = self.batch_generator.insert(
                [[bonus_token]],
                max_tokens=[new_remaining],
                caches=[kv_cache],
            )
            removed = False  # re-inserted successfully

            # 6. Update uid maps
            new_uid = new_uids[0]
            del self.uid_to_request_id[old_uid]
            self.request_id_to_uid[request_id] = new_uid
            self.uid_to_request_id[new_uid] = request_id

            # 7. Accumulate stats
            self._pld_spec_accepted += num_accept
            self._pld_spec_wasted += num_drafts - num_accept

            self._pld_win_attempts += 1
            self._pld_win_accepted += num_accept
            if num_accept == num_drafts:
                self._pld_win_full += 1
            if num_accept == 0:
                self._pld_win_zero += 1
            tokens_this_round = num_accept + 1  # accepted drafts + bonus
            self._pld_win_tokens += tokens_this_round
            self._pld_maybe_log_summary()

            extra = list(drafts[:num_accept]) + [bonus_token]
            logger.debug(
                "[PLD-spec] accepted=%d/%d bonus=%d",
                num_accept,
                num_drafts,
                bonus_token,
            )
            return extra

        except Exception as exc:
            logger.warning(
                "[PLD-spec] Failed for %s: %s", request_id, exc, exc_info=False
            )
            return []

        finally:
            # Guarantee: if removed but not re-inserted, do an emergency
            # re-insert so the request is never orphaned in self.running.
            if removed:
                try:
                    cache_arg = [kv_cache] if kv_cache is not None else None
                    em_uids = self.batch_generator.insert(
                        [[last_token]],
                        max_tokens=[max(1, remaining - 1)],
                        caches=cache_arg,
                    )
                    em_uid = em_uids[0]
                    self.uid_to_request_id.pop(old_uid, None)
                    self.request_id_to_uid[request_id] = em_uid
                    self.uid_to_request_id[em_uid] = request_id
                    logger.warning(
                        "[PLD-spec] Emergency re-insert for %s (uid %d→%d)",
                        request_id,
                        old_uid,
                        em_uid,
                    )
                except Exception as em_exc:
                    logger.error(
                        "[PLD-spec] Emergency re-insert failed for %s: %s — "
                        "request may stall",
                        request_id,
                        em_exc,
                    )
                    self.uid_to_request_id.pop(old_uid, None)

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

        # Process deferred aborts FIRST — these are requests where the
        # client disconnected mid-generation. We deferred the
        # batch_generator.remove() call to avoid touching Metal command
        # buffers that were still in-flight. Now that we're at the top
        # of step(), the previous batch_generator.next() has completed
        # and Metal has synchronized, so it's safe to remove.
        if self._pending_aborts:
            self._process_pending_aborts()

        # Schedule waiting requests (errors here propagate immediately —
        # these are logic errors, not cache corruption)
        scheduled = self._schedule_waiting()
        output.scheduled_request_ids = [r.request_id for r in scheduled]
        output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)

        # Run generation step with cache error recovery
        if self.batch_generator is not None and self.running:
            for attempt in range(max_retries + 1):
                try:
                    responses = self.batch_generator.next()
                    output.has_work = True

                    if responses:
                        if isinstance(responses, tuple):
                            # mlx_lm >= 0.31.2 returns
                            # (prompt_responses, generation_responses).
                            # PromptProcessingBatch.Response objects have no
                            # .token and must not drive the request lifecycle.
                            # Only forward generation responses that carry a
                            # .token attribute to _process_batch_responses().
                            flat_responses = []
                            for r in responses:
                                if isinstance(r, list):
                                    flat_responses.extend(r)
                                elif r is not None:
                                    flat_responses.append(r)
                            responses = [
                                r for r in flat_responses if hasattr(r, "token")
                            ]

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

        # ── Deferred SSM re-derive (idle-time processing) ── vmlx#103
        # For thinking models (gen_prompt_len > 0), the SSM companion store
        # queues a re-derive task instead of skipping entirely. We run the
        # re-derive here ONLY when the scheduler is fully idle — no running
        # requests, no waiting requests, and no unprocessed-prefill
        # requests. The forward pass uses the Metal GPU so it can't overlap
        # with active or queued work; without these guards the re-derive
        # could starve queued requests by holding the GPU for a full second
        # prefill while they wait for their first token (vmlx#103).
        # The re-derive runs a separate prefill pass on just the prompt
        # tokens (no thinking/output contamination) and stores the clean
        # SSM state for future prefix cache hits.
        _has_unprocessed = bool(getattr(self, "unprocessed_requests", []))
        _has_waiting = bool(getattr(self, "waiting", []))
        if (
            self._is_hybrid
            and not self.running
            and not _has_waiting
            and not _has_unprocessed
            and hasattr(self, "_ssm_rederive_queue")
            and self._ssm_rederive_queue
            and self._ssm_state_cache is not None
        ):
            # Process ONE task per step to avoid long GPU stalls
            tokens, prompt_len, orig_request_id = self._ssm_rederive_queue.pop(0)
            try:
                logger.info(
                    f"SSM re-derive: running deferred prefill for "
                    f"{orig_request_id} ({prompt_len} prompt tokens, "
                    f"{len(self._ssm_rederive_queue)} remaining in queue)"
                )
                clean_cache = self._prefill_for_prompt_only_cache(tokens)
                if clean_cache is not None:
                    # Extract SSM layers from the clean cache.
                    kv_set = set(self._hybrid_kv_positions or [])
                    ssm_layers = []
                    for layer_idx, c in enumerate(clean_cache):
                        if layer_idx not in kv_set:
                            if hasattr(c, "cache") and isinstance(c.cache, list):
                                from copy import deepcopy
                                import mlx.core as mx

                                cloned = deepcopy(c)
                                cloned.cache = [
                                    mx.contiguous(mx.array(a))
                                    if a is not None
                                    else None
                                    for a in c.cache
                                ]
                                ssm_layers.append(cloned)
                            else:
                                ssm_layers.append(c)
                    if ssm_layers:
                        self._ssm_state_cache.store(tokens, prompt_len, ssm_layers)
                        logger.info(
                            f"SSM re-derive: stored clean companion for "
                            f"{orig_request_id}: {len(ssm_layers)} SSM layers, "
                            f"{prompt_len}-token key (next fetch will hit)"
                        )
                    del clean_cache
                    try:
                        import mlx.core as mx

                        mx.clear_memory_cache()
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"SSM re-derive failed for {orig_request_id}: {e}")

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
            "ewma_ttft_seconds": round(self._ewma_ttft, 3),
        }
        # Include cache stats
        if self.block_aware_cache is not None:
            stats["paged_cache"] = self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            stats["memory_aware_cache"] = self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            stats["prefix_cache"] = self.prefix_cache.get_stats()
        ssm_stats = self._get_ssm_cache_stats()
        if ssm_stats is not None:
            stats["ssm_companion_cache"] = ssm_stats
        return stats

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        base: Optional[Dict[str, Any]] = None
        if self.block_aware_cache is not None:
            base = self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            base = self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            base = self.prefix_cache.get_stats()
        ssm_stats = self._get_ssm_cache_stats()
        if base is not None and ssm_stats is not None:
            base = dict(base)
            base["ssm_companion_cache"] = ssm_stats
        # Surface L2 prompt disk cache (hits / misses / entries / TQ-native counters)
        # under a dedicated sub-key. This is the prompt-level disk cache manager
        # (`DiskCacheManager`), separate from the block-level L2 disk store tracked
        # by paged_cache's own `disk_hits` counter. Before this fix, prompt-level
        # L2 restores worked but weren't visible in `/v1/cache/stats`, so users
        # mistook the counter gap for a functional regression.
        if self.disk_cache is not None:
            try:
                disk_stats = self.disk_cache.stats()
                if base is None:
                    base = {}
                else:
                    base = dict(base)
                base["disk_cache"] = disk_stats
            except Exception:
                pass
        return base

    def _get_ssm_cache_stats(self) -> Optional[Dict[str, Any]]:
        """A3→A1-001: surface SSM companion cache footprint so users see the
        real cache memory cost on hybrid models. Without this, Nemotron 120B
        can silently consume ~32 GB of SSM state beyond the prefix-cache
        budget — a hidden OOM on small-memory machines.

        Returns None if no SSM cache exists. Otherwise reports entries,
        max_entries, and approximate bytes (sum of layer cache nbytes across
        all stored entries — best-effort, since the SSM companion stores
        deepcopied per-layer state).
        """
        cache = getattr(self, "_ssm_state_cache", None)
        if cache is None:
            return None
        try:
            store = getattr(cache, "_store", None)
            entries = len(store) if store is not None else 0
            max_entries = getattr(cache, "_max_entries", None) or getattr(
                cache, "max_entries", 0
            )
            nbytes = 0
            if store is not None:
                for v in store.values():
                    states = v[0] if isinstance(v, tuple) and v else v
                    if isinstance(states, list):
                        for layer in states:
                            arrs = getattr(layer, "cache", None)
                            if isinstance(arrs, list):
                                for a in arrs:
                                    nb = getattr(a, "nbytes", 0)
                                    if isinstance(nb, int):
                                        nbytes += nb

                            # Add sizes for lengths array and legacy state arrays
                            lens = getattr(layer, "lengths", None)
                            if lens is not None:
                                nb = getattr(lens, "nbytes", 0)
                                if isinstance(nb, int):
                                    nbytes += nb

                            state = getattr(layer, "state", None)
                            if isinstance(state, (list, tuple)):
                                for s in state:
                                    nb = getattr(s, "nbytes", 0)
                                    if isinstance(nb, int):
                                        nbytes += nb
            return {
                "entries": entries,
                "max_entries": int(max_entries) if max_entries else 0,
                "nbytes": nbytes,
                "nbytes_mb": round(nbytes / (1024 * 1024), 2),
            }
        except Exception as _e:
            return {"entries": 0, "max_entries": 0, "nbytes": 0, "error": str(_e)}

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
        try:
            self.batch_generator.close()
        except Exception:
            pass
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
        if hasattr(self, "_n_kv_heads_cached"):
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

        # Drop the per-process state machine factory LRU so a subsequent
        # model load doesn't accidentally serve a stale matcher built for
        # the previous model's tokenizer / parser id pair (audit
        # 2026-04-08, ISSUE-A2-002 — was previously dead code).
        try:
            from .state_machine import reset_factory_cache

            reset_factory_cache()
        except Exception:
            pass

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")
