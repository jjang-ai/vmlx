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

from .persistence_outcome import LEDGER as _PERSIST
import logging
import os
import random
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .mllm_scheduler import _resolve_prefix_cache_byte_budget
from .utils.ssm_companion_cache import DEFAULT_SSM_COMPANION_ENTRIES


from mlx_lm.generate import BatchGenerator, generation_stream
from .sampling import make_minimax_m3_sampler, make_sampler
from .block_disk_store import BlockDiskStore
from .disk_cache import DiskCacheManager
from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from .mlx_memory import clear_mlx_memory_cache
from .paged_cache import PagedCacheManager
from .prefix_cache import (
    BlockAwarePrefixCache,
    DSV4_APPEND_SAFE_CHECKPOINT_POLICY,
    PAGED_CACHE_SCHEMA_VERSION,
    PrefixCacheManager,
    build_block_cache_namespace,
    expected_cache_layer_count,
    compute_model_cache_key,
    looped_cache_identity_scope,
    runtime_cache_fingerprint,
)
from .errors import PromptTooLongError
from .prompt_lookup import NgramIndex, find_draft_tokens, pld_stats
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .mllm_batch_generator import HybridSSMStateCache, _fix_hybrid_cache
from .state_machine import SequenceStateMachine, make_state_machine
from .logprobs import format_token_logprobs_for_output as _format_token_logprobs_for_output
from .utils.mamba_cache import (
    ensure_mamba_support,
    register_generation_logprobs,
    unregister_generation_logprobs,
)
from .utils.single_batch_generator import SingleBatchGenerator
from .utils.prefix_hit import (
    disk_prefix_hit_tail_and_cached_tokens as _shared_disk_prefix_hit,
    prefix_hit_tail_and_cached_tokens as _shared_prefix_hit,
)
from .utils.head_dim_detection import (
    choose_supported_kv_group_size,
    detect_cache_head_dims,
)
from .utils.hybrid_tq_cache import is_turboquant_make_cache
from .utils.cache_types import (
    ATTENTION_CACHE_CLASS_NAMES,
    CUMULATIVE_CACHE_CLASS_NAMES,
    describe_runtime_cache_layout,
    expand_cache_class_names,
)
from .utils.ssm_companion_disk_store import SSMCompanionDiskStore
from .utils.cache_extent import logical_truncate_target
from .utils.memory_limits import (
    get_effective_metal_working_set_bytes,
    get_metal_ws_guard_threshold,
)

logger = logging.getLogger(__name__)

def prefix_cache_key_tokens(request: Any) -> list:
    """The token sequence a prefix-cache entry is keyed on.

    Chat templates append assistant-role tokens at the end (e.g.
    ``<|im_start|>assistant\n<think>\n``) which differ on every subsequent
    turn, so including them in the key causes a 100% miss rate in multi-turn
    conversations. The key therefore strips ``_gen_prompt_len`` trailing tokens.

    This existed as three hand-copied copies of the same four lines
    (the disk-store paths and the generic paged store). Keeping ONE copy matters
    beyond tidiness: the boundary this returns is also where a warm turn resumes
    computing, so anything that needs to reproduce a warm turn's arithmetic —
    such as splitting a cold prefill at the same place — has to agree with it
    exactly. A near-miss proxy is not good enough; deriving the width from
    ``_gen_prompt_len`` alone regressed Laguna-S, whose cache covers the whole
    prompt and whose correct width is 0.
    """
    tokens = list(getattr(request, "prompt_token_ids", None) or [])
    gpl = int(getattr(request, "_gen_prompt_len", 0) or 0)
    if 0 < gpl < len(tokens):
        tokens = tokens[:-gpl]
    return tokens



def _call_with_optional_cache_extra(
    method: Callable[..., Any],
    *args: Any,
    cache_extra_keys: Any = None,
    **kwargs: Any,
) -> Any:
    """Preserve legacy cache call shapes unless a discriminator is present."""
    if cache_extra_keys is not None:
        kwargs["cache_extra_keys"] = cache_extra_keys
    return method(*args, **kwargs)


def _append_looped_cache_identity_scope(scope: str, model: Any) -> str:
    looped_scope = looped_cache_identity_scope(model)
    return f"{scope}:{looped_scope}" if looped_scope else scope


def _typed_paged_cache_detail(
    cache_type: str, *, disk_hit: bool, disk_only: bool = False
) -> str:
    """Describe a typed hit without losing its L2 promotion source.

    Mirrors ``_block_cache_detail``'s disk-only handling: with paged RAM
    disabled there is no paged tier to credit, and every hit is by definition
    an SSD restore, so the ``paged+``/``+disk`` decorations would misreport the
    backend on /health, the cache pill, and the logs.
    """
    if disk_only:
        return f"block-disk+{cache_type}"
    base = f"paged+{cache_type}"
    return f"{base}+disk" if disk_hit else base


def _block_cache_detail(paged_cache_manager: Any, *, disk_hit: bool) -> str:
    """Report the effective block backend without calling disk-only RAM paged."""
    if bool(getattr(paged_cache_manager, "disk_only", False)):
        return "block-disk"
    return "paged+disk" if disk_hit else "paged"


def _prompt_snapshot_backend_limit_bytes(
    *,
    memory_aware_cache: Any = None,
    disk_cache: Any = None,
    block_aware_cache: Any = None,
) -> Optional[int]:
    """Return the largest enabled single-entry RAM/L2 snapshot budget.

    DSV4 prompt snapshots are donated to the block-aware paged/L2 hierarchy,
    not only the legacy memory/disk caches.  A zero block-RAM ceiling means
    unbounded resident storage when paged RAM is enabled; a zero disk ceiling
    means an explicitly unbounded L2.  Either unbounded destination removes
    the finite admission cap.  Legacy backend semantics remain unchanged.
    """

    limits: list[int] = []
    unbounded = False

    if memory_aware_cache is not None:
        memory_max = int(
            memory_aware_cache.get_stats().get("max_bytes", 0) or 0
        )
        if memory_max > 0:
            limits.append(int(memory_max * 0.95))

    if disk_cache is not None:
        disk_max = int(getattr(disk_cache, "max_size_bytes", 0) or 0)
        if disk_max <= 0:
            unbounded = True
        else:
            limits.append(disk_max)

    paged = getattr(block_aware_cache, "paged_cache", None)
    if paged is not None:
        if not bool(getattr(paged, "disk_only", False)):
            resident_max = int(
                getattr(paged, "max_resident_bytes", 0) or 0
            )
            if resident_max <= 0:
                unbounded = True
            else:
                limits.append(int(resident_max * 0.95))
        block_disk = getattr(paged, "_disk_store", None)
        if block_disk is not None:
            block_disk_max = int(
                getattr(block_disk, "max_size_bytes", 0) or 0
            )
            if block_disk_max <= 0:
                unbounded = True
            else:
                limits.append(block_disk_max)

    if unbounded:
        return None
    return max(limits) if limits else None


def _m3_vl_cached_prefix_covers_media_tokens(model: Any, request: Any) -> bool:
    """Return True only when restored M3 state contains every media splice.

    A media-salted paged prefix is safe to replay without pixel tensors once
    the cached boundary is strictly past the final image/video placeholder.
    Partial hits that end inside the media span must take the original atomic
    full-prompt vision forward instead.
    """
    cached_tokens = max(0, int(getattr(request, "cached_tokens", 0) or 0))
    prompt_tokens = list(getattr(request, "prompt_token_ids", None) or [])
    if cached_tokens <= 0 or not prompt_tokens:
        return False

    config = getattr(model, "args", None) or getattr(model, "config", None)
    media_token_ids: set[int] = set()
    if getattr(request, "pixel_values", None) is not None:
        media_token_ids.add(int(getattr(config, "image_token_index", 200025)))
    if getattr(request, "pixel_values_videos", None) is not None:
        media_token_ids.add(int(getattr(config, "video_token_index", 200026)))
    if not media_token_ids:
        return False

    media_positions = [
        index for index, token in enumerate(prompt_tokens) if int(token) in media_token_ids
    ]
    return bool(media_positions) and cached_tokens > max(media_positions)

# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption (must be specific to avoid
# matching unrelated errors — e.g., "cache" alone would match any error
# mentioning cache files, directories, or variables).
# Force the generation-prompt strip OFF for every family, not just the
# mixed-SWA / openpangu models that hard-disable it. The strip replays suffix
# tokens from an earlier cache boundary, which is documented to cause small
# distribution drift on MiMo V2.5; this switch is how that can be MEASURED on
# any other strip-active family instead of assumed.
_GEN_PROMPT_STRIP_DISABLED = os.environ.get(
    "VMLX_DISABLE_GEN_PROMPT_STRIP", ""
).strip().lower() in {"1", "true", "yes", "on"}


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
            # Without the original keep/max_size/offset/_idx tuple there is no
            # proof that tensor order is temporal rather than already wrapped
            # circular order. Never invent metadata for a rotating cache.
            return None
        try:
            keep = int(orig_meta[0])
            max_size = int(orig_meta[1])
            offset = int(orig_meta[2])
            idx = int(orig_meta[3])
        except (ValueError, TypeError):
            return None
        if (
            keep < 0
            or max_size <= 0
            or offset < 0
            or keep > offset
            or offset > max_size
            or idx != offset
            or safe_len < keep
            or safe_len > offset
        ):
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


def _align_attention_state_dict(
    state_dict: Dict[str, Any],
    target: int,
) -> Optional[Dict[str, Any]]:
    """Slice one attention KV layer's state to ``target`` tokens.

    Handles the two position-indexed layouts the extractor emits: a plain
    ``(keys, values)`` pair of arrays, and the ``QuantizedKVCache`` form where
    each side is a tuple of component arrays sharing a sequence axis.

    Returns ``None`` when the layer cannot be *proven* alignable -- unknown
    layout, mismatched key/value lengths, a state shorter than ``target``, or a
    rotating buffer that has already wrapped.  Callers must treat that as
    "skip the store" rather than publishing a longer state under a shorter key.
    """
    if int(target) <= 0:
        return None
    state = state_dict.get("state")
    if not (isinstance(state, tuple) and len(state) == 2):
        return None
    keys, values = state
    cls_name = state_dict.get("class_name", "")
    safe = int(target)

    if (
        hasattr(keys, "shape")
        and hasattr(values, "shape")
        and len(keys.shape) in (3, 4)
        and len(values.shape) == len(keys.shape)
    ):
        seq_dim = 2 if len(keys.shape) == 4 else 1
        if int(keys.shape[seq_dim]) != int(values.shape[seq_dim]):
            return None
        if int(keys.shape[seq_dim]) < safe:
            return None
        if len(keys.shape) == 4:
            keys = keys[:, :, :safe, :]
            values = values[:, :, :safe, :]
        else:
            keys = keys[:, :safe, :]
            values = values[:, :safe, :]
    elif (
        isinstance(keys, (tuple, list))
        and isinstance(values, (tuple, list))
        and len(keys) >= 1
        and len(values) == len(keys)
    ):
        lengths = []
        for key_part, value_part in zip(keys, values):
            if (
                not hasattr(key_part, "shape")
                or not hasattr(value_part, "shape")
                or len(key_part.shape) < 2
                or len(value_part.shape) < 2
            ):
                return None
            if int(key_part.shape[-2]) != int(value_part.shape[-2]):
                return None
            lengths.append(int(key_part.shape[-2]))
        if not lengths or len(set(lengths)) != 1 or lengths[0] < safe:
            return None
        keys = tuple(t[..., :safe, :] for t in keys)
        values = tuple(t[..., :safe, :] for t in values)
    else:
        return None

    new_meta = _rebuild_meta_state_after_truncation(
        cls_name, state_dict.get("meta_state", ()), safe
    )
    if new_meta is None:
        return None
    return {**state_dict, "state": (keys, values), "meta_state": new_meta}


def _is_recognized_attention_layout(state: Any) -> bool:
    """Whether ``state`` is one of the KV layouts the store knows how to slice.

    Only used to tell two failures apart in the log: a layout we understand but
    refused to align (a wrapped rotating buffer -- expected), versus a layout we
    do not recognise at all (a real gap worth chasing). Collapsing both into one
    "unknown format" warning would have hidden the second behind the first.
    """
    if not (isinstance(state, tuple) and len(state) == 2):
        return False
    keys, values = state
    if (
        hasattr(keys, "shape")
        and hasattr(values, "shape")
        and len(keys.shape) in (3, 4)
        and len(values.shape) == len(keys.shape)
    ):
        return True
    return (
        isinstance(keys, (tuple, list))
        and isinstance(values, (tuple, list))
        and len(keys) >= 1
        and len(values) == len(keys)
        and all(hasattr(t, "shape") and len(t.shape) >= 2 for t in keys)
        and all(hasattr(t, "shape") and len(t.shape) >= 2 for t in values)
    )


def _align_cache_list_state_dict(
    state_dict: Dict[str, Any],
    target: int,
) -> Optional[Dict[str, Any]]:
    """Align every sub-cache of an extracted ``CacheList`` wrapper to ``target``.

    A ``CacheList`` layer carries ``state=None`` plus real nested payloads in
    ``sub_caches``, so passing the wrapper through unchanged would store the
    longer nested states under the shortened key.  This walks into them and
    applies the same per-layer rule the flat path uses.

    Families this reaches (all of which build one ``CacheList`` per layer, so
    every layer hits this path):
      - ``CacheList(KVCache(), KVCache())`` -- deepseek_v32, longcat_flash,
        longcat_flash_ngram.  Both subs are position-indexed and align.
      - ``CacheList(ArraysCache(...), KVCache())`` -- falcon_h1, baichuan_m1.
        The cumulative sub cannot be sliced to a token boundary, so this
        returns ``None`` and names the offending class for the caller's log.

    Returns ``None`` when any sub-cache cannot be proven alignable.
    """
    if int(target) <= 0:
        return None
    subs = state_dict.get("sub_caches")
    if not isinstance(subs, (list, tuple)) or not subs:
        return None

    aligned_subs = []
    for sub in subs:
        if not isinstance(sub, dict):
            return None
        sub_cls = sub.get("class_name", "")
        sub_state = sub.get("state")
        if sub_cls in CUMULATIVE_CACHE_CLASS_NAMES:
            # Cumulative state under a shortened key would replay a prefix whose
            # recurrent state ran past it. The flat path can drop these because
            # the companion re-derives them; a nested one has no companion slot,
            # so refuse the whole layer rather than store a half-valid CacheList.
            return None
        if sub_state is None or sub_state == ():
            # A proven no-state placeholder encodes no token position.
            aligned_subs.append(sub)
            continue
        aligned = _align_attention_state_dict(sub, target)
        if aligned is None:
            return None
        aligned_subs.append(aligned)

    return {**state_dict, "sub_caches": aligned_subs}


def _blocking_cache_list_sub_class(state_dict: Dict[str, Any]) -> str:
    """Name the sub-cache class that made a ``CacheList`` unalignable.

    Purely for diagnostics: a store that silently declines is indistinguishable
    from a family that has no cache, which is exactly how falcon_h1 went a full
    campaign with zero prefix reuse and no obvious signal.
    """
    subs = state_dict.get("sub_caches")
    if not isinstance(subs, (list, tuple)):
        return "<no sub_caches>"
    for sub in subs:
        if isinstance(sub, dict) and sub.get("class_name") in CUMULATIVE_CACHE_CLASS_NAMES:
            return str(sub.get("class_name"))
    return ",".join(
        str(sub.get("class_name", "?")) if isinstance(sub, dict) else type(sub).__name__
        for sub in subs
    )


def _truncate_minimax_m3_state_dict(
    state_dict: Dict[str, Any],
    target: int,
) -> Optional[Dict[str, Any]]:
    """Slice all three MiniMax-M3 MSA cache lanes to ``target`` tokens.

    MiniMax-M3 sparse layers persist ``(keys, values, idx_keys)``.  The paged
    store removes the chat template's generation trailer from its token key;
    leaving this three-tensor state untouched stores more positional cache
    entries than the key represents.  A later prefix hit then replays the tool
    result/user tail at the wrong positions and can continue an older task.

    Return ``None`` rather than storing a partial/malformed MSA state.  The
    caller treats that as a cache-store miss and the next request prefills
    safely.
    """
    if state_dict.get("class_name") != "MiniMaxM3SparseCache":
        return None
    state = state_dict.get("state")
    if not (isinstance(state, (tuple, list)) and len(state) == 3):
        return None
    keys, values, idx_keys = state
    if any(value is None or not hasattr(value, "shape") for value in state):
        return None
    if any(len(value.shape) != 4 for value in state):
        return None
    lengths = (
        int(keys.shape[2]),
        int(values.shape[2]),
        int(idx_keys.shape[2]),
    )
    if int(target) <= 0 or len(set(lengths)) != 1 or lengths[0] < int(target):
        return None
    safe = int(target)
    meta = _rebuild_meta_state_after_truncation(
        "MiniMaxM3SparseCache",
        state_dict.get("meta_state", ()),
        safe,
    )
    if meta is None:
        return None
    return {
        **state_dict,
        "state": (
            keys[:, :, :safe, :],
            values[:, :, :safe, :],
            idx_keys[:, :, :safe, :],
        ),
        "meta_state": meta,
    }


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based


# Queue SSM re-derive for every non-empty paged KV store. A higher threshold
# made short hybrid prompts write KV blocks that could never become full
# KV+SSM hits, so repeated Ling/Bailing requests always re-prefilled despite
# a paged cache hit. The companion cache LRU and queue cap bound memory.
SSM_REDERIVE_MIN_TOKENS = 1
SSM_REDERIVE_QUEUE_CAP = 8


class IdleTaskResult(Enum):
    """Outcome of one idle-maintenance task invocation (vmlx#245).

    DONE dequeues the task; PARKED re-queues it at the front so it retries at
    the next idle iteration (used when foreground work arrives mid-task).
    """

    DONE = "done"
    PARKED = "parked"


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 1
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings
    prefill_batch_size: int = 512
    completion_batch_size: int = 512
    prefill_step_size: int = 2048

    # Multimodal processor-output RAM LRU. Production defaults off: entries
    # retain pixel/video tensors and grew active MLX memory once per media turn
    # even in the SSD-only cache profile. This is separate from prefix-cache
    # RAM and must be explicitly opted into for diagnostics.
    enable_vision_cache: bool = False
    vision_cache_size: int = 16

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
    cache_memory_mb: Optional[int] = None  # None = auto-detect (20% of available RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting
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
    kv_cache_quantization_explicit: bool = False

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

    # SSM companion cache budget for hybrid (Mamba/GatedDelta) models. Mirrors
    # `MLLMSchedulerConfig`. Entries can be tens/hundreds of MB, so production
    # defaults are deliberately conservative and byte-bound.
    ssm_state_cache_size: int = DEFAULT_SSM_COMPANION_ENTRIES
    ssm_state_cache_max_mb: Optional[int] = 0

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

        # Loader thread executor — every step()/BatchGenerator call must
        # run on the SAME thread that loaded the model, because MLX Metal
        # streams are thread-local. JANGTQ DSV4-Flash (and other JANGTQ
        # bundles) allocate kernels on the loader thread; running step()
        # on the asyncio event-loop thread crashes with
        # `RuntimeError: There is no Stream(gpu, N) in current thread.`
        # MLLMScheduler has had this since 2026-04-25 (see _step_executor
        # there). Engine_core._engine_loop reads this attribute and, when
        # present, dispatches scheduler.step() through it.
        self._step_executor = self.config.step_executor

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, Request] = {}  # Running requests by ID
        self.requests: Dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished
        self._pending_aborts: Set[str] = set()  # Deferred aborts (processed in step())
        # Aborted requests eligible for prompt-snapshot salvage into the
        # prefix cache (processed alongside _pending_aborts in step()).
        self._abort_salvage_requests: Dict[str, Request] = {}

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
        self._ttft_sample_count: int = 0
        self._ttft_alpha: float = 0.1
        # SSD-only prefix admission must compare restoration with recomputing
        # the reusable prefix.  These rates are learned from this exact loaded
        # model/runtime; cache bytes per token and prefill speed differ too much
        # across native KV, mixed-SWA, compressed, sparse, and typed caches for
        # one family-blind throughput constant to be safe.
        self._cache_admission_prefill_seconds_per_token: float = 0.0
        self._cache_admission_prefill_sample_count: int = 0
        self._cache_admission_prefill_reference_tokens: int = 0
        self._cache_admission_disk_seconds_per_token: float = 0.0
        self._cache_admission_disk_sample_count: int = 0
        self._cache_admission_disk_reference_tokens: int = 0
        self._cache_admission_alpha: float = 0.25
        self._cache_admission_max_comparable_ratio: float = 2.0

        self._model_type_for_runtime = self._detect_model_type_for_runtime(model)
        self._uses_openpangu_cache = self._model_type_for_runtime == "openpangu_v2"
        self._uses_glm5_next_cache = (
            self._model_type_for_runtime in ("glm5_next", "glm5_next_text")
        )
        self._glm5_next_cache_unsupported = False
        self._prefix_cache_requested = bool(self.config.enable_prefix_cache)
        self._prompt_disk_cache_requested = bool(self.config.enable_disk_cache)
        self._block_disk_cache_requested = bool(self.config.enable_block_disk_cache)
        if self._uses_glm5_next_cache:
            # GLM's 34 KDA layers carry cumulative convolution/recurrent state,
            # while its 11 MLA layers carry positional KV plus packed DSA and
            # compressed pool history. Arbitrary paged blocks cannot rebuild
            # that mixed boundary. Use one immutable N-1 typed snapshot and the
            # prompt-disk codec; generic block/TQ routes remain fail-closed.
            if self.config.use_paged_cache or self.config.enable_block_disk_cache:
                logger.warning(
                    "glm5_next native cache does not support generic paged/block "
                    "reuse; disabling those lanes while preserving exact typed "
                    "memory/prompt-disk reuse"
                )
            self.config.use_paged_cache = False
            self.config.enable_block_disk_cache = False
            self.config.use_memory_aware_cache = True
            if self.config.enable_prefix_cache:
                logger.info(
                    "glm5_next exact typed prefix cache enabled: N-1 "
                    "snapshots round-trip KDA conv/recurrent and MLA/DSA state"
                )
        if self._uses_openpangu_cache:
            # OpenPanguV2LayerCache contains path-dependent convolution state in
            # addition to MLA latent KV, DSA indexer, and rotating SWA state.
            # Exact N-1 prompt-boundary snapshots now have a typed clone/disk
            # codec. Generic block paging is still unsafe because an arbitrary
            # block cannot reconstruct cumulative conv state. Force the exact
            # memory-aware lane while preserving the user's prefix + prompt-L2
            # choices, and fail closed on paged/block L2.
            if self.config.use_paged_cache or self.config.enable_block_disk_cache:
                logger.warning(
                    "openpangu_v2 exact typed cache does not support generic "
                    "paged/block reuse; disabling paged and block-disk while "
                    "preserving exact memory/prompt-disk reuse"
                )
            self.config.use_paged_cache = False
            self.config.enable_block_disk_cache = False
            self.config.use_memory_aware_cache = True
            if self.config.enable_prefix_cache:
                logger.info(
                    "openpangu_v2 exact typed prefix cache enabled: N-1 snapshots "
                    "round-trip MLA KV, DSA indexer, rotating SWA metadata, and "
                    "all causal-conv states; reverse truncation remains disabled"
                )
        # Track if model uses mixed cache types. DSV4's DeepseekV4Cache and
        # ZAYA's CCA CacheList are first-class typed cache contracts, not SSM
        # companion-cache rows.
        self._uses_dsv4_cache = self._model_uses_dsv4_cache(model)
        self._uses_zaya_cache = (
            self._model_type_for_runtime == "zaya"
            or self._model_uses_zaya_cache(model)
        )
        self._is_hybrid = self._is_hybrid_model(model)
        self._uses_m3_msa_cache = self._model_uses_m3_msa_cache(model)
        # Do not silently widen repetition-penalty history for families whose
        # chat templates contain EOS/turn-boundary sentinels in the prompt.
        # A 512-token lookback penalizes legitimate stop tokens on MiniMax/Ling
        # multi-turn chats and can create the continuation loops it was meant
        # to prevent.
        self._long_repetition_context = self._model_type_for_runtime in {
            "deepseek_v4",
        }
        _model_make_cache = getattr(model, "make_cache", None)
        self._tq_active = bool(
            _model_make_cache and is_turboquant_make_cache(_model_make_cache)
        )
        self._hybrid_live_tq_policy = getattr(
            _model_make_cache, "_vmlx_hybrid_tq_policy", None
        )
        self._hybrid_live_tq_attention_layers = list(
            getattr(_model_make_cache, "_vmlx_hybrid_tq_attention_layers", ()) or []
        )
        self._hybrid_live_tq_companion_layers = list(
            getattr(_model_make_cache, "_vmlx_hybrid_tq_companion_layers", ()) or []
        )
        # compress_after gates whether attention-KV TQ actually live-encodes
        # during decode (0 => objects-only, no live encode). Capture it so
        # capabilities can report the truthful mode instead of assuming decode.
        self._hybrid_live_tq_compress_after = int(
            getattr(_model_make_cache, "_vmlx_tq_compress_after", 0) or 0
        )
        self._hybrid_tq_auto_policy = getattr(
            _model_make_cache, "_vmlx_tq_auto_policy", None
        )
        self._hybrid_tq_default_key_bits = int(
            getattr(_model_make_cache, "_vmlx_tq_default_key_bits", 0) or 0
        )
        self._hybrid_tq_default_value_bits = int(
            getattr(_model_make_cache, "_vmlx_tq_default_value_bits", 0) or 0
        )
        self._tq_batch_api = self._turboquant_cache_supports_batch_api(model)
        self._log_runtime_cache_contract(model)
        if self._tq_active and self._tq_batch_api:
            logger.info(
                "TurboQuantKVCache live decode preserving configured batching "
                "(batch cache API=turboquant_kv_v1)."
            )
        elif self._tq_active:
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
                    "TurboQuantKVCache live decode is single-sequence only "
                    "with this jang_tools build (missing batch API "
                    "turboquant_kv_v1); overriding %s.",
                    ", ".join(changed),
                )

        # Single-batch-native cache families (sweep 2026-07-05): openPangu's
        # OpenPanguV2LayerCache carries path-dependent conv states and Zaya's
        # CCA conv_state is shaped for batch=1 — feeding either a 2-sequence
        # batch corrupts or crashes the decode step ("[reshape] Cannot
        # reshape array of size 128 into shape (1,1,1,64)" aborting all
        # in-flight requests on Zaya; openPangu previously aborted earlier at
        # the missing filter()). Mirror the TurboQuant guard above: force
        # serial scheduling so concurrent clients queue instead of crashing.
        #
        # MiniMax-M3 (MiniMaxM3SparseCache) is the same class of hazard: its
        # attention derives positions from a SCALAR cache.offset
        # (`mx.arange(offset, offset + Sq)`) and the Lightning-Indexer's
        # `isinstance(cache, MiniMaxM3SparseCache)` guard only fires on the
        # native single-batch cache. Under BatchGenerator the per-sequence
        # BatchKVCache makes `offset` an array, so `mx.arange` raises
        # `TypeError: arange(): incompatible function arguments` and the engine
        # loop aborts EVERY in-flight request (live-repro'd 2026-07-08 with two
        # concurrent >2048-token requests at --max-num-seqs 2). It only stays
        # correct because max_num_seqs defaults to 1; pin it like the others so
        # a user-set --max-num-seqs>1 queues serially instead of DoS-ing.
        _single_batch_native_family = (
            self._model_type_for_runtime
            in {"glm5_next", "glm5_next_text", "openpangu_v2", "zaya"}
            or self._uses_zaya_cache
            or self._uses_m3_msa_cache
        )
        if _single_batch_native_family:
            changed = []
            if self.config.max_num_seqs != 1:
                changed.append(f"max_num_seqs {self.config.max_num_seqs}->1")
                self.config.max_num_seqs = 1
            if self.config.prefill_batch_size != 1:
                changed.append(
                    f"prefill_batch_size {self.config.prefill_batch_size}->1"
                )
                self.config.prefill_batch_size = 1
            if self.config.completion_batch_size != 1:
                changed.append(
                    f"completion_batch_size {self.config.completion_batch_size}->1"
                )
                self.config.completion_batch_size = 1
            if changed:
                logger.warning(
                    "%s cache is single-sequence native (path-dependent "
                    "conv/CCA state); overriding %s so concurrent requests "
                    "queue serially instead of corrupting the batch.",
                    self._model_type_for_runtime,
                    ", ".join(changed),
                )

        # mlxstudio#138: surface the precedence when both knobs are set.
        # Without VMLX_DISABLE_TQ_KV, the loader's TQ patch wins because
        # BatchGenerator runs `model.make_cache()` and gets TurboQuantKVCache;
        # the scheduler's q4/q8 wrap below still installs but only patches
        # `QuantizedKVCache.size()` upstream — it never wraps a TQ cache.
        # Make the precedence visible so the user knows their flag is
        # being shadowed and how to opt out.
        if self._tq_active and self.config.kv_cache_quantization != "none":
            if getattr(self.config, "kv_cache_quantization_explicit", False):
                logger.warning(
                    f"--kv-cache-quantization='{self.config.kv_cache_quantization}' "
                    f"requested but jang_config.turboquant.enabled=true is in effect "
                    f"(TurboQuantKVCache patched make_cache). The bundle's calibrated "
                    f"TQ takes precedence; your flag will not change live-decode KV "
                    f"bit-width. Set VMLX_DISABLE_TQ_KV=1 to skip TQ and let q4/q8 "
                    f"take effect."
                )
            else:
                logger.info(
                    "KV cache auto mode: TurboQuantKVCache is active for live decode; "
                    f"stored prefix/cache snapshots still use "
                    f"{self.config.kv_cache_quantization} when applicable."
                )

        if (
            self._uses_m3_msa_cache
            and self.config.kv_cache_quantization != "none"
        ):
            logger.info(
                "MiniMax-M3 MSA cache (MiniMaxM3SparseCache, Lightning-Indexer "
                "idx_keys) detected — disabling generic KV cache quantization "
                "(was: %s). The append-only MSA dual-cache is structurally "
                "incompatible with q4/q8 quantized KV (SDPA mask dtype must "
                "promote to bfloat16).",
                self.config.kv_cache_quantization,
            )
            self.config.kv_cache_quantization = "none"
        if (
            self._uses_zaya_cache
            and self.config.kv_cache_quantization != "none"
        ):
            logger.info(
                "ZAYA/CCA typed cache detected — disabling generic KV cache "
                "quantization (was: %s). ZAYA CCA can only quantize the "
                "standard KV subcache after a typed partial codec proves "
                "conv_state/prev_hs parity.",
                self.config.kv_cache_quantization,
            )
            self.config.kv_cache_quantization = "none"
        elif (
            self._is_hybrid
            and not self._uses_dsv4_cache
            and not self._uses_zaya_cache
            and self.config.kv_cache_quantization != "none"
        ):
            if (
                self._tq_active
                and self._hybrid_live_tq_policy == "attention_kv_only"
            ):
                logger.info(
                    "Hybrid/path-dependent cache model detected — live "
                    "TurboQuant KV active for attention KVCache layers; q4/q8 "
                    "remains storage-boundary quantization for prefix/paged/L2. "
                    "Hybrid non-KV state is preserved full precision and "
                    "restored through the SSM companion cache or clean-prefill "
                    "rederive (stored_kv_quantization=%s).",
                    self.config.kv_cache_quantization,
                )
            else:
                logger.info(
                    "Hybrid/path-dependent cache model detected — using q4/q8 only at cache storage boundaries "
                    "(attention KVCache layers). Hybrid non-KV state is preserved full precision "
                    "and restored through the SSM companion cache or clean-prefill rederive "
                    "(stored_kv_quantization=%s).",
                    self.config.kv_cache_quantization,
                )

        # Mixed-attention models (Gemma 4 = sliding + full) require preserving
        # RotatingKVCache metadata through truncation, paged blocks, and L2
        # restore. The old blanket prefix-cache bypass hid that path entirely;
        # keep detection only as a diagnostic now that RotatingKVCache-aware
        # truncation/reconstruction is implemented.
        self._mixed_attention_cache_model = False
        try:
            self._mixed_attention_cache_model = self._model_has_mixed_attention(model)
            if self._mixed_attention_cache_model:
                logger.info(
                    "LLM mixed-attention model detected (e.g. Gemma 4 sliding+full). "
                    "Prefix cache remains enabled; RotatingKVCache metadata will be "
                    "preserved during truncation and paged/L2 reconstruction."
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
        if (
            self._is_hybrid
            and not self._uses_dsv4_cache
            and not self._uses_zaya_cache
            and self.config.enable_prefix_cache
            and hasattr(model, "make_cache")
        ):
            try:
                from mlx_lm.models.cache import KVCache as _KVC

                _template = model.make_cache()
                self._hybrid_num_layers = len(_template)
                # The companion splits layers into "attention" and "SSM" and
                # restores each from a different lane, so it can only describe a
                # model where a layer is one or the other. A parallel hybrid --
                # falcon_h1 and baichuan_m1 build CacheList(ArraysCache, KVCache)
                # for EVERY layer -- violates that: each layer is both at once.
                _kv_positions: List[int] = []
                _parallel_hybrid_layers: List[int] = []
                for _i, _t in enumerate(_template):
                    _names = expand_cache_class_names([_t])
                    _has_kv = bool(_names & ATTENTION_CACHE_CLASS_NAMES)
                    _has_cumulative = bool(_names & CUMULATIVE_CACHE_CLASS_NAMES)
                    if _has_kv and _has_cumulative:
                        _parallel_hybrid_layers.append(_i)
                    elif _has_kv:
                        _kv_positions.append(_i)
                if _parallel_hybrid_layers:
                    # Leaving _ssm_state_cache as None keeps every companion path
                    # disabled (they all guard on it). That matters: an empty
                    # kv_positions list would make the restore loop treat EVERY
                    # layer as SSM and overwrite it, turning "no prefix cache"
                    # into a corrupted one.
                    self._hybrid_kv_positions = None
                    logger.warning(
                        "Hybrid SSM companion disabled: %d/%d layers carry "
                        "attention and cumulative state together (parallel "
                        "hybrid), which the per-layer companion split cannot "
                        "represent. Prefix reuse stays off for this model "
                        "rather than restoring mismatched layers.",
                        len(_parallel_hybrid_layers),
                        self._hybrid_num_layers,
                    )
                else:
                    self._hybrid_kv_positions = _kv_positions
                    # Honour SchedulerConfig budgets. The old default of 50
                    # entries let hybrid users grow several GB of resident SSM
                    # state on short unique prompts; entry count alone is not
                    # enough because entry size scales with architecture and
                    # prompt.
                    _ssm_cache_size = max(
                        0,
                        int(
                            getattr(
                                self.config,
                                "ssm_state_cache_size",
                                DEFAULT_SSM_COMPANION_ENTRIES,
                            )
                        ),
                    )
                    _ssm_cache_max_mb = getattr(
                        self.config, "ssm_state_cache_max_mb", 0
                    )
                    _ssm_model_key = compute_model_cache_key(
                        model,
                        model_path=self.config.model_path,
                        smelt_enabled=self.config.smelt_enabled,
                        smelt_pct=self.config.smelt_pct,
                        tq_enabled=bool(self._tq_active),
                        kv_quant_bits=(
                            4 if self.config.kv_cache_quantization == "q4"
                            else 8 if self.config.kv_cache_quantization == "q8"
                            else 0
                        ),
                    )
                    self._ssm_state_cache = HybridSSMStateCache(
                        max_entries=_ssm_cache_size,
                        model_key=_ssm_model_key,
                        max_bytes=(
                            int(_ssm_cache_max_mb) * 1024 * 1024
                            if _ssm_cache_max_mb is not None
                            else None
                        ),
                    )
                    logger.info(
                        f"Hybrid SSM cache: {len(self._hybrid_kv_positions)}/"
                        f"{self._hybrid_num_layers} KV layers, "
                        f"SSM companion enabled (entries={_ssm_cache_size}, "
                        f"max_mb={_ssm_cache_max_mb}, "
                        f"model_key={_ssm_model_key[:12]})"
                    )
            except Exception as _e:
                logger.warning(f"Failed to init hybrid SSM cache layout: {_e}")
        elif self._uses_zaya_cache:
            if self.config.enable_prefix_cache and self.config.use_paged_cache:
                logger.info(
                    "ZAYA/CCA typed paged prefix cache enabled — terminal "
                    "blocks store CCA conv_state + prev_hs and block disk L2 "
                    "uses zaya_cca_v1 serialization. Generic TurboQuant KV "
                    "remains disabled for this family."
                )
            else:
                logger.info(
                    "ZAYA/CCA cache contract detected but prefix cache is disabled; "
                    "CCA prompt-state lookup/store/re-derive is disabled for this run"
                )
        elif self._is_hybrid and not self._uses_dsv4_cache:
            logger.info(
                "Hybrid SSM cache detected but prefix cache is disabled; "
                "SSM companion lookup/store/re-derive is disabled for this run"
            )

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

        # Auto-detect hybrid models (MambaCache + KVCache). They cannot use the
        # legacy memory-aware cache, but Block Disk L2 is now a complete block
        # backend even when paged RAM is explicitly disabled: attention KV is
        # restored from SSD and full-precision SSM state comes from the typed
        # companion L2 or clean-prefill rederive.
        # In-RAM paged cache is OFF for every family: SSD block-disk L2 is the
        # only cache tier this product ships. Neither of the two cases below may
        # switch a RAM tier back on. Where an architecture genuinely cannot be
        # served from L2, the answer is NO REUSE -- never a silent RAM tier and
        # never an unsafe restore.
        if (
            self.config.enable_prefix_cache
            and not self.config.use_paged_cache
            and self._uses_zaya_cache
        ):
            # zaya_cca_v1 records need KV + conv_state + prev_hs together. The
            # memory-aware and legacy lanes cannot hold that, and block-disk-only
            # is not wired for the CCA contract either. Disabling ONLY the
            # memory-aware lane is not enough: the init chain then falls through
            # to the LEGACY PrefixCacheManager, which is just as unable to carry
            # CCA state. Take no prefix lane at all rather than an unsafe one —
            # ZAYA re-prefills cleanly. Costs speed, never correctness.
            logger.warning(
                "ZAYA/CCA typed cache has no safe prefix lane without paged "
                "RAM, and paged RAM is OFF for every family (SSD block-disk L2 "
                "is the only tier). Disabling prefix reuse for ZAYA entirely: "
                "every turn re-prefills cleanly rather than restore incomplete "
                "conv_state/prev_hs. Wire the CCA contract into the block-disk "
                "lane to get reuse back."
            )
            self.config.use_memory_aware_cache = False
            self.config.enable_prefix_cache = False
            self.config.enable_block_disk_cache = False
        elif (
            self.config.enable_prefix_cache
            and not self.config.use_paged_cache
            and self.config.use_memory_aware_cache
            and self._is_hybrid
            and not self.config.enable_block_disk_cache
        ):
            # Hybrid MambaCache/GatedDelta state is path-dependent; the
            # memory-aware lane would reuse it incorrectly. With Block Disk L2
            # switched off there is no correct backend, so take the reuse loss.
            logger.info(
                "Non-standard cache model detected (MambaCache/hybrid layers) "
                "with Block Disk L2 disabled. Paged RAM stays OFF (SSD L2 is "
                "the only tier); disabling the memory-aware lane. Enable "
                "--enable-block-disk-cache to get hybrid prefix reuse back."
            )
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
        # Symmetric with MLLMScheduler._detect_mla — walks model +
        # language_model + inner .model wrappers, inspects args/config/
        # text_config + _raw_config, so wrapped models (Kimi K2.6 mlx_vlm
        # around DeepseekV3, glm_moe_dsa inheriting deepseek_v32, mistral4
        # text wrappers) are caught the same way the LLM scheduler's other
        # cache-record validator already does.
        _is_mla = self._detect_mla()
        # User opt-in override for non-DSV4 MLA auto-disable. DeepSeek V3 /
        # Mistral 4 stash compressed latents in KV — quantizing them again
        # is double-lossy and harms quality. DSV4 is stricter: its native
        # DeepseekV4Cache already owns the heterogeneous SWA + CSA/HCA cache
        # compression contract (optionally PoolQuantizedV4Cache for compressed
        # pools), so generic q4/q8 prefix-cache quantization is always forced
        # off for DSV4.
        _mla_kvq_env = os.environ.get("VMLX_ALLOW_MLA_KV_QUANT")
        _allow_mla_kvq = _mla_kvq_env in ("1", "true", "True", "yes", "on")
        _is_dsv4_composite = self._uses_dsv4_cache
        if (
            (self._uses_openpangu_cache or self._uses_glm5_next_cache)
            and self.config.kv_cache_quantization != "none"
        ):
            logger.info(
                "%s native mixed cache detected — forcing generic KV cache "
                "quantization off (was: %s). MLA latents and path-dependent conv "
                "state must round-trip through the typed full-precision codec.",
                "glm5_next" if self._uses_glm5_next_cache else "openPangu",
                self.config.kv_cache_quantization,
            )
            self.config.kv_cache_quantization = "none"
        elif (
            self.config.kv_cache_quantization != "none"
            and _is_dsv4_composite
        ):
            logger.info(
                "DSV4 composite cache detected — forcing generic KV cache "
                "quantization off (was: %s). DSV4 uses native "
                "DeepseekV4Cache SWA + CSA/HCA state; optional "
                "DSV4_POOL_QUANT compresses the CSA/HCA pools without "
                "replacing the composite cache with QuantizedKVCache.",
                self.config.kv_cache_quantization,
            )
            self.config.kv_cache_quantization = "none"
        elif (
            self.config.kv_cache_quantization != "none"
            and _is_mla
            and not _is_dsv4_composite
            and not _allow_mla_kvq
        ):
            logger.info(
                f"MLA model detected (kv_lora_rank > 0) — disabling KV cache quantization "
                f"(was: {self.config.kv_cache_quantization}). MLA stores compressed latents "
                f"that should not be further quantized. Set VMLX_ALLOW_MLA_KV_QUANT=1 to "
                f"override if you accept the quality risk."
            )
            self.config.kv_cache_quantization = "none"
        elif (
            self.config.kv_cache_quantization != "none"
            and _is_mla
            and not _is_dsv4_composite
            and _allow_mla_kvq
        ):
            logger.warning(
                f"MLA model + KV cache quantization='{self.config.kv_cache_quantization}' "
                f"requested via VMLX_ALLOW_MLA_KV_QUANT=1 — running double-lossy KV quant "
                f"on compressed latents. Expect some output drift; turn off if quality matters."
            )
        if (
            self._tq_active
            and not getattr(self.config, "kv_cache_quantization_explicit", False)
        ):
            # Auto mode uses the model's native TurboQuant codec at prefix/paged
            # storage boundaries. Applying generic q4 first would double-quantize
            # the same attention KV and erase the per-layer TQ seed/bit policy.
            logger.info(
                "KV cache auto mode: native TurboQuant owns stored attention KV; "
                "generic q4/q8 prefix quantization is suppressed"
            )
        elif self.config.kv_cache_quantization != "none":
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

        if self._uses_dsv4_cache and self.config.use_paged_cache:
            # DSV4's native delta records are cut at a HARD-CODED 256 tokens
            # (DSV4_NATIVE_BLOCK_SIZE); the store cutter walks the CONFIGURED
            # block size and demands an exact interval match per block. Any
            # other block size makes the very first lookup raise
            # "DSV4 block transport has no interval (0, N)" and EVERY store of
            # EVERY request aborts — 100% store-dead, surfaced only as a
            # per-request warning that names no cause.
            #
            # The 256 force used to live only in cli.serve_command, inside a
            # try whose except logs at DEBUG. Generator selection is a
            # SEPARATE, class-based detection, so a registry lookup that threw
            # left a 64-block manager driving native-256 records. This is the
            # opt-out-equals-failed-detection shape, and the cure is to
            # reconcile where every launcher converges rather than in one
            # command function. Advise and correct; never refuse.
            try:
                from .utils.dsv4_batch_generator import DSV4_NATIVE_BLOCK_SIZE
            except Exception:
                DSV4_NATIVE_BLOCK_SIZE = 256
            _configured_block = int(
                getattr(self.config, "paged_cache_block_size", 0) or 0
            )
            if _configured_block != DSV4_NATIVE_BLOCK_SIZE:
                logger.info(
                    "DSV4 paged block size reconciled %s -> %s: the native "
                    "delta records are cut at %s tokens and a block cutter at "
                    "any other size cannot match a single interval, which "
                    "would drop 100%% of cache stores.",
                    _configured_block,
                    DSV4_NATIVE_BLOCK_SIZE,
                    DSV4_NATIVE_BLOCK_SIZE,
                )
                self.config.paged_cache_block_size = DSV4_NATIVE_BLOCK_SIZE
            logger.info(
                "DSV4 DeepseekV4Cache-aware paged prefix cache enabled — "
                "terminal blocks store full SWA+CSA/HCA composite state and "
                "block disk L2 uses deepseek_v4_v10_delta nested-state serialization "
                "with N-1 prompt-token keys."
            )

        if self.config.enable_prefix_cache:
            logger.info(
                "Prefix cache requires continuous batching — enabled automatically"
            )
            if self.config.use_paged_cache or self.config.enable_block_disk_cache:
                _block_disk_only = bool(
                    self.config.enable_block_disk_cache
                    and not self.config.use_paged_cache
                )
                # Create optional block-level disk store (L2)
                block_disk_store = None
                block_disk_store_error: Exception | None = None
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
                        import hashlib

                        # Include quant + runtime cache shape in hash to prevent
                        # cross-config cache poisoning (same fix as prompt disk
                        # cache — C3, extended for DSV4 tri-mode cache schema).
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
                        dsv4_scope = ""
                        zaya_scope = ""
                        if self._uses_dsv4_cache:
                            # Include the unsafe-override env in the scope so
                            # safe/default runs can NEVER share namespace with
                            # `VMLX_DSV4_TRUST_TRIMMED_CACHE=1` debug runs that
                            # store post-generation contaminated state.
                            _unsafe_trim = (
                                "1"
                                if os.environ.get(
                                    "VMLX_DSV4_TRUST_TRIMMED_CACHE", "0"
                                ).lower() in ("1", "true", "yes")
                                else "0"
                            )
                            dsv4_scope = (
                                f":dsv4_long_ctx={os.environ.get('DSV4_LONG_CTX', '0')}"
                                f":dsv4_pool_quant={os.environ.get('DSV4_POOL_QUANT', '')}"
                                f":dsv4_unsafe_trim={_unsafe_trim}"
                                f":dsv4_paged_block_size={self.config.paged_cache_block_size}"
                                ":dsv4_cache_schema=deepseek_v4_v10_delta"
                                ":dsv4_append_safe_checkpoint="
                                f"{DSV4_APPEND_SAFE_CHECKPOINT_POLICY}"
                            )
                        elif self._uses_zaya_cache:
                            zaya_scope = ":zaya_cache_schema=zaya_cca_v1"
                        # Bind persisted block payloads to the loaded bundle,
                        # not only its filesystem path.  Prefix-cache keys
                        # already use this content-derived identity (config /
                        # JANG metadata, weight artifacts, runtime schema, and
                        # loader/cache representation); block L2 must use the
                        # same identity so an in-place bundle replacement
                        # cannot refault stale tensors from the old model.
                        block_scope_key = build_block_cache_namespace(
                            model=self.model,
                            model_path=self.config.model_path,
                            quant_tag=quant_tag,
                            tq_native_tag=tq_native_tag,
                            smelt_enabled=self.config.smelt_enabled,
                            smelt_pct=self.config.smelt_pct,
                            tq_enabled=(
                                self._tq_active
                                and not self._uses_dsv4_cache
                                and not self._uses_zaya_cache
                            ),
                            kv_quant_bits=self._kv_cache_bits,
                            dsv4_scope=dsv4_scope,
                            zaya_scope=zaya_scope,
                        )
                        model_hash = hashlib.sha256(
                            block_scope_key.encode()
                        ).hexdigest()[:12]
                        cache_dir = os.path.join(cache_root, model_hash)
                    else:
                        logger.warning(
                            "Block disk cache: model_path not set, using isolated "
                            "'default' namespace."
                        )
                        cache_dir = os.path.join(cache_root, "default")
                    # Derive expected layer count
                    # from model config and pass to BlockDiskStore so the
                    # validator can hard-reject wrong-model L2 entries.
                    _expected_n_layers = self._expected_cache_layer_count()
                    try:
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
                            expected_num_layers=_expected_n_layers,
                            global_cache_root=cache_root,
                            allow_legacy_hashed_namespaces=(
                                _default_block_cache_root
                            ),
                            allow_legacy_direct_namespace=(
                                not _default_block_cache_root
                            ),
                        )
                        if self.config.step_executor is not None:
                            block_disk_store.set_load_executor(
                                self.config.step_executor
                            )
                        logger.info(
                            f"Block disk cache enabled: dir={cache_dir}, "
                            f"max={self.config.block_disk_cache_max_gb}GB, "
                            f"expected_layers={_expected_n_layers}"
                        )
                        if (
                            self._is_hybrid
                            and not self._uses_dsv4_cache
                            and not self._uses_zaya_cache
                            and self._ssm_state_cache is not None
                        ):
                            try:
                                _ssm_disk = SSMCompanionDiskStore(
                                    directory=os.path.join(cache_dir, "ssm_companion"),
                                    budget_bytes=block_disk_store.max_size_bytes,
                                    global_budget=block_disk_store.global_budget,
                                )
                                self._ssm_state_cache.attach_disk_store(_ssm_disk)
                                logger.info(
                                    "Hybrid SSM companion L2 enabled: dir=%s; "
                                    "shares aggregate root max=%.3gGB",
                                    _ssm_disk.directory,
                                    self.config.block_disk_cache_max_gb,
                                )
                            except Exception as _ssm_disk_e:
                                logger.warning(
                                    "Hybrid SSM companion L2 init failed; "
                                    "continuing with in-memory SSM companion only: %s",
                                    _ssm_disk_e,
                                )
                    except Exception as e:
                        logger.error(
                            f"Failed to initialize block disk cache at {cache_dir}: {e}. "
                            "Continuing without disk cache.",
                            exc_info=True,
                        )
                        block_disk_store = None
                        # Keep the cause. Without it the disk-ONLY path below
                        # raises a fatal "could not be initialized" with no
                        # explanation, and the actual reason is a logger.error
                        # forty lines earlier that a reader has no reason to
                        # connect to the crash.
                        block_disk_store_error = e

                # Use paged cache for memory efficiency.
                # Give the block pool the SAME RAM-byte ceiling the memory-aware
                # path uses (default 20% of available RAM, 32 GB hard cap) so the
                # in-RAM block KV mirror can't ratchet resident GPU memory upward
                # with distinct prefixes. Without this, the pool grew to a flat
                # max_cache_blocks regardless of per-model KV size (measured
                # +3.7 GB vs +98 MB for the memory-aware path on the same
                # workload). enforce_byte_budget() evicts only free (ref==0)
                # cached blocks, disk-L2 write-through first.
                from .memory_cache import resolve_paged_resident_policy

                if _block_disk_only and block_disk_store is None:
                    raise RuntimeError(
                        "Block disk-only cache was requested but its SSD store "
                        "could not be initialized; refusing to substitute a RAM "
                        f"backend. Cause: {block_disk_store_error!r}"
                    ) from block_disk_store_error

                _paged_resident_budget, _explicit_zero_cache = (
                    resolve_paged_resident_policy(self.config, _block_disk_only)
                )
                if _explicit_zero_cache:
                    logger.info(
                        "cache-memory-mb=0 requested: paged RAM payloads "
                        "disabled (frugal); blocks restore transiently."
                    )
                try:
                    self.paged_cache_manager = PagedCacheManager(
                        block_size=self.config.paged_cache_block_size,
                        max_blocks=self.config.max_cache_blocks,
                        disk_store=block_disk_store,
                        max_resident_bytes=_paged_resident_budget,
                        disk_only=_block_disk_only,
                        frugal=_explicit_zero_cache,
                    )
                    if _block_disk_only:
                        logger.info(
                            "Block disk-only prefix backend: paged RAM disabled, "
                            "max_index_blocks=%d; payloads restore transiently from SSD",
                            self.config.max_cache_blocks,
                        )
                    else:
                        logger.info(
                            "Paged cache RAM ceiling: %.0f MB (%.0f%% of available); "
                            "block pool max_blocks=%d",
                            _paged_resident_budget / (1024 * 1024),
                            self.config.cache_memory_percent * 100,
                            self.config.max_cache_blocks,
                        )
                    self.block_aware_cache = BlockAwarePrefixCache(
                        model=model,
                        paged_cache_manager=self.paged_cache_manager,
                        model_path=self.config.model_path,
                        smelt_enabled=self.config.smelt_enabled,
                        smelt_pct=self.config.smelt_pct,
                        tq_enabled=(
                            self._tq_active
                            and not self._uses_dsv4_cache
                            and not self._uses_zaya_cache
                        ),
                        kv_quant_bits=self._kv_cache_bits,
                        uses_dsv4_cache=self._uses_dsv4_cache,
                        uses_zaya_cache=self._uses_zaya_cache,
                        mixed_attention_cache_model=(
                            self._mixed_attention_cache_model
                        ),
                    )
                except Exception:
                    self._cleanup_failed_block_cache_initialization(
                        block_disk_store=block_disk_store,
                    )
                    raise
                if self._uses_dsv4_cache:
                    logger.info(
                        "DSV4 native composite block index enabled: "
                        "block_size=%s, max_blocks=%s "
                        "(not generic paged KV; records deepseek_v4_v10_delta "
                        "SWA+CSA/HCA state)",
                        self.config.paged_cache_block_size,
                        self.config.max_cache_blocks,
                    )
                elif _block_disk_only:
                    logger.info(
                        "Block disk-only cache enabled: block_size=%s, "
                        "max_index_blocks=%s, persistent_ram_payloads=0",
                        self.config.paged_cache_block_size,
                        self.config.max_cache_blocks,
                    )
                else:
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
                # The isolate-clone handed back by fetch() slices MLX arrays, and
                # MLX pins each one to the creating thread's stream. fetch() is
                # reached from add_request() on the API thread, so the clone must
                # be built on the llm-worker that owns the generation stream —
                # otherwise decode dies with "no Stream(gpu, 0) in current thread"
                # and the request collapses into an empty 200 (F11, Laguna).
                if self.config.step_executor is not None:
                    self.memory_aware_cache.set_clone_executor(self.config.step_executor)
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
                    # Derive when unset. SchedulerConfig defaults
                    # prefix_cache_max_bytes to None, so passing it straight
                    # through left this cache bounded by ENTRY COUNT only —
                    # 100 whole-KV snapshots, which at 90k context is hundreds
                    # of GB. Same hole the MLLM path had; same derivation.
                    max_bytes=_resolve_prefix_cache_byte_budget(self.config),
                    model_path=self.config.model_path,
                    smelt_enabled=self.config.smelt_enabled,
                    smelt_pct=self.config.smelt_pct,
                    tq_enabled=(
                        self._tq_active
                        and not self._uses_dsv4_cache
                        and not self._uses_zaya_cache
                    ),
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
            n_layers = self._expected_cache_layer_count() or 0
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
                # Include layer count to invalidate on architecture change.
                scope_key = (
                    f"{self.config.model_path}:quant={quant_tag}:layers={n_layers}"
                    f":tq_native={tq_native_tag}"
                    f":prefix_cache_schema={PAGED_CACHE_SCHEMA_VERSION}"
                    f":{runtime_cache_fingerprint()}"
                )
                scope_key = _append_looped_cache_identity_scope(
                    scope_key,
                    self.model,
                )
                model_hash = hashlib.sha256(scope_key.encode()).hexdigest()[:12]
                model_slug = os.path.basename(self.config.model_path.rstrip("/"))
                cache_dir = os.path.join(base_dir, f"{model_slug}_{model_hash}")
            else:
                cache_dir = base_dir
            self.disk_cache = DiskCacheManager(
                cache_dir=cache_dir,
                max_size_gb=self.config.disk_cache_max_gb,
                allow_tq_native=bool(self._tq_active),
                # Pass expected layer count
                # so the safetensors header validator can hard-reject
                # wrong-model L2 entries before mx.load triggers
                # multi-hundred-GB metal::malloc.
                expected_num_layers=int(n_layers) if n_layers else None,
                required_cache_class=(
                    "OpenPanguV2LayerCache"
                    if self._uses_openpangu_cache
                    else "MiniMaxM3SparseCache"
                    if self._uses_m3_msa_cache
                    else None
                ),
                required_cache_classes=(
                    ("Glm5KDACache", "Glm5MLACache")
                    if self._uses_glm5_next_cache
                    else None
                ),
            )
            if self.config.step_executor is not None:
                self.disk_cache.set_load_executor(self.config.step_executor)
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
        self._cache_hit_requests = 0
        self._cache_hit_tokens = 0
        self._cache_hit_tokens_by_detail: Dict[str, int] = {}
        self._cache_reuse_skips = 0
        self._cache_reuse_skip_tokens = 0
        self._last_cache_reuse_skip: Optional[Dict[str, Any]] = None
        self._cache_reuse_partial_downgrades = 0
        self._cache_reuse_partial_tokens = 0
        self._last_cache_reuse_partial: Optional[Dict[str, Any]] = None
        self._last_cache_selection: Optional[Dict[str, Any]] = None
        self._last_cache_execution: Optional[Dict[str, Any]] = None
        self._tq_decoder_warmup_stats: Optional[Dict[str, Any]] = None

        # Periodic Metal memory cache cleanup timer.
        # During sustained multi-request traffic, self.running is never empty
        # so _cleanup_finished's clear_mlx_memory_cache() never triggers.
        # This timer ensures Metal's internal allocator cache gets flushed
        # periodically (every 60s) even during continuous load.
        self._last_metal_gc_time = time.monotonic()
        self._metal_gc_interval = 60.0  # seconds

    @staticmethod
    def _model_has_mixed_attention(model: Any) -> bool:
        """Return True for cache layouts with sliding/rotating attention.

        Config metadata catches Gemma/Step-style mixed-attention models. Some
        bundles (notably MiMo V2.5) do not expose that metadata reliably, but
        their runtime cache is authoritative: if ``model.make_cache()`` returns
        RotatingKVCache layers, post-generation truncation is not a safe prompt
        boundary store. Use the clean N-1 prompt-prefill cache contract instead.
        """
        def _cfg_value(cfg: Any, key: str) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key)
            return getattr(cfg, key, None)

        def _cache_has_rotating(cache_obj: Any) -> bool:
            if cache_obj is None:
                return False
            if isinstance(cache_obj, (list, tuple)):
                return any(_cache_has_rotating(c) for c in cache_obj)
            sub = getattr(cache_obj, "caches", None)
            if isinstance(sub, (list, tuple)):
                return any(_cache_has_rotating(c) for c in sub)
            return type(cache_obj).__name__ in {
                "RotatingKVCache",
                "BatchRotatingKVCache",
            }

        candidates = []
        for attr in ('args', 'config'):
            cfg = getattr(model, attr, None)
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
        try:
            return _cache_has_rotating(model.make_cache())
        except Exception:
            return False

    @staticmethod
    def _model_uses_dsv4_cache(model: Any) -> bool:
        """Return True when model.make_cache() contains DeepseekV4Cache."""
        if not hasattr(model, "make_cache"):
            return False
        try:
            return any(Scheduler._is_dsv4_cache_object(c) for c in (model.make_cache() or []))
        except Exception:
            return False

    @staticmethod
    def _model_uses_zaya_cache(model: Any) -> bool:
        """Return True when model.make_cache() contains ZAYA CCA cache slots."""
        if not hasattr(model, "make_cache"):
            return False
        try:
            cache = model.make_cache() or []
            names = [type(c).__name__ for c in cache]
            return "ZayaNoStateCache" in names or (
                "CacheList" in names
                and any("zaya" in str(type(c).__module__).lower() for c in cache)
            )
        except Exception:
            return False

    @staticmethod
    def _model_uses_m3_msa_cache(model: Any) -> bool:
        """Return True when model.make_cache() contains MiniMaxM3SparseCache (MSA).

        MiniMax-M3's append-only MSA dual-cache (keys/values + Lightning-Indexer
        idx_keys) is structurally incompatible with generic q4/q8 KV quantization:
        the quantized KV path yields an SDPA mask whose dtype does not promote to
        the bfloat16 output ("Mask type must promote to output type bfloat16").
        """
        if not hasattr(model, "make_cache"):
            return False
        try:
            return any(type(c).__name__ == "MiniMaxM3SparseCache" for c in (model.make_cache() or []))
        except Exception:
            return False

    @staticmethod
    def _turboquant_cache_supports_batch_api(model: Any) -> bool:
        """Return True when every TurboQuant cache declares real batch support."""
        if not hasattr(model, "make_cache"):
            return False
        try:
            cache = model.make_cache() or []
        except Exception:
            return False
        tq_slots = [c for c in cache if type(c).__name__ == "TurboQuantKVCache"]
        if not tq_slots:
            return False
        required_methods = ("extend", "filter", "extract", "prepare", "finalize")
        for slot in tq_slots:
            if getattr(slot, "_vmlx_batch_api", None) != "turboquant_kv_v1":
                return False
            if not all(callable(getattr(slot, name, None)) for name in required_methods):
                return False
        return True

    def _log_runtime_cache_contract(self, model: Any) -> None:
        """Record and log instantiated per-layer cache classes."""
        try:
            # This is the exact factory used by both mlx-lm BatchGenerator and
            # vMLX SingleBatchGenerator. It delegates to model.make_cache()
            # for native architectures and creates the generator's default
            # KVCache-per-layer layout otherwise. Calling model.make_cache()
            # directly and returning early when it did not exist made ordinary
            # full-KV models (live MiniMax-M2.7) invisible even though the
            # generator instantiates 62 real KVCache slots.
            from mlx_lm.models.cache import make_prompt_cache

            delegated_to_model_make_cache = hasattr(model, "make_cache")
            cache = make_prompt_cache(model) or []
            runtime_layout = describe_runtime_cache_layout(cache, model=model)
            runtime_layout["source"] = "instantiated_runtime_cache_factory"
            runtime_layout["factory"] = "mlx_lm.models.cache.make_prompt_cache"
            runtime_layout["delegated_to_model_make_cache"] = (
                delegated_to_model_make_cache
            )
            dtype_status = getattr(
                model,
                "_vmlx_quant_metadata_dtype_harmonization",
                None,
            )
            if isinstance(dtype_status, dict):
                runtime_layout["parameter_dtype_harmonization"] = dict(dtype_status)
            self._runtime_cache_layout = runtime_layout
            self._runtime_cache_num_layers = runtime_layout["layer_count"]
            self._runtime_cache_kv_positions = (
                None
                if runtime_layout["parallel_layer_indices"]
                else runtime_layout["attention_layer_indices"]
            )
            self._runtime_cache_parallel_hybrid_positions = runtime_layout[
                "parallel_layer_indices"
            ]
            logger.info(
                "Runtime cache layout: model_type=%s layers=%d layout=%s",
                self._model_type_for_runtime or "unknown",
                runtime_layout["layer_count"],
                ";".join(
                    f"{idx}:{slot_type}"
                    for idx, slot_type in enumerate(runtime_layout["slot_types"])
                ),
            )
        except Exception as exc:
            logger.debug("Runtime cache layout logging skipped: %s", exc)

    @staticmethod
    def _is_dsv4_cache_class_name(class_name: str) -> bool:
        return class_name in {"DeepseekV4Cache", "PoolQuantizedV4Cache"}

    @staticmethod
    def _is_dsv4_cache_object(cache: Any) -> bool:
        try:
            return any(
                Scheduler._is_dsv4_cache_class_name(cls.__name__)
                for cls in type(cache).__mro__
            )
        except Exception:
            return False

    @staticmethod
    def _is_hybrid_model(model: Any) -> bool:
        """Check if model uses non-standard cache types requiring block-aware reuse.

        Returns True for:
        - Hybrid models (mixed KVCache + MambaCache layers)
        - Pure Mamba/SSM models (all MambaCache/ArraysCache layers)

        These models cannot use memory-aware cache (which needs truncatable KVCache).
        They use either paged RAM blocks or disk-only blocks paired with the typed
        SSM companion cache/rederive path.
        """
        if not hasattr(model, "make_cache"):
            return False
        try:
            cache = model.make_cache()
            # Resolve CacheList wrappers to what they actually contain. Reading
            # the wrapper name and discarding it classified falcon_h1 (every
            # layer a CacheList around a cumulative ArraysCache) as plain KV,
            # which cost it the SSM companion and all prefix reuse.
            cache_types = expand_cache_class_names(cache)
            # Standard KV-only models don't need special handling.
            # Match any class name ending with "KVCache" (e.g., KVCache,
            # RotatingKVCache, QuantizedKVCache, ChunkedKVCache) so future
            # KV cache variants are handled automatically without hardcoding.
            # MiniMax-M3's MiniMaxM3SparseCache is a KVCache subclass: append-only,
            # position-truncatable (its trim() slices keys/values/idx_keys by offset).
            # It is NOT SSM/cumulative, so treat it as KV — otherwise M3 is mis-routed
            # through the SSM-companion hybrid path (wrong + per-request re-derive overhead).
            kv_types = {
                t for t in cache_types
                if t == "KVCache" or t.endswith("KVCache") or t == "MiniMaxM3SparseCache"
            }
            if any(Scheduler._is_dsv4_cache_class_name(t) for t in cache_types):
                return False
            # openPangu v2 owns a path-dependent composite cache: MLA latent KV,
            # DSA indexer state, rotating SWA state, and causal-convolution state.
            # Its per-layer wrapper is intentionally not a KVCache subclass, but
            # it is also not an SSM companion cache.  Classifying it as hybrid
            # silently forces paged cache and advertises an inapplicable async
            # SSM-rederive contract.
            if "OpenPanguV2LayerCache" in cache_types or cache_types & {
                "Glm5KDACache",
                "Glm5MLACache",
            }:
                return False
            if cache_types and cache_types == kv_types:
                return False
            # DSV4 composite attention is handled by _uses_dsv4_cache above,
            # not by the hybrid SSM companion path.
            # No CacheList discard here: expand_cache_class_names() already
            # resolved wrappers, so a surviving "CacheList" means its contents
            # were unreadable and must not be assumed KV-only.
            return bool(cache_types - kv_types)
        except Exception as e:
            raise RuntimeError(
                "make_cache() failed during cache-architecture detection; "
                "refusing to classify the model as plain KV"
            ) from e

    def _block_store_activity_probe(self) -> bool:
        """True while any request is waiting or running (block-store maintenance gate)."""

        return bool(getattr(self, "waiting", None) or getattr(self, "running", None))

    def _expected_cache_layer_count(self) -> Optional[int]:
        """Return the number of cache-bearing layers for validation/scope keys.

        Some hybrid models have transformer blocks that do not own cache state
        (Nemotron-H MoE/MLP blocks are the current live example: 52 blocks but
        29 Mamba/attention cache entries). L2/paged validators must compare
        against the cache contract, not raw ``num_hidden_layers``.
        """
        return expected_cache_layer_count(
            self.model, getattr(self, "_hybrid_num_layers", None)
        )

    def _detect_mla(self) -> bool:
        """Detect Multi-head Latent Attention (compressed-latent KV).

        Mirrors ``MLLMScheduler._detect_mla`` so the LLM scheduler's KV-cache
        quantization gate catches wrapped MLA models (Kimi K2.6 mlx_vlm
        wrapper around DeepseekV3, glm_moe_dsa inheriting deepseek_v32,
        mistral4 text wrappers, deepseek_v4 / DSV4 backbones). The earlier
        inline check only inspected ``self.model.args`` — wrappers expose
        the relevant config via ``language_model.config.text_config`` and
        were silently treated as non-MLA, leaving generic q4/q8 prefix-cache
        quantization on top of an already compressed latent.
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
                raw = getattr(cfg, "_raw_config", None)
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
            candidates = [self.model]
            lm = getattr(self.model, "language_model", None)
            if lm is not None and lm not in candidates:
                candidates.append(lm)
            for obj in list(candidates):
                inner = getattr(obj, "model", None)
                if inner is not None and inner not in candidates:
                    candidates.append(inner)

            cfgs: list[Any] = []
            for obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(obj, attr, None)
                    if cfg is None:
                        continue
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
                if model_type in ("mistral4", "deepseek_v4"):
                    return True
        except Exception:
            pass
        return False

    def _detect_cache_head_dims(self) -> Tuple[int, ...]:
        """Detect cache tensor trailing dims for KV quant validation."""
        try:
            return detect_cache_head_dims(self.model)
        except Exception as e:
            logger.debug(f"Could not detect cache head dims: {e}")
            return ()

    def _detect_head_dim(self) -> Optional[int]:
        """Detect the model's primary KV cache trailing dim from config."""
        dims = self._detect_cache_head_dims()
        if dims:
            return dims[0]
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
            # TWO-PASS: MLA detection first, then num_key_value_heads.
            # Most MLA families store compressed latent cache with H=1, but
            # Ling/Bailing's mlx_lm implementation expands and stores full
            # per-head KV in MLAAttention.update_and_fetch(). Treating it as
            # H=1 sliced valid (1,32,T,D) cache down to (1,1,T,D) and caused
            # cache-hit append crashes.
            for model_obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    model_type = str(getattr(cfg, "model_type", "") or "").lower()
                    if not model_type:
                        tc = getattr(cfg, "text_config", None)
                        if tc is not None:
                            model_type = str(
                                getattr(tc, "model_type", "") or ""
                            ).lower()
                    kv_lora_rank = getattr(cfg, "kv_lora_rank", 0)
                    if not kv_lora_rank:
                        tc = getattr(cfg, "text_config", None)
                        if tc is not None:
                            kv_lora_rank = getattr(tc, "kv_lora_rank", 0)
                    if kv_lora_rank and kv_lora_rank > 0:
                        if model_type in ("bailing_hybrid", "bailing_moe_v2_5"):
                            n_kv = int(getattr(cfg, "num_attention_heads", 0) or 0)
                            if not n_kv:
                                tc = getattr(cfg, "text_config", None)
                                if tc is not None:
                                    n_kv = int(
                                        getattr(tc, "num_attention_heads", 0) or 0
                                    )
                        else:
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
                        if not n_kv:
                            tc = getattr(cfg, "text_config", None)
                            if tc is not None:
                                n_kv = getattr(tc, "num_key_value_heads", 0) or getattr(
                                    tc, "num_kv_heads", 0
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

    def _detect_allowed_n_kv_heads(self) -> set[int]:
        """Return all KV head counts that are valid for this model."""
        if hasattr(self, "_allowed_n_kv_heads_cached"):
            return self._allowed_n_kv_heads_cached
        allowed: set[int] = set()
        primary = self._detect_n_kv_heads()
        if primary > 0:
            allowed.add(primary)
        try:
            candidates = [self.model]
            lm = getattr(self.model, "language_model", None)
            if lm is not None:
                candidates.append(lm)
            mm = getattr(self.model, "model", None)
            if mm is not None and mm is not self.model:
                candidates.append(mm)
            for model_obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    for field in (
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

    def _detect_config_int_field(self, fields: tuple[str, ...]) -> int:
        try:
            candidates = [self.model]
            lm = getattr(self.model, "language_model", None)
            if lm is not None:
                candidates.append(lm)
            mm = getattr(self.model, "model", None)
            if mm is not None and mm is not self.model:
                candidates.append(mm)
            for model_obj in candidates:
                for attr in ("args", "config", "text_config"):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    for field in fields:
                        val = getattr(cfg, field, None)
                        if val is None:
                            tc = getattr(cfg, "text_config", None)
                            if tc is not None:
                                val = getattr(tc, field, None)
                        if isinstance(val, int) and val > 0:
                            return val
        except Exception:
            pass
        return 0

    def _cache_head_slice_target(self, class_name: str, actual_heads: int) -> int:
        """Return target H for safe GQA-inflation slicing, or 0 to keep as-is.

        Mixed full/SWA families such as Gemma 4 and MiMo V2 can legitimately
        use different KV head counts by cache class. Extraction must not slice
        an 8-head full-attention KVCache down to the smaller 4-head SWA count.
        """
        allowed = self._detect_allowed_n_kv_heads()
        if allowed and actual_heads in allowed:
            return 0
        primary = self._detect_n_kv_heads()
        if primary <= 0 or actual_heads <= primary:
            return 0
        if len(allowed) > 1:
            if "Rotating" in str(class_name):
                target = self._detect_config_int_field(
                    (
                        "swa_num_key_value_heads",
                        "num_swa_key_value_heads",
                        "sliding_num_key_value_heads",
                        "local_num_key_value_heads",
                    )
                )
            else:
                target = self._detect_config_int_field(
                    (
                        "num_global_key_value_heads",
                        "global_num_key_value_heads",
                    )
                )
            return target if target > 0 and actual_heads > target else 0
        return primary

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

        # Validate cache trailing-dim compatibility with group_size.
        # mx.quantize() requires group_size to divide the last dimension.
        cache_head_dims = self._detect_cache_head_dims()
        head_dim = cache_head_dims[0] if cache_head_dims else None
        if cache_head_dims:
            adjusted_group_size = choose_supported_kv_group_size(
                cache_head_dims, group_size
            )
            if adjusted_group_size is None:
                logger.error(
                    "KV cache quantization: no supported group_size found for "
                    f"cache_head_dims={cache_head_dims}. Disabling KV cache quantization."
                )
                return
            if adjusted_group_size != group_size:
                logger.warning(
                    f"KV cache quantization: group_size={group_size} does not divide "
                    f"cache_head_dims={cache_head_dims} or is unsupported. "
                    f"Auto-adjusting to group_size={adjusted_group_size}."
                )
                group_size = adjusted_group_size
            logger.info(
                "KV cache quantization validated: "
                f"cache_head_dims={cache_head_dims}, group_size={group_size}"
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
            import numpy as np
        except ImportError:
            return cache

        bits = self._kv_cache_bits
        group_size = self._kv_cache_group_size
        def _packed_cpu(parts):
            # Stored prefix state must outlive the request's Metal stream.
            # Keep the already-quantized q4/q8 payload, scales, and zeros as
            # independent NumPy buffers; fetch converts them back on the model
            # owner worker immediately before dequantization.
            # Paged cache reconstructs blocks immediately on the owner worker
            # and its typed serializer expects MLX arrays, so retain MLX there.
            if getattr(self, "block_aware_cache", None) is not None:
                return tuple(parts)
            return tuple(np.array(part) for part in parts)
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
                            qkv.keys = _packed_cpu(
                                mx.quantize(sc.keys, group_size=group_size, bits=bits)
                            )
                            qkv.values = _packed_cpu(
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
                    qkv.keys = _packed_cpu(
                        mx.quantize(layer_cache.keys, group_size=group_size, bits=bits)
                    )
                    qkv.values = _packed_cpu(
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
            elif self._is_dsv4_cache_object(layer_cache):
                # DSV4's cache is not a plain KV payload. DeepseekV4Cache
                # combines SWA local RotatingKVCache with cumulative CSA/HCA
                # compressed-pool state, and PoolQuantizedV4Cache already
                # owns the only supported DSV4-native pool compression. Do not
                # wrap any component in generic QuantizedKVCache for prefix,
                # paged, or L2 storage; doing so changes the composite cache
                # type contract and risks stale local/pool mismatches.
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
        def _mlx_parts(parts):
            return tuple(x if isinstance(x, mx.array) else mx.array(x) for x in parts)

        for i, layer_cache in enumerate(cache):
            if _CacheList is not None and isinstance(layer_cache, _CacheList):
                # MoE: recurse into each sub-cache
                dequantized_subs = []
                for sc in layer_cache.caches:
                    if isinstance(sc, QuantizedKVCache):
                        if sc.keys is not None:
                            try:
                                kv = KVCache()
                                qk = _mlx_parts(sc.keys)
                                qv = _mlx_parts(sc.values)
                                kv.keys = mx.dequantize(
                                    qk[0], qk[1], qk[2],
                                    sc.group_size,
                                    sc.bits,
                                )
                                kv.values = mx.dequantize(
                                    qv[0], qv[1], qv[2],
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
                        qk = _mlx_parts(layer_cache.keys)
                        qv = _mlx_parts(layer_cache.values)
                        kv.keys = mx.dequantize(
                            qk[0], qk[1], qk[2],
                            layer_cache.group_size,
                            layer_cache.bits,
                        )
                        kv.values = mx.dequantize(
                            qv[0], qv[1], qv[2],
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
            elif self._is_dsv4_cache_object(layer_cache):
                local = getattr(layer_cache, "local", None)
                if isinstance(local, QuantizedKVCache):
                    try:
                        if local.keys is None:
                            result.append(layer_cache)
                            continue
                        from jang_tools.dsv4.mlx_model import DeepseekV4Cache
                        from mlx_lm.models.cache import RotatingKVCache

                        def _as_mx_array(x):
                            if x is None:
                                return None
                            if isinstance(x, mx.array):
                                # Re-home arrays onto the current scheduler
                                # worker stream. DSV4 cache hits are looked up
                                # from the API thread but consumed by the
                                # llm-worker; lazy arrays tied to the API
                                # thread's Stream(gpu, 0) crash on replay.
                                return x + mx.zeros_like(x)
                            if hasattr(x, "shape"):
                                return mx.array(x)
                            return x

                        def _as_mx_tree(x):
                            if isinstance(x, tuple):
                                return tuple(_as_mx_tree(v) for v in x)
                            if isinstance(x, list):
                                return [_as_mx_tree(v) for v in x]
                            return _as_mx_array(x)

                        q_keys = tuple(_as_mx_array(x) for x in local.keys)
                        q_values = tuple(_as_mx_array(x) for x in local.values)
                        keys = mx.dequantize(
                            q_keys[0],
                            q_keys[1],
                            q_keys[2],
                            local.group_size,
                            local.bits,
                        )
                        values = mx.dequantize(
                            q_values[0],
                            q_values[1],
                            q_values[2],
                            local.group_size,
                            local.bits,
                        )
                        try:
                            mx.eval(keys, values)
                        except Exception:
                            pass

                        rotating_meta = tuple(
                            getattr(layer_cache, "_vmlx_dsv4_local_meta_state", ()) or ()
                        )
                        sliding_window = int(
                            getattr(layer_cache, "_vmlx_dsv4_sliding_window", 0)
                            or (rotating_meta[1] if len(rotating_meta) >= 2 else 128)
                        )
                        keep = int(
                            getattr(layer_cache, "_vmlx_dsv4_keep", 0)
                            or (rotating_meta[0] if len(rotating_meta) >= 1 else 0)
                        )
                        local_rot = RotatingKVCache(max_size=sliding_window, keep=keep)
                        local_rot.state = (keys, values)
                        if rotating_meta:
                            try:
                                local_rot.meta_state = rotating_meta
                            except Exception:
                                local_rot.offset = int(getattr(local, "offset", 0) or 0)
                                local_rot._idx = min(local_rot.offset, keys.shape[-2])
                        else:
                            local_rot.offset = int(getattr(local, "offset", 0) or 0)
                            local_rot._idx = min(local_rot.offset, keys.shape[-2])

                        restored = DeepseekV4Cache(
                            sliding_window=sliding_window,
                            compress_ratio=getattr(layer_cache, "compress_ratio", None),
                        )
                        restored.local = local_rot
                        # Direct dict read below bypasses the state-property
                        # flush; drain any deferred pool tokens first.
                        if hasattr(layer_cache, "flush_pool_pending"):
                            layer_cache.flush_pool_pending()
                        restored.compressor_state = {
                            k: _as_mx_tree(v)
                            for k, v in (
                                getattr(layer_cache, "compressor_state", {}) or {}
                            ).items()
                        }
                        restored.indexer_state = {
                            k: _as_mx_tree(v)
                            for k, v in (
                                getattr(layer_cache, "indexer_state", {}) or {}
                            ).items()
                        }
                        eval_args = [keys, values]
                        for state in (
                            restored.compressor_state,
                            restored.indexer_state,
                        ):
                            for value in state.values():
                                if isinstance(value, mx.array):
                                    eval_args.append(value)
                        try:
                            mx.eval(*eval_args)
                        except Exception:
                            pass
                        result.append(restored)
                    except Exception as e:
                        logger.warning(
                            f"DSV4 local SWA dequantization failed for layer {i}: {e}. "
                            f"Treating as cache miss."
                        )
                        return None
                else:
                    result.append(layer_cache)
            else:
                result.append(layer_cache)
        return result

    def _prefill_for_prompt_only_cache(
        self,
        prompt_tokens: List[int],
        should_stop: Optional[Callable[[], bool]] = None,
        capture_dsv4_deltas: bool = False,
        base_cache: Optional[List[Any]] = None,
        base_token_count: int = 0,
    ) -> Optional[List[Any]]:
        """
        Run a prefill-only forward pass to get cache state for the given tokens.

        For hybrid models (MambaCache + KVCache), MambaCache is cumulative
        and can't be truncated from post-generation state. This method runs
        a separate prefill pass to capture cache state with exactly the given
        tokens, without output token contamination.

        Args:
            prompt_tokens: Token IDs to prefill (typically prompt[:-1])
            should_stop: Optional poll evaluated before and between prefill
                chunks. When it returns True the partial prefill is abandoned
                and None is returned, so idle maintenance yields before
                submitting stale work after a foreground request arrives
                (vmlx#245).

        Returns:
            List of cache objects with state for exactly the given tokens,
            or None on failure or preemption
        """
        if not prompt_tokens:
            return None
        try:
            import mlx.core as mx

            # Resuming from an existing checkpoint turns a whole-prefix derive
            # into a delta. The recurrent state advances the same way it does
            # across an ordinary chunk boundary, so there is no
            # contiguous-from-zero requirement -- the RESUME path already
            # generates from checkpoint states, and chunked SSM re-derive is
            # default ON off the back of a byte-exactness A/B.
            _resume_at = 0
            if base_cache is not None and int(base_token_count or 0) > 0:
                if int(base_token_count) >= len(prompt_tokens):
                    # Nothing to advance; the caller already has this boundary.
                    return list(base_cache)
                if capture_dsv4_deltas:
                    # The DSV4 capture loop stamps anchors from a cache it owns
                    # from token zero. Resuming would leave the delta chain
                    # without its parent records, so decline rather than store
                    # something that cannot be composed later.
                    logger.debug(
                        "Ignoring base cache for a DSV4 delta-capture derive; "
                        "anchors must be stamped from a full pass."
                    )
                else:
                    fresh_cache = base_cache
                    _resume_at = int(base_token_count)
            if _resume_at == 0:
                fresh_cache = self.model.make_cache()

            if capture_dsv4_deltas:
                # Shadow re-key chains are later extended from native
                # DSV4 checkpoints. A generic full-cache snapshot would store
                # ``deepseek_v4_pending`` parents which cannot be composed
                # with the generator's immutable delta records. Reuse the
                # owning generator's capture loop so periodic and append-safe
                # anchors are stamped while the live cache is at the exact
                # boundary. Never fall back to a generic shadow store.
                from .utils.dsv4_batch_generator import DSV4BatchGenerator

                generator = getattr(self, "batch_generator", None)
                if (
                    not self._uses_dsv4_cache
                    or not isinstance(generator, DSV4BatchGenerator)
                    or not generator._delta_capture_admitted(
                        fresh_cache,
                        len(prompt_tokens),
                    )
                ):
                    del fresh_cache
                    return None
                last_logits = generator._prefill_last_logits(
                    prompt_tokens,
                    fresh_cache,
                    capture_block_deltas=True,
                    force_terminal_anchor=True,
                    should_stop=should_stop,
                )
                if last_logits is None:
                    del fresh_cache
                    clear_mlx_memory_cache(log=logger)
                    return None
                return fresh_cache

            # Process in chunks to avoid Metal GPU timeout. DSV4 must use the
            # same architecture-aligned policy as DSV4BatchGenerator; the old
            # single-shot exception timed out at long context and diverged from
            # the actual request path.
            #
            # The non-DSV4 branch mirrors the live prefill step for the same
            # reason. Chunking is mechanically safe (measured on Lightning
            # 30B nemotron_h, 4600 tokens: layer-1 diffs start at 1e-6, no
            # uninitialised-state garbage; dense attention is byte-identical
            # chunked vs one-shot) — but SSM scan numerics re-associate at
            # chunk boundaries and amplify through depth (worst |diff| 1.45
            # vs one-shot). A companion is byte-comparable to a cold prefill
            # ONLY when derived with the SAME step the live path uses, so a
            # hardcoded 2048 silently diverged whenever the user overrode
            # prefill_step_size.
            if self._uses_dsv4_cache:
                from .utils.dsv4_batch_generator import (
                    dsv4_effective_prefill_step,
                    dsv4_prefill_step_policy,
                )

                configured_step, single_shot = dsv4_prefill_step_policy(
                    getattr(getattr(self, "config", None), "prefill_step_size", 2048)
                )
                chunk_size = dsv4_effective_prefill_step(
                    configured_step,
                    len(prompt_tokens),
                    single_shot=single_shot,
                )
            else:
                chunk_size = max(
                    1,
                    int(
                        getattr(
                            getattr(self, "config", None), "prefill_step_size", 2048
                        )
                        or 2048
                    ),
                )
            for start in range(_resume_at, len(prompt_tokens), chunk_size):
                if should_stop is not None and should_stop():
                    del fresh_cache
                    clear_mlx_memory_cache(log=logger)
                    return None
                chunk = prompt_tokens[start : start + chunk_size]
                input_ids = mx.array([chunk])
                _ = self.model(input_ids, cache=fresh_cache)
                # Materialize after each chunk to prevent massive lazy graph
                eval_args = []
                def _collect_tree_arrays(obj):
                    if obj is None:
                        return
                    if isinstance(obj, (tuple, list)):
                        for item in obj:
                            _collect_tree_arrays(item)
                        return
                    if hasattr(obj, "shape"):
                        eval_args.append(obj)

                def _collect_cache_arrays(cache_obj):
                    if cache_obj is None:
                        return
                    if hasattr(cache_obj, "caches") and isinstance(
                        getattr(cache_obj, "caches", None), (list, tuple)
                    ):
                        for sub_cache in cache_obj.caches:
                            _collect_cache_arrays(sub_cache)
                        return
                    if Scheduler._is_dsv4_cache_object(cache_obj):
                        state = (
                            cache_obj.storage_state
                            if type(cache_obj).__name__ == "PoolQuantizedV4Cache"
                            and getattr(type(cache_obj), "storage_state", None)
                            is not None
                            else cache_obj.state
                        )
                        _collect_tree_arrays(state)
                        return
                    if hasattr(cache_obj, "keys") and cache_obj.keys is not None:
                        # QuantizedKVCache: keys/values are tuples of arrays
                        if isinstance(cache_obj.keys, tuple):
                            eval_args.extend(cache_obj.keys)
                            eval_args.extend(cache_obj.values)
                        else:
                            eval_args.extend([cache_obj.keys, cache_obj.values])
                        idx_keys = getattr(cache_obj, "idx_keys", None)
                        if idx_keys is not None:
                            eval_args.append(idx_keys)
                    elif hasattr(cache_obj, "cache") and isinstance(
                        cache_obj.cache, list
                    ):
                        for arr in cache_obj.cache:
                            if hasattr(arr, "shape"):
                                eval_args.append(arr)

                for c in fresh_cache:
                    _collect_cache_arrays(c)
                if eval_args:
                    mx.eval(*eval_args)
                # Return each chunk's transients to the allocator before the
                # next chunk — the MLLM twin of this loop got this in
                # e8b2b6087 after retained transients (~18GB per 2048-token
                # chunk) stacked into a background memory wave that killed
                # the serve process; this lane (ZAYA / mixed-SWA / M3
                # deferred clean stores, hybrid-text idle rederive) runs the
                # same post-turn work and needs the same discipline.
                mx.clear_cache()

            return fresh_cache
        except Exception as e:
            logger.warning(f"Prefill-only pass failed: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _dsv4_trace_timing(
        self,
        event: str,
        start: float,
        request_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        if not self._uses_dsv4_cache:
            return
        if os.environ.get("VMLINUX_DSV4_TRACE_TIMINGS", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        try:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            logger.info(
                "DSV4 timing: component=scheduler event=%s request_id=%s elapsed_ms=%.3f %s",
                event,
                request_id,
                elapsed_ms,
                extra,
            )
        except Exception:
            pass

    def _dsv4_remaining_thinking_soft_cap(self, request: Any) -> Optional[int]:
        # DSV4 answer-reserve soft cap, decremented by lifetime output so a
        # re-inserted request resumes with the correct remaining thinking
        # budget (mirrors Request.remaining_output_budget semantics).
        cap = getattr(request, "_dsv4_thinking_soft_cap", None)
        if not cap:
            return None
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            return None
        if cap <= 0:
            return None
        return max(1, cap - request.total_output_tokens)

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

    @staticmethod
    def _detect_model_type_for_runtime(model: Any) -> str:
        """Best-effort model_type string for scheduler runtime policy.

        Server-side registry metadata is not plumbed into SchedulerConfig.
        Cache/repetition policy still needs to distinguish loop-prone
        families such as DSV4, MiniMax-M2, and Ling/Bailing. Prefer the loaded
        model config, then class/module names as a fallback.
        """
        candidates: list[Any] = []
        for obj in (model, getattr(model, "model", None), getattr(model, "_model", None)):
            if obj is not None:
                candidates.append(obj)
        for obj in candidates:
            cfg = getattr(obj, "config", None)
            if isinstance(cfg, dict):
                mt = cfg.get("model_type")
                if isinstance(mt, str) and mt:
                    return mt.lower()
                text_cfg = cfg.get("text_config")
                if isinstance(text_cfg, dict):
                    mt = text_cfg.get("model_type")
                    if isinstance(mt, str) and mt:
                        return mt.lower()
            elif cfg is not None:
                mt = getattr(cfg, "model_type", None)
                if isinstance(mt, str) and mt:
                    return mt.lower()
        names = []
        for obj in candidates:
            cls = type(obj)
            names.append(getattr(cls, "__name__", ""))
            names.append(getattr(cls, "__module__", ""))
        joined = " ".join(n.lower() for n in names if n)
        if "deepseek_v4" in joined or "deepseekv4" in joined or "dsv4" in joined:
            return "deepseek_v4"
        if "minimax_m2" in joined:
            return "minimax_m2"
        if "bailing" in joined or "ling" in joined:
            return "bailing_hybrid"
        if "glm5_next" in joined:
            return "glm5_next"
        return ""

    def _canonicalize_tags(self, text: str) -> str:
        """Rewrite variant-suffixed special tags (`</think:opensource>`) to the
        bare spelling every downstream parser matches. Identity for bundles
        that don't use a variant. See `special_tag_dialect`."""
        canon = getattr(self, "_tag_canonicalizer", None)
        if canon is None:
            from .special_tag_dialect import build_canonical_map, compile_canonicalizer

            canon = compile_canonicalizer(build_canonical_map(self._actual_tokenizer))
            self._tag_canonicalizer = canon if canon is not None else False
        return canon(text) if canon else text

    @staticmethod
    def _encode_prompt_text(tokenizer: Any, prompt: str, add_special_tokens: Optional[bool]) -> List[int]:
        """Encode a scheduler prompt with optional explicit special-token mode."""
        if add_special_tokens is None:
            return tokenizer.encode(prompt)
        try:
            return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        except TypeError:
            # Some processor/tokenizer shims do not expose the HF kwarg. Fall
            # back to their native encode path rather than rejecting the request.
            return tokenizer.encode(prompt)

    def _get_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            from .special_tag_dialect import make_streaming_detokenizer

            self._detokenizer_pool[request_id] = make_streaming_detokenizer(
                self._actual_tokenizer
            )
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
            cache_extra_keys = getattr(request, "_cache_extra_keys", None)
            if active_layer == "prefix":
                _call_with_optional_cache_extra(
                    self.prefix_cache.store_cache,
                    tokens_seg,
                    cache_seg,
                    cache_type=role,
                    cache_extra_keys=cache_extra_keys,
                )
            elif active_layer == "memory":
                _call_with_optional_cache_extra(
                    self.memory_aware_cache.store,
                    tokens_seg,
                    cache_seg,
                    cache_type=role,
                    cache_extra_keys=cache_extra_keys,
                )
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

    def _store_glm_prompt_boundary(self, uid: int, prompt_cache: List[Any]) -> bool:
        """Detach GLM's exact N-1 state to SSD before the final prompt token.

        Retaining that expanded MLA boundary in Metal while processing the
        final token exceeds the stock M5 Max working-set cap. DiskCacheManager
        converts the typed state to CPU-owned buffers synchronously; waiting
        for its writer here releases those buffers and makes the entry visible
        before decode or a following request can race it.
        """

        request_id = self.uid_to_request_id.get(int(uid))
        request = self.running.get(request_id) if request_id is not None else None
        if request is None:
            logger.warning(
                "GLM prompt-boundary cache skipped: no request owns uid=%s", uid
            )
            return False
        if getattr(request, "_bypass_prefix_cache", False):
            return False
        if self.disk_cache is None:
            logger.warning(
                "GLM prompt-boundary cache skipped for %s: prompt SSD L2 is disabled",
                request_id,
            )
            return False

        prompt_tokens = list(request.prompt_token_ids or [])
        if len(prompt_tokens) <= 1:
            return False
        stored = _call_with_optional_cache_extra(
            self.disk_cache.store,
            prompt_tokens,
            prompt_cache,
            cache_type=self._pick_cache_type_for_request(request),
            cache_extra_keys=getattr(request, "_cache_extra_keys", None),
        )
        request._glm_prompt_boundary_disk_store = bool(stored)
        request._glm_prompt_boundary_cache_layers = len(prompt_cache)
        if not stored:
            logger.warning(
                "GLM exact N-1 prompt SSD store rejected for %s; generation "
                "continues without a reusable cache",
                request_id,
            )
            return False

        durable = self.disk_cache.flush_pending_writes(
            prompt_tokens,
            cache_extra_keys=getattr(request, "_cache_extra_keys", None),
        )
        if not durable:
            request._glm_prompt_boundary_disk_store = False
            logger.warning(
                "GLM exact N-1 prompt SSD write failed for %s; generation "
                "continues without a reusable cache",
                request_id,
            )
            return False
        logger.info(
            "GLM exact N-1 prompt boundary persisted before final token for "
            "%s (%d layers, %d prompt tokens)",
            request_id,
            len(prompt_cache),
            len(prompt_tokens),
        )
        return True

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

    def _insert_accepts_gen_prompt_lens(self) -> bool:
        """Does the ACTIVE generator's insert() take gen_prompt_lens?

        The cold-prefill-split kwarg exists on SingleBatchGenerator and
        DSV4BatchGenerator, but native-MTP text models run the stock
        mlx_lm.BatchGenerator, which does not take it. Passing it
        unconditionally raised TypeError on EVERY insert for such models and
        the retry path re-queued the request forever (caught live by the
        release soak on Nemotron-Omni). Probed once per generator instance.
        """
        cached = getattr(self, "_gen_prompt_lens_supported_for", None)
        gen = self.batch_generator
        if cached is not None and cached[0] is gen:
            return cached[1]
        import inspect

        try:
            supported = "gen_prompt_lens" in inspect.signature(gen.insert).parameters
        except (TypeError, ValueError):
            supported = False
        if not supported:
            logger.info(
                "Generator %s.insert() does not take gen_prompt_lens — "
                "cold-prefill split kwarg omitted (generator owns its own "
                "prefill)",
                type(gen).__name__,
            )
        self._gen_prompt_lens_supported_for = (gen, supported)
        return supported

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        if getattr(self, "_uses_m3_msa_cache", False):
            sampler = make_minimax_m3_sampler(
                temp=sampling_params.temperature,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p,
                top_k=sampling_params.top_k,
            )
            logger.info(
                "MiniMax-M3 MSA cache detected — using runtime-compatible "
                "fp32 logits sampler (temp=%s, top_p=%s, top_k=%s); generic "
                "MLX-LM logprob sampler bypassed for M3.",
                sampling_params.temperature,
                sampling_params.top_p,
                sampling_params.top_k,
            )
        else:
            sampler = make_sampler(
                temp=sampling_params.temperature,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p,
                top_k=sampling_params.top_k,
            )
        if float(sampling_params.temperature or 0.0) == 0.0:
            try:
                setattr(sampler, "_vmlx_accepts_logits", True)
            except Exception:
                pass

        # Build logits processors (e.g., repetition penalty).
        #
        # mlx_lm.BatchGenerator applies global processors to the entire prompt
        # context, while mlx_lm.generate_step applies repetition penalty from
        # the final prompt token onward. Install per-request wrappers in
        # _schedule_waiting() so continuous batching and DSV4 custom generation
        # match generate_step without suppressing useful prompt vocabulary on
        # long non-English/code prompts.
        logits_processors = None
        if (
            sampling_params.repetition_penalty
            and sampling_params.repetition_penalty != 1.0
        ):
            from mlx_lm.sample_utils import make_logits_processors

            _rep_context_size = 512 if self._long_repetition_context else 20
            logits_processors = make_logits_processors(
                repetition_penalty=sampling_params.repetition_penalty,
                repetition_context_size=_rep_context_size,
            )

        stop_tokens = self._get_stop_tokens()
        # Add custom stop token IDs
        if sampling_params.stop_token_ids:
            stop_tokens.update(sampling_params.stop_token_ids)

        # Typed prompt snapshots are deep-copied before decode because their
        # path-dependent state cannot be rewound. Give every single-active
        # generator (including the early-return DSV4 path below) the largest
        # single-entry limit of the enabled RAM/disk backends so it can reject
        # an oversize boundary before paying for that copy.
        _prompt_snapshot_max_bytes = _prompt_snapshot_backend_limit_bytes(
            memory_aware_cache=self.memory_aware_cache,
            disk_cache=self.disk_cache,
            block_aware_cache=self.block_aware_cache,
        )

        # DSV4-Flash family bypass. mlx_lm.BatchGenerator's prefill /
        # decode loop calls mx.eval / mx.async_eval / inputs.tolist() on
        # tensors that carry Stream(gpu, N) metadata from MLX C++ internal
        # kernel scheduling — those streams are bound to threads other
        # than the worker, so the worker can't materialise the tensors.
        # Live-traced 18 mitigation iterations (synchronize patches, CPU-
        # stream copies, internal-stream pre-warm, etc.); none survived.
        # Use the DSV4-native generator that calls model() forward + sample
        # in a single pinned stream context per step. Single-batch only
        # (max_num_seqs must be 1).
        try:
            # Sniff the model class name + module path. DSV4 model class
            # comes from jang_tools.dsv4.mlx_model.{Model,DeepseekV4Model}.
            _model_for_sniff = getattr(self.model, "_model", self.model)
            _cls = type(_model_for_sniff)
            _cls_name = _cls.__name__.lower()
            _mod_name = (_cls.__module__ or "").lower()
            _is_dsv4 = (
                "dsv4" in _mod_name
                or "deepseek_v4" in _mod_name
                or "deepseekv4" in _cls_name
                or "dsv4" in _cls_name
            )
            if _is_dsv4:
                from .utils.dsv4_batch_generator import DSV4BatchGenerator
                logger.info(
                    "DSV4-Flash family detected — using DSV4BatchGenerator "
                    "instead of mlx_lm.BatchGenerator (stream-thread bypass, "
                    "single-batch only)"
                )
                return DSV4BatchGenerator(
                    model=self.model,
                    max_tokens=sampling_params.max_tokens,
                    stop_tokens=stop_tokens,
                    sampler=sampler,
                    logits_processors=None,
                    prefill_batch_size=1,
                    completion_batch_size=1,
                    prefill_step_size=self.config.prefill_step_size,
                    capture_prompt_snapshot=self.block_aware_cache is not None,
                    prompt_snapshot_max_bytes=_prompt_snapshot_max_bytes,
                )
        except Exception as _dsv4_err:
            logger.debug(f"DSV4 generator detection failed: {_dsv4_err}")

        try:
            from .native_mtp import model_has_native_mtp_runtime

            if model_has_native_mtp_runtime(self.model):
                logger.info(
                    "Native MTP head detected — using mlx_lm.BatchGenerator "
                    "even with max_num_seqs=1 so the draft/verify runtime and "
                    "SSM rollback path are active."
                )
                generator = BatchGenerator(
                    model=self.model,
                    max_tokens=sampling_params.max_tokens,
                    stop_tokens=stop_tokens,
                    sampler=sampler,
                    logits_processors=None,
                    prefill_batch_size=self.config.prefill_batch_size,
                    completion_batch_size=self.config.completion_batch_size,
                    prefill_step_size=self.config.prefill_step_size,
                )
                if self._uses_glm5_next_cache:
                    generator._vmlx_prompt_boundary_callback = (
                        self._store_glm_prompt_boundary
                    )
                return generator
        except Exception as _mtp_gen_err:
            logger.debug(f"Native MTP generator detection failed: {_mtp_gen_err}")

        if int(getattr(self.config, "max_num_seqs", 1) or 1) <= 1:
            logger.info(
                "max_num_seqs=1 — using vMLX SingleBatchGenerator "
                "(raw native cache, scheduler-owned single-active path)"
            )
            return SingleBatchGenerator(
                model=self.model,
                max_tokens=sampling_params.max_tokens,
                stop_tokens=stop_tokens,
                sampler=sampler,
                logits_processors=None,
                prefill_batch_size=1,
                completion_batch_size=1,
                prefill_step_size=self.config.prefill_step_size,
                prompt_snapshot_max_bytes=_prompt_snapshot_max_bytes,
            )

        return BatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            logits_processors=None,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
        )

    @staticmethod
    def _wrap_generated_only_logits_processor(
        processor: Callable[[Any, Any], Any],
        skip_prefix_tokens: int,
    ) -> Callable[[Any, Any], Any]:
        """Trim prefill-only prompt tokens before applying a logits processor.

        mlx_lm.generate_step evaluates repetition penalty against the final
        prompt token plus generated tokens. BatchGenerator's TokenBuffer includes
        every prompt token by default. Keeping the prompt-wide context changes
        greedy output for prompt-heavy tasks (Ling/Bailing Russian Three.js
        prompt: first token becomes "Организации" instead of "Вот"). This
        wrapper preserves generated-only semantics while still letting
        BatchGenerator own prefill/decode scheduling.
        """

        skip = max(int(skip_prefix_tokens or 0), 0)

        def _wrapped(tokens: Any, logits: Any) -> Any:
            if skip <= 0:
                return processor(tokens, logits)
            try:
                # Preserve at least one token. generate_step includes the last
                # prompt token in the first processor context, then appends
                # generated tokens on later steps.
                n = len(tokens)
                trim = min(skip, max(n - 1, 0))
                if trim > 0:
                    tokens = tokens[trim:]
            except Exception:
                pass
            return processor(tokens, logits)

        return _wrapped

    def _request_logits_processors(
        self,
        request: Request,
        tokens_to_process: List[int],
    ) -> Optional[List[Callable[[Any, Any], Any]]]:
        """Build BatchGenerator processors with generate_step-compatible context."""
        processors: List[Callable[[Any, Any], Any]] = []
        rep = request.sampling_params.repetition_penalty
        if rep and rep != 1.0:
            from mlx_lm.sample_utils import make_logits_processors

            rep_context_size = 512 if self._long_repetition_context else 20
            repetition_processors = make_logits_processors(
                repetition_penalty=rep,
                repetition_context_size=rep_context_size,
            )
            skip_prefix_tokens = max(len(tokens_to_process) - 1, 0)
            processors.extend(
                self._wrap_generated_only_logits_processor(p, skip_prefix_tokens)
                for p in repetition_processors
            )

        params = request.sampling_params
        from .utils.token_logits_processors import (
            make_openai_token_penalty_processor,
        )

        openai_processor = make_openai_token_penalty_processor(
            logit_bias=params.logit_bias,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
        )
        if openai_processor is not None:
            processors.append(openai_processor)
        return processors or None

    def _request_seeded_sampler(self, request: Request):
        """Return a stable request-local sampler when the API supplied a seed."""
        seed = request.sampling_params.seed
        if seed is None:
            return None
        cached = getattr(request, "_seeded_sampler", None)
        if cached is not None:
            return cached
        params = request.sampling_params
        if getattr(self, "_uses_m3_msa_cache", False):
            sampler = make_minimax_m3_sampler(
                temp=params.temperature,
                top_p=params.top_p,
                min_p=params.min_p,
                top_k=params.top_k,
                seed=int(seed),
            )
        else:
            sampler = make_sampler(
                temp=params.temperature,
                top_p=params.top_p,
                min_p=params.min_p,
                top_k=params.top_k,
                seed=int(seed),
            )
        request._seeded_sampler = sampler
        return sampler

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> bool:
        """Ensure BatchGenerator exists with compatible settings."""
        sampler_params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
            sampling_params.top_k,
            sampling_params.repetition_penalty,
        )
        cleared_cache = False

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
                return False

            # Clear all prefix caches when BatchGenerator changes —
            # BatchKVCache objects and block tables are tied to their generator instance
            if self.batch_generator is not None:
                if self.block_aware_cache is not None:
                    # Block-aware paged cache stores extracted per-layer state
                    # and reconstructs fresh cache objects on the scheduler
                    # worker. It is not sampler-bound, so preserve it across
                    # BatchGenerator recreation; clearing here creates stale
                    # hit bookkeeping and destroys legitimate prefix reuse.
                    logger.debug(
                        "Preserving paged cache across BatchGenerator recreation"
                    )
                elif self.memory_aware_cache is not None:
                    logger.debug(
                        "Clearing memory-aware cache: BatchGenerator being recreated"
                    )
                    self.memory_aware_cache.clear()
                    cleared_cache = True
                elif self.prefix_cache is not None:
                    logger.debug(
                        "Clearing prefix cache: BatchGenerator being recreated"
                    )
                    self.prefix_cache.clear()
                    cleared_cache = True

            self.batch_generator = self._create_batch_generator(sampling_params)
            self._current_sampler_params = sampler_params
        return cleared_cache

    def warm_tq_storage_decoders(self) -> Dict[str, Any]:
        """Eagerly materialize native TQ restore codecs on the model worker.

        This runs during engine start, before readiness is advertised, so the
        first paged/L2 hit does not become the accidental synchronization point
        for per-layer seeded TurboQuant decoder state.
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
        if not (
            self._tq_active
            and self.config.enable_prefix_cache
            and hasattr(self.model, "make_cache")
        ):
            self._tq_decoder_warmup_stats = stats
            return stats

        started = time.perf_counter()
        cache_layers = None
        try:
            from .tq_disk_store import warm_tq_decoder_states

            cache_layers = self.model.make_cache() or []
            model_args = getattr(self.model, "args", None)
            probe_heads = int(getattr(model_args, "num_key_value_heads", 1) or 1)
            stats.update(
                warm_tq_decoder_states(cache_layers, probe_heads=probe_heads)
            )
            stats["enabled"] = bool(stats.get("configs"))
            stats["seconds"] = round(time.perf_counter() - started, 6)
            logger.info(
                "TurboQuant storage decoder startup warmup: configs=%d "
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
            logger.warning("TurboQuant storage decoder startup warmup failed: %s", exc)
        finally:
            del cache_layers
        self._tq_decoder_warmup_stats = stats
        return stats

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
        try:
            from .cache_record_validator import reject_live_cache_or_warn
            return reject_live_cache_or_warn(cache, source="scheduler-live-cache")
        except Exception:
            if cache is None or cache == []:
                return False
            if isinstance(cache, list):
                return all(c is not None for c in cache)
            return True

    @staticmethod
    def _cache_reuse_budget_fraction() -> float:
        """Fraction of currently available RAM allowed for cache merge scratch."""
        raw = os.environ.get("VMLX_CACHE_REUSE_BUDGET_FRACTION", "0.85")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.85
        return max(0.10, min(0.95, value))

    @staticmethod
    def _prefix_hit_tail_and_cached_tokens(
        *,
        fetch_tokens: List[int],
        remaining: List[int],
        gen_prompt_suffix: List[int],
    ) -> Tuple[List[int], int]:
        """Thin adapter over the shared N-1 prefix-hit arithmetic."""
        return _shared_prefix_hit(
            key_tokens=fetch_tokens,
            remaining=remaining,
            gen_prompt_suffix=gen_prompt_suffix,
        )

    def _dsv4_snapshot_store_below_threshold(self, request) -> bool:
        """True when a DSV4 prompt snapshot is too short to store on its own.

        Short prompts now capture snapshots solely to seed the extended
        decode-time chain. When that chain never reached a block boundary
        (short output too), storing the tiny N-1 snapshot would recreate the
        cost the snapshot threshold exists to avoid — fall through to the
        deferred path, which logs the skip.
        """
        if not self._uses_dsv4_cache:
            return False
        prompt_tokens = list(request.prompt_token_ids or [])
        gpl = int(getattr(request, "_gen_prompt_len", 0) or 0)
        if 0 < gpl < len(prompt_tokens):
            prompt_tokens = prompt_tokens[:-gpl]
        key_len = max(0, len(prompt_tokens) - 1)
        try:
            from .utils.dsv4_batch_generator import (
                dsv4_prompt_snapshot_min_tokens as _min_tokens,
            )

            min_tokens = _min_tokens()
        except Exception:
            min_tokens = 256
        return key_len < min_tokens

    def _log_dsv4_extended_match_divergence(self, request) -> None:
        """Log where an incoming prompt diverges from the last extended store.

        Block-hash matching only reports reuse at block granularity; when the
        panel's re-rendered history drifts from the raw fed+generated stream
        (the #93 lead), this pinpoints the exact first differing token so the
        glue defect can be fixed instead of guessed at.
        """
        stored = getattr(self, "_dsv4_last_extended_key_tokens", None)
        if not stored:
            return
        prompt = list(request.prompt_token_ids or [])
        matched = int(getattr(request, "_cache_matched_tokens", 0) or 0)
        limit = min(len(stored), len(prompt))
        if limit <= 0 or matched >= limit:
            return
        first_diff = next(
            (i for i in range(limit) if stored[i] != prompt[i]), limit
        )
        if first_diff >= limit and len(prompt) >= len(stored):
            return
        lo = max(0, first_diff - 8)
        hi = min(limit, first_diff + 8)
        stored_win = stored[lo:hi]
        prompt_win = prompt[lo:hi]
        decode = getattr(
            getattr(self, "_actual_tokenizer", None) or self.tokenizer,
            "decode",
            None,
        )
        try:
            stored_txt = decode(stored_win) if decode else ""
            prompt_txt = decode(prompt_win) if decode else ""
        except Exception:
            stored_txt = prompt_txt = "<decode failed>"
        logger.info(
            "DSV4 extended-match divergence: first_diff=%d matched=%d "
            "stored_len=%d prompt_len=%d window=[%d:%d) stored_ids=%s "
            "prompt_ids=%s stored_text=%r prompt_text=%r",
            first_diff,
            matched,
            len(stored),
            len(prompt),
            lo,
            hi,
            stored_win,
            prompt_win,
            stored_txt,
            prompt_txt,
        )

    @staticmethod
    def _disk_prefix_hit_tail_and_cached_tokens(
        *,
        fetch_tokens: List[int],
        matched_tokens: List[int],
        gen_prompt_suffix: List[int],
    ) -> Tuple[List[int], int]:
        """Thin adapter over the shared disk L2 partial-hit arithmetic."""
        return _shared_disk_prefix_hit(
            fetch_tokens=fetch_tokens,
            matched_tokens=matched_tokens,
            gen_prompt_suffix=gen_prompt_suffix,
        )

    def _cache_selection_hot_advantage_threshold(self) -> int:
        raw = os.environ.get("VMLINUX_CACHE_SELECTION_HOT_ADVANTAGE_TOKENS", "64")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 64

    def _update_cache_admission_rate(
        self,
        name: str,
        sample: float,
        reference_tokens: int,
    ) -> None:
        if sample <= 0.0 or reference_tokens <= 0:
            return
        value_attr = f"_cache_admission_{name}_seconds_per_token"
        count_attr = f"_cache_admission_{name}_sample_count"
        reference_attr = f"_cache_admission_{name}_reference_tokens"
        current = max(0.0, float(getattr(self, value_attr, 0.0) or 0.0))
        count = max(0, int(getattr(self, count_attr, 0) or 0))
        prior_reference = max(
            0,
            int(getattr(self, reference_attr, 0) or 0),
        )
        comparable_ratio = max(
            1.0,
            float(self._cache_admission_max_comparable_ratio or 2.0),
        )
        same_context_scale = bool(
            prior_reference > 0
            and reference_tokens <= prior_reference * comparable_ratio
            and prior_reference <= reference_tokens * comparable_ratio
        )
        if count == 0 or not same_context_scale:
            updated = float(sample)
            updated_count = 1
        else:
            alpha = min(
                1.0,
                max(0.0, float(getattr(self, "_cache_admission_alpha", 0.25))),
            )
            updated = alpha * float(sample) + (1.0 - alpha) * current
            updated_count = count + 1
        setattr(self, value_attr, updated)
        setattr(self, count_attr, updated_count)
        # Use the most recent context scale as the comparison anchor. Prefill
        # cost is not linear at long context, so a short historical sample must
        # never veto a much larger restart restore merely because its local
        # seconds/token happened to be lower.
        setattr(self, reference_attr, int(reference_tokens))

    def _record_disk_admission_sample(
        self,
        *,
        cached_tokens: int,
        retrieval_seconds: float,
    ) -> None:
        tokens = max(0, int(cached_tokens or 0))
        seconds = max(0.0, float(retrieval_seconds or 0.0))
        if tokens > 0 and seconds > 0.0:
            self._update_cache_admission_rate("disk", seconds / tokens, tokens)

    def _record_clean_prefill_admission_sample(
        self,
        request: Request,
        ttft_seconds: float,
    ) -> None:
        execution = getattr(request, "_cache_execution", None)
        if not isinstance(execution, dict):
            return
        if int(execution.get("cached_tokens", 0) or 0) > 0:
            return
        prefill_tokens = max(0, int(execution.get("prefill_tokens", 0) or 0))
        reference_tokens = max(
            prefill_tokens,
            int(execution.get("prompt_tokens", 0) or 0),
        )
        seconds = max(0.0, float(ttft_seconds or 0.0))
        if prefill_tokens > 0 and seconds > 0.0:
            self._update_cache_admission_rate(
                "prefill",
                seconds / prefill_tokens,
                reference_tokens,
            )

    def _record_cache_admission_first_token_sample(
        self,
        request: Request,
        *,
        scheduled_ttft_seconds: float,
        admission_first_token_seconds: Optional[float] = None,
    ) -> None:
        """Finish request-local admission timing when the first token arrives.

        A persisted block hit can read its tensor payloads in either of two
        places.  On a fresh process, ``fetch_cache()`` hydrates them while
        resolving the candidate on the API thread; once its metadata is warm,
        worker reconstruction performs the reads instead.  Admission must use
        both intervals or identical SSD work appears artificially cheap after
        restart.

        The accepted hit also measures its uncached tail on this exact loaded
        model.  Removing cache preparation from scheduled TTFT yields a
        conservative tail-prefill/first-token sample, so one restart refault
        seeds both sides of the *next* SSD-versus-clean decision.
        """
        execution = getattr(request, "_cache_execution", None)
        if not isinstance(execution, dict):
            return

        scheduled_ttft = max(0.0, float(scheduled_ttft_seconds or 0.0))
        candidate_lookup = max(
            0.0,
            float(execution.get("candidate_lookup_seconds", 0.0) or 0.0),
        )
        worker_cache = max(
            0.0,
            float(execution.get("total_worker_cache_seconds", 0.0) or 0.0),
        )
        admission_first_token = max(
            0.0,
            float(admission_first_token_seconds)
            if admission_first_token_seconds is not None
            else candidate_lookup + scheduled_ttft,
        )
        execution["candidate_lookup_seconds"] = round(candidate_lookup, 6)
        execution["admission_first_token_seconds"] = round(
            admission_first_token,
            6,
        )

        cached_tokens = max(0, int(execution.get("cached_tokens", 0) or 0))
        selection = execution.get("selection")
        selected_paged = bool(
            isinstance(selection, dict) and selection.get("selected") == "paged"
        )
        disk_only_hit = bool(
            cached_tokens > 0
            and execution.get("cache_reuse_applied") is True
            and selected_paged
            and bool(getattr(self.paged_cache_manager, "disk_only", False))
        )
        if disk_only_hit:
            ssd_retrieval = candidate_lookup + worker_cache
            execution["ssd_retrieval_seconds"] = round(ssd_retrieval, 6)
            self._record_disk_admission_sample(
                cached_tokens=cached_tokens,
                retrieval_seconds=ssd_retrieval,
            )

            tail_tokens = max(
                0,
                int(execution.get("uncached_prompt_tokens", 0) or 0),
            )
            prompt_tokens = max(
                tail_tokens,
                int(execution.get("prompt_tokens", 0) or 0),
            )
            tail_prefill = max(0.0, scheduled_ttft - worker_cache)
            if tail_tokens > 0 and tail_prefill > 0.0:
                self._update_cache_admission_rate(
                    "prefill",
                    tail_prefill / tail_tokens,
                    prompt_tokens,
                )
        else:
            self._record_clean_prefill_admission_sample(
                request,
                scheduled_ttft,
            )

        request._cache_execution = {
            key: value for key, value in execution.items() if value is not None
        }
        self._last_cache_execution = dict(request._cache_execution)

    def _should_clean_prefill_over_disk_only(
        self,
        *,
        paged_cached_tokens: int,
        paged_cold_tokens: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        cached = max(0, int(paged_cached_tokens or 0))
        cold = max(0, int(paged_cold_tokens or 0))
        prefill_rate = max(
            0.0,
            float(
                getattr(
                    self,
                    "_cache_admission_prefill_seconds_per_token",
                    0.0,
                )
                or 0.0
            ),
        )
        disk_rate = max(
            0.0,
            float(
                getattr(
                    self,
                    "_cache_admission_disk_seconds_per_token",
                    0.0,
                )
                or 0.0
            ),
        )
        prefill_samples = max(
            0,
            int(
                getattr(self, "_cache_admission_prefill_sample_count", 0) or 0
            ),
        )
        disk_samples = max(
            0,
            int(getattr(self, "_cache_admission_disk_sample_count", 0) or 0),
        )
        prefill_reference = max(
            0,
            int(
                getattr(self, "_cache_admission_prefill_reference_tokens", 0)
                or 0
            ),
        )
        disk_reference = max(
            0,
            int(
                getattr(self, "_cache_admission_disk_reference_tokens", 0) or 0
            ),
        )
        comparable_ratio = max(
            1.0,
            float(
                getattr(self, "_cache_admission_max_comparable_ratio", 2.0)
                or 2.0
            ),
        )

        def _comparable(reference: int) -> bool:
            return bool(
                cached > 0
                and reference > 0
                and cached <= reference * comparable_ratio
                and reference <= cached * comparable_ratio
            )

        comparable = bool(
            prefill_samples > 0
            and disk_samples > 0
            and _comparable(prefill_reference)
            and _comparable(disk_reference)
        )
        estimated_disk = disk_rate * cold
        estimated_prefill = prefill_rate * cached
        decision: Dict[str, Any] = {
            "prefill_seconds_per_token": round(prefill_rate, 9),
            "disk_seconds_per_token": round(disk_rate, 9),
            "prefill_cost_samples": prefill_samples,
            "disk_cost_samples": disk_samples,
            "prefill_reference_tokens": prefill_reference,
            "disk_reference_tokens": disk_reference,
            "cost_history_comparable": comparable,
            "max_comparable_ratio": comparable_ratio,
            "estimated_disk_seconds": round(estimated_disk, 6),
            "estimated_prefill_seconds": round(estimated_prefill, 6),
        }
        if comparable and prefill_rate > 0.0 and disk_rate > 0.0:
            reject = estimated_disk >= estimated_prefill
            decision["cost_reason"] = (
                "estimated_clean_prefill_faster"
                if reject
                else "estimated_disk_restore_faster"
            )
            return reject, decision
        decision["cost_reason"] = (
            "insufficient_cost_history"
            if prefill_samples == 0 or disk_samples == 0
            else "noncomparable_context_scale"
        )
        return False, decision

    def _detach_paged_candidate(self, request_id: str, block_table: Any) -> None:
        """Release a fetched block table that admission chose not to consume."""
        if self.block_aware_cache is None:
            return
        try:
            self.block_aware_cache.paged_cache.release_request_refs(block_table)
            # BlockAwarePrefixCache and its PagedCache each own an index entry.
            # Drop both exactly as the existing warm-prefix preference path did.
            detach = getattr(self.block_aware_cache, "detach_request", None)
            if callable(detach):
                detach(request_id)
            self.block_aware_cache.paged_cache.detach_request(request_id)
        except Exception:
            pass

    def _paged_cold_block_tokens(self, block_table: Any) -> int:
        if self.block_aware_cache is None or not block_table:
            return 0
        paged_cache = getattr(self.block_aware_cache, "paged_cache", None)
        blocks = getattr(paged_cache, "allocated_blocks", {}) or {}
        total = 0
        for block_id in getattr(block_table, "block_ids", []) or []:
            block = blocks.get(block_id)
            if block is not None and getattr(block, "cache_data_from_disk", False):
                total += max(0, int(getattr(block, "token_count", 0) or 0))
        return total

    def _warm_prefix_alternative(
        self,
        fetch_tokens: List[int],
        gen_prompt_suffix: List[int],
    ) -> Tuple[Any | None, List[int], int, str | None]:
        cache = None
        remaining: List[int] = []
        detail: str | None = None
        if self.memory_aware_cache is not None:
            cache, raw_remaining = self.memory_aware_cache.fetch(fetch_tokens)
            if cache:
                remaining, cached_tokens = self._prefix_hit_tail_and_cached_tokens(
                    fetch_tokens=fetch_tokens,
                    remaining=raw_remaining or [],
                    gen_prompt_suffix=gen_prompt_suffix,
                )
                return cache, remaining, cached_tokens, "memory"
        if self.prefix_cache is not None:
            cache, raw_remaining = self.prefix_cache.fetch_cache(fetch_tokens)
            if cache:
                remaining, cached_tokens = self._prefix_hit_tail_and_cached_tokens(
                    fetch_tokens=fetch_tokens,
                    remaining=raw_remaining or [],
                    gen_prompt_suffix=gen_prompt_suffix,
                )
                return cache, remaining, cached_tokens, "prefix"
        return None, [], 0, detail

    def _should_prefer_warm_prefix_over_paged(
        self,
        *,
        paged_cached_tokens: int,
        paged_cold_tokens: int,
        warm_cached_tokens: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        threshold = self._cache_selection_hot_advantage_threshold()
        hot_advantage = (
            max(0, int(paged_cached_tokens or 0))
            - max(0, int(paged_cold_tokens or 0))
            - max(0, int(warm_cached_tokens or 0))
        )
        decision = {
            "paged_cached_tokens": max(0, int(paged_cached_tokens or 0)),
            "paged_cold_tokens": max(0, int(paged_cold_tokens or 0)),
            "warm_cached_tokens": max(0, int(warm_cached_tokens or 0)),
            "hot_advantage_tokens": hot_advantage,
            "hot_advantage_threshold_tokens": threshold,
        }
        return warm_cached_tokens > 0 and hot_advantage < threshold, decision

    def _remaining_tokens_after_cached_prefix(
        self,
        request: Request,
        cached_tokens: int,
    ) -> List[int]:
        """Return the prefill tail after reusing a possibly-shrunk prefix.

        `add_request()` strips generation-prompt suffix tokens from the cache
        lookup key, then appends the suffix back to the remaining prefill tail.
        Memory-pressure shrinking must preserve that contract or thinking /
        assistant rail suffixes can be accidentally dropped on partial reuse.
        """
        prompt_tokens = list(request.prompt_token_ids or [])
        if not prompt_tokens:
            return []
        cached_tokens = max(0, int(cached_tokens or 0))
        gen_prompt_len = int(getattr(request, "_gen_prompt_len", 0) or 0)
        if (
            0 < gen_prompt_len < len(prompt_tokens)
            and not getattr(self, "_mixed_attention_cache_model", False)
        ):
            fetch_tokens = prompt_tokens[:-gen_prompt_len]
            suffix = prompt_tokens[-gen_prompt_len:]
            cached_tokens = min(cached_tokens, len(fetch_tokens))
            return list(fetch_tokens[cached_tokens:]) + list(suffix)
        cached_tokens = min(cached_tokens, len(prompt_tokens))
        return list(prompt_tokens[cached_tokens:])

    def _release_unusable_paged_hit(self, request: Request) -> None:
        """Drop refs and optimistic credit for a paged hit not actually used."""
        if self.block_aware_cache is not None:
            adjust = getattr(self.block_aware_cache, "adjust_cache_hit_credit", None)
            if callable(adjust):
                try:
                    adjust(request.request_id, accepted_tokens=0)
                except Exception as exc:
                    logger.debug(
                        "Request %s: failed to roll back unused cache-hit credit: %s",
                        request.request_id,
                        exc,
                    )
        block_table = getattr(request, "block_table", None)
        if block_table is not None and self.block_aware_cache is not None:
            try:
                paged_cache = getattr(self.block_aware_cache, "paged_cache", None)
                if paged_cache is not None and hasattr(
                    paged_cache, "release_request_refs"
                ):
                    paged_cache.release_request_refs(block_table)
                if hasattr(self.block_aware_cache, "detach_request"):
                    self.block_aware_cache.detach_request(request.request_id)
            except Exception as exc:
                logger.debug(
                    "Request %s: failed to release unused paged-cache refs: %s",
                    request.request_id,
                    exc,
                )
        request.block_table = None
        request.shared_prefix_blocks = 0
        request._cache_matched_tokens = 0
        request._cache_checkpoint_tokens = 0
        request._cache_replayed_tokens = 0

    def _accept_paged_hit_credit(self, request: Request, accepted_tokens: int) -> None:
        """Reconcile a fetched block hit to the prefix generation really used."""
        if self.block_aware_cache is None:
            return
        adjust = getattr(self.block_aware_cache, "adjust_cache_hit_credit", None)
        if callable(adjust):
            try:
                adjust(request.request_id, accepted_tokens=accepted_tokens)
            except Exception as exc:
                logger.debug(
                    "Request %s: failed to reconcile cache-hit credit: %s",
                    request.request_id,
                    exc,
                )

    def _retarget_ssm_rederive_to_paged_boundary(
        self,
        request_id: str,
        store_tokens: List[int],
        block_table: Any,
    ) -> None:
        """Align a queued clean SSM snapshot with the KV prefix actually stored.

        A small block pool can persist only a prefix of a longer prompt.  The
        companion queue was previously keyed to the full N-1 prompt, which can
        never pair with the shorter KV chain.  Replace that request's pending
        task with the real block-table boundary so the next exact/longer prompt
        can consume every durable KV page instead of remaining pinned to an old
        shorter SSM checkpoint.
        """
        if not getattr(self, "_is_hybrid", False):
            return
        queue = getattr(self, "_ssm_rederive_queue", None)
        if not queue:
            return
        try:
            boundary = int(getattr(block_table, "num_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        if boundary <= 0 or boundary >= len(store_tokens):
            return
        retained = [
            item
            for item in queue
            if not (len(item) >= 3 and item[2] == request_id)
        ]
        if len(retained) == len(queue):
            return
        retained.append((list(store_tokens[:boundary]), boundary, request_id))
        self._ssm_rederive_queue = retained
        logger.info(
            "SSM companion: retargeted deferred re-derive for %s from %d "
            "to stored paged boundary %d tokens",
            request_id,
            len(store_tokens),
            boundary,
        )

    def _persist_hybrid_ssd_companion(
        self, request: Request, store_tokens: List[int], block_table: Any
    ) -> None:
        """Publish the native half of a text-hybrid SSD prefix before its fence.

        RAM capacity zero must not disable SSD write-through. Deferring the
        companion until idle (or deriving it on the next hit) leaves a durable
        KV-only prefix behind the tool-call publication barrier. Work here is
        request-owned and aligned to the prefix the block store actually kept.
        Other typed/native cache contracts retain their own publication paths.
        """
        companion = getattr(self, "_ssm_state_cache", None)
        if (
            not getattr(self, "_is_hybrid", False)
            or getattr(self, "_uses_dsv4_cache", False)
            or getattr(self, "_uses_zaya_cache", False)
            or getattr(self, "_mixed_attention_cache_model", False)
            or companion is None
            or not companion.disk_enabled
        ):
            return
        boundary = int(getattr(block_table, "num_tokens", 0) or 0)
        if boundary <= 0 or boundary > len(store_tokens):
            raise RuntimeError("hybrid SSD companion has no valid retained KV boundary")
        # This text-only rederive cannot reproduce media embeddings. Do not
        # publish a complete-state claim under a media-salted KV identity.
        if (
            getattr(request, "images", None)
            or getattr(request, "videos", None)
            or getattr(request, "pixel_values", None) is not None
            or getattr(request, "pixel_values_videos", None) is not None
        ):
            raise RuntimeError("hybrid SSD companion requires its media-aware owner")
        tokens = list(store_tokens[:boundary])
        started = time.perf_counter()
        seed_len = 0
        if not companion.has_complete(tokens, boundary):
            reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
            if reconstructed is not None and getattr(self, "_kv_cache_bits", 0):
                reconstructed = self._dequantize_cache_for_use(reconstructed)
            seed, seed_len = self._seed_cache_from_ssm_checkpoint(
                request, reconstructed=reconstructed or [],
                ssm_tokens=tokens, fetch_num=boundary,
            )
            clean = self._prefill_for_prompt_only_cache(
                tokens, base_cache=seed, base_token_count=seed_len,
            )
            if clean is None or len(clean) != self._hybrid_num_layers:
                raise RuntimeError("hybrid SSD companion clean boundary unavailable")
            kv_positions = set(self._hybrid_kv_positions or [])
            states = [layer for i, layer in enumerate(clean) if i not in kv_positions]
            if not states:
                raise RuntimeError("hybrid SSD companion native layers unavailable")
            # store() detaches and materializes each native layer; never donate
            # mutable post-generation state or retain an unbounded RAM mirror.
            companion.store(tokens, boundary, states)
        key = companion._key(tokens, boundary)
        if not companion._disk.wait_for_write(key, timeout=30.0):
            raise RuntimeError("hybrid SSD companion write failed or timed out")
        self._ssm_rederive_queue = [
            item for item in getattr(self, "_ssm_rederive_queue", [])
            if not (len(item) >= 3 and item[2] == request.request_id)
        ]
        logger.info(
            "Hybrid SSD companion terminal: request=%s tokens=%d seed_tokens=%d "
            "hash=%s durable=true elapsed_ms=%.3f",
            request.request_id, boundary, seed_len, key[:12],
            (time.perf_counter() - started) * 1000.0,
        )

    def _cache_reuse_contract(self) -> str:
        """Name the active prompt-cache state contract for telemetry/policy."""
        if self._uses_dsv4_cache:
            return "deepseek_v4_composite"
        if self._uses_zaya_cache:
            return "zaya_cca"
        if self._is_hybrid:
            return "hybrid_ssm"
        try:
            if getattr(self, "_mixed_attention_cache_model", False):
                return "mixed_swa_kv"
        except Exception:
            pass
        if getattr(self, "_tq_active", False):
            return "turboquant_kv"
        return "plain_kv"

    def _memory_fit_target_cached_tokens(
        self,
        *,
        original_cached_tokens: int,
        cache_bytes: int,
        available_bytes: int,
        multiplier: float,
        block_size: int,
        budget_fraction: Optional[float] = None,
    ) -> int:
        """Compute a block-aligned cached-token target that fits memory budget."""
        if (
            original_cached_tokens <= 0
            or cache_bytes <= 0
            or available_bytes <= 0
            or multiplier <= 0
            or block_size <= 0
        ):
            return 0
        if budget_fraction is None:
            budget_fraction = self._cache_reuse_budget_fraction()
        budget_fraction = max(0.10, min(0.95, float(budget_fraction)))
        bytes_per_token = max(float(cache_bytes) / float(original_cached_tokens), 1.0)
        cache_budget_bytes = float(available_bytes) * budget_fraction / float(
            multiplier
        )
        target_tokens = int(cache_budget_bytes / bytes_per_token)
        target_tokens = min(target_tokens, original_cached_tokens - 1)
        target_tokens = (target_tokens // block_size) * block_size
        return target_tokens if target_tokens >= block_size else 0

    def _cache_reuse_cache_format(self, cache: Any) -> str:
        """Describe the actual cache object used for memory-fit accounting."""

        flags: set[str] = set()

        def _visit(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, dict):
                for value in obj.values():
                    _visit(value)
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)
                return

            cls_name = type(obj).__name__
            if cls_name == "QuantizedKVCache":
                flags.add("quantized_kv")
                return
            if cls_name == "TurboQuantKVCache":
                flags.add("turboquant_kv")
                return
            if self._is_dsv4_cache_object(obj):
                local = getattr(obj, "local", None)
                if type(local).__name__ == "QuantizedKVCache":
                    flags.add("dsv4_quantized_local")
                else:
                    flags.add("dsv4_composite")
                return

            caches = getattr(obj, "caches", None)
            if isinstance(caches, (list, tuple)):
                for item in caches:
                    _visit(item)
                return

            if cls_name in ("KVCache", "RotatingKVCache", "BatchKVCache") or (
                hasattr(obj, "keys") and hasattr(obj, "values")
            ):
                flags.add("full_precision_kv")
            elif cls_name in ("MambaCache", "ArraysCache"):
                flags.add("state_cache")

        _visit(cache)
        if not flags:
            return "unknown"
        if len(flags) == 1:
            return next(iter(flags))
        order = (
            "turboquant_kv",
            "quantized_kv",
            "dsv4_quantized_local",
            "dsv4_composite",
            "full_precision_kv",
            "state_cache",
        )
        return "+".join(flag for flag in order if flag in flags)

    def _cache_reuse_quant_bits(self, cache: Any) -> int:
        """Return the smallest quantized-KV bit width present in a cache tree."""

        bits_seen: list[int] = []

        def _maybe_bits(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, dict):
                for value in obj.values():
                    _maybe_bits(value)
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _maybe_bits(item)
                return

            cls_name = type(obj).__name__
            if cls_name == "QuantizedKVCache":
                raw_bits = getattr(obj, "bits", None) or getattr(
                    self,
                    "_kv_cache_bits",
                    0,
                )
                try:
                    raw_bits = int(raw_bits or 0)
                except (TypeError, ValueError):
                    raw_bits = 0
                if raw_bits > 0:
                    bits_seen.append(raw_bits)
                return

            if self._is_dsv4_cache_object(obj):
                _maybe_bits(getattr(obj, "local", None))
                return

            caches = getattr(obj, "caches", None)
            if isinstance(caches, (list, tuple)):
                for item in caches:
                    _maybe_bits(item)

        _maybe_bits(cache)
        return min(bits_seen) if bits_seen else 0

    def _cache_merge_memory_multiplier(self, cache: Any) -> float:
        """Scratch-memory multiplier for merging this specific cache object.

        A global q4/q8 setting is not enough here: some fetch paths already
        dequantize a hit before the memory-fit check. Charging those full
        precision caches as q4/q8 makes the scheduler drop valid long-prefix
        hits and full-prefill large prompts.
        """

        quant_bits = self._cache_reuse_quant_bits(cache)
        if quant_bits and quant_bits <= 4:
            return 5.0
        if quant_bits and quant_bits <= 8:
            return 3.0
        return 2.0

    def _hybrid_ssm_fetch_tokens(self, request: Request) -> List[int]:
        """Return the token key used for hybrid SSM companion cache lookups."""
        explicit = getattr(request, "_hybrid_ssm_fetch_tokens", None)
        if explicit is not None:
            return list(explicit)
        return prefix_cache_key_tokens(request)

    def _fetch_block_aligned_ssm_checkpoint(
        self,
        request: Request,
        *,
        max_len: int,
        block_size: int,
        exact_boundary_already_missed: bool = False,
        retain_unaligned_checkpoint: bool = False,
    ) -> Optional[Tuple[int, List[Any]]]:
        """Find a hybrid SSM checkpoint aligned to paged-KV block trimming.

        KV block tables can only be trimmed to whole blocks. SSM state is
        cumulative, so pairing KV@128 with SSM@150 would corrupt the next
        forward pass. This helper accepts only checkpoints whose token length
        equals the block-aligned KV length, or an exact SSM checkpoint at that
        aligned length.
        """
        prefetched_attr = "_hybrid_ssm_prefetched_checkpoint"

        def _clear_prefetched_checkpoint() -> None:
            try:
                delattr(request, prefetched_attr)
            except AttributeError:
                pass

        if retain_unaligned_checkpoint:
            # Request objects can be retried. Never let a prior attempt's
            # detached recurrent state leak into a later lookup.
            _clear_prefetched_checkpoint()

        if (
            self._ssm_state_cache is None
            or max_len <= 0
            or block_size <= 0
        ):
            request._cache_reuse_partial_unavailable_reason = "ssm_checkpoint_unavailable"
            return None
        fetch_longest = getattr(self._ssm_state_cache, "fetch_longest_prefix", None)
        fetch_exact = getattr(self._ssm_state_cache, "fetch", None)
        if fetch_longest is None:
            request._cache_reuse_partial_unavailable_reason = "ssm_longest_prefix_unavailable"
            return None

        ssm_tokens = self._hybrid_ssm_fetch_tokens(request)
        search_len = (int(max_len) // block_size) * block_size
        while search_len >= block_size:
            skip_exact_probe = bool(
                exact_boundary_already_missed and search_len == int(max_len)
            )
            try:
                if skip_exact_probe:
                    hit = fetch_longest(
                        ssm_tokens,
                        search_len,
                        exact_boundary_already_missed=True,
                    )
                else:
                    hit = fetch_longest(ssm_tokens, search_len)
            except TypeError as exc:
                # Preserve compatibility with custom/legacy companion-cache
                # implementations that predate the duplicate-probe hint.
                if (
                    not skip_exact_probe
                    or "exact_boundary_already_missed" not in str(exc)
                ):
                    raise
                hit = fetch_longest(ssm_tokens, search_len)
            if hit is None:
                request._cache_reuse_partial_unavailable_reason = (
                    "no_block_aligned_ssm_checkpoint"
                )
                return None
            checkpoint_len, states, is_complete = hit
            checkpoint_len = int(checkpoint_len or 0)
            if checkpoint_len > search_len:
                request._cache_reuse_partial_unavailable_reason = (
                    "invalid_ssm_checkpoint_length"
                )
                return None
            aligned_len = (checkpoint_len // block_size) * block_size
            if checkpoint_len <= 0 or aligned_len <= 0:
                request._cache_reuse_partial_unavailable_reason = (
                    "invalid_ssm_checkpoint_length"
                )
                return None
            if is_complete and checkpoint_len == aligned_len:
                _clear_prefetched_checkpoint()
                return checkpoint_len, states
            if (
                retain_unaligned_checkpoint
                and is_complete
                and not hasattr(request, prefetched_attr)
            ):
                # The aligned RESUME lane cannot use this state because its KV
                # table can only be trimmed at whole-block boundaries. The
                # delta-derive fallback can use the same detached state at any
                # boundary, so retain it for that immediately following path
                # instead of physically loading it from SSD a second time.
                setattr(
                    request,
                    prefetched_attr,
                    (int(max_len), checkpoint_len, states, True),
                )
            if fetch_exact is not None and aligned_len > 0:
                exact = fetch_exact(ssm_tokens, aligned_len)
                if exact is not None:
                    exact_states, exact_complete = exact
                    if exact_complete:
                        _clear_prefetched_checkpoint()
                        return aligned_len, exact_states
            search_len = aligned_len - block_size
        request._cache_reuse_partial_unavailable_reason = (
            "no_block_aligned_ssm_checkpoint"
        )
        return None

    def _record_cache_reuse_partial(
        self,
        *,
        request: Request,
        cache_to_use: Any,
        cache_bytes: int,
        available_bytes: int,
        multiplier: float,
        budget_fraction: float,
        original_cached_tokens: int,
        used_cached_tokens: int,
        remaining_tokens: List[int],
        cache_contract: str,
    ) -> None:
        # Paged lookup credits the full prefix it found before worker-side
        # memory fitting. Reconcile that optimistic credit to the smaller
        # block-aligned prefix generation actually consumes.
        self._accept_paged_hit_credit(request, used_cached_tokens)
        budget_bytes = float(available_bytes) * float(budget_fraction)
        if original_cached_tokens > 0 and cache_bytes > 0:
            used_cache_bytes = float(cache_bytes) * (
                float(used_cached_tokens) / float(original_cached_tokens)
            )
        else:
            used_cache_bytes = 0.0
        self._cache_reuse_partial_downgrades += 1
        self._cache_reuse_partial_tokens += used_cached_tokens
        self._last_cache_reuse_partial = {
            "request_id": request.request_id,
            "reason": "insufficient_memory_for_full_cache_merge",
            "cache_contract": cache_contract,
            "cache_format": self._cache_reuse_cache_format(cache_to_use),
            "available_bytes": int(available_bytes),
            "cache_bytes": int(cache_bytes),
            "budget_bytes": budget_bytes,
            "original_needed_bytes": float(cache_bytes) * float(multiplier),
            "used_cache_bytes": used_cache_bytes,
            "used_needed_bytes": used_cache_bytes * float(multiplier),
            "original_needed_mb": round((cache_bytes * multiplier) / 1048576, 1),
            "budget_mb": round(budget_bytes / 1048576, 1),
            "available_mb": round(available_bytes / 1048576, 1),
            "original_cache_mb": round(cache_bytes / 1048576, 1),
            "used_cache_mb": round(used_cache_bytes / 1048576, 1),
            "used_needed_mb": round((used_cache_bytes * multiplier) / 1048576, 1),
            "multiplier": multiplier,
            "budget_fraction": budget_fraction,
            "kv_cache_bits": getattr(self, "_kv_cache_bits", 0),
            "original_cached_tokens": original_cached_tokens,
            "used_cached_tokens": used_cached_tokens,
            "dropped_cached_tokens": original_cached_tokens - used_cached_tokens,
            "tail_tokens": len(remaining_tokens),
            "prompt_tokens": len(request.prompt_token_ids or []),
            "cache_type": type(cache_to_use).__name__,
        }

    def _record_cache_reuse_skip(
        self,
        *,
        request: Request,
        cache_to_use: Any,
        cache_bytes: int,
        available_bytes: int,
        needed_bytes: float,
        merge_budget_bytes: float,
        multiplier: float,
        budget_fraction: float,
    ) -> None:
        """Record an all-or-nothing cache-reuse fallback with enough context.

        This path should be rare: `_shrink_paged_cache_for_memory()` gets the
        first chance to keep a safe block-aligned prefix. When it cannot, the
        user-facing record must say why partial reuse was unavailable and how
        much prompt will be fully prefetched, otherwise a 10-minute TTFT looks
        like a generic cache miss instead of a specific cache-contract failure.
        """
        cached_tokens = int(getattr(request, "cached_tokens", 0) or 0)
        remaining_tokens = len(getattr(request, "remaining_tokens", []) or [])
        prompt_tokens = len(request.prompt_token_ids or [])
        partial_reason = getattr(
            request,
            "_cache_reuse_partial_unavailable_reason",
            None,
        )
        cache_contract = getattr(
            request,
            "_cache_reuse_contract",
            self._cache_reuse_contract(),
        )
        self._cache_reuse_skips += 1
        self._cache_reuse_skip_tokens += cached_tokens
        self._last_cache_reuse_skip = {
            "request_id": request.request_id,
            "reason": "insufficient_memory_for_cache_merge",
            "action": "full_prefill",
            "needed_mb": round(float(needed_bytes) / 1048576, 1),
            "budget_mb": round(float(merge_budget_bytes) / 1048576, 1),
            "available_mb": round(float(available_bytes) / 1048576, 1),
            "cache_mb": round(float(cache_bytes) / 1048576, 1),
            "multiplier": multiplier,
            "budget_fraction": budget_fraction,
            "kv_cache_bits": getattr(self, "_kv_cache_bits", 0),
            "cached_tokens": cached_tokens,
            "dropped_cached_tokens": cached_tokens,
            "remaining_tokens": remaining_tokens,
            "full_prefill_tokens": prompt_tokens,
            "prompt_tokens": prompt_tokens,
            "cache_type": type(cache_to_use).__name__,
            "cache_contract": cache_contract,
            "cache_format": self._cache_reuse_cache_format(cache_to_use),
            "partial_reuse_available": False,
            "partial_reuse_unavailable_reason": partial_reason,
        }
        logger.warning(
            "Request %s: skipping cache reuse after partial-reuse attempt "
            "failed (need %.0fMB, budget %.0fMB, available %.0fMB, cached "
            "%s tokens, contract=%s, format=%s, partial_reason=%s); "
            "full-prefilling %s prompt tokens",
            request.request_id,
            float(needed_bytes) / 1048576,
            float(merge_budget_bytes) / 1048576,
            float(available_bytes) / 1048576,
            cached_tokens,
            cache_contract,
            self._cache_reuse_cache_format(cache_to_use),
            partial_reason or "unknown",
            prompt_tokens,
        )

    def _record_scheduled_cache_hit(self, request: Request) -> None:
        """Record cache-hit tokens after the request is admitted to generation."""
        try:
            cached_tokens = int(getattr(request, "cached_tokens", 0) or 0)
        except (TypeError, ValueError):
            cached_tokens = 0
        if cached_tokens <= 0:
            return
        detail = str(getattr(request, "_cache_detail", "") or "unknown")
        if (
            detail != "unknown"
            and getattr(request, "_tq_native_cache_hit", False)
            and "+tq-native" not in detail
        ):
            detail = f"{detail}+tq-native"
            request._cache_detail = detail
        execution = getattr(request, "_cache_execution", None)
        if isinstance(execution, dict):
            execution["cache_detail"] = detail
            self._last_cache_execution = dict(execution)
        self._cache_hit_requests += 1
        self._cache_hit_tokens += cached_tokens
        self._cache_hit_tokens_by_detail[detail] = (
            self._cache_hit_tokens_by_detail.get(detail, 0) + cached_tokens
        )
        finalize = getattr(
            self.block_aware_cache, "finalize_cache_hit_credit", None
        ) if self.block_aware_cache is not None else None
        if callable(finalize):
            finalize(request.request_id)

    def _shrink_hybrid_ssm_paged_cache_for_memory(
        self,
        *,
        request: Request,
        cache_to_use: Any,
        cache_bytes: int,
        available_bytes: int,
        multiplier: float,
        budget_fraction: float,
        original_cached_tokens: int,
        target_tokens: int,
        block_size: int,
    ) -> Tuple[Optional[Any], Optional[List[int]]]:
        """Memory-fit reuse for hybrid SSM cache with KV/SSM alignment."""
        checkpoint = self._fetch_block_aligned_ssm_checkpoint(
            request,
            max_len=target_tokens,
            block_size=block_size,
        )
        if checkpoint is None:
            if not hasattr(request, "_cache_reuse_partial_unavailable_reason"):
                request._cache_reuse_partial_unavailable_reason = (
                    "no_block_aligned_ssm_checkpoint"
                )
            return None, None
        checkpoint_len, _states = checkpoint
        trimmed_table = self.block_aware_cache.trim_block_table(
            request.request_id,
            checkpoint_len,
        )
        if trimmed_table is None or not getattr(trimmed_table, "block_ids", None):
            request._cache_reuse_partial_unavailable_reason = "kv_trim_failed"
            return None, None
        used_cached_tokens = int(getattr(trimmed_table, "num_tokens", 0) or 0)
        if used_cached_tokens != checkpoint_len:
            logger.warning(
                "Request %s: hybrid memory-fit cache rejected because KV trim "
                "(%s tokens) did not match SSM checkpoint (%s tokens)",
                request.request_id,
                used_cached_tokens,
                checkpoint_len,
            )
            request._cache_reuse_partial_unavailable_reason = (
                "kv_ssm_checkpoint_misaligned"
            )
            return None, None
        try:
            reconstructed = self.block_aware_cache.reconstruct_cache(trimmed_table)
        except Exception as exc:
            logger.warning(
                "Request %s: failed to reconstruct hybrid memory-fit KV cache "
                "(target %s/%s tokens): %s",
                request.request_id,
                used_cached_tokens,
                original_cached_tokens,
                exc,
            )
            request._cache_reuse_partial_unavailable_reason = "kv_reconstruct_failed"
            return None, None
        if reconstructed is not None and getattr(self, "_kv_cache_bits", 0):
            reconstructed = self._dequantize_cache_for_use(reconstructed)
        if reconstructed is None:
            request._cache_reuse_partial_unavailable_reason = "kv_dequant_failed"
            return None, None

        request.block_table = trimmed_table
        request.cached_tokens = used_cached_tokens
        request.shared_prefix_blocks = len(getattr(trimmed_table, "block_ids", []) or [])
        request.remaining_tokens = self._remaining_tokens_after_cached_prefix(
            request,
            used_cached_tokens,
        )
        finalized = self._finalize_hybrid_paged_cache_on_worker(
            request,
            reconstructed,
        )
        if finalized is None or not self._validate_cache(finalized):
            request._cache_reuse_partial_unavailable_reason = "hybrid_finalize_failed"
            return None, None
        request.prompt_cache = finalized
        remaining_tokens = request.remaining_tokens or []
        if len(remaining_tokens) == 0 and request.prompt_token_ids:
            remaining_tokens = list(request.prompt_token_ids[-1:])
            request.remaining_tokens = remaining_tokens
        self._record_cache_reuse_partial(
            request=request,
            cache_to_use=finalized,
            cache_bytes=cache_bytes,
            available_bytes=available_bytes,
            multiplier=multiplier,
            budget_fraction=budget_fraction,
            original_cached_tokens=original_cached_tokens,
            used_cached_tokens=used_cached_tokens,
            remaining_tokens=remaining_tokens,
            cache_contract="hybrid_ssm",
        )
        logger.warning(
            "Request %s: hybrid SSM cache reuse memory-fit to %s/%s tokens; "
            "prefilling %s tail tokens (budget %.0fMB, available %.0fMB)",
            request.request_id,
            used_cached_tokens,
            original_cached_tokens,
            len(remaining_tokens),
            (available_bytes * budget_fraction) / 1048576,
            available_bytes / 1048576,
        )
        return finalized, remaining_tokens

    def _shrink_paged_cache_for_memory(
        self,
        *,
        request: Request,
        cache_to_use: Any,
        cache_bytes: int,
        available_bytes: int,
        multiplier: float,
        budget_fraction: Optional[float] = None,
    ) -> Tuple[Optional[Any], Optional[List[int]]]:
        """Shrink a paged-cache hit so cache reuse survives memory pressure.

        The old behavior was all-or-nothing: if a full cached-prefix merge
        needed more temporary memory than was currently available, the scheduler
        discarded the entire hit and prefetched the full prompt again. That is
        catastrophic for long multi-turn chats. For ordinary positional KV/TQ
        cache we can safely keep a block-aligned prefix, release the trailing
        request refs, and prefill only the tail.

        Native path-dependent caches are never silently treated as plain KV.
        Hybrid SSM takes the companion-cache checkpoint path below; DSV4
        composite and ZAYA CCA remain explicit no-generic-shrink contracts.
        """
        if (
            self.block_aware_cache is None
            or getattr(request, "block_table", None) is None
            or not getattr(request.block_table, "block_ids", None)
        ):
            request._cache_reuse_contract = self._cache_reuse_contract()
            request._cache_reuse_partial_unavailable_reason = (
                "non_paged_cache_no_block_table"
            )
            return None, None
        cache_contract = self._cache_reuse_contract()
        request._cache_reuse_contract = cache_contract
        request._cache_reuse_partial_unavailable_reason = None

        original_cached_tokens = int(getattr(request, "cached_tokens", 0) or 0)
        if original_cached_tokens <= 0 or cache_bytes <= 0 or available_bytes <= 0:
            return None, None
        if multiplier <= 0:
            return None, None

        block_size = int(getattr(self.block_aware_cache, "block_size", 0) or 0)
        if block_size <= 0:
            return None, None

        if budget_fraction is None:
            budget_fraction = self._cache_reuse_budget_fraction()
        budget_fraction = max(0.10, min(0.95, float(budget_fraction)))

        target_tokens = self._memory_fit_target_cached_tokens(
            original_cached_tokens=original_cached_tokens,
            cache_bytes=cache_bytes,
            available_bytes=available_bytes,
            multiplier=multiplier,
            block_size=block_size,
            budget_fraction=budget_fraction,
        )
        if target_tokens <= 0:
            request._cache_reuse_partial_unavailable_reason = (
                "memory_budget_below_one_block"
            )
            return None, None

        if cache_contract in ("deepseek_v4_composite", "zaya_cca"):
            terminal_tag = (
                "deepseek_v4"
                if cache_contract == "deepseek_v4_composite"
                else "zaya_cca"
            )
            trim_terminal = getattr(
                self.block_aware_cache,
                "trim_block_table_to_terminal_state",
                None,
            )
            if not callable(trim_terminal):
                request._cache_reuse_partial_unavailable_reason = (
                    "no_terminal_path_dependent_checkpoint"
                )
                return None, None
            trimmed_table = trim_terminal(
                request.request_id,
                target_tokens,
                terminal_tag,
            )
            if trimmed_table is None or not getattr(trimmed_table, "block_ids", None):
                request._cache_reuse_partial_unavailable_reason = (
                    "no_terminal_path_dependent_checkpoint"
                )
                return None, None

            used_cached_tokens = int(getattr(trimmed_table, "num_tokens", 0) or 0)
            if used_cached_tokens <= 0 or used_cached_tokens >= original_cached_tokens:
                request._cache_reuse_partial_unavailable_reason = (
                    "terminal_trim_did_not_reduce_cache"
                )
                return None, None

            try:
                trimmed_cache = self.block_aware_cache.reconstruct_cache(trimmed_table)
            except Exception as exc:
                logger.warning(
                    "Request %s: failed to reconstruct terminal memory-fit "
                    "%s cache (target %s/%s tokens): %s",
                    request.request_id,
                    cache_contract,
                    used_cached_tokens,
                    original_cached_tokens,
                    exc,
                )
                request._cache_reuse_partial_unavailable_reason = (
                    "terminal_reconstruct_failed"
                )
                return None, None

            if not self._validate_cache(trimmed_cache):
                request._cache_reuse_partial_unavailable_reason = (
                    "terminal_validate_failed"
                )
                return None, None

            remaining_tokens = self._remaining_tokens_after_cached_prefix(
                request,
                used_cached_tokens,
            )
            if len(remaining_tokens) == 0 and request.prompt_token_ids:
                remaining_tokens = list(request.prompt_token_ids[-1:])

            request.block_table = trimmed_table
            request.cached_tokens = used_cached_tokens
            request.shared_prefix_blocks = len(
                getattr(trimmed_table, "block_ids", []) or []
            )
            request.remaining_tokens = remaining_tokens
            request.prompt_cache = trimmed_cache

            self._record_cache_reuse_partial(
                request=request,
                cache_to_use=trimmed_cache,
                cache_bytes=cache_bytes,
                available_bytes=available_bytes,
                multiplier=multiplier,
                budget_fraction=budget_fraction,
                original_cached_tokens=original_cached_tokens,
                used_cached_tokens=used_cached_tokens,
                remaining_tokens=remaining_tokens,
                cache_contract=cache_contract,
            )
            logger.warning(
                "Request %s: full %s cache reuse needs %.0fMB but only %.0fMB "
                "is budgeted (%.0fMB available); using terminal-safe partial "
                "prefix cache %s/%s tokens and prefilling %s tail tokens",
                request.request_id,
                cache_contract,
                (cache_bytes * multiplier) / 1048576,
                (available_bytes * budget_fraction) / 1048576,
                available_bytes / 1048576,
                used_cached_tokens,
                original_cached_tokens,
                len(remaining_tokens),
            )
            return trimmed_cache, remaining_tokens

        if cache_contract == "hybrid_ssm":
            return self._shrink_hybrid_ssm_paged_cache_for_memory(
                request=request,
                cache_to_use=cache_to_use,
                cache_bytes=cache_bytes,
                available_bytes=available_bytes,
                multiplier=multiplier,
                budget_fraction=budget_fraction,
                original_cached_tokens=original_cached_tokens,
                target_tokens=target_tokens,
                block_size=block_size,
            )

        trimmed_table = self.block_aware_cache.trim_block_table(
            request.request_id,
            target_tokens,
        )
        if trimmed_table is None or not getattr(trimmed_table, "block_ids", None):
            request._cache_reuse_partial_unavailable_reason = "kv_trim_failed"
            return None, None

        used_cached_tokens = int(getattr(trimmed_table, "num_tokens", 0) or 0)
        if used_cached_tokens <= 0 or used_cached_tokens >= original_cached_tokens:
            request._cache_reuse_partial_unavailable_reason = (
                "kv_trim_did_not_reduce_cache"
            )
            return None, None

        try:
            trimmed_cache = self.block_aware_cache.reconstruct_cache(trimmed_table)
        except Exception as exc:
            logger.warning(
                "Request %s: failed to reconstruct memory-fit partial cache "
                "(target %s/%s tokens): %s",
                request.request_id,
                used_cached_tokens,
                original_cached_tokens,
                exc,
            )
            request._cache_reuse_partial_unavailable_reason = "kv_reconstruct_failed"
            return None, None

        if not self._validate_cache(trimmed_cache):
            logger.warning(
                "Request %s: memory-fit partial cache failed validation "
                "(target %s/%s tokens)",
                request.request_id,
                used_cached_tokens,
                original_cached_tokens,
            )
            request._cache_reuse_partial_unavailable_reason = "kv_validate_failed"
            return None, None

        remaining_tokens = self._remaining_tokens_after_cached_prefix(
            request,
            used_cached_tokens,
        )
        if len(remaining_tokens) == 0 and request.prompt_token_ids:
            remaining_tokens = list(request.prompt_token_ids[-1:])

        request.block_table = trimmed_table
        request.cached_tokens = used_cached_tokens
        request.shared_prefix_blocks = len(getattr(trimmed_table, "block_ids", []) or [])
        request.remaining_tokens = remaining_tokens
        request.prompt_cache = trimmed_cache

        self._record_cache_reuse_partial(
            request=request,
            cache_to_use=trimmed_cache,
            cache_bytes=cache_bytes,
            available_bytes=available_bytes,
            multiplier=multiplier,
            budget_fraction=budget_fraction,
            original_cached_tokens=original_cached_tokens,
            used_cached_tokens=used_cached_tokens,
            remaining_tokens=remaining_tokens,
            cache_contract=cache_contract,
        )
        logger.warning(
            "Request %s: full cache reuse needs %.0fMB but only %.0fMB is "
            "budgeted (%.0fMB available); using partial prefix cache %s/%s "
            "tokens and prefilling %s tail tokens instead of falling back to "
            "a full prefill",
            request.request_id,
            (cache_bytes * multiplier) / 1048576,
            (available_bytes * budget_fraction) / 1048576,
            available_bytes / 1048576,
            used_cached_tokens,
            original_cached_tokens,
            len(remaining_tokens),
        )
        return trimmed_cache, remaining_tokens

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
            cls_name = type(layer_cache).__name__
            if cls_name == "MiniMaxM3SparseCache":
                try:
                    from .models.minimax_m3.cache import clone_minimax_m3_sparse
                except Exception:
                    return None

                def _copy_m3_slice(value):
                    try:
                        orig_dtype = getattr(value, "dtype", None)
                        copied = _from_numpy(_to_numpy(value), orig_dtype)
                        return copied
                    except Exception:
                        return value

                new_cache = clone_minimax_m3_sparse(
                    layer_cache,
                    target_len,
                    copy_fn=_copy_m3_slice,
                    require_idx_keys=True,
                )
                if new_cache is None:
                    return None
                truncated.append(new_cache)
                continue
            if cls_name == "TurboQuantKVCache":
                # Preserve the native TQ identity and exact codec policy across
                # the N-1 prompt-boundary copy. The generic positional branch
                # below intentionally creates KVCache, which used to erase
                # key/value bits and seed before paged-block extraction. That
                # made Auto write full-precision block L2 records even though
                # live health truthfully reported TurboQuant objects active.
                # ``state`` returns the complete decoded KV view, including a
                # compressed prefix when live encoding is enabled; ``keys`` by
                # itself may contain only the active float window.
                try:
                    from jang_tools.turboquant.cache import TurboQuantKVCache

                    k, v = layer_cache.state
                    if not hasattr(k, "ndim") or not hasattr(v, "ndim"):
                        return None
                    if k.ndim == 4:
                        safe_target = logical_truncate_target(layer_cache, target_len, int(k.shape[2]))
                        np_k, np_v = _to_numpy(k), _to_numpy(v)
                        tk = _from_numpy(
                            np_k[:, :, :safe_target, :], getattr(k, "dtype", None)
                        )
                        tv = _from_numpy(
                            np_v[:, :, :safe_target, :], getattr(v, "dtype", None)
                        )
                    elif k.ndim == 3:
                        safe_target = logical_truncate_target(layer_cache, target_len, int(k.shape[1]))
                        np_k, np_v = _to_numpy(k), _to_numpy(v)
                        tk = _from_numpy(
                            np_k[:, :safe_target, :], getattr(k, "dtype", None)
                        )
                        tv = _from_numpy(
                            np_v[:, :safe_target, :], getattr(v, "dtype", None)
                        )
                    else:
                        return None
                    if safe_target <= 0:
                        return None
                    new_cache = TurboQuantKVCache(
                        key_dim=int(k.shape[-1]),
                        value_dim=int(v.shape[-1]),
                        key_bits=int(getattr(layer_cache, "key_bits", 8)),
                        value_bits=int(getattr(layer_cache, "value_bits", 8)),
                        seed=int(getattr(layer_cache, "_seed", 42)),
                        compress_after=int(
                            getattr(layer_cache, "compress_after", 0) or 0
                        ),
                        sink_tokens=int(getattr(layer_cache, "sink_tokens", 0) or 0),
                    )
                    new_cache.keys = tk
                    new_cache.values = tv
                    new_cache.offset = safe_target
                    new_cache.step = safe_target
                    truncated.append(new_cache)
                    continue
                except Exception as exc:
                    logger.warning(
                        "TurboQuant prompt-boundary truncation failed: %s", exc
                    )
                    return None
            if Scheduler._is_dsv4_cache_class_name(cls_name):
                try:
                    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

                    local = getattr(layer_cache, "local", None)
                    sliding_window = int(getattr(local, "max_size", 128) or 128)
                    compress_ratio = getattr(layer_cache, "compress_ratio", None)
                    current_len = int(getattr(layer_cache, "offset", 0) or 0)
                    to_trim = max(0, current_len - target_len)

                    # SAFETY: DeepseekV4Cache is a composite of THREE
                    # attention components (SWA + CSA + HCA), each with
                    # its own rewind constraint. Trimming a post-
                    # generation live cache back to the prompt boundary
                    # is unsafe in all three:
                    #
                    # (1) SWA local — RotatingKVCache as `self.local`.
                    #     Cannot be rewound after the circular buffer
                    #     wraps (offset > max_size). _idx goes negative
                    #     and replay applies output-side tokens at wrong
                    #     positions → looping decode (verified live
                    #     2026-05-05 with prompt 29 + output 600,
                    #     sliding_window=128 → offset=629, idx=-501).
                    #     Same constraint as plain RotatingKVCache at
                    #     scheduler.py:72 (`_rebuild_meta_state_after_
                    #     truncation` returns None when wrapped). The
                    #     DSV4 branch was bypassing that check.
                    #
                    # (2) CSA compressor pool — cumulative buffer of
                    #     `pooled` rows summarizing every `compress_ratio`
                    #     raw positions. After generation, pool contains
                    #     prompt-side AND output-side rows interleaved
                    #     by the compressor's chunk boundaries.
                    #     `trim(n)` drops `max(1, n // ratio)` trailing
                    #     rows (jang_tools dsv4/mlx_model.py:527) but
                    #     the boundary often does NOT align with the
                    #     prompt/output split. Even an aligned trim
                    #     leaves the kept rows whose `key/value` were
                    #     computed from a window that may have included
                    #     output tokens — semantically wrong.
                    #
                    # (3) HCA/indexer pool — same cumulative behavior
                    #     and same trim approximation as CSA. Wrong
                    #     indexer state mis-routes attention sparsely
                    #     across output rather than prompt.
                    #
                    # The clean fix is to capture a prompt-boundary
                    # snapshot BEFORE decode starts (or async re-derive
                    # after the request completes). Until that path
                    # exists, refuse ALL post-generation DSV4 cache
                    # stores. Caller falls through to full prefill on
                    # next turn, which is correct.
                    #
                    # Override: VMLX_DSV4_TRUST_TRIMMED_CACHE=1 to keep
                    # the (broken) v1.5.13 store-always behavior for
                    # benchmarking. NOT recommended for production.
                    _trust_trim = os.environ.get(
                        "VMLX_DSV4_TRUST_TRIMMED_CACHE", "0"
                    ) in ("1", "true", "yes")
                    if to_trim > 0 and not _trust_trim:
                        logger.info(
                            f"DSV4 prompt cache store SKIPPED: "
                            f"current_len={current_len}, "
                            f"target_len={target_len}, "
                            f"to_trim={to_trim} tokens. "
                            f"DeepseekV4Cache (SWA+CSA+HCA composite) "
                            f"cannot be safely rewound from post-"
                            f"generation state — SWA RotatingKVCache "
                            f"wraps at offset>{sliding_window} and "
                            f"CSA/HCA pool buffers are cumulative "
                            f"across the entire window. Returning None "
                            f"forces clean full prefill on next turn. "
                            f"Override: VMLX_DSV4_TRUST_TRIMMED_CACHE=1 "
                            f"(NOT recommended)."
                        )
                        return None

                    new_cache = DeepseekV4Cache(
                        sliding_window=sliding_window,
                        compress_ratio=compress_ratio,
                    )
                    new_cache.state = layer_cache.state
                    try:
                        new_cache.meta_state = layer_cache.meta_state
                    except Exception:
                        pass
                    if to_trim:
                        new_cache.trim(to_trim)
                    truncated.append(new_cache)
                    continue
                except Exception as e:
                    logger.debug(
                        f"DeepseekV4Cache prompt truncation failed: {e}"
                    )
                    return None
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
                        safe_target = logical_truncate_target(layer_cache, target_len, k[0].shape[-2])
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
                            safe_target = logical_truncate_target(layer_cache, target_len, k.shape[2])
                            np_k, np_v = _to_numpy(k), _to_numpy(v)
                            new_cache.keys = _from_numpy(
                                np_k[:, :, :safe_target, :], k.dtype
                            )
                            new_cache.values = _from_numpy(
                                np_v[:, :, :safe_target, :], v.dtype
                            )
                        elif ndim == 3:
                            safe_target = logical_truncate_target(layer_cache, target_len, k.shape[1])
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
                if (
                    isinstance(layer_cache, dict)
                    and layer_cache.get("dsv4_block_records")
                ):
                    # DSV4BatchGenerator already detached immutable native
                    # pool deltas and exact SWA/buffer anchors at their true
                    # prefill boundaries. Passing this transport through
                    # avoids re-materializing one full composite snapshot at
                    # terminal dispatch.
                    extracted.append(dict(layer_cache))
                    cls_name = str(layer_cache.get("class_name") or "")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    continue
                cls_name = type(layer_cache).__name__
                # CacheList (MoE models like DeepSeek V3.2, Falcon H1):
                # wrapper with .caches attribute containing sub-caches.
                # Extract each sub-cache's state and store as a list.
                if hasattr(layer_cache, "caches") and isinstance(
                    getattr(layer_cache, "caches", None), (list, tuple)
                ):
                    sub_states = []
                    all_ok = True
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
                            ):
                                sk, sv = sub_state
                                if (
                                    hasattr(sk, "shape")
                                    and len(sk.shape) == 4
                                ):
                                    target_h = self._cache_head_slice_target(
                                        type(sub_cache).__name__,
                                        int(sk.shape[1]),
                                    )
                                    if target_h > 0:
                                        sub_state = (
                                            sk[:, :target_h, :, :],
                                            sv[:, :target_h, :, :],
                                        )
                                elif (
                                    isinstance(sk, (tuple, list))
                                    and len(sk) >= 1
                                    and hasattr(sk[0], "shape")
                                    and len(sk[0].shape) == 4
                                ):
                                    target_h = self._cache_head_slice_target(
                                        type(sub_cache).__name__,
                                        int(sk[0].shape[1]),
                                    )
                                    if target_h > 0:
                                        sub_state = (
                                            tuple(t[:, :target_h, :, :] for t in sk),
                                            tuple(t[:, :target_h, :, :] for t in sv),
                                        )
                            sub_states.append(
                                {
                                    "state": sub_state,
                                    "meta_state": sub_cache.meta_state,
                                    "class_name": type(sub_cache).__name__,
                                    **(
                                        {
                                            "tq_config": {
                                                "key_bits": int(sub_cache.key_bits),
                                                "value_bits": int(sub_cache.value_bits),
                                                "seed": int(getattr(sub_cache, "_seed", 42)),
                                            }
                                        }
                                        if type(sub_cache).__name__
                                        == "TurboQuantKVCache"
                                        else {}
                                    ),
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
                elif type(layer_cache).__name__ == "ZayaNoStateCache":
                    cls_name = type(layer_cache).__name__
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    extracted.append(
                        {
                            "state": (),
                            "meta_state": (),
                            "class_name": cls_name,
                            "no_state": True,
                        }
                    )
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
                elif (
                    (
                        self._is_dsv4_cache_class_name(cls_name)
                        and hasattr(layer_cache, "meta_state")
                    )
                    or (
                        hasattr(layer_cache, "state")
                        and hasattr(layer_cache, "meta_state")
                    )
                ):
                    # PoolQuantizedV4Cache.state materializes its retained q8
                    # pools as BF16 for attention. Cache transport must use the
                    # native encoded tree or restore would quantize those rows
                    # a second time and change their codes.
                    state_attr = (
                        "storage_state"
                        if cls_name == "PoolQuantizedV4Cache"
                        and hasattr(layer_cache, "storage_state")
                        else "state"
                    )
                    state = getattr(layer_cache, state_attr)
                    meta = layer_cache.meta_state  # (offset,) as strings
                    if self._is_dsv4_cache_class_name(cls_name):
                        # DSV4 may store q4/q8-compressed local SWA KV inside
                        # the composite cache. In that case .meta_state belongs
                        # to QuantizedKVCache, not the real RotatingKVCache.
                        # Use the saved rotating metadata for reconstruction
                        # and carry quant metadata separately in cache_meta.
                        saved_rot_meta = getattr(
                            layer_cache, "_vmlx_dsv4_local_meta_state", None
                        )
                        if saved_rot_meta:
                            meta = tuple(saved_rot_meta)

                    # Normalize GQA head inflation from BatchKVCache.merge().
                    # merge() broadcasts H to max across all caches in the batch,
                    # but the true KV head count is smaller for GQA/MQA models.
                    # Slice away the inflated heads before storing.
                    # Handles both standard KVCache (plain tensors) and
                    # QuantizedKVCache (tuple-of-tuples: (data, scales, zeros)).
                    if isinstance(state, tuple) and len(state) == 2:
                        keys, values = state
                        if hasattr(keys, "shape") and len(keys.shape) == 4:
                            # Standard KVCache: keys/values are 4D tensors.
                            target_h = self._cache_head_slice_target(
                                cls_name,
                                int(keys.shape[1]),
                            )
                            if target_h > 0:
                                orig_h = keys.shape[1]
                                keys = keys[:, :target_h, :, :]
                                values = values[:, :target_h, :, :]
                                state = (keys, values)
                                if i == 0:
                                    logger.debug(
                                        f"GQA head normalization: sliced H "
                                        f"{orig_h} → {target_h}"
                                    )
                        elif isinstance(keys, (tuple, list)) and len(keys) >= 1:
                            # QuantizedKVCache: keys/values are tuples of
                            # (data, scales, zeros) — check first component.
                            first_k = keys[0]
                            if hasattr(first_k, "shape") and len(first_k.shape) == 4:
                                target_h = self._cache_head_slice_target(
                                    cls_name,
                                    int(first_k.shape[1]),
                                )
                                if target_h > 0:
                                    orig_h = first_k.shape[1]
                                    keys = tuple(t[:, :target_h, :, :] for t in keys)
                                    values = tuple(t[:, :target_h, :, :] for t in values)
                                    state = (keys, values)
                                    if i == 0:
                                        logger.debug(
                                            f"GQA head normalization (quantized): "
                                            f"sliced H {orig_h} → {target_h}"
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
                    entry = {
                        "state": state,
                        "meta_state": meta,
                        "class_name": cls_name,
                    }
                    if cls_name == "TurboQuantKVCache":
                        entry["tq_config"] = {
                            "key_bits": int(getattr(layer_cache, "key_bits", 8)),
                            "value_bits": int(getattr(layer_cache, "value_bits", 8)),
                            "seed": int(getattr(layer_cache, "_seed", 42)),
                        }
                    if self._is_dsv4_cache_class_name(cls_name):
                        entry["compress_ratio"] = getattr(
                            layer_cache, "compress_ratio", None
                        )
                        local_quant_meta = getattr(
                            layer_cache, "_vmlx_dsv4_local_quant_meta", None
                        )
                        if local_quant_meta:
                            entry["local_quant_meta"] = tuple(local_quant_meta)
                        try:
                            entry["sliding_window"] = getattr(
                                getattr(layer_cache, "local", None),
                                "max_size",
                                None,
                            )
                        except Exception:
                            entry["sliding_window"] = None
                        if entry["sliding_window"] is None:
                            entry["sliding_window"] = getattr(
                                layer_cache, "_vmlx_dsv4_sliding_window", None
                            )
                        entry["pool_quant"] = (
                            cls_name == "PoolQuantizedV4Cache"
                        )
                        if state_attr == "storage_state":
                            entry["pool_storage_schema"] = (
                                state[0]
                                if isinstance(state, (tuple, list)) and state
                                else None
                            )
                    extracted.append(entry)
                else:
                    logger.debug(
                        f"Layer {i} ({type(layer_cache).__name__}) lacks state/meta_state"
                    )
                    failed += 1
            except Exception as e:
                logger.warning(f"Failed to extract state from layer {i}: {e}")
                cls_name = type(layer_cache).__name__
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                extracted.append(
                    {
                        "state": None,
                        "meta_state": None,
                        "class_name": cls_name,
                    }
                )
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
                add_special_tokens = getattr(request, "_encode_add_special_tokens", None)
                # Handle both tokenizers and processors (for MLLM models)
                if hasattr(self.tokenizer, "encode"):
                    request.prompt_token_ids = self._encode_prompt_text(
                        self.tokenizer,
                        request.prompt,
                        add_special_tokens,
                    )
                elif hasattr(self.tokenizer, "tokenizer") and hasattr(
                    self.tokenizer.tokenizer, "encode"
                ):
                    # Processor wraps tokenizer (e.g., Qwen3VLProcessor)
                    request.prompt_token_ids = self._encode_prompt_text(
                        self.tokenizer.tokenizer,
                        request.prompt,
                        add_special_tokens,
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

        max_prompt_tokens = int(getattr(request, "_max_prompt_tokens", 0) or 0)
        if max_prompt_tokens > 0 and len(request.prompt_token_ids) > max_prompt_tokens:
            raise PromptTooLongError(
                len(request.prompt_token_ids),
                max_prompt_tokens,
                source="tokenized prompt",
                request_id=request.request_id,
            )

        # Bound output by the model's declared context (prompt + output must
        # not run past the positional ceiling); a binding clamp logs a clear
        # context-exhaustion notice instead of silently degrading.
        from vmlx_engine.context_limits import clamp_output_to_declared_context

        sampling_params = getattr(request, "sampling_params", None)
        if sampling_params is not None and getattr(
            sampling_params, "max_tokens", None
        ) is not None:
            sampling_params.max_tokens = clamp_output_to_declared_context(
                len(request.prompt_token_ids),
                sampling_params.max_tokens,
                request_id=str(request.request_id),
            )

        if self._uses_dsv4_cache:
            from vmlx_engine.utils.dsv4_batch_generator import (
                dsv4_max_prefill_tokens,
            )

            _dsv4_max_prefill = dsv4_max_prefill_tokens()
            if _dsv4_max_prefill > 0 and len(request.prompt_token_ids) > _dsv4_max_prefill:
                raise ValueError(
                    "DeepSeek V4 Flash JANGTQ long-prefill guard: prompt has "
                    f"{len(request.prompt_token_ids)} tokens, max safe prefill is "
                    f"{_dsv4_max_prefill}. The Python DSV4 path defaults to bounded "
                    "prefill chunks for SWA+CSA/HCA memory safety after installed-app "
                    "cache-hit validation; legacy single-shot remains available with "
                    "DSV4_PREFILL_STEP_SIZE=0 for diagnostics. Very large prompts still scale with "
                    "accumulated compressor/indexer pool work. "
                    "Unset DSV4_MAX_PREFILL_TOKENS (or set it to 0) to rely on "
                    "the architecture-aware admission limit for this machine."
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
        if (
            getattr(self, "_mixed_attention_cache_model", False)
            or getattr(self, "_uses_openpangu_cache", False)
            or getattr(self, "_uses_glm5_next_cache", False)
        ):
            # Mixed full/SWA models must use the full effective prompt as the
            # cache key for strict logprob equivalence. Stripping the generation
            # prompt replays several suffix tokens from an earlier RotatingKV
            # boundary; MiMo V2.5 shows small but real distribution drift there.
            _gpl_fetch = 0
        elif _GEN_PROMPT_STRIP_DISABLED:
            # Same reasoning, forced on for ANY family. Exists because the drift
            # above was only ever established on MiMo V2.5, and whether a given
            # strip-active family pays it is a measurement nobody could run
            # without a switch. Costs prefix reuse across generation-prompt
            # variants, so it is opt-in.
            _gpl_fetch = 0
        if 0 < _gpl_fetch < len(_full_tokens_list):
            _fetch_tokens = _full_tokens_list[:-_gpl_fetch]
            _gpl_suffix_tokens = _full_tokens_list[-_gpl_fetch:]
        else:
            _fetch_tokens = _full_tokens_list
            _gpl_suffix_tokens = []
        _cache_extra_keys = getattr(request, "_cache_extra_keys", None)

        _record_fetch_bypass = getattr(
            self.block_aware_cache, "record_fetch_bypass", None
        )
        if _bypass and callable(_record_fetch_bypass):
            _record_fetch_bypass(
                request.request_id,
                attempted_tokens=len(_fetch_tokens),
            )

        if self.block_aware_cache is not None and not _bypass:
            # Use paged cache
            request._paged_disk_hit = False
            try:
                _paged_disk_hits_before = int(
                    getattr(
                        getattr(self.block_aware_cache.paged_cache, "stats", None),
                        "disk_hits",
                        0,
                    )
                    or 0
                )
            except Exception:
                _paged_disk_hits_before = 0
            _paged_candidate_lookup_start = time.perf_counter()
            request._cache_admission_start_time = _paged_candidate_lookup_start
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                _fetch_tokens,
                cache_extra_keys=_cache_extra_keys,
            )
            request._cache_candidate_lookup_seconds = round(
                max(0.0, time.perf_counter() - _paged_candidate_lookup_start),
                6,
            )
            try:
                _paged_disk_hits_after = int(
                    getattr(
                        getattr(self.block_aware_cache.paged_cache, "stats", None),
                        "disk_hits",
                        0,
                    )
                    or 0
                )
                request._paged_disk_hit = (
                    _paged_disk_hits_after > _paged_disk_hits_before
                )
            except Exception:
                request._paged_disk_hit = False
            # Re-append gpl suffix to remaining so model sees template trailer.
            if _gpl_suffix_tokens:
                remaining = list(remaining or []) + list(_gpl_suffix_tokens)
            if block_table and block_table.num_tokens > 0:
                paged_cold_tokens = self._paged_cold_block_tokens(block_table)
                _typed_disk_only = bool(
                    getattr(self.paged_cache_manager, "disk_only", False)
                )
                if _typed_disk_only:
                    # Disk-only keeps hash/chain metadata in L1 but never the
                    # cache tensors.  Per-block ``cache_data_from_disk`` flips
                    # only during worker reconstruction, so every candidate
                    # token is cold even when that pre-reconstruct marker says
                    # false (the live Laguna selector incorrectly reported 0).
                    paged_cold_tokens = int(block_table.num_tokens)
                warm_cache = None
                warm_remaining: List[int] = []
                warm_cached_tokens = 0
                warm_detail: str | None = None
                # Paged/TurboQuant hits with cold/disk-promoted blocks can have
                # worse TTFT than a smaller hot prefix hit. Compare a conservative
                # hot-token advantage before committing to reconstruction.
                if (
                    (paged_cold_tokens > 0 or getattr(self, "_tq_active", False))
                    and not self._uses_dsv4_cache
                    and not self._uses_zaya_cache
                    and not _cache_extra_keys
                ):
                    (
                        warm_cache,
                        warm_remaining,
                        warm_cached_tokens,
                        warm_detail,
                    ) = self._warm_prefix_alternative(
                        _fetch_tokens,
                        _gpl_suffix_tokens,
                    )
                prefer_warm, selection = self._should_prefer_warm_prefix_over_paged(
                    paged_cached_tokens=int(getattr(block_table, "num_tokens", 0) or 0),
                    paged_cold_tokens=paged_cold_tokens,
                    warm_cached_tokens=warm_cached_tokens,
                )
                prefer_clean = False
                if _typed_disk_only:
                    prefer_clean, disk_cost = (
                        self._should_clean_prefill_over_disk_only(
                            paged_cached_tokens=int(block_table.num_tokens),
                            paged_cold_tokens=paged_cold_tokens,
                        )
                    )
                    selection.update(disk_cost)
                if prefer_clean:
                    self._detach_paged_candidate(request.request_id, block_table)
                    request._cache_selection_attempted_tokens = int(
                        block_table.num_tokens
                    )
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = list(request.prompt_token_ids)
                    request._cache_detail = None
                    request.block_table = None
                    request.shared_prefix_blocks = 0
                    request._paged_block_table_needs_worker_reconstruct = False
                    selection.update(
                        {
                            "selected": "clean_prefill",
                            "rejected": "paged",
                            "reason": selection.get("cost_reason"),
                        }
                    )
                    request._cache_selection = selection
                    self._last_cache_selection = selection
                    logger.info(
                        "Request %s: selected clean prefill over SSD-only hit "
                        "(cached=%d, estimated_disk=%.6fs, "
                        "estimated_prefill=%.6fs, reason=%s)",
                        request.request_id,
                        selection["paged_cached_tokens"],
                        selection["estimated_disk_seconds"],
                        selection["estimated_prefill_seconds"],
                        selection["reason"],
                    )
                elif prefer_warm and warm_cache is not None and warm_detail is not None:
                    self._detach_paged_candidate(request.request_id, block_table)
                    request.prompt_cache = warm_cache
                    request.cached_tokens = warm_cached_tokens
                    request.remaining_tokens = warm_remaining
                    request._cache_detail = warm_detail
                    request.block_table = None
                    request.shared_prefix_blocks = 0
                    request._paged_block_table_needs_worker_reconstruct = False
                    if getattr(self, "_kv_cache_bits", 0):
                        request._prompt_cache_needs_worker_dequant = True
                    selection.update(
                        {
                            "selected": warm_detail,
                            "rejected": "paged",
                            "reason": "cold_paged_reconstruction_cost",
                        }
                    )
                    request._cache_selection = selection
                    self._last_cache_selection = selection
                    logger.info(
                        "Request %s: selected %s cache over paged hit "
                        "(paged_cached=%d, cold_paged=%d, warm_cached=%d, "
                        "hot_advantage=%d < threshold=%d)",
                        request.request_id,
                        warm_detail,
                        selection["paged_cached_tokens"],
                        selection["paged_cold_tokens"],
                        selection["warm_cached_tokens"],
                        selection["hot_advantage_tokens"],
                        selection["hot_advantage_threshold_tokens"],
                    )
                else:
                    paged_selection_reason = (
                        selection.get("cost_reason")
                        if _typed_disk_only
                        else "paged_hot_advantage_sufficient"
                        if warm_detail is not None
                        else "no_warm_prefix_alternative"
                    )
                    selection.update(
                        {
                            "selected": "paged",
                            "rejected": warm_detail,
                            "reason": paged_selection_reason,
                        }
                    )
                    request._cache_selection = selection
                    self._last_cache_selection = selection
                    # Defer reconstruction to the scheduler llm-worker. Both
                    # in-RAM block tensors and SSM companion tensors are MLX arrays
                    # created on that worker stream; reconstructing them in
                    # add_request() on the API thread causes fake misses or stream
                    # errors. L2 disk promotion also creates MLX arrays, so keeping
                    # one reconstruction path avoids thread-dependent behavior.
                    request.block_table = block_table
                    request.cached_tokens = block_table.num_tokens
                    request._cache_matched_tokens = int(
                        block_table.matched_tokens
                        if block_table.matched_tokens is not None
                        else block_table.num_tokens
                    )
                    request._cache_checkpoint_tokens = int(
                        block_table.checkpoint_tokens
                        if block_table.checkpoint_tokens is not None
                        else block_table.num_tokens
                    )
                    request._cache_replayed_tokens = int(
                        block_table.replayed_tokens or 0
                    )
                    request.shared_prefix_blocks = len(block_table.block_ids)
                    request.remaining_tokens = remaining
                    request._paged_block_table_needs_worker_reconstruct = True
                    if (
                        self._is_hybrid
                        and not self._uses_dsv4_cache
                        and not self._uses_zaya_cache
                    ):
                        request._hybrid_prompt_cache_needs_worker_ssm = True
                        request._hybrid_ssm_fetch_tokens = list(_fetch_tokens)
                    if getattr(self, "_kv_cache_bits", 0):
                        request._prompt_cache_needs_worker_dequant = True
                    if self._uses_dsv4_cache:
                        request._cache_detail = _typed_paged_cache_detail(
                            "dsv4",
                            disk_hit=bool(getattr(request, "_paged_disk_hit", False)),
                            disk_only=_typed_disk_only,
                        )
                    elif self._uses_zaya_cache:
                        request._cache_detail = _typed_paged_cache_detail(
                            "zaya_cca",
                            disk_hit=bool(getattr(request, "_paged_disk_hit", False)),
                            disk_only=_typed_disk_only,
                        )
                    else:
                        request._cache_detail = _block_cache_detail(
                            self.paged_cache_manager,
                            disk_hit=bool(getattr(request, "_paged_disk_hit", False)),
                        )
                    _block_backend_label = (
                        "block disk-only"
                        if bool(getattr(self.paged_cache_manager, "disk_only", False))
                        else "paged"
                    )
                    logger.info(
                        "Request %s: %s cache hit, matched_tokens=%d "
                        "checkpoint_tokens=%d replayed_tokens=%d in %d blocks, "
                        "%d remaining to process (worker reconstruct pending)",
                        request.request_id,
                        _block_backend_label,
                        request._cache_matched_tokens,
                        request._cache_checkpoint_tokens,
                        request._cache_replayed_tokens,
                        request.shared_prefix_blocks,
                        len(remaining),
                    )
                    if self._uses_dsv4_cache:
                        self._log_dsv4_extended_match_divergence(request)
            else:
                request.remaining_tokens = request.prompt_token_ids
                _block_backend_label = (
                    "block disk-only"
                    if bool(getattr(self.paged_cache_manager, "disk_only", False))
                    else "paged"
                )
                logger.info(
                    f"Request {request.request_id}: {_block_backend_label} cache miss, "
                    f"processing all {len(request.prompt_token_ids)} tokens"
                )
                if self._uses_dsv4_cache:
                    self._log_dsv4_extended_match_divergence(request)
        elif (
            self.memory_aware_cache is not None
            and not _bypass
        ):
            # Use memory-aware prefix cache (gpl-stripped fetch; suffix re-attached)
            cache, remaining = _call_with_optional_cache_extra(
                self.memory_aware_cache.fetch,
                _fetch_tokens,
                cache_extra_keys=_cache_extra_keys,
            )
            remaining, cached_tokens = self._prefix_hit_tail_and_cached_tokens(
                fetch_tokens=_fetch_tokens,
                remaining=remaining or [],
                gen_prompt_suffix=_gpl_suffix_tokens,
            )
            if cache:
                # Dequantize for BatchGenerator compatibility
                if getattr(self, "_kv_cache_bits", 0):
                    # Dequantization creates/evaluates MLX arrays. add_request()
                    # runs on the API thread, while live decode runs on the
                    # scheduler's llm-worker; doing this here can leave tensors
                    # tied to the API thread's Stream(gpu, 0). Defer stored
                    # q4/q8 cache dequant to _schedule_waiting() on the worker.
                    request._prompt_cache_needs_worker_dequant = True
                if cache is None:
                    # Dequantization failed — treat as cache miss
                    request.remaining_tokens = request.prompt_token_ids
                    logger.info(
                        f"Request {request.request_id}: dequantization failed, "
                        f"treating as cache miss"
                    )
                else:
                    request.prompt_cache = cache
                    request.cached_tokens = cached_tokens
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
            cache, remaining = _call_with_optional_cache_extra(
                self.prefix_cache.fetch_cache,
                _fetch_tokens,
                cache_extra_keys=_cache_extra_keys,
            )
            remaining, cached_tokens = self._prefix_hit_tail_and_cached_tokens(
                fetch_tokens=_fetch_tokens,
                remaining=remaining or [],
                gen_prompt_suffix=_gpl_suffix_tokens,
            )
            if cache:
                # Dequantize for BatchGenerator compatibility
                if getattr(self, "_kv_cache_bits", 0):
                    # Defer q4/q8 cache dequant to the scheduler worker. See
                    # the memory-aware branch above for the thread-stream reason.
                    request._prompt_cache_needs_worker_dequant = True
                if cache is None:
                    # Dequantization failed — treat as cache miss
                    request.remaining_tokens = request.prompt_token_ids
                    logger.info(
                        f"Request {request.request_id}: dequantization failed, "
                        f"treating as cache miss"
                    )
                else:
                    request.prompt_cache = cache
                    request.cached_tokens = cached_tokens
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
        if (
            request.prompt_cache is None
            and self.disk_cache is not None
            and not _bypass
            and not getattr(request, "_paged_block_table_needs_worker_reconstruct", False)
        ):
            _disk_fetch_tokens = list(request.prompt_token_ids)
            _gpl = getattr(request, "_gen_prompt_len", 0) or 0
            if (
                getattr(self, "_uses_openpangu_cache", False)
                or getattr(self, "_uses_glm5_next_cache", False)
            ):
                # The typed snapshot is the exact full effective prompt's N-1
                # state.  Do not strip/replay template trailer tokens against a
                # different causal-conv boundary.
                _gpl = 0
            if _gpl > 0 and _gpl < len(_disk_fetch_tokens):
                _disk_fetch_tokens = _disk_fetch_tokens[:-_gpl]
                _disk_suffix_tokens = list(request.prompt_token_ids[-_gpl:])
            else:
                _disk_suffix_tokens = []
            _disk_matched_tokens = list(_disk_fetch_tokens)
            if hasattr(self.disk_cache, "fetch_longest_prefix"):
                disk_cache, _disk_matched_tokens = _call_with_optional_cache_extra(
                    self.disk_cache.fetch_longest_prefix,
                    _disk_fetch_tokens,
                    cache_extra_keys=_cache_extra_keys,
                )
                _disk_matched_tokens = list(_disk_matched_tokens or [])
            else:
                if _cache_extra_keys is None:
                    disk_cache = self.disk_cache.fetch(_disk_fetch_tokens)
                else:
                    disk_cache = self.disk_cache.fetch(
                        _disk_fetch_tokens,
                        cache_extra_keys=_cache_extra_keys,
                    )
            if disk_cache is not None:
                # Disk cache stores full-precision N-1 tokens (last prompt token re-fed on hit)
                # Dequantize if KV cache quantization is active (disk stores full precision
                # but may have been quantized before storage in some paths)
                _disk_needs_worker_dequant = False
                if getattr(self, "_kv_cache_bits", 0):
                    # As with L1 memory/prefix hits, do not create/evaluate
                    # dequantized MLX arrays on the API thread. Worker-side
                    # scheduling owns the stream-safe replay path.
                    request._prompt_cache_needs_worker_dequant = True
                    _disk_needs_worker_dequant = True
                if disk_cache is None:
                    # Dequantization failed — treat as full cache miss
                    logger.info(
                        f"Request {request.request_id}: disk cache dequantization "
                        f"failed, treating as cache miss"
                    )
                else:
                    request.prompt_cache = disk_cache
                    if (
                        _disk_matched_tokens
                        and len(_disk_matched_tokens) < len(_disk_fetch_tokens)
                    ):
                        remaining, cached_tokens = (
                            self._disk_prefix_hit_tail_and_cached_tokens(
                                fetch_tokens=_disk_fetch_tokens,
                                matched_tokens=_disk_matched_tokens,
                                gen_prompt_suffix=_disk_suffix_tokens,
                            )
                        )
                    else:
                        if not _disk_matched_tokens:
                            _disk_matched_tokens = list(_disk_fetch_tokens)
                        remaining, cached_tokens = self._prefix_hit_tail_and_cached_tokens(
                            fetch_tokens=_disk_matched_tokens,
                            remaining=[],
                            gen_prompt_suffix=_disk_suffix_tokens,
                        )
                    request.cached_tokens = cached_tokens
                    request.remaining_tokens = remaining
                    # Annotate cache_detail: "disk+tq" for TQ-native 26x-compressed
                    # files, "disk" for standard float16 format.
                    _tq_disk = (
                        hasattr(self.disk_cache, "_last_fetch_tq_native")
                        and self.disk_cache._last_fetch_tq_native
                    )
                    request._tq_native_cache_hit = bool(_tq_disk)
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
                    if getattr(self, "_kv_cache_bits", 0) and not _disk_needs_worker_dequant:
                        try:
                            l1_data = self._quantize_cache_for_storage(disk_cache)
                        except Exception:
                            pass  # Store full-precision on quant failure
                    if _tq_disk and self.block_aware_cache is not None:
                        # A TQ-native prompt file has already paid the packed ->
                        # attention-KV restore cost. Synchronously extracting and
                        # re-slicing every decoded layer into paged blocks here
                        # delayed first-token delivery and stored those blocks as
                        # plain KV. Serve this request from the restored prompt
                        # cache; the normal completion path will populate the
                        # paged cache from the model's re-wrapped TQ cache.
                        request._tq_disk_direct_restore = True
                    elif self.block_aware_cache is not None and not _disk_needs_worker_dequant:
                        try:
                            extracted = self._extract_cache_states(l1_data)
                            if extracted:
                                # Prompt-level disk L2 stores the cache under the
                                # full prompt key but the cache payload itself is
                                # prompt_len - 1, because the last prompt token is
                                # re-fed to obtain first-token logits. Mirror that
                                # N-1 key when backfilling the block cache. Then
                                # route the current request through the paged path
                                # so memory pressure can shrink the hit instead of
                                # discarding it and full-prefilling a huge prompt.
                                _block_store_tokens = (
                                    _disk_matched_tokens[:-1]
                                    if len(_disk_matched_tokens) > 1
                                    else list(_disk_matched_tokens)
                                )
                                _block_table = self.block_aware_cache.store_cache(
                                    request.request_id,
                                    _block_store_tokens,
                                    extracted,
                                    cache_type=_l1_type,
                                )
                                if _block_table is not None:
                                    request.prompt_cache = None
                                    request.block_table = _block_table
                                    request.cached_tokens = int(
                                        getattr(_block_table, "num_tokens", 0) or 0
                                    )
                                    request.shared_prefix_blocks = len(
                                        getattr(_block_table, "block_ids", []) or []
                                    )
                                    request.remaining_tokens = (
                                        self._remaining_tokens_after_cached_prefix(
                                            request,
                                            request.cached_tokens,
                                        )
                                    )
                                    request._paged_block_table_needs_worker_reconstruct = True
                                    request._cache_detail = (
                                        "paged+disk+tq" if _tq_disk else "paged+disk"
                                    )
                                else:
                                    # Clean up request table entry. Release request
                                    # refs so blocks become cached-but-free.
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
                    elif (
                        self.memory_aware_cache is not None
                        and not _disk_needs_worker_dequant
                        and not getattr(self, "_uses_openpangu_cache", False)
                        and not getattr(self, "_uses_glm5_next_cache", False)
                    ):
                        try:
                            _l1_store_tokens = (
                                _disk_matched_tokens[:-1]
                                if len(_disk_matched_tokens) > 1
                                else list(_disk_matched_tokens)
                            )
                            _call_with_optional_cache_extra(
                                self.memory_aware_cache.store,
                                _l1_store_tokens,
                                l1_data,
                                cache_type=_l1_type,
                                cache_extra_keys=_cache_extra_keys,
                            )
                        except Exception:
                            pass
                    elif self.prefix_cache is not None and not _disk_needs_worker_dequant:
                        try:
                            _l1_store_tokens = (
                                _disk_matched_tokens[:-1]
                                if len(_disk_matched_tokens) > 1
                                else list(_disk_matched_tokens)
                            )
                            _call_with_optional_cache_extra(
                                self.prefix_cache.store_cache,
                                _l1_store_tokens,
                                l1_data,
                                cache_type=_l1_type,
                                cache_extra_keys=_cache_extra_keys,
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

    def request_progress(self, request_id: str) -> Optional[int]:
        """Monotonic count of processed tokens, or None if unknown.

        This used to return ``num_computed_tokens + total_output_tokens`` and
        describe itself as "prefilled tokens plus generated tokens". It was
        neither.

        ``num_computed_tokens`` is incremented in exactly ONE place —
        ``Request.append_output_token`` — so it counts output tokens and
        nothing advances it during prefill. The sum was therefore **2x the
        generated token count**, reported to operators as "%d tokens" by the
        two "still progressing" log lines. ``server.py`` had already worked
        this out at the consumer ("it counts OUTPUT tokens only") and relies on
        a prefilling request reporting 0; the producer's docstring was never
        corrected to match.

        Worse, it was not monotonic. ``_reschedule_running_requests`` zeroes
        ``num_computed_tokens`` on a recovery restart while deliberately
        preserving ``total_output_tokens``, so the counter HALVED mid-request.
        The timeout logic only credits ``progress > last_progress``: after a
        restart a healthy request had to regenerate everything it had already
        produced before it registered as alive again, and in the meantime it
        matched neither the "progressing" branch nor the "no reading" grace
        branch — so it could be killed as wedged while generating normally.
        A recovery restart is precisely when that must not happen.

        ``total_output_tokens`` alone is accurate and never resets for generic
        text generators. DSV4 is the exception: its native generator exposes
        the logical composite-cache offset after every bounded prefill chunk,
        so the timeout can observe real SWA/CSA/HCA progress instead of treating
        a near-context prefill as an unreadable zero for several blind windows.

        ``MLLMScheduler.request_progress`` deliberately returns something
        different — ``num_prompt_tokens + total_output_tokens`` — because
        ``MLLMRequest`` has no ``num_computed_tokens`` and its value genuinely
        is prefill plus generation.

        ⚠️ An earlier version of this docstring asserted "both are monotonic,
        which is the only property the callers require". That was FALSE and was
        caught in review: the MLLM path summed ``num_output_tokens``, which is
        derived from ``len(output_tokens)``, and its retry path clears that
        list — so it had the identical go-backwards defect this fix removed
        here. Both now use a lifetime counter that survives a retry. Do not
        restate the invariant without re-checking both sides.
        """
        request = self.requests.get(request_id)
        if request is None:
            return None
        lifetime_output = int(getattr(request, "total_output_tokens", 0) or 0)
        generator = getattr(self, "batch_generator", None)
        batch_uid = getattr(request, "batch_uid", None)
        generator_probe = getattr(generator, "request_progress", None)
        if batch_uid is not None and callable(generator_probe):
            try:
                native_progress = generator_probe(batch_uid)
                if native_progress is not None:
                    return max(lifetime_output, int(native_progress))
            except Exception:
                # Liveness telemetry must never break generation. The stable
                # lifetime output count remains a safe fallback.
                pass
        return lifetime_output

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
        # PLD owns two request-keyed token-history structures.  Normal
        # completion and cache-error rescheduling already drop them, but the
        # abort path (client disconnect, cancellation, or engine error) used
        # to return without doing so.  In a long-lived server every aborted
        # long-context request therefore retained its complete n-gram index.
        # Purge before the idempotence check so a repeated cleanup can also
        # repair state left by an older/incomplete abort.
        self._pld_pending.pop(request_id, None)
        self._pld_ngram_indices.pop(request_id, None)

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
            # Keep the request object so the deferred abort can publish its
            # clean prompt-boundary snapshot into the prefix cache (#58) —
            # otherwise an identical retry re-prefills the whole prompt.
            if (
                self.block_aware_cache is not None
                and not getattr(request, "_bypass_prefix_cache", False)
                and int(getattr(request, "cached_tokens", 0) or 0)
                < len(request.prompt_token_ids)
            ):
                self._abort_salvage_requests[request.request_id] = request

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
            clear_mlx_memory_cache(log=logger)

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
            salvage_req = self._abort_salvage_requests.pop(request_id, None)
            if uid is not None:
                unregister_generation_logprobs(self.model, uid)
                if self.batch_generator is not None:
                    # Salvage the clean prompt-boundary KV snapshot BEFORE
                    # remove() drops the generator request (#58). Metal has
                    # synchronized by this point so the snapshot is safe to
                    # detach and store.
                    if (
                        salvage_req is not None
                        and self.block_aware_cache is not None
                        and hasattr(self.batch_generator, "take_prompt_snapshots")
                    ):
                        try:
                            snaps = self.batch_generator.take_prompt_snapshots(
                                [uid]
                            )
                            if uid in snaps:
                                snapshot, key_tokens = snaps[uid]
                                self._store_aborted_prompt_snapshot(
                                    request_id, salvage_req, snapshot, key_tokens
                                )
                        except Exception as e:
                            logger.warning(
                                f"Prompt-snapshot salvage failed for "
                                f"{request_id}: {e}"
                            )
                    try:
                        self.batch_generator.remove([uid])
                    except Exception as e:
                        logger.warning(
                            f"Deferred abort remove failed for {request_id}: {e}"
                        )
                self.uid_to_request_id.pop(uid, None)
            logger.debug(f"Processed deferred abort for {request_id}")

    def _store_aborted_prompt_snapshot(
        self,
        request_id: str,
        request: "Request",
        snapshot: List[Any],
        key_tokens: List[int],
    ) -> None:
        """Publish an aborted request's prompt-boundary KV into the prefix
        cache (#58).

        The snapshot is the clean prompt-boundary state captured after
        prefill and before decode, so it matches the same N-1 key contract
        as the completion-path store. Mirrors that path: optional TQ
        quantization, extract, store, then settle the stored blocks to
        cached-but-free so memory pressure can reclaim them.
        """
        cached = int(getattr(request, "cached_tokens", 0) or 0)
        if len(key_tokens) <= cached:
            return
        _t_store = time.perf_counter()
        cache_for_store = snapshot
        if getattr(self, "_kv_cache_bits", 0):
            cache_for_store = self._quantize_cache_for_storage(cache_for_store)
        extracted = self._extract_cache_states(cache_for_store)
        if not extracted:
            return
        store_kwargs: Dict[str, Any] = {
            "cache_type": self._pick_cache_type_for_request(request),
        }
        if getattr(request, "_cache_extra_keys", None):
            store_kwargs["cache_extra_keys"] = dict(request._cache_extra_keys)
        if (
            self._is_hybrid
            and not self._uses_dsv4_cache
            and not self._uses_zaya_cache
            and not self._mixed_attention_cache_model
        ):
            store_kwargs["store_cumulative_state"] = False
        self.block_aware_cache.store_cache(
            request_id,
            list(key_tokens),
            extracted,
            **store_kwargs,
        )
        self._dsv4_trace_timing(
            "abort_salvage_store",
            _t_store,
            request_id,
            tokens=len(key_tokens),
            layers=len(extracted),
        )
        logger.info(
            "Salvaged aborted request %s prompt KV into prefix cache "
            "(%d cache-key tokens, %d layers)",
            request_id,
            len(key_tokens),
            len(extracted),
        )
        # store_cache() re-registered this request in _request_tables with
        # the stored blocks at ref_count>=1. Settle them to cached-but-free
        # (same idiom as the completion-path post-store release).
        try:
            _stored_entry = self.block_aware_cache._request_tables.pop(
                request_id, None
            )
            self.block_aware_cache.paged_cache.release_request_refs(
                _stored_entry.block_table if _stored_entry else None
            )
            self.block_aware_cache.paged_cache.detach_request(request_id)
            if self.block_aware_cache.paged_cache.enforces_byte_budget:
                self.block_aware_cache.paged_cache.enforce_byte_budget()
        except Exception as _rel_e:
            logger.debug(
                "Post-salvage paged ref release failed for %s: %s",
                request_id,
                _rel_e,
            )

    def has_requests(self) -> bool:
        """Check for generation work or deferred abort cleanup.

        The async engine loop only calls ``step()`` while this returns true.
        A last-request abort removes the request from ``running`` but defers
        BatchGenerator removal until the next safe scheduler step.  Treat that
        cleanup as work or the loop goes idle with the generator UID and its
        native cache still retained indefinitely.

        Foreground work ONLY. Idle maintenance (e.g. the deferred SSM
        re-derive queue) is surfaced through ``has_idle_tasks()`` and drained
        by the engine loop's idle branch AFTER responses are finalized
        (vmlx#245) — it must never keep ``step()`` on the response path.
        """
        return bool(
            self.waiting
            or self.running
            or self._pending_aborts
        )

    # ── Post-response idle-task hook (vmlx#245) ──────────────────────────
    # Maintenance work that is a pure cache optimization (e.g. the deferred
    # SSM companion re-derive) must run AFTER responses are finalized and
    # must yield to newly-arrived foreground requests. The engine loop
    # drains AT MOST ONE task per idle iteration, dispatched on the same
    # executor as step() (Metal streams are thread-local). Tasks must be
    # safe to skip: shutdown/model-eject drops the queue.

    def register_idle_task(
        self,
        fn: Callable[[], "IdleTaskResult"],
        *,
        name: str = "",
    ) -> None:
        """Queue ``fn`` to run at engine idle, after responses are delivered.

        ``fn`` runs on the step executor and returns ``IdleTaskResult.DONE``
        to dequeue or ``IdleTaskResult.PARKED`` to re-queue at the front and
        retry at the next idle iteration. Exceptions drop the task.
        """
        if not hasattr(self, "_idle_tasks"):
            self._idle_tasks = deque()
        self._idle_tasks.append((name or getattr(fn, "__name__", "idle-task"), fn))

    def has_idle_tasks(self) -> bool:
        """Whether idle maintenance work is queued (engine idle-branch gate)."""
        return bool(getattr(self, "_idle_tasks", None))

    def run_one_idle_task(self) -> bool:
        """Run at most one queued idle task. Returns True if one ran.

        Must be called on the step executor (Metal stream affinity), and only
        when the scheduler has no foreground work — the engine loop's idle
        branch guarantees both.
        """
        tasks = getattr(self, "_idle_tasks", None)
        if not tasks:
            return False
        name, fn = tasks.popleft()
        try:
            result = fn()
        except Exception as e:
            logger.warning(f"Idle task {name!r} failed (dropped): {e}")
            return True
        if result is IdleTaskResult.PARKED:
            tasks.appendleft((name, fn))
        return True

    def _begin_foreground_admission(self, request_id: str) -> None:
        """Mark a foreground request before scheduler lookup can begin.

        EngineCore may need to wait for the previous request's terminal cache
        cleanup before calling ``add_request``.  Idle cache-maintenance work
        must still yield during that gap or it can occupy Metal with a full
        shadow prefill before the waiting request reaches ``self.waiting``.
        """
        pending = getattr(self, "_foreground_admissions_pending", None)
        if pending is None:
            pending = set()
            self._foreground_admissions_pending = pending
        pending.add(str(request_id))

    def _end_foreground_admission(self, request_id: str) -> None:
        """Clear a foreground-admission marker after admission or failure."""
        pending = getattr(self, "_foreground_admissions_pending", None)
        if pending is not None:
            pending.discard(str(request_id))

    def _foreground_pending(self) -> bool:
        """Cheap poll: has foreground work arrived? (idle-task park signal)"""
        return bool(
            self.waiting
            or self.running
            or getattr(self, "unprocessed_requests", None)
            or self._pending_aborts
            or getattr(self, "_foreground_admissions_pending", None)
        )

    def _ensure_ssm_rederive_idle_task(self) -> None:
        """Register the SSM re-derive drain task once per queue lifetime."""
        if getattr(self, "_ssm_rederive_task_queued", False):
            return
        self._ssm_rederive_task_queued = True
        self.register_idle_task(self._drain_one_ssm_rederive, name="ssm-rederive")

    def _drain_one_ssm_rederive(self) -> "IdleTaskResult":
        """Idle-task body: process ONE deferred SSM re-derive queue entry.

        ── Deferred SSM re-derive (idle-time processing) ── vmlx#103/#245
        For thinking models (gen_prompt_len > 0), the SSM companion store
        queues a re-derive task instead of skipping entirely. The re-derive
        runs a separate prefill pass on just the prompt tokens (no
        thinking/output contamination) and stores the clean SSM state for
        future prefix cache hits. It runs ONLY at engine idle — the forward
        pass uses the Metal GPU so it can't overlap with active or queued
        work — and parks between prefill chunks when foreground work
        arrives, bounding the unpreemptable window to one chunk (vmlx#245).
        """
        queue = getattr(self, "_ssm_rederive_queue", None)
        if (
            not queue
            or not self._is_hybrid
            or self._uses_zaya_cache
            or not self.config.enable_prefix_cache
            or self._ssm_state_cache is None
        ):
            # Drained, or this scheduler can never consume the queue —
            # either way the task is complete. Drop stale entries so the
            # engine loop doesn't spin on unservable work.
            if queue is not None:
                queue.clear()
            self._ssm_rederive_task_queued = False
            return IdleTaskResult.DONE
        if self._foreground_pending():
            return IdleTaskResult.PARKED
        # Process ONE entry per idle iteration to avoid long GPU stalls.
        tokens, prompt_len, orig_request_id = queue.pop(0)
        if self._ssm_state_cache.has_complete(tokens, prompt_len):
            # Stored between enqueue and this idle tick (typical after a cache
            # HIT restored from that very entry, or an identical request
            # completed first). The deferred prefill would recompute
            # byte-identical state — a full wasted prompt-length prefill that
            # starves the next request's TTFT. Same probe the MLLM drain has
            # carried since 1e8602b40; this text path never got it.
            logger.info(
                "SSM re-derive skipped at idle for %s: complete companion "
                "already stored at %d-token key",
                orig_request_id,
                prompt_len,
            )
            if queue:
                return IdleTaskResult.PARKED
            self._ssm_rederive_task_queued = False
            return IdleTaskResult.DONE
        try:
            logger.info(
                f"SSM re-derive: running deferred prefill for "
                f"{orig_request_id} ({prompt_len} prompt tokens, "
                f"{len(queue)} remaining in queue)"
            )
            parked = False

            def _park_poll() -> bool:
                nonlocal parked
                parked = self._foreground_pending()
                return parked

            clean_cache = self._prefill_for_prompt_only_cache(
                tokens, should_stop=_park_poll
            )
            if parked and clean_cache is None:
                # Foreground arrived mid-prefill: partial work was discarded;
                # retry this entry at the next idle iteration.
                queue.insert(0, (tokens, prompt_len, orig_request_id))
                logger.info(
                    f"SSM re-derive: parked for foreground work "
                    f"({orig_request_id}, {prompt_len} prompt tokens)"
                )
                return IdleTaskResult.PARKED
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
                clear_mlx_memory_cache(log=logger)
        except Exception as e:
            logger.warning(f"SSM re-derive failed for {orig_request_id}: {e}")
        if queue:
            return IdleTaskResult.PARKED
        self._ssm_rederive_task_queued = False
        return IdleTaskResult.DONE

    # ── DSV4 shadow re-key (idle-time predicted-transcript store) ────────
    # With builtin tools OFF the DSV4 encoder strips prior-turn reasoning
    # from the rendered history, so the reasoning-inclusive chain stored at
    # completion cannot match the next turn past the assistant rail — the
    # next turn re-prefills the whole visible transcript cold. The visible
    # render is byte-predictable from this turn's fed prompt and output
    # (`prompt[:-1] + </think> + answer + eos`), so at engine idle we
    # prefill that predicted sequence and store it as a prompt-only chain
    # keyed exactly like the next prompt's prefix.

    @staticmethod
    def _dsv4_predict_shadow_rekey_tokens(
        prompt_ids: Sequence[int], output_ids: Sequence[int]
    ) -> Optional[List[int]]:
        """Predicted next-turn visible-transcript prefix, or None when the
        turn is ineligible.

        Eligibility is derived from the tokens alone — no render stamp
        needed: a reasoning-keeping (tools-on) render leaves prior-turn
        ``<think>`` openers in the fed history, while a reasoning-dropping
        render never feeds THINK_OPEN outside the terminal generation rail.
        """
        from .utils.dsv4_batch_generator import (
            DSV4_EOS_ID,
            THINK_CLOSE_ID,
            THINK_OPEN_ID,
        )

        prompt_ids = list(prompt_ids or [])
        output_ids = list(output_ids or [])
        if not prompt_ids or not output_ids:
            return None
        # Thinking rail must be armed (effort-prompt variants that append
        # tokens after <think> are out of scope for v1 — skip cleanly).
        if prompt_ids[-1] != THINK_OPEN_ID:
            return None
        # Prior-turn reasoning present in the fed history means this is a
        # reasoning-keeping render; the extended store already covers the
        # exact fed sequence and a stripped shadow chain would never match.
        if THINK_OPEN_ID in prompt_ids[:-1]:
            return None
        try:
            close_idx = output_ids.index(THINK_CLOSE_ID)
        except ValueError:
            return None
        visible = list(output_ids[close_idx + 1 :])
        while visible and visible[-1] == DSV4_EOS_ID:
            visible.pop()
        if not visible:
            return None
        return prompt_ids[:-1] + [THINK_CLOSE_ID] + visible + [DSV4_EOS_ID]

    def _queue_dsv4_shadow_rekey(
        self,
        request_id: str,
        request: "Request",
        finish_reason: Optional[str],
    ) -> None:
        """Queue an idle-time shadow re-key for an eligible finished turn."""
        try:
            from .utils.dsv4_batch_generator import (
                dsv4_prompt_snapshot_min_tokens,
                dsv4_shadow_rekey_enabled,
                dsv4_shadow_rekey_max_tokens,
            )

            if (
                not self._uses_dsv4_cache
                or self.block_aware_cache is None
                or not self.config.enable_prefix_cache
                or finish_reason != "stop"
                or getattr(request, "_bypass_prefix_cache", False)
                # Tools-on renders keep a distinct template/tool prefix and
                # already extend the normal native chain. Building a stripped
                # tools-off prediction here cannot serve that continuation and
                # only burns a redundant post-response prefill.
                or getattr(request, "_vmlx_tools_present", False)
                or not dsv4_shadow_rekey_enabled()
            ):
                return
            predicted = self._dsv4_predict_shadow_rekey_tokens(
                getattr(request, "prompt_token_ids", None) or [],
                getattr(request, "output_token_ids", None) or [],
            )
            if predicted is None:
                return
            if len(predicted) < dsv4_prompt_snapshot_min_tokens():
                # Below one cache block nothing is storable, and the next
                # turn re-prefills this many tokens for free anyway.
                return
            cap = dsv4_shadow_rekey_max_tokens()
            if cap > 0 and len(predicted) > cap:
                logger.info(
                    "DSV4 shadow re-key skipped for %s: predicted chain "
                    "%d tokens exceeds cap %d.",
                    request_id,
                    len(predicted),
                    cap,
                )
                return
            entry = (
                predicted,
                request_id,
                self._pick_cache_type_for_request(request),
                dict(request._cache_extra_keys)
                if getattr(request, "_cache_extra_keys", None)
                else None,
            )
            queue = getattr(self, "_dsv4_shadow_rekey_queue", None)
            if queue is None:
                queue = []
                self._dsv4_shadow_rekey_queue = queue
            queue.append(entry)
            # Newest-wins bound: idle GPU time should chase the latest
            # conversation state, not a backlog of superseded turns.
            while len(queue) > 2:
                dropped = queue.pop(0)
                logger.debug(
                    "DSV4 shadow re-key queue bound: dropped stale entry "
                    "for %s",
                    dropped[1],
                )
            logger.info(
                "DSV4 shadow re-key queued for %s: %d predicted "
                "visible-transcript tokens (reasoning-dropping render).",
                request_id,
                len(predicted),
            )
            self._ensure_dsv4_shadow_rekey_idle_task()
        except Exception as e:
            logger.debug(
                f"DSV4 shadow re-key queue failed for {request_id}: {e}"
            )

    def _ensure_dsv4_shadow_rekey_idle_task(self) -> None:
        """Register the shadow re-key drain task once per queue lifetime."""
        if getattr(self, "_dsv4_shadow_rekey_task_queued", False):
            return
        self._dsv4_shadow_rekey_task_queued = True
        self.register_idle_task(
            self._drain_one_dsv4_shadow_rekey, name="dsv4-shadow-rekey"
        )

    def _drain_one_dsv4_shadow_rekey(self) -> "IdleTaskResult":
        """Idle-task body: prefill + store ONE predicted-transcript chain.

        Mirrors the deferred SSM re-derive discipline (vmlx#245): runs only
        at engine idle, parks between prefill chunks when foreground work
        arrives, and processes one queue entry per idle iteration.
        """
        queue = getattr(self, "_dsv4_shadow_rekey_queue", None)
        if (
            not queue
            or not self._uses_dsv4_cache
            or self.block_aware_cache is None
            or not self.config.enable_prefix_cache
        ):
            if queue is not None:
                queue.clear()
            self._dsv4_shadow_rekey_task_queued = False
            return IdleTaskResult.DONE
        if self._foreground_pending():
            return IdleTaskResult.PARKED
        predicted, orig_request_id, cache_type, extra_keys = queue.pop(0)
        try:
            logger.info(
                "DSV4 shadow re-key: prefilling predicted visible "
                "transcript for %s (%d tokens, %d remaining in queue)",
                orig_request_id,
                len(predicted),
                len(queue),
            )
            parked = False

            def _park_poll() -> bool:
                nonlocal parked
                parked = self._foreground_pending()
                return parked

            fresh_cache = self._prefill_for_prompt_only_cache(
                predicted,
                should_stop=_park_poll,
                capture_dsv4_deltas=True,
            )
            if parked and fresh_cache is None:
                # Foreground arrived mid-prefill: partial work discarded;
                # retry this entry at the next idle iteration.
                queue.insert(
                    0, (predicted, orig_request_id, cache_type, extra_keys)
                )
                logger.info(
                    "DSV4 shadow re-key: parked for foreground work (%s)",
                    orig_request_id,
                )
                return IdleTaskResult.PARKED
            if fresh_cache is not None:
                self._store_dsv4_shadow_chain(
                    f"{orig_request_id}-rekey",
                    predicted,
                    fresh_cache,
                    cache_type,
                    extra_keys,
                )
                del fresh_cache
                clear_mlx_memory_cache(log=logger)
        except Exception as e:
            logger.warning(
                f"DSV4 shadow re-key failed for {orig_request_id}: {e}"
            )
        if queue:
            return IdleTaskResult.PARKED
        self._dsv4_shadow_rekey_task_queued = False
        return IdleTaskResult.DONE

    def _store_dsv4_shadow_chain(
        self,
        store_id: str,
        key_tokens: List[int],
        cache: List[Any],
        cache_type: str,
        extra_keys: Optional[Dict[str, Any]],
    ) -> None:
        """Store an idle-prefilled predicted chain, then settle refs.

        The shadow prefill must donate the same native delta transport used by
        request-time DSV4 snapshots. Generic composite snapshots produce
        pending parent pages that cannot be extended by native deltas. Store
        under a synthetic id, then release the blocks to cached-but-free so
        memory pressure can reclaim them.
        """
        _t_store = time.perf_counter()
        from .utils.dsv4_batch_generator import DSV4BatchGenerator

        transport = DSV4BatchGenerator._dsv4_delta_transport(cache)
        if not transport:
            logger.warning(
                "DSV4 shadow re-key skipped for %s: native delta transport "
                "was not captured",
                store_id,
            )
            return
        expected_end = len(key_tokens)
        try:
            intervals = tuple(
                (int(start), int(end))
                for start, end in transport[0]["dsv4_record_intervals"]
            )
            complete = bool(
                intervals
                and intervals[0][0] == 0
                and intervals[-1][1] == expected_end
                and all(
                    start >= 0
                    and end > start
                    and end - start <= 256
                    and (index == 0 or start == intervals[index - 1][1])
                    for index, (start, end) in enumerate(intervals)
                )
                and all(
                    tuple(
                        (int(start), int(end))
                        for start, end in layer["dsv4_record_intervals"]
                    )
                    == intervals
                    for layer in transport
                )
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            complete = False
        if not complete:
            logger.warning(
                "DSV4 shadow re-key skipped for %s: native delta chain does "
                "not cover one contiguous [0, %d) interval",
                store_id,
                expected_end,
            )
            return
        extracted = self._extract_cache_states(transport)
        if not extracted:
            return
        store_kwargs: Dict[str, Any] = {"cache_type": cache_type}
        if extra_keys:
            store_kwargs["cache_extra_keys"] = dict(extra_keys)
        self.block_aware_cache.store_cache(
            store_id,
            list(key_tokens),
            extracted,
            **store_kwargs,
        )
        self._dsv4_trace_timing(
            "shadow_rekey_store",
            _t_store,
            store_id,
            tokens=len(key_tokens),
            layers=len(extracted),
        )
        logger.info(
            "DSV4 shadow re-key stored predicted visible-transcript chain "
            "for %s (%d cache-key tokens, %d layers)",
            store_id,
            len(key_tokens),
            len(extracted),
        )
        # store_cache() registered the synthetic id in _request_tables with
        # the stored blocks at ref_count>=1. Settle them to cached-but-free
        # (same idiom as the completion-path post-store release).
        try:
            _stored_entry = self.block_aware_cache._request_tables.pop(
                store_id, None
            )
            self.block_aware_cache.paged_cache.release_request_refs(
                _stored_entry.block_table if _stored_entry else None
            )
            self.block_aware_cache.paged_cache.detach_request(store_id)
            if self.block_aware_cache.paged_cache.enforces_byte_budget:
                self.block_aware_cache.paged_cache.enforce_byte_budget()
        except Exception as _rel_e:
            logger.debug(
                "Post-shadow-store paged ref release failed for %s: %s",
                store_id,
                _rel_e,
            )

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _cleanup_failed_block_cache_initialization(
        self,
        *,
        block_disk_store: Optional[Any],
    ) -> None:
        """Release asynchronous L2 resources after partial cache construction."""
        ssm_state_cache = getattr(self, "_ssm_state_cache", None)
        ssm_disk_store = (
            getattr(ssm_state_cache, "_disk", None)
            if ssm_state_cache is not None
            else None
        )
        if ssm_disk_store is not None:
            try:
                ssm_disk_store.shutdown(timeout=None)
            except Exception as exc:
                logger.warning(
                    "Failed to stop SSM companion after cache init failure: %s",
                    exc,
                )
            finally:
                ssm_state_cache._disk = None

        manager = getattr(self, "paged_cache_manager", None)
        manager_disk_store = getattr(manager, "_disk_store", None)
        owned_block_store = manager_disk_store or block_disk_store
        if owned_block_store is not None:
            try:
                owned_block_store.shutdown()
            except Exception as exc:
                logger.warning(
                    "Failed to stop block disk cache after cache init failure: %s",
                    exc,
                )
        if manager is not None and manager_disk_store is not None:
            manager._disk_store = None
        self.paged_cache_manager = None
        self.block_aware_cache = None

    def shutdown(self) -> None:
        """Shutdown the scheduler and flush disk caches. Idempotent."""
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        # Idle maintenance is a pure cache optimization — safe to skip.
        # Drop queued tasks so no forward pass runs after teardown begins
        # (vmlx#245 shutdown contract).
        if getattr(self, "_idle_tasks", None):
            self._idle_tasks.clear()
        if getattr(self, "_ssm_rederive_queue", None):
            self._ssm_rederive_queue.clear()
        self._ssm_rederive_task_queued = False

        # Flush prompt-level disk cache (DiskCacheManager)
        if getattr(self, "disk_cache", None) is not None:
            logger.info("Shutting down prompt disk cache...")
            self.disk_cache.shutdown()
            logger.info("Prompt disk cache shutdown complete")

        # Flush hybrid SSM companion writes before releasing the shared
        # aggregate-budget lease owned by BlockDiskStore.  SSM publication is
        # asynchronous and uses that same coordinator for atomic accounting.
        if self._ssm_state_cache is not None:
            ssm_disk_store = getattr(self._ssm_state_cache, "_disk", None)
            if ssm_disk_store is not None:
                logger.info("Shutting down SSM companion disk cache...")
                ssm_disk_store.shutdown(timeout=None)
                self._ssm_state_cache._disk = None
                logger.info("SSM companion disk cache shutdown complete")

        # Flush block-level disk cache (BlockDiskStore)
        if hasattr(self, "paged_cache_manager") and self.paged_cache_manager:
            disk_store = getattr(self.paged_cache_manager, "_disk_store", None)
            if disk_store is not None:
                logger.info("Shutting down block disk cache...")
                disk_store.shutdown()
                logger.info("Block disk cache shutdown complete")

    def _seed_cache_from_ssm_checkpoint(
        self,
        request: Request,
        *,
        reconstructed: List[Any],
        ssm_tokens: List[int],
        fetch_num: int,
    ) -> Tuple[Optional[List[Any]], int]:
        """Build a cache positioned at the newest complete SSM checkpoint.

        Returns ``(seed_cache, checkpoint_len)``, or ``(None, 0)`` when no
        usable checkpoint exists and the caller must derive from scratch.

        Unlike the vmlx#91 RESUME ladder this deliberately accepts a checkpoint
        at ANY length, aligned or not. RESUME has to trim the KV back to the
        checkpoint, so a non-block-aligned one would leave SSM and KV
        disagreeing about position; here the KV hit is left alone and only the
        recurrent state is advanced, so alignment is irrelevant. That
        distinction is the whole point -- companion stores land on gpl-stripped
        prompt boundaries, which are essentially never multiples of 64, so the
        aligned-only ladder misses them and falls through to a full re-derive.
        """
        if not reconstructed or int(fetch_num or 0) <= 0:
            return None, 0
        prefetched_attr = "_hybrid_ssm_prefetched_checkpoint"
        prefetched = getattr(request, prefetched_attr, None)
        try:
            delattr(request, prefetched_attr)
        except AttributeError:
            pass
        cache = self._ssm_state_cache
        fetch_longest = getattr(cache, "fetch_longest_prefix", None)
        hit = None
        if isinstance(prefetched, tuple) and len(prefetched) == 4:
            prefetched_max_len, ck_len, ck_states, ck_complete = prefetched
            if int(prefetched_max_len or 0) == int(fetch_num):
                hit = (ck_len, ck_states, ck_complete)
        if hit is None:
            if cache is None or not callable(fetch_longest):
                return None, 0
            try:
                hit = fetch_longest(ssm_tokens, fetch_num)
            except Exception:
                return None, 0
        if not hit:
            return None, 0
        try:
            ck_len, ck_states, ck_complete = int(hit[0]), hit[1], bool(hit[2])
        except Exception:
            return None, 0
        # An incomplete entry was stored after the generation-prompt suffix was
        # processed; resuming from it would bake that contamination into every
        # future exact hit at fetch_num.
        if not ck_complete or not ck_states or not (0 < ck_len < fetch_num):
            return None, 0

        kv_positions = list(self._hybrid_kv_positions or [])
        if not kv_positions or len(kv_positions) != len(reconstructed):
            return None, 0

        try:
            import mlx.core as mx
            from .utils.cache_extent import cache_offset
        except Exception:
            return None, 0

        sliced: List[Any] = []
        for layer in reconstructed:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                return None, 0
            # OFFSET is the only token-count authority. Restored KV buffers are
            # zero-padded up to the cache step, so keys.shape is routinely
            # LARGER than the logical length -- reading the shape here is the
            # exact mistake that silently emptied dots3 answers.
            live = int(cache_offset(layer) or 0)
            if live < ck_len:
                return None, 0
            seq_axis = 1 if keys.ndim == 3 else 2
            if int(keys.shape[seq_axis]) < ck_len:
                return None, 0
            try:
                clone = type(layer)()
            except Exception:
                # Only plain append-only KV can be sliced to an arbitrary
                # position. Anything needing constructor arguments is a typed
                # or windowed cache whose state is not position-sliceable.
                return None, 0
            if seq_axis == 1:
                clone.keys = keys[:, :ck_len, :]
                clone.values = values[:, :ck_len, :]
            else:
                clone.keys = keys[..., :ck_len, :]
                clone.values = values[..., :ck_len, :]
            clone.offset = ck_len
            sliced.append(clone)

        seed = self._compose_hybrid_cache_from_parts(sliced, ck_states)
        if seed is None:
            return None, 0
        mx.eval([c.keys for c in sliced] + [c.values for c in sliced])
        logger.info(
            "Request %s: deriving SSM companion for %d tokens as a %d-token "
            "DELTA from a checkpoint at %d, instead of re-running the whole "
            "prefix.",
            request.request_id,
            fetch_num,
            fetch_num - ck_len,
            ck_len,
        )
        return seed, ck_len

    def _compose_hybrid_cache_from_parts(
        self,
        attention_caches: List[Any],
        ssm_states: List[Any],
    ) -> Optional[List[Any]]:
        """Interleave attention caches and SSM states into a full layer list."""
        kv_positions = list(self._hybrid_kv_positions or [])
        total = int(getattr(self, "_hybrid_num_layers", 0) or 0)
        if not kv_positions or total <= 0:
            return None
        if len(attention_caches) != len(kv_positions):
            return None
        kv_set = set(kv_positions)
        ssm_slots = [i for i in range(total) if i not in kv_set]
        if len(ssm_slots) != len(ssm_states):
            return None
        composed: List[Any] = [None] * total
        for pos, layer in zip(kv_positions, attention_caches):
            if not (0 <= int(pos) < total):
                return None
            composed[int(pos)] = layer
        for pos, state in zip(ssm_slots, ssm_states):
            composed[pos] = state
        if any(c is None for c in composed):
            return None
        return composed

    def _finalize_hybrid_paged_cache_on_worker(
        self,
        request: Request,
        reconstructed: List[Any],
    ) -> Optional[List[Any]]:
        """Attach SSM companion state to a paged KV hit on the llm-worker."""
        block_table = getattr(request, "block_table", None)
        fetch_num = getattr(block_table, "num_tokens", 0) if block_table else 0
        ssm_tokens = list(
            getattr(request, "_hybrid_ssm_fetch_tokens", None)
            or request.prompt_token_ids
            or []
        )
        request._hybrid_prompt_cache_needs_worker_ssm = False

        if (
            not self._is_hybrid
            or self._uses_dsv4_cache
            or self._uses_zaya_cache
            or self._ssm_state_cache is None
            or self.block_aware_cache is None
            or block_table is None
            or fetch_num <= 0
        ):
            return reconstructed

        logger.info(
            "SSM companion worker fetch: req=%s fetch_num=%d cache_size=%d "
            "tokens_tail=%s",
            request.request_id,
            fetch_num,
            len(getattr(self._ssm_state_cache, "_store", {})),
            ssm_tokens[max(0, fetch_num - 8):fetch_num],
        )

        ssm_states = None
        try:
            entry = self._ssm_state_cache.fetch(ssm_tokens, fetch_num)
        except Exception as e:
            logger.debug(
                "SSM companion worker fetch failed for %s: %s",
                request.request_id,
                e,
            )
            entry = None

        if entry is not None:
            ssm_states, is_complete = entry
            if not is_complete:
                logger.info(
                    f"SSM companion for {request.request_id}: "
                    "is_complete=False (gpl-contaminated), rejecting hit — "
                    "full prefill"
                )
                ssm_states = None

        # vmlx#91 RESUME: exact SSM miss, but a shorter complete companion may
        # still be a valid prefix. Trim KV blocks to that checkpoint and re-run
        # only the tail.
        if not ssm_states:
            resume_disabled = os.environ.get("VMLX_DISABLE_SSM_PREFIX_RESUME") in (
                "1",
                "true",
                "True",
                "yes",
                "on",
            )
            missed_ck = None
            block_size = int(getattr(self.block_aware_cache, "block_size", 0) or 0)
            if not resume_disabled and block_size > 0:
                try:
                    missed_ck = self._fetch_block_aligned_ssm_checkpoint(
                        request,
                        max_len=fetch_num,
                        block_size=block_size,
                        # The exact worker lookup immediately above already
                        # missed. Do not probe that same SSD key again, and
                        # carry any complete non-aligned checkpoint into the
                        # delta-seed path instead of refaulting it twice.
                        exact_boundary_already_missed=True,
                        retain_unaligned_checkpoint=True,
                    )
                except Exception:
                    missed_ck = None
            if missed_ck is not None:
                ck_len, ck_states = missed_ck
                trimmed = self.block_aware_cache.trim_block_table(
                    request.request_id, ck_len
                )
                if (
                    trimmed is not None
                    and trimmed.num_tokens > 0
                    and trimmed.num_tokens == ck_len
                ):
                    rereconstructed = self.block_aware_cache.reconstruct_cache(trimmed)
                    if rereconstructed is not None and getattr(self, "_kv_cache_bits", 0):
                        rereconstructed = self._dequantize_cache_for_use(
                            rereconstructed
                        )
                    if rereconstructed is not None:
                        block_table = trimmed
                        request.block_table = trimmed
                        request.cached_tokens = trimmed.num_tokens
                        request.shared_prefix_blocks = len(trimmed.block_ids)
                        reconstructed = rereconstructed
                        ssm_states = ck_states
                        request.remaining_tokens = self._remaining_tokens_after_cached_prefix(
                            request,
                            trimmed.num_tokens,
                        )
                        self._accept_paged_hit_credit(request, trimmed.num_tokens)
                        logger.info(
                            f"Request {request.request_id}: vmlx#91 RESUME — "
                            f"trimmed KV to {trimmed.num_tokens} tokens "
                            f"(block-aligned with SSM checkpoint), "
                            f"prefill tail: {len(request.remaining_tokens)} tokens"
                        )
                elif trimmed is not None:
                    logger.info(
                        "Request %s: vmlx#91 RESUME skipped — KV trim (%s) "
                        "did not match SSM checkpoint (%s)",
                        request.request_id,
                        getattr(trimmed, "num_tokens", 0),
                        ck_len,
                    )

        if not ssm_states and fetch_num > 0 and len(ssm_tokens) >= fetch_num:
            boundary_tokens = list(ssm_tokens[:fetch_num])
            # Deriving the companion over the WHOLE prefix makes a cache HIT
            # cost as much forward compute as a cold prefill: a 12k paged hit
            # would re-run all ~12k tokens here and then still prefill the
            # visible tail, while usage reports cached_tokens=12k. The hit
            # saves nothing it claims to save.
            #
            # The RESUME ladder above already declined, but only because it
            # needs a checkpoint that is BLOCK-ALIGNED -- it has to trim the KV
            # back to the checkpoint, and SSM@ck with KV@floor64(ck)
            # double-applies the gap. That alignment requirement does not apply
            # here: this path advances the SSM state up to fetch_num and leaves
            # the KV hit untouched, so ANY complete checkpoint below fetch_num
            # is a valid starting point.
            _seed_cache, _seed_len = self._seed_cache_from_ssm_checkpoint(
                request,
                reconstructed=reconstructed,
                ssm_tokens=ssm_tokens,
                fetch_num=fetch_num,
            )
            try:
                clean_cache = self._prefill_for_prompt_only_cache(
                    boundary_tokens,
                    base_cache=_seed_cache,
                    base_token_count=_seed_len,
                )
                if clean_cache is not None:
                    kv_set = set(self._hybrid_kv_positions or [])
                    derived_states = []
                    for layer_idx, c in enumerate(clean_cache):
                        if layer_idx in kv_set:
                            continue
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
                            derived_states.append(cloned)
                        else:
                            derived_states.append(c)
                    if derived_states:
                        ssm_states = derived_states
                        try:
                            self._ssm_state_cache.store(
                                boundary_tokens,
                                fetch_num,
                                derived_states,
                            )
                        except Exception:
                            pass
                        logger.info(
                            "Request %s: synchronously derived SSM companion "
                            "for block-aligned paged hit (%d tokens, %d layers)",
                            request.request_id,
                            fetch_num,
                            len(derived_states),
                        )
                    del clean_cache
                    clear_mlx_memory_cache(log=logger)
            except Exception as e:
                logger.info(
                    "Request %s: synchronous SSM companion derive failed for "
                    "%d-token paged hit: %s",
                    request.request_id,
                    fetch_num,
                    e,
                )

        if not ssm_states:
            try:
                boundary_tokens = list(ssm_tokens[:fetch_num])
                if (
                    boundary_tokens
                    and self._ssm_state_cache is not None
                    and len(getattr(self._ssm_state_cache, "_store", {}))
                    < self._ssm_state_cache.max_entries
                ):
                    if not hasattr(self, "_ssm_rederive_queue"):
                        self._ssm_rederive_queue = []
                    if (
                        (boundary_tokens, fetch_num, request.request_id)
                        not in self._ssm_rederive_queue
                    ):
                        if len(self._ssm_rederive_queue) >= SSM_REDERIVE_QUEUE_CAP:
                            self._ssm_rederive_queue.pop(0)
                        self._ssm_rederive_queue.append(
                            (boundary_tokens, fetch_num, request.request_id)
                        )
                        self._ensure_ssm_rederive_idle_task()
                        logger.info(
                            "SSM companion: queued block-boundary re-derive "
                            "for %s (%d cache-key tokens after hybrid miss)",
                            request.request_id,
                            fetch_num,
                        )
            except Exception as e:
                logger.debug(
                    "SSM companion: failed to queue block-boundary re-derive "
                    "for %s: %s",
                    request.request_id,
                    e,
                )
            logger.info(
                f"Request {request.request_id}: hybrid paged MISS — "
                f"{fetch_num} KV tokens cached but no usable SSM companion, "
                "full prefill (blocks kept cached for future sessions)"
            )
            self._release_unusable_paged_hit(request)
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            return None

        # The KV offset (from the restored block tensors) and the companion
        # boundary (the block table's token count) come from independent
        # sources.  Pairing them unequal desynchronizes the attention and
        # recurrent halves; fail closed to a full prefill instead.
        from .utils.cache_extent import hybrid_kv_boundary_mismatch

        _kv_mismatch = hybrid_kv_boundary_mismatch(reconstructed, fetch_num)
        if _kv_mismatch is not None:
            logger.warning(
                "Request %s: hybrid restore REJECTED — KV layer %d restored at "
                "offset %d but the SSM companion boundary is %d; pairing them "
                "would desynchronize attention and recurrent state. Full "
                "prefill instead.",
                request.request_id,
                _kv_mismatch[0],
                _kv_mismatch[1],
                int(fetch_num or 0),
            )
            self._release_unusable_paged_hit(request)
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            return None

        full_cache = _fix_hybrid_cache(
            reconstructed,
            self.model,
            kv_positions=self._hybrid_kv_positions,
            num_model_layers=self._hybrid_num_layers,
        )
        if full_cache is None:
            logger.warning(
                f"Request {request.request_id}: hybrid cache expansion failed"
            )
            self._release_unusable_paged_hit(request)
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            return None

        kv_set = set(self._hybrid_kv_positions or [])
        ssm_idx = 0
        for layer_i in range(len(full_cache)):
            if layer_i not in kv_set and ssm_idx < len(ssm_states):
                full_cache[layer_i] = ssm_states[ssm_idx]
                ssm_idx += 1
        tq_native_blocks = int(
            getattr(self.block_aware_cache, "_last_reconstruct_tq_blocks", 0) or 0
        )
        if getattr(self, "_tq_active", False):
            try:
                from .mllm_batch_generator import _recompress_to_tq

                full_cache = _recompress_to_tq(full_cache, self.model)
            except Exception as exc:
                logger.warning(
                    "Request %s: failed to restore TurboQuant cache objects after "
                    "paged reconstruction: %s",
                    request.request_id,
                    exc,
                )
                return None
        request._tq_native_cache_hit = tq_native_blocks > 0
        request.prompt_cache = full_cache
        if bool(
            getattr(getattr(self, "paged_cache_manager", None), "disk_only", False)
        ):
            request._cache_detail = "block-disk+ssm"
        else:
            request._cache_detail = (
                "paged+ssm+disk"
                if getattr(request, "_paged_disk_hit", False)
                else "paged+ssm"
            )
        if request._tq_native_cache_hit:
            request._cache_detail += "+tq-native"
        request._cache_detail_ssm_layers = ssm_idx
        logger.info(
            f"Request {request.request_id}: hybrid block-cache HIT — "
            f"{getattr(request.block_table, 'num_tokens', fetch_num)} tokens "
            f"(KV + {ssm_idx} SSM layers)"
        )
        return full_cache

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
                active_mem, max_mem = get_effective_metal_working_set_bytes(mx)
                guard_threshold = get_metal_ws_guard_threshold()
                if max_mem > 0 and active_mem > 0 and len(self.running) > 0:
                    if active_mem / max_mem * 100.0 >= guard_threshold:
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
            try:
                _attempted_cached_tokens = int(
                    getattr(
                        request,
                        "_cache_selection_attempted_tokens",
                        getattr(request, "cached_tokens", 0),
                    )
                    or 0
                )
            except (TypeError, ValueError):
                _attempted_cached_tokens = 0
            _cache_fallback_reason: Optional[str] = None
            _initial_block_table = getattr(request, "block_table", None)
            _initial_cache_blocks = len(
                getattr(_initial_block_table, "block_ids", []) or []
            )

            # Ensure we have a batch generator
            _cache_cleared = self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            if _cache_cleared and (
                request.prompt_cache is not None
                or getattr(request, "_paged_block_table_needs_worker_reconstruct", False)
            ):
                logger.info(
                    "Request %s: prefix cache hit invalidated by BatchGenerator "
                    "recreation; falling back to full prefill",
                    request.request_id,
                )
                self._release_unusable_paged_hit(request)
                request.prompt_cache = None
                request.block_table = None
                request.cached_tokens = 0
                request.shared_prefix_blocks = 0
                request.remaining_tokens = request.prompt_token_ids
                request._cache_detail = None
                request._paged_block_table_needs_worker_reconstruct = False
                request._hybrid_prompt_cache_needs_worker_ssm = False
                request._prompt_cache_needs_worker_dequant = False
                _cache_fallback_reason = "batch_generator_recreated"

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
            _prompt_tokens = int(
                getattr(request, "num_prompt_tokens", 0)
                or len(getattr(request, "prompt_token_ids", []) or [])
            )
            _accepted_cached_tokens = int(
                getattr(request, "cached_tokens", 0) or 0
            )
            cache_execution: Dict[str, Any] = {
                "request_id": request.request_id,
                "cache_detail": getattr(request, "_cache_detail", None),
                "prompt_tokens": _prompt_tokens,
                "attempted_cached_tokens": _attempted_cached_tokens,
                "cached_tokens": _accepted_cached_tokens,
                "matched_tokens": int(
                    getattr(request, "_cache_matched_tokens", 0) or 0
                ),
                "checkpoint_tokens": int(
                    getattr(request, "_cache_checkpoint_tokens", 0) or 0
                ),
                "replayed_tokens": int(
                    getattr(request, "_cache_replayed_tokens", 0) or 0
                ),
                "uncached_prompt_tokens": max(
                    _prompt_tokens - _accepted_cached_tokens, 0
                ),
                "prefill_tokens": len(tokens_to_process),
                "blocks": _initial_cache_blocks,
                "selection": getattr(request, "_cache_selection", None),
                "candidate_lookup_seconds": round(
                    max(
                        0.0,
                        float(
                            getattr(
                                request,
                                "_cache_candidate_lookup_seconds",
                                0.0,
                            )
                            or 0.0
                        ),
                    ),
                    6,
                ),
                "cache_outcome": (
                    "hit"
                    if _accepted_cached_tokens > 0
                    else "discarded"
                    if _attempted_cached_tokens > 0
                    else "miss"
                ),
                "cache_reuse_applied": bool(
                    cache_to_use is not None and _accepted_cached_tokens > 0
                ),
                "reconstructed": False,
                "dequantized": False,
                "reconstruction_seconds": 0.0,
                "dequantization_seconds": 0.0,
                "total_worker_cache_seconds": 0.0,
            }
            if getattr(request, "_tq_disk_direct_restore", False):
                cache_execution["tq_disk_direct_restore"] = True
            _cache_worker_start = time.perf_counter()

            if getattr(request, "_paged_block_table_needs_worker_reconstruct", False):
                block_table = getattr(request, "block_table", None)
                # A soft cap means this family cannot close its think rail, so
                # the server will immediately re-issue the same prompt as a
                # visible-answer pass over the identical block table. Retain a
                # pristine copy now so that pass skips the replay, which is the
                # dominant cost of the reasoning-to-content stall at depth.
                if self.block_aware_cache is not None:
                    arm_memo = getattr(self.block_aware_cache, "arm_reconstruct_memo", None)
                    if callable(arm_memo):
                        arm_memo(
                            bool(getattr(request, "_dsv4_thinking_soft_cap", None))
                        )
                _t_reconstruct = time.perf_counter()
                cache_to_use = (
                    self.block_aware_cache.reconstruct_cache(block_table)
                    if self.block_aware_cache is not None and block_table is not None
                    else None
                )
                self._dsv4_trace_timing(
                    "reconstruct_cache",
                    _t_reconstruct,
                    request.request_id,
                    cached_tokens=getattr(block_table, "num_tokens", 0) if block_table else 0,
                    blocks=len(getattr(block_table, "block_ids", []) or []) if block_table else 0,
                    ok=cache_to_use is not None,
                    memo_hit=bool(
                        getattr(self.block_aware_cache, "_last_reconstruct_memo_hit", False)
                    ) if self.block_aware_cache is not None else False,
                )
                if cache_execution is not None:
                    cache_execution["reconstruction_seconds"] = round(
                        time.perf_counter() - _t_reconstruct,
                        6,
                    )
                    cache_execution["reconstructed"] = cache_to_use is not None
                    cache_execution["reconstruction_ok"] = cache_to_use is not None
                # In frugal paged mode fetch_cache() can match an L1 chain whose
                # tensor payloads were intentionally released after L2
                # write-through. The actual disk reads then happen here, on the
                # worker, after add_request() took its fetch-time counter sample.
                # Promote that worker-side source fact into the request detail so
                # UI/API usage does not misreport a disk restore as RAM-only.
                _reconstruct_disk_blocks = int(
                    getattr(
                        self.block_aware_cache,
                        "_last_reconstruct_disk_blocks",
                        0,
                    )
                    or 0
                )
                if cache_to_use is not None and _reconstruct_disk_blocks > 0:
                    request._paged_disk_hit = True
                    _worker_detail = str(
                        getattr(request, "_cache_detail", "")
                        or _block_cache_detail(
                            self.paged_cache_manager,
                            disk_hit=True,
                        )
                    )
                    # Disk-only labels ("block-disk", "block-disk+dsv4") already
                    # say every hit came from SSD; decorating them again yields
                    # nonsense like "block-disk+dsv4+disk".
                    if not bool(
                        getattr(self.paged_cache_manager, "disk_only", False)
                    ) and "+disk" not in _worker_detail:
                        _worker_detail += "+disk"
                    request._cache_detail = _worker_detail
                    if cache_execution is not None:
                        cache_execution["cache_detail"] = _worker_detail
                        cache_execution["disk_blocks"] = _reconstruct_disk_blocks
                    logger.info(
                        "Request %s: worker reconstructed %d block(s) from L2",
                        request.request_id,
                        _reconstruct_disk_blocks,
                    )
                request._paged_block_table_needs_worker_reconstruct = False
                if cache_to_use is None:
                    # mlxstudio#73: a failed reconstruct follows a fetch that
                    # owns block refs; release them before falling back.
                    logger.info(
                        f"Request {request.request_id}: worker-side paged cache "
                        "reconstruction failed, treating as cache miss"
                    )
                    # The lookup was optimistic; reconstruction did not save any
                    # prefill work. Release refs and roll back hit telemetry.
                    self._release_unusable_paged_hit(request)
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                    request._hybrid_prompt_cache_needs_worker_ssm = False
                    request._prompt_cache_needs_worker_dequant = False
                    _cache_fallback_reason = "paged_reconstruction_failed"
                else:
                    request.prompt_cache = cache_to_use

                # TQ-native paged/L2 records are decoded to ordinary KVCache
                # objects by BlockAwarePrefixCache so the packed storage codec
                # never leaks into attention.  A model whose native live cache
                # is TurboQuantKVCache still needs the decoded arrays re-wrapped
                # in that class before generation.  The storage-only policy has
                # compress_after=0, so this does not re-encode or change the
                # restored values; it restores the model's proven allocation and
                # update path.  Leaving Laguna/Qwen full-KV hits as plain KVCache
                # made identical 48-token warm prompts decode at 7-8 tok/s after
                # a 24.5 tok/s cold pass even though reconstruction took 0.11s.
                _tq_native_blocks = int(
                    getattr(
                        self.block_aware_cache,
                        "_last_reconstruct_tq_blocks",
                        0,
                    )
                    or 0
                )
                if (
                    cache_to_use is not None
                    and _tq_native_blocks > 0
                    and getattr(self, "_tq_active", False)
                    and not getattr(self, "_is_hybrid", False)
                ):
                    _t_tq_rewrap = time.perf_counter()
                    _tq_rewrapped = False
                    try:
                        from .mllm_batch_generator import _recompress_to_tq

                        _rewrapped_cache = _recompress_to_tq(
                            cache_to_use,
                            self.model,
                        )
                        _tq_rewrapped = any(
                            type(layer).__name__ == "TurboQuantKVCache"
                            for layer in (_rewrapped_cache or [])
                        )
                        if _tq_rewrapped:
                            cache_to_use = _rewrapped_cache
                            request.prompt_cache = cache_to_use
                            request._tq_native_cache_hit = True
                    except Exception as exc:
                        logger.warning(
                            "Request %s: failed to re-wrap %d decoded TQ-native "
                            "paged blocks for live cache-class parity; continuing "
                            "with decoded KVCache: %s",
                            request.request_id,
                            _tq_native_blocks,
                            exc,
                        )
                    if cache_execution is not None:
                        cache_execution["tq_native_blocks"] = _tq_native_blocks
                        cache_execution["tq_rewrapped"] = _tq_rewrapped
                        cache_execution["tq_rewrap_seconds"] = round(
                            time.perf_counter() - _t_tq_rewrap,
                            6,
                        )

            # DSV4 q4/q8 prefix cache restore must happen on the llm-worker
            # that owns the MLX stream. add_request() runs on the API/event
            # loop thread, so only block lookup and lightweight object
            # reconstruction are allowed there.
            if (
                cache_to_use is not None
                and getattr(request, "_prompt_cache_needs_worker_dequant", False)
            ):
                _t_dequant = time.perf_counter()
                cache_to_use = self._dequantize_cache_for_use(cache_to_use)
                if cache_execution is not None:
                    cache_execution["dequantization_seconds"] = round(
                        time.perf_counter() - _t_dequant,
                        6,
                    )
                    cache_execution["dequantized"] = cache_to_use is not None
                    cache_execution["dequantization_ok"] = cache_to_use is not None
                if cache_to_use is None:
                    logger.info(
                        f"Request {request.request_id}: worker-side DSV4 cache "
                        f"dequantization failed, treating as cache miss"
                    )
                    self._release_unusable_paged_hit(request)
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                    _cache_fallback_reason = "cache_dequantization_failed"
                else:
                    request.prompt_cache = cache_to_use
                    request._prompt_cache_needs_worker_dequant = False

            if (
                cache_to_use is not None
                and getattr(request, "_hybrid_prompt_cache_needs_worker_ssm", False)
            ):
                cache_to_use = self._finalize_hybrid_paged_cache_on_worker(
                    request, cache_to_use
                )
                tokens_to_process = (
                    request.remaining_tokens
                    if request.remaining_tokens is not None
                    else request.prompt_token_ids
                )
                if len(tokens_to_process) == 0:
                    tokens_to_process = request.prompt_token_ids[-1:]

            # Prompt-level TQ L2 restores arrive as decoded KVCache objects.
            # Restore the model's native live cache class on the worker without
            # the synchronous API-thread paged backfill that previously delayed
            # SSE/Electron TTFT and lost the TQ storage policy.
            if (
                cache_to_use is not None
                and getattr(request, "_tq_disk_direct_restore", False)
                and getattr(self, "_tq_active", False)
                and not any(
                    type(layer).__name__ == "TurboQuantKVCache"
                    for layer in (cache_to_use or [])
                )
            ):
                _t_tq_rewrap = time.perf_counter()
                _tq_rewrapped = False
                try:
                    from .mllm_batch_generator import _recompress_to_tq

                    rewrapped = _recompress_to_tq(cache_to_use, self.model)
                    _tq_rewrapped = any(
                        type(layer).__name__ == "TurboQuantKVCache"
                        for layer in (rewrapped or [])
                    )
                    if _tq_rewrapped:
                        cache_to_use = rewrapped
                        request.prompt_cache = rewrapped
                except Exception as exc:
                    logger.warning(
                        "Request %s: failed to re-wrap direct TQ disk restore: %s",
                        request.request_id,
                        exc,
                    )
                if cache_execution is not None:
                    cache_execution["tq_rewrapped"] = _tq_rewrapped
                    cache_execution["tq_rewrap_seconds"] = round(
                        time.perf_counter() - _t_tq_rewrap,
                        6,
                    )

            # Validate cache before using it
            if cache_to_use is not None:
                if not self._validate_cache(cache_to_use):
                    logger.warning(
                        f"Request {request.request_id}: invalid cache, "
                        f"proceeding without cache"
                    )
                    self._release_unusable_paged_hit(request)
                    cache_to_use = None
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                    _cache_fallback_reason = "cache_validation_failed"
                else:
                    # Check memory: _merge_caches doubles cache memory temporarily
                    # Skip cache if available memory is tight
                    try:
                        from .memory_cache import estimate_kv_cache_memory

                        cache_bytes = estimate_kv_cache_memory(cache_to_use)
                        import psutil

                        avail = psutil.virtual_memory().available
                        # Memory amplification during merge depends on the cache
                        # object we are about to insert, not just the configured
                        # q4/q8 storage mode. Some fetch paths already dequantize
                        # before reaching this point.
                        multiplier = self._cache_merge_memory_multiplier(cache_to_use)
                        needed = cache_bytes * multiplier
                        budget_fraction = self._cache_reuse_budget_fraction()
                        merge_budget = avail * budget_fraction
                        if needed > merge_budget:
                            (
                                partial_cache,
                                partial_tokens_to_process,
                            ) = self._shrink_paged_cache_for_memory(
                                request=request,
                                cache_to_use=cache_to_use,
                                cache_bytes=cache_bytes,
                                available_bytes=avail,
                                multiplier=multiplier,
                                budget_fraction=budget_fraction,
                            )
                            if partial_cache is not None:
                                cache_to_use = partial_cache
                                tokens_to_process = partial_tokens_to_process or (
                                    request.remaining_tokens
                                    if request.remaining_tokens is not None
                                    else request.prompt_token_ids
                                )
                                if len(tokens_to_process) == 0:
                                    tokens_to_process = request.prompt_token_ids[-1:]
                            else:
                                self._record_cache_reuse_skip(
                                    request=request,
                                    cache_to_use=cache_to_use,
                                    cache_bytes=cache_bytes,
                                    available_bytes=avail,
                                    needed_bytes=needed,
                                    merge_budget_bytes=merge_budget,
                                    multiplier=multiplier,
                                    budget_fraction=budget_fraction,
                                )
                                self._release_unusable_paged_hit(request)
                                cache_to_use = None
                                request.prompt_cache = None
                                request.cached_tokens = 0
                                request.remaining_tokens = request.prompt_token_ids
                                tokens_to_process = request.prompt_token_ids
                                _cache_fallback_reason = "cache_memory_budget_exceeded"
                    except ImportError:
                        pass  # psutil is a required dep but handle gracefully
                    except Exception as e:
                        logger.debug(f"Memory check failed, skipping: {e}")

            # Insert into BatchGenerator with optional cache.
            # Wrapped in try/except to prevent lost requests — if insert fails
            # completely, put the request back in the waiting queue.
            # M3 VL (additive, gated): a cold/partial media request must see the
            # FULL prompt atomically with its vision tensors. A media-salted
            # paged hit may omit those tensors only when the restored prefix is
            # strictly past every image/video placeholder; the cached MSA state
            # then already owns the vision splice and only the text tail is fed.
            _m3vl_pv = getattr(request, "pixel_values", None)
            _m3vl_grid = getattr(request, "image_grid_thw", None)
            _m3vl_pv_video = getattr(request, "pixel_values_videos", None)
            _m3vl_video_grid = getattr(request, "video_grid_thw", None)
            _m3vl_active = (
                (_m3vl_pv is not None or _m3vl_pv_video is not None)
                and self.batch_generator.__class__.__name__ == "SingleBatchGenerator"
            )
            _m3vl_cache_replay = bool(
                _m3vl_active
                and cache_to_use is not None
                and getattr(request, "_cache_extra_keys", None)
                and _m3_vl_cached_prefix_covers_media_tokens(self.model, request)
            )
            if _m3vl_active and cache_to_use is not None and not _m3vl_cache_replay:
                logger.info(
                    "Request %s: M3 VL cache prefix ends before the final media "
                    "placeholder; releasing hit and using atomic full vision prefill",
                    request.request_id,
                )
                self._release_unusable_paged_hit(request)
                cache_to_use = None
                request.prompt_cache = None
                request.cached_tokens = 0
                request.remaining_tokens = request.prompt_token_ids
                _cache_fallback_reason = "media_placeholders_outside_cached_prefix"
            if _m3vl_active and not _m3vl_cache_replay:
                tokens_to_process = list(request.prompt_token_ids)

            try:
                try:
                    insert_kwargs = {}
                    request_sampler = self._request_seeded_sampler(request)
                    if request_sampler is not None:
                        insert_kwargs["samplers"] = [request_sampler]
                    if _m3vl_active:
                        if _m3vl_cache_replay:
                            cached_prefix_len = max(
                                0, int(getattr(request, "cached_tokens", 0) or 0)
                            )
                            insert_kwargs["all_tokens"] = [
                                list(request.prompt_token_ids[:cached_prefix_len])
                            ]
                        else:
                            if _m3vl_pv is not None:
                                insert_kwargs["pixel_values"] = [_m3vl_pv]
                                insert_kwargs["image_grid_thw"] = [_m3vl_grid]
                            if _m3vl_pv_video is not None:
                                insert_kwargs["pixel_values_videos"] = [_m3vl_pv_video]
                                insert_kwargs["video_grid_thw"] = [_m3vl_video_grid]
                        request_processors = self._request_logits_processors(
                            request,
                            list(request.prompt_token_ids)
                            if _m3vl_cache_replay
                            else list(tokens_to_process),
                        )
                        if request_processors is not None:
                            insert_kwargs["logits_processors"] = [request_processors]
                    elif (
                        self.batch_generator.__class__.__name__
                        == "DSV4BatchGenerator"
                    ):
                        # DSV4's custom generator applies repetition penalty
                        # itself during its pinned-stream decode loop. Pass the
                        # full original prompt so prefix-cache hit paths and
                        # normal prefill paths use the same logits-processor
                        # context instead of only the re-fed tail token. The
                        # processor itself must still use generate_step
                        # semantics: final prompt token plus generated output,
                        # not the full user prompt. Exact-copy/code prompts put
                        # target identifiers in the prompt; penalizing that
                        # entire prompt corrupts valid repeats.
                        insert_kwargs["all_tokens"] = [request.prompt_token_ids]
                        # A request-level cache bypass must suppress DSV4's
                        # N-1 snapshot and native block-delta capture as well
                        # as scheduler lookup/store.  The generator otherwise
                        # sees the globally enabled block cache and can attempt
                        # an export for a request that explicitly opted out.
                        insert_kwargs["capture_prompt_snapshots"] = [
                            not bool(
                                getattr(request, "_bypass_prefix_cache", False)
                            )
                        ]
                        insert_kwargs["prompt_snapshot_tail_tokens"] = [
                            max(
                                1,
                                int(
                                    getattr(request, "_gen_prompt_len", 0) or 0
                                )
                                + 1,
                            )
                        ]
                        _soft_cap = self._dsv4_remaining_thinking_soft_cap(request)
                        if _soft_cap is not None:
                            insert_kwargs["thinking_soft_caps"] = [_soft_cap]
                        request_processors = self._request_logits_processors(
                            request, list(request.prompt_token_ids)
                        )
                        if request_processors is not None:
                            insert_kwargs["logits_processors"] = [
                                request_processors
                            ]
                    else:
                        single_batch_cached_prefix: Optional[List[int]] = None
                        if (
                            cache_to_use is not None
                            and self.batch_generator.__class__.__name__
                            == "SingleBatchGenerator"
                        ):
                            cached_prefix_len = max(
                                0, int(getattr(request, "cached_tokens", 0) or 0)
                            )
                            if cached_prefix_len > 0 and request.prompt_token_ids:
                                single_batch_cached_prefix = list(
                                    request.prompt_token_ids[
                                        : min(
                                            cached_prefix_len,
                                            len(request.prompt_token_ids),
                                        )
                                    ]
                                )
                                insert_kwargs["all_tokens"] = [
                                    single_batch_cached_prefix
                                ]
                        processor_context_tokens = (
                            list(request.prompt_token_ids)
                            if single_batch_cached_prefix is not None
                            else list(tokens_to_process)
                        )
                        request_processors = self._request_logits_processors(
                            request, processor_context_tokens
                        )
                        if request_processors is not None:
                            insert_kwargs["logits_processors"] = [
                                request_processors
                            ]
                    # Where a warm turn would re-feed after restoring. Lets
                    # the cold prefill split at the same boundary so the two
                    # arms are numerically equivalent. CAPABILITY-PROBED:
                    # native-MTP text models use the stock mlx_lm
                    # BatchGenerator, whose insert() does not take this kwarg
                    # — passing it unconditionally wedged EVERY request on
                    # such models in a TypeError retry loop (caught live by
                    # the release soak on Nemotron-Omni; the qwen MTP line
                    # serves via the MLLM lane and never hit this). The
                    # split feature simply does not apply to a generator
                    # that owns its own prefill.
                    if self._insert_accepts_gen_prompt_lens():
                        insert_kwargs["gen_prompt_lens"] = [
                            int(getattr(request, "_gen_prompt_len", 0) or 0)
                        ]
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        # Remaining budget, not the full cap: a cache-error
                        # reschedule re-enters here after clearing output_token_ids,
                        # and must not restart a fresh max_tokens (see
                        # Request.remaining_output_budget).
                        max_tokens=[request.remaining_output_budget],
                        caches=[cache_to_use] if cache_to_use else None,
                        **insert_kwargs,
                    )
                except Exception as e:
                    # Cache-related insertion failure - retry without cache
                    if cache_to_use is not None:
                        logger.warning(
                            f"Request {request.request_id}: cache insertion failed "
                            f"({type(e).__name__}: {e}), retrying without cache"
                        )
                        self._release_unusable_paged_hit(request)
                        cache_to_use = None
                        request.prompt_cache = None
                        request.cached_tokens = 0
                        request.remaining_tokens = request.prompt_token_ids
                        tokens_to_process = request.prompt_token_ids
                        _cache_fallback_reason = "cache_insert_failed"
                        insert_kwargs = {}
                        request_sampler = self._request_seeded_sampler(request)
                        if request_sampler is not None:
                            insert_kwargs["samplers"] = [request_sampler]
                        if _m3vl_active:
                            if _m3vl_pv is not None:
                                insert_kwargs["pixel_values"] = [_m3vl_pv]
                                insert_kwargs["image_grid_thw"] = [_m3vl_grid]
                            if _m3vl_pv_video is not None:
                                insert_kwargs["pixel_values_videos"] = [
                                    _m3vl_pv_video
                                ]
                                insert_kwargs["video_grid_thw"] = [
                                    _m3vl_video_grid
                                ]
                            request_processors = self._request_logits_processors(
                                request, list(tokens_to_process)
                            )
                            if request_processors is not None:
                                insert_kwargs["logits_processors"] = [
                                    request_processors
                                ]
                        elif (
                            self.batch_generator.__class__.__name__
                            == "DSV4BatchGenerator"
                        ):
                            insert_kwargs["all_tokens"] = [request.prompt_token_ids]
                            insert_kwargs["capture_prompt_snapshots"] = [
                                not bool(
                                    getattr(
                                        request,
                                        "_bypass_prefix_cache",
                                        False,
                                    )
                                )
                            ]
                            insert_kwargs["prompt_snapshot_tail_tokens"] = [
                                max(
                                    1,
                                    int(
                                        getattr(request, "_gen_prompt_len", 0)
                                        or 0
                                    )
                                    + 1,
                                )
                            ]
                            _soft_cap = self._dsv4_remaining_thinking_soft_cap(
                                request
                            )
                            if _soft_cap is not None:
                                insert_kwargs["thinking_soft_caps"] = [_soft_cap]
                            request_processors = self._request_logits_processors(
                                request, list(request.prompt_token_ids)
                            )
                            if request_processors is not None:
                                insert_kwargs["logits_processors"] = [
                                    request_processors
                                ]
                        else:
                            request_processors = self._request_logits_processors(
                                request, list(tokens_to_process)
                            )
                            if request_processors is not None:
                                insert_kwargs["logits_processors"] = [
                                    request_processors
                                ]
                        uids = self.batch_generator.insert(
                            [tokens_to_process],
                            # Remaining budget, not the full cap (see the
                            # cached-insert site above and
                            # Request.remaining_output_budget).
                            max_tokens=[request.remaining_output_budget],
                            caches=None,
                            **insert_kwargs,
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
                if request.sampling_params.logprobs:
                    register_generation_logprobs(self.model, uid)
                request.status = RequestStatus.RUNNING
                _final_cached_tokens = int(
                    getattr(request, "cached_tokens", 0) or 0
                )
                cache_execution.update(
                    {
                        "cache_detail": getattr(request, "_cache_detail", None),
                        "cached_tokens": _final_cached_tokens,
                        "uncached_prompt_tokens": max(
                            _prompt_tokens - _final_cached_tokens, 0
                        ),
                        # This is the actual tail handed to the generator. An
                        # exact prefix hit still forwards one kickoff token.
                        "prefill_tokens": len(tokens_to_process),
                        "cache_outcome": (
                            "hit"
                            if _final_cached_tokens > 0
                            else "discarded"
                            if _attempted_cached_tokens > 0
                            else "miss"
                        ),
                        "cache_reuse_applied": bool(
                            cache_to_use is not None and _final_cached_tokens > 0
                        ),
                        "total_worker_cache_seconds": (
                            round(
                                max(
                                    0.0,
                                    time.perf_counter() - _cache_worker_start,
                                ),
                                6,
                            )
                            if _attempted_cached_tokens > 0
                            else 0.0
                        ),
                    }
                )
                if _final_cached_tokens == 0 and _attempted_cached_tokens > 0:
                    cache_execution["fallback_reason"] = (
                        _cache_fallback_reason or "cache_candidate_discarded"
                    )
                request._cache_execution = {
                    key: value
                    for key, value in cache_execution.items()
                    if value is not None
                }
                # Publish misses and discarded hits too. Otherwise a cold
                # request leaves an older hit masquerading as "last".
                self._last_cache_execution = dict(request._cache_execution)
                self.running[request.request_id] = request
                scheduled.append(request)

                # H1 parity: Add per-request stop tokens to shared batch generator
                # Track additions so they can be removed on cleanup.
                # Upstream mlx_lm BatchGenerator (continuous batching / native
                # MTP) has NO `stop_tokens` attribute — only our
                # SingleBatchGenerator / DSV4BatchGenerator do. Unguarded, any
                # request carrying stop_token_ids raised AttributeError out of
                # _schedule_waiting into step() (issue #229-#236 review). For
                # generators without the shared set, per-request stops are
                # enforced scheduler-side in the decode loop instead.
                if (
                    request.sampling_params.stop_token_ids
                    and self.batch_generator is not None
                    and hasattr(self.batch_generator, "stop_tokens")
                ):
                    new_tokens = set(request.sampling_params.stop_token_ids)
                    self.batch_generator.stop_tokens.update(new_tokens)
                    request._added_stop_tokens = new_tokens

                self.total_prompt_tokens += request.num_prompt_tokens
                self._record_scheduled_cache_hit(request)
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

    def _terminal_cache_capture_enabled(self, request: Request, response: Any) -> bool:
        """Return whether a finished response has a real cache-store owner.

        Batch generators expose ``prompt_cache`` even when every prefix-cache
        tier is disabled.  Treating that attribute alone as permission to
        capture state made path-dependent families run a second full prompt
        prefill after terminal delivery with nowhere to publish the result.
        Require both the configured prefix-cache contract and an instantiated
        L1/L2 store before doing any snapshot extraction or deferred re-prefill.
        """
        if getattr(request, "_bypass_prefix_cache", False):
            return False
        if not hasattr(response, "prompt_cache"):
            return False
        if not bool(getattr(self.config, "enable_prefix_cache", False)):
            return False
        return any(
            getattr(self, name, None) is not None
            for name in (
                "block_aware_cache",
                "memory_aware_cache",
                "prefix_cache",
                "disk_cache",
            )
        )

    def _record_ttft_sample(self, ttft_seconds: float) -> None:
        """Seed the TTFT EWMA from its first observation, then smooth it.

        Treating the initial ``0.0`` telemetry value as a real historical
        sample reports the first request at only ``alpha`` times its actual
        TTFT (10% with the production alpha). The zero is an empty-state
        sentinel, so the first real sample owns the baseline.
        """
        sample = float(ttft_seconds)
        if self._ttft_sample_count == 0:
            self._ewma_ttft = sample
        else:
            self._ewma_ttft = (
                self._ttft_alpha * sample
                + (1 - self._ttft_alpha) * self._ewma_ttft
            )
        self._ttft_sample_count += 1

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
                    _first_token_time = time.perf_counter()
                    ttft = _first_token_time - request._schedule_time
                    self._record_ttft_sample(ttft)
                    _admission_start = getattr(
                        request,
                        "_cache_admission_start_time",
                        None,
                    )
                    self._record_cache_admission_first_token_sample(
                        request,
                        scheduled_ttft_seconds=ttft,
                        admission_first_token_seconds=(
                            _first_token_time - _admission_start
                            if isinstance(_admission_start, (int, float))
                            else None
                        ),
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
                and not request.sampling_params.logprobs
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
            # Per-request stop_token_ids fallback for batch generators WITHOUT
            # a shared `stop_tokens` set (upstream mlx_lm BatchGenerator under
            # continuous batching / native MTP). Our SingleBatchGenerator /
            # DSV4BatchGenerator stop inside the generator via the shared set;
            # upstream cannot, so the request's stop tokens would otherwise be
            # silently IGNORED (issue #229-#236 review). Checked before the
            # detokenizer add, so the stop token never leaks into new_text —
            # identical surface behavior to the generator-side stop.
            if (
                not is_stop
                and request.sampling_params.stop_token_ids
                and not hasattr(self.batch_generator, "stop_tokens")
                and response.token in request.sampling_params.stop_token_ids
            ):
                is_stop = True
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
            if request.sampling_params.logprobs:
                token_logprobs = _format_token_logprobs_for_output(
                    response.token,
                    getattr(response, "logprobs", None),
                    request.sampling_params.top_logprobs,
                )
                if token_logprobs is not None:
                    request.output_logprobs.append(token_logprobs)

            _detail = getattr(request, "_cache_detail", "")
            if (
                _detail
                and getattr(request, "_tq_native_cache_hit", False)
                and "+tq-native" not in _detail
            ):
                _detail += "+tq-native"
            execution = getattr(request, "_cache_execution", None)
            if isinstance(execution, dict) and _detail:
                execution["cache_detail"] = _detail
                request._cache_execution = dict(execution)
                self._last_cache_execution = dict(request._cache_execution)
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token] + spec_tokens,
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=request.cached_tokens,
                cache_detail=_detail,
                logprobs=(
                    list(request.output_logprobs)
                    if request.sampling_params.logprobs
                    else None
                ),
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
                if self._terminal_cache_capture_enabled(request, response):
                    try:
                        # CLEAN PROMPT-BOUNDARY SNAPSHOT (DSV4 fast path).
                        #
                        # DSV4BatchGenerator captures a deep-copy of the
                        # cache state IMMEDIATELY after prefill — before
                        # decode mutates the live cache. That snapshot is
                        # the *correct* prompt-boundary state for prefix
                        # cache + L2 disk store, free of SWA wrap and
                        # CSA/HCA pool drift. When present, prefer it
                        # over the live `prompt_cache` and skip the
                        # truncation guard entirely.
                        snapshot_cache = getattr(
                            response, "prompt_cache_snapshot", None
                        )
                        # DSV4 EXTENDED STORE: zero-copy delta transport that
                        # continues the prompt chain through decode, ending at
                        # a 256-token boundary. When valid it preempts both
                        # the prompt-boundary snapshot store and the cache-hit
                        # store-skip: the next turn re-renders prior output in
                        # its history (tools keep reasoning), so storing the
                        # exact fed sequence turns that hot re-prefill into a
                        # block-aligned hit.
                        extended_cache = getattr(
                            response, "extended_cache_snapshot", None
                        )
                        extended_cache_tokens = int(
                            getattr(response, "extended_cache_tokens", 0) or 0
                        )

                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache:
                            request._extracted_cache_from_prompt_snapshot = False
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
                                    prompt_len = len(request.prompt_token_ids)
                                    extended_store_armed = False
                                    if (
                                        extended_cache is not None
                                        and self._uses_dsv4_cache
                                    ):
                                        # Store-policy cap (#168, default ON):
                                        # tools-off renders strip prior
                                        # reasoning from replayed history, so
                                        # the reasoning-inclusive extended key
                                        # diverges at the assistant boundary —
                                        # fall through to the prompt-snapshot
                                        # store the next turn CAN match.
                                        from .utils.dsv4_batch_generator import (
                                            dsv4_extended_store_capped,
                                        )

                                        if dsv4_extended_store_capped(
                                            bool(
                                                getattr(
                                                    request,
                                                    "_vmlx_tools_present",
                                                    False,
                                                )
                                            )
                                        ):
                                            logger.info(
                                                "DSV4 extended store capped at "
                                                "prompt boundary (reasoning-"
                                                "stripping render, #168 store "
                                                "policy); using prompt-snapshot "
                                                "store."
                                            )
                                            extended_cache = None
                                            extended_cache_tokens = 0
                                    if (
                                        extended_cache is not None
                                        and extended_cache_tokens > 0
                                        and self._uses_dsv4_cache
                                    ):
                                        full_fed_sequence = list(
                                            request.prompt_token_ids
                                        ) + list(request.output_token_ids)
                                        try:
                                            _ext_ivals = (
                                                extended_cache[0].get(
                                                    "dsv4_record_intervals"
                                                )
                                                or ()
                                            )
                                        except Exception:
                                            _ext_ivals = ()
                                        _ext_chain_end = (
                                            int(_ext_ivals[-1][1])
                                            if _ext_ivals
                                            else -1
                                        )
                                        if (
                                            extended_cache_tokens
                                            <= len(full_fed_sequence)
                                            and _ext_chain_end
                                            == extended_cache_tokens
                                        ):
                                            request._extracted_cache_key_tokens = (
                                                full_fed_sequence[
                                                    :extended_cache_tokens
                                                ]
                                            )
                                            cache_for_extract = extended_cache
                                            extended_store_armed = True
                                            request._dsv4_extended_store_used = True
                                            self._dsv4_last_extended_key_tokens = list(
                                                request._extracted_cache_key_tokens
                                            )
                                            logger.info(
                                                "DSV4 prefix cache store using "
                                                "extended prefill+decode delta "
                                                "chain (%d cache-key tokens: "
                                                "prompt=%d, generated_covered=%d).",
                                                extended_cache_tokens,
                                                prompt_len,
                                                max(
                                                    0,
                                                    extended_cache_tokens
                                                    - prompt_len,
                                                ),
                                            )
                                        else:
                                            logger.warning(
                                                "DSV4 extended store rejected: "
                                                "boundary=%d fed_sequence=%d "
                                                "chain_end=%d; falling back to "
                                                "prompt snapshot store.",
                                                extended_cache_tokens,
                                                len(full_fed_sequence),
                                                _ext_chain_end,
                                            )
                                    if extended_store_armed:
                                        pass  # cache_for_extract set above
                                    elif (
                                        snapshot_cache is not None
                                        and not (
                                            getattr(
                                                self, "_uses_m3_msa_cache", False
                                            )
                                            and int(
                                                getattr(
                                                    request, "cached_tokens", 0
                                                )
                                                or 0
                                            )
                                            > 0
                                        )
                                        and not self._dsv4_snapshot_store_below_threshold(
                                            request
                                        )
                                    ):
                                        # Snapshot was captured at the
                                        # prompt boundary — use it
                                        # DIRECTLY. No truncation, no
                                        # rewind, no guard.
                                        snapshot_family = (
                                            "ZAYA"
                                            if self._uses_zaya_cache
                                            else "DSV4"
                                            if self._uses_dsv4_cache
                                            else "Path-dependent"
                                        )
                                        logger.info(
                                            f"{snapshot_family} prefix cache store using "
                                            f"clean prompt-boundary snapshot "
                                            f"({len(snapshot_cache)} layers, "
                                            f"prompt_len={prompt_len}). No "
                                            f"truncation needed."
                                        )
                                        cache_for_extract = snapshot_cache
                                    elif self._uses_dsv4_cache:
                                        # DSV4 cache-hit kickoff responses can
                                        # arrive without a generator-captured
                                        # prompt snapshot. The live cache has
                                        # then processed the remaining prompt
                                        # tail plus generated tokens, so
                                        # trimming it would rewind SWA without
                                        # proving CSA/HCA pool correctness.
                                        #
                                        # If this request already came from a
                                        # paged-prefix hit, do NOT synchronously
                                        # re-prefill the whole expanded prompt
                                        # just to donate a longer prefix. A real
                                        # live row showed a 7,719-token hit with
                                        # only 38 remaining tokens still spent
                                        # ~70s in this post-hit re-prefill
                                        # before returning. The existing N-1
                                        # terminal DeepseekV4Cache record is the
                                        # correctness-safe cache point; later
                                        # requests can reuse it and re-feed the
                                        # cheap tail until an async extension
                                        # store exists.
                                        if (
                                            int(
                                                getattr(
                                                    request, "cached_tokens", 0
                                                )
                                                or 0
                                            )
                                            > 0
                                        ):
                                            logger.info(
                                                "DSV4 prefix cache store skipped "
                                                "for cache-hit request %s "
                                                "(cached_tokens=%s, prompt_len=%s): "
                                                "avoiding synchronous full "
                                                "prompt-boundary re-prefill.",
                                                request.request_id,
                                                getattr(request, "cached_tokens", 0),
                                                len(request.prompt_token_ids),
                                            )
                                            cache_for_extract = None
                                            request._dsv4_cache_hit_store_skipped = True
                                        else:
                                            # No snapshot and no prefix hit:
                                            # re-derive the exact N-1 cache-key
                                            # state instead of trimming the live
                                            # post-generation cache.
                                            dsv4_prompt_tokens = list(
                                                request.prompt_token_ids
                                            )
                                            _gpl_d = (
                                                getattr(request, "_gen_prompt_len", 0)
                                                or 0
                                            )
                                            if 0 < _gpl_d < len(dsv4_prompt_tokens):
                                                dsv4_prompt_tokens = dsv4_prompt_tokens[
                                                    :-_gpl_d
                                                ]
                                            dsv4_key_tokens = (
                                                dsv4_prompt_tokens[:-1]
                                                if len(dsv4_prompt_tokens) > 1
                                                else []
                                            )
                                            try:
                                                from .utils.dsv4_batch_generator import (
                                                    dsv4_prompt_snapshot_min_tokens as _resolve_dsv4_prompt_snapshot_min_tokens,
                                                )

                                                dsv4_prompt_snapshot_min_tokens = (
                                                    _resolve_dsv4_prompt_snapshot_min_tokens()
                                                )
                                            except Exception:
                                                dsv4_prompt_snapshot_min_tokens = 256
                                            if (
                                                len(dsv4_key_tokens)
                                                < dsv4_prompt_snapshot_min_tokens
                                            ):
                                                logger.info(
                                                    "DSV4 prefix cache store skipped "
                                                    "for request %s: prompt below snapshot/store threshold "
                                                    "(cache_key_tokens=%d < %d).",
                                                    request.request_id,
                                                    len(dsv4_key_tokens),
                                                    dsv4_prompt_snapshot_min_tokens,
                                                )
                                                cache_for_extract = None
                                                request._dsv4_short_prompt_store_skipped = True
                                            elif dsv4_key_tokens:
                                                logger.info(
                                                    "DSV4 prefix cache store using "
                                                    "deferred clean prompt-boundary re-prefill "
                                                    "(%d cache-key tokens from %d "
                                                    "prompt tokens) after terminal dispatch.",
                                                    len(dsv4_key_tokens),
                                                    len(dsv4_prompt_tokens),
                                                )
                                                request._deferred_prompt_cache = {
                                                    "family": "DSV4",
                                                    "mode": "paged",
                                                    "key_tokens": list(
                                                        dsv4_key_tokens
                                                    ),
                                                }
                                                cache_for_extract = None
                                            else:
                                                cache_for_extract = None
                                    elif self._uses_zaya_cache:
                                        # ZAYA CCA cache is path-dependent:
                                        # CacheList(KVCache, ArraysCache)
                                        # stores standard KV plus CCA
                                        # conv_state/prev_hs. The live cache
                                        # returned after decode has already
                                        # absorbed generated tokens in the CCA
                                        # state and cannot be rewound safely.
                                        #
                                        # Store the same N-1 key used by the
                                        # paged-prefix path by running a clean
                                        # prompt-only prefill for exactly those
                                        # key tokens. This is slower on cold
                                        # store but preserves exact typed CCA
                                        # semantics and enables correct paged/L2
                                        # reuse on the next identical prefix.
                                        zaya_prompt_tokens = list(
                                            request.prompt_token_ids
                                        )
                                        _gpl_z = (
                                            getattr(request, "_gen_prompt_len", 0)
                                            or 0
                                        )
                                        if 0 < _gpl_z < len(zaya_prompt_tokens):
                                            zaya_prompt_tokens = zaya_prompt_tokens[
                                                :-_gpl_z
                                            ]
                                        zaya_key_tokens = (
                                            zaya_prompt_tokens[:-1]
                                            if len(zaya_prompt_tokens) > 1
                                            else []
                                        )
                                        if zaya_key_tokens:
                                            logger.info(
                                                "ZAYA prefix cache store using "
                                                "deferred clean prompt-boundary re-prefill "
                                                "(%d cache-key tokens from %d "
                                                "prompt tokens) after terminal dispatch.",
                                                len(zaya_key_tokens),
                                                len(zaya_prompt_tokens),
                                            )
                                            request._deferred_prompt_cache = {
                                                "family": "ZAYA",
                                                "mode": "paged",
                                                "key_tokens": list(zaya_key_tokens),
                                            }
                                            cache_for_extract = None
                                        else:
                                            cache_for_extract = None
                                    elif getattr(
                                        self, "_mixed_attention_cache_model", False
                                    ):
                                        # Mixed full/sliding-window attention
                                        # models (Step3p7, Gemma4, MiMo, etc.)
                                        # use RotatingKVCache for SWA layers.
                                        # Once the live cache has advanced
                                        # through decode, prompt-boundary rewind
                                        # is unsafe when the rotating window has
                                        # wrapped. Use the same N-1 clean
                                        # prompt-boundary prefill contract as
                                        # other path-dependent cache families
                                        # instead of trimming live post-decode
                                        # state.
                                        mixed_prompt_tokens = list(
                                            request.prompt_token_ids
                                        )
                                        _gpl_mixed = (
                                            getattr(request, "_gen_prompt_len", 0)
                                            or 0
                                        )
                                        if (
                                            0 < _gpl_mixed < len(mixed_prompt_tokens)
                                            and not getattr(
                                                self,
                                                "_mixed_attention_cache_model",
                                                False,
                                            )
                                        ):
                                            mixed_prompt_tokens = mixed_prompt_tokens[
                                                :-_gpl_mixed
                                            ]
                                        mixed_key_tokens = (
                                            mixed_prompt_tokens[:-1]
                                            if len(mixed_prompt_tokens) > 1
                                            else []
                                        )
                                        if mixed_key_tokens:
                                            logger.info(
                                                "Mixed-SWA prefix cache store using "
                                                "deferred clean prompt-boundary re-prefill "
                                                "(%d cache-key tokens from %d "
                                                "prompt tokens) after terminal dispatch.",
                                                len(mixed_key_tokens),
                                                len(mixed_prompt_tokens),
                                            )
                                            request._deferred_prompt_cache = {
                                                "family": "Mixed-SWA",
                                                "mode": "paged",
                                                "key_tokens": list(mixed_key_tokens),
                                            }
                                            cache_for_extract = None
                                        else:
                                            cache_for_extract = None
                                    elif getattr(
                                        self, "_uses_m3_msa_cache", False
                                    ):
                                        # A SingleBatchGenerator snapshot captured
                                        # after replaying an M3 paged hit is at the
                                        # prompt boundary, but it was extended from
                                        # reconstructed sparse MSA state.  Live
                                        # deterministic tool loops showed that
                                        # persisting this hit-derived extension can
                                        # make the next post-tool hit repeat the tool
                                        # call.  The object-cache path already
                                        # rederives M3 hit extensions; paged/L2 must
                                        # use the same clean full-prefill contract.
                                        m3_prompt_tokens = list(
                                            request.prompt_token_ids
                                        )
                                        _gpl_m3 = (
                                            getattr(request, "_gen_prompt_len", 0)
                                            or 0
                                        )
                                        if 0 < _gpl_m3 < len(m3_prompt_tokens):
                                            m3_prompt_tokens = m3_prompt_tokens[
                                                :-_gpl_m3
                                            ]
                                        m3_key_tokens = (
                                            m3_prompt_tokens[:-1]
                                            if len(m3_prompt_tokens) > 1
                                            else []
                                        )
                                        if (
                                            int(
                                                getattr(
                                                    request, "cached_tokens", 0
                                                )
                                                or 0
                                            )
                                            > 0
                                            and m3_key_tokens
                                        ):
                                            logger.info(
                                                "MiniMax-M3 paged prefix store using "
                                                "deferred clean prompt-boundary re-prefill "
                                                "(%d cache-key tokens from %d prompt "
                                                "tokens) after terminal dispatch.",
                                                len(m3_key_tokens),
                                                len(m3_prompt_tokens),
                                            )
                                            request._deferred_prompt_cache = {
                                                "family": "MiniMax-M3",
                                                "mode": "paged",
                                                "key_tokens": list(m3_key_tokens),
                                            }
                                            cache_for_extract = None
                                        else:
                                            # Cold M3 requests may safely use the
                                            # generator's clean prompt snapshot.
                                            cache_for_extract = (
                                                snapshot_cache
                                                if snapshot_cache is not None
                                                else self._truncate_cache_to_prompt_length(
                                                    raw_cache, prompt_len
                                                )
                                            )
                                    else:
                                        # Paged cache: truncate to N-1 tokens so the
                                        # last prompt token can be re-fed on cache hit.
                                        # Without this, the last token's KV would be
                                        # duplicated with wrong positional encoding.
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
                                        _t_extract = time.perf_counter()
                                        extracted_cache = self._extract_cache_states(
                                            cache_for_extract
                                        )
                                        self._dsv4_trace_timing(
                                            "extract_cache_states",
                                            _t_extract,
                                            request_id,
                                            layers=len(extracted_cache or []),
                                            snapshot=bool(snapshot_cache is not None),
                                        )
                                        # L2 disk: TQ recompress a COPY for 26x
                                        # smaller disk files. The original cache
                                        # objects are unchanged (extract already ran).
                                        if (
                                            self.disk_cache is not None
                                            and not self._is_hybrid
                                            # Extended-store transports end past
                                            # the prompt boundary; the legacy
                                            # disk key is prompt-stripped and
                                            # would mismatch the chain state.
                                            and not extended_store_armed
                                        ):
                                            try:
                                                from .mllm_batch_generator import (
                                                    _recompress_to_tq,
                                                )

                                                tq_for_disk = _recompress_to_tq(
                                                    cache_for_extract, self.model
                                                )
                                                _disk_store_tokens = (
                                                    prefix_cache_key_tokens(request)
                                                )
                                                _call_with_optional_cache_extra(
                                                    self.disk_cache.store,
                                                    _disk_store_tokens,
                                                    tq_for_disk,
                                                    cache_type=self._pick_cache_type_for_request(
                                                        request
                                                    ),
                                                    cache_extra_keys=getattr(
                                                        request,
                                                        "_cache_extra_keys",
                                                        None,
                                                    ),
                                                )
                                            except Exception as de:
                                                logger.debug(
                                                    f"Disk cache store failed for "
                                                    f"{request_id}: {de}"
                                                )
                                        if extracted_cache:
                                            from .utils.turboquant_config import (
                                                turboquant_cache_telemetry,
                                            )

                                            self._last_turboquant_cache = (
                                                turboquant_cache_telemetry(extracted_cache)
                                            )
                                            if self._last_turboquant_cache.get(
                                                "object_layers", 0
                                            ):
                                                logger.info(
                                                    "TurboQuant live telemetry: %s",
                                                    self._last_turboquant_cache,
                                                )
                                            request._extracted_cache = extracted_cache
                                            request._extracted_cache_from_prompt_snapshot = bool(
                                                snapshot_cache is not None
                                                or getattr(
                                                    request,
                                                    "_extracted_cache_from_prompt_snapshot",
                                                    False,
                                                )
                                            )
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
                                    elif getattr(request, "_deferred_prompt_cache", None):
                                        # The terminal output owns this scheduler
                                        # step. Cleanup will materialize and store
                                        # the clean typed cache after dispatch.
                                        pass
                                    elif getattr(
                                        request,
                                        "_dsv4_cache_hit_store_skipped",
                                        False,
                                    ) or getattr(
                                        request,
                                        "_dsv4_short_prompt_store_skipped",
                                        False,
                                    ):
                                        pass
                                    else:
                                        logger.warning(
                                            f"Cannot produce prompt-only cache for "
                                            f"{request_id}, skipping paged cache store"
                                        )
                            else:
                                if getattr(
                                    self, "_mixed_attention_cache_model", False
                                ):
                                    mixed_prompt_tokens = list(
                                        request.prompt_token_ids
                                    )
                                    _gpl_mixed = (
                                        getattr(request, "_gen_prompt_len", 0)
                                        or 0
                                    )
                                    if (
                                        0 < _gpl_mixed < len(mixed_prompt_tokens)
                                        and not getattr(
                                            self,
                                            "_mixed_attention_cache_model",
                                            False,
                                        )
                                    ):
                                        mixed_prompt_tokens = mixed_prompt_tokens[
                                            :-_gpl_mixed
                                        ]
                                    mixed_key_tokens = (
                                        mixed_prompt_tokens[:-1]
                                        if len(mixed_prompt_tokens) > 1
                                        else []
                                    )
                                    if mixed_key_tokens:
                                        logger.info(
                                            "Mixed-SWA prefix cache store using "
                                            "deferred clean prompt-boundary re-prefill "
                                            "(%d cache-key tokens from %d prompt "
                                            "tokens, object cache) after terminal dispatch.",
                                            len(mixed_key_tokens),
                                            len(mixed_prompt_tokens),
                                        )
                                        request._deferred_prompt_cache = {
                                            "family": "Mixed-SWA",
                                            "mode": "object",
                                            "key_tokens": list(mixed_key_tokens),
                                        }
                                        request._extracted_cache = None
                                    else:
                                        request._extracted_cache = None
                                elif (
                                    getattr(self, "_uses_openpangu_cache", False)
                                    or getattr(self, "_uses_glm5_next_cache", False)
                                ):
                                    # SingleBatchGenerator captured this exact
                                    # N-1 typed boundary before consuming the
                                    # final prompt token. Never fall back to the
                                    # live post-decode path-dependent state.
                                    _typed_family = (
                                        "glm5_next"
                                        if getattr(
                                            self, "_uses_glm5_next_cache", False
                                        )
                                        else "openPangu"
                                    )
                                    if snapshot_cache is not None:
                                        request._extracted_cache = snapshot_cache
                                        request._extracted_cache_key_tokens = list(
                                            request.prompt_token_ids[:-1]
                                        )
                                        request._extracted_cache_from_prompt_snapshot = True
                                        logger.info(
                                            "%s prefix store using exact typed "
                                            "N-1 prompt snapshot (%d layers, %d key tokens)",
                                            _typed_family,
                                            len(snapshot_cache),
                                            len(request._extracted_cache_key_tokens),
                                        )
                                    elif getattr(
                                        request,
                                        "_glm_prompt_boundary_disk_store",
                                        False,
                                    ):
                                        request._extracted_cache = None
                                        logger.info(
                                            "glm5_next exact typed N-1 boundary "
                                            "was already persisted to prompt SSD "
                                            "before the final token (%d layers)",
                                            int(
                                                getattr(
                                                    request,
                                                    "_glm_prompt_boundary_cache_layers",
                                                    0,
                                                )
                                                or 0
                                            ),
                                        )
                                    else:
                                        request._extracted_cache = None
                                        logger.warning(
                                            "%s request produced no exact N-1 typed "
                                            "snapshot; skipping cache store",
                                            _typed_family,
                                        )
                                elif getattr(self, "_uses_m3_msa_cache", False):
                                    # MiniMax-M3 MSA is path-dependent: a cache
                                    # hit restores K/V/idx_keys, then tail replay
                                    # extends all three. Persisting that
                                    # hit-derived extension as a new longer
                                    # prefix drifted from a clean full-prefill
                                    # state in live multi-turn tests: the next
                                    # exact hit answered an earlier turn. Rebuild
                                    # the N-1 store payload from a clean
                                    # prompt-only prefill instead.
                                    m3_prompt_tokens = list(
                                        request.prompt_token_ids
                                    )
                                    _gpl_m3 = (
                                        getattr(request, "_gen_prompt_len", 0)
                                        or 0
                                    )
                                    if 0 < _gpl_m3 < len(m3_prompt_tokens):
                                        m3_prompt_tokens = m3_prompt_tokens[
                                            :-_gpl_m3
                                        ]
                                    m3_key_tokens = (
                                        m3_prompt_tokens[:-1]
                                        if len(m3_prompt_tokens) > 1
                                        else []
                                    )
                                    if (
                                        int(
                                            getattr(request, "cached_tokens", 0)
                                            or 0
                                        )
                                        > 0
                                        and m3_key_tokens
                                    ):
                                        logger.info(
                                            "MiniMax-M3 prefix cache store using "
                                            "deferred clean prompt-boundary re-prefill "
                                            "(%d cache-key tokens from %d "
                                            "prompt tokens) after terminal dispatch.",
                                            len(m3_key_tokens),
                                            len(m3_prompt_tokens),
                                        )
                                        request._deferred_prompt_cache = {
                                            "family": "MiniMax-M3",
                                            "mode": "object",
                                            "key_tokens": list(m3_key_tokens),
                                        }
                                        request._extracted_cache = None
                                    else:
                                        request._extracted_cache = (
                                            snapshot_cache
                                            if snapshot_cache is not None
                                            else raw_cache
                                        )
                                else:
                                    # Standard object-cache path. Some native
                                    # cache families (M3 MSA, DSV4) provide a
                                    # clean prompt-boundary snapshot because the
                                    # live cache has already advanced through
                                    # decode/lookahead by the time we finish.
                                    # Prefer that snapshot here too; paged-cache
                                    # already does this above.
                                    request._extracted_cache = (
                                        snapshot_cache
                                        if snapshot_cache is not None
                                        else raw_cache
                                    )
                        else:
                            logger.info(
                                f"No cache returned from BatchGenerator for {request_id}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to extract cache for {request_id}: {e}")

                # Tools-off renders strip prior reasoning, so the chain
                # stored above cannot match the next turn's rendered
                # history. Queue an idle-time shadow re-key of the
                # predicted visible transcript (no-op unless eligible).
                self._queue_dsv4_shadow_rekey(request_id, request, finish_reason)

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

    def _mixed_swa_deferred_seed(self, request: Request, key_tokens: List[int]):
        """Refault the accepted immutable prefix, never the live decode cache.

        Cleanup calls this on the model worker before releasing request block
        references. No new prefix lookup, cross-request table, or resident KV
        mirror is introduced. Unsupported layouts keep the clean-from-zero path.
        """
        if not getattr(self, "_mixed_attention_cache_model", False):
            return None, 0
        # Weight quantization is independent of cache quantization. Repeated
        # lossy cache round-trips need their own drift qualification.
        if getattr(self, "_kv_cache_bits", 0) or getattr(self, "_tq_active", False):
            return None, 0
        if os.environ.get("VMLX_DISABLE_MIXED_SWA_CLEAN_RESUME", "").lower() in (
            "1", "true", "yes", "on",
        ):
            return None, 0
        try:
            from copy import deepcopy
            from mlx_lm.models.cache import KVCache, RotatingKVCache

            owner = getattr(self, "block_aware_cache", None)
            table = getattr(request, "block_table", None)
            entry = getattr(owner, "_request_tables", {}).get(request.request_id)
            if table is None or getattr(entry, "block_table", None) is not table:
                return None, 0
            boundary = int(table.num_tokens)
            if not (0 < boundary <= len(key_tokens)) or boundary != request.cached_tokens:
                return None, 0
            seed = owner.reconstruct_cache(table)
            if seed is None:
                return None, 0
            if len(seed) != len(self.model.layers) or any(
                type(layer) not in (KVCache, RotatingKVCache)
                or int(layer.offset) != boundary
                for layer in seed
            ):
                return None, 0
            return deepcopy(seed), boundary
        except Exception as exc:
            logger.warning(
                "Mixed-SWA clean-prefix seed unavailable for %s; using full prefill: %s",
                request.request_id, exc,
            )
            return None, 0

    def _materialize_deferred_prompt_cache(
        self, request_id: str, request: Request
    ) -> None:
        """Build a path-dependent clean prefix only after terminal dispatch.

        M3 sparse MSA, ZAYA CCA, DSV4 composite state, and mixed full/SWA cache
        layouts cannot safely donate post-decode state.  Their clean N-1
        re-prefill used to run inside ``_process_batch_responses`` and therefore
        delayed terminal SSE/UI delivery by the full prefill duration.
        EngineCore dispatches terminal outputs before ``_cleanup_finished`` and
        blocks next-turn admission until cleanup completes, making this the safe
        ownership boundary for the shared work.
        """
        deferred = getattr(request, "_deferred_prompt_cache", None)
        if not isinstance(deferred, dict):
            return

        # Consume the marker first so an exception cannot accidentally retry the
        # expensive re-prefill during a later cleanup path.
        request._deferred_prompt_cache = None
        family = str(deferred.get("family") or "path-dependent")
        key_tokens = list(deferred.get("key_tokens") or [])
        mode = deferred.get("mode")
        if not key_tokens or mode not in {"paged", "object"}:
            logger.warning(
                "Ignoring invalid deferred %s cache descriptor for %s",
                family,
                request_id,
            )
            return

        try:
            logger.info(
                "%s %s prefix cleanup now running deferred clean "
                "prompt-boundary re-prefill (%d cache-key tokens) after "
                "terminal dispatch.",
                family,
                mode,
                len(key_tokens),
            )
            seed, seed_len = (
                self._mixed_swa_deferred_seed(request, key_tokens)
                if family == "Mixed-SWA" and mode == "paged"
                else (None, 0)
            )
            if seed is not None:
                logger.info(
                    "Mixed-SWA clean-prefix resume for %s: checkpoint=%d target=%d suffix=%d",
                    request_id, seed_len, len(key_tokens), len(key_tokens) - seed_len,
                )
                clean_cache = self._prefill_for_prompt_only_cache(
                    key_tokens, base_cache=seed, base_token_count=seed_len,
                )
                if clean_cache is None:
                    seed = None
                    logger.warning(
                        "Mixed-SWA clean-prefix resume failed for %s; retrying full prefill",
                        request_id,
                    )
                    clean_cache = self._prefill_for_prompt_only_cache(key_tokens)
            else:
                clean_cache = self._prefill_for_prompt_only_cache(key_tokens)
            if clean_cache is None:
                logger.warning(
                    "Cannot produce deferred %s prompt-only %s cache for %s; "
                    "skipping prefix store",
                    family,
                    mode,
                    request_id,
                )
                request._extracted_cache = None
                return

            request._extracted_cache_key_tokens = list(key_tokens)
            request._extracted_cache_from_prompt_snapshot = True
            if mode == "object":
                request._extracted_cache = clean_cache
                return

            if getattr(self, "_kv_cache_bits", 0):
                clean_cache = self._quantize_cache_for_storage(clean_cache)
            extract_started = time.perf_counter()
            extracted_cache = self._extract_cache_states(clean_cache)
            if getattr(self, "_uses_dsv4_cache", False):
                self._dsv4_trace_timing(
                    "extract_cache_states",
                    extract_started,
                    request_id,
                    layers=len(extracted_cache or []),
                    deferred_family=family,
                )
            if not extracted_cache:
                logger.warning(
                    "Deferred %s cache extraction returned empty for %s",
                    family,
                    request_id,
                )
                request._extracted_cache = None
                return

            # Preserve the legacy prompt-level disk manager mirror when it is
            # configured.  BlockDiskStore persistence still happens below via
            # BlockAwarePrefixCache.store_cache().
            if (
                self.disk_cache is not None
                and not self._is_hybrid
            ):
                try:
                    from .mllm_batch_generator import _recompress_to_tq

                    tq_for_disk = _recompress_to_tq(clean_cache, self.model)
                    disk_store_tokens = prefix_cache_key_tokens(request)
                    _call_with_optional_cache_extra(
                        self.disk_cache.store,
                        disk_store_tokens,
                        tq_for_disk,
                        cache_type=self._pick_cache_type_for_request(request),
                        cache_extra_keys=getattr(
                            request,
                            "_cache_extra_keys",
                            None,
                        ),
                    )
                except Exception as exc:
                    logger.debug(
                        "Deferred %s disk cache store failed for %s: %s",
                        family,
                        request_id,
                        exc,
                    )

            from .utils.turboquant_config import turboquant_cache_telemetry

            self._last_turboquant_cache = turboquant_cache_telemetry(extracted_cache)
            request._extracted_cache = extracted_cache
            logger.info(
                "Extracted %d deferred %s layer states for request %s",
                len(extracted_cache),
                family,
                request_id,
            )
        except Exception as exc:
            request._extracted_cache = None
            logger.warning(
                "Deferred %s prompt cache materialization failed for %s: %s",
                family,
                request_id,
                exc,
                exc_info=True,
            )

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        # This runs on the worker thread and gates the next admission (the
        # engine only clears _terminal_cleanup_complete once it finishes), so
        # its total is on the critical path between two back-to-back requests
        # — e.g. the DSV4 answer-pass flip. Timed so that path can be
        # attributed instead of guessed at.
        _cleanup_start = time.perf_counter()
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

            # Cacheability is decided by the prompt prefix, not by output
            # length. Short-output requests still benefit from caching their
            # prompt KV for the next turn that shares the same prefix —
            # multi-turn chat with brief replies is the canonical case.
            # Benchmarks that legitimately want no-store set
            # ``_bypass_prefix_cache`` explicitly via cache_salt /
            # skip_prefix_cache below.
            _output_len = getattr(request, "num_output_tokens", 0) if request else 0
            _skip_cache_store = False
            # Hard cache bypass from the API request (cache_salt / skip_prefix_cache).
            # This overrides all heuristics — the benchmark client asked for
            # guaranteed fresh execution, so nothing gets stored either.
            if request is not None and getattr(request, "_bypass_prefix_cache", False):
                _skip_cache_store = True
            if (
                request is not None
                and self._uses_dsv4_cache
                and request.status == RequestStatus.FINISHED_LENGTH_CAPPED
                and not getattr(request, "_extracted_cache_from_prompt_snapshot", False)
            ):
                # DSV4 DeepseekV4Cache includes CSA/HCA pool state in addition
                # to local SWA KV. After a length-capped decode, the live cache
                # has advanced through generated tokens. Trimming positional KV
                # back to the prompt boundary is not enough to prove the
                # compressor/indexer pool state is also at the prompt boundary,
                # and live exact-repeat tests showed those stores can produce
                # immediate stop on cache hit. Do not let capped generations
                # donate prefix blocks. Clean prompt-boundary snapshots captured
                # before decode are exempt because they are already at N-1 and do
                # not include generated-token SWA/CSA/HCA drift.
                _skip_cache_store = True
            # No DSV4-specific short-prompt skip. Other families store paged
            # cache at any prompt length and rely on the standard LRU
            # eviction (max_blocks budget) to bound memory. The earlier
            # 512-token threshold was based on a gigabyte/entry estimate
            # that turned out to be wrong — DSV4 composite state per
            # prompt is tens of MB, not GB, with the v6 nested-state
            # schema. Treat DSV4 like every other model.
            if _skip_cache_store and request is not None:
                logger.debug(
                    f"Skipping cache store for {request_id}: "
                    f"output_len={_output_len}, status={request.status.name}"
                )
                _PERSIST.record(request_id, "skipped", f"output_len={_output_len} status={request.status.name}")
                if hasattr(request, "_extracted_cache"):
                    request._extracted_cache = None

            if request is not None:
                if _skip_cache_store:
                    request._deferred_prompt_cache = None
                else:
                    self._materialize_deferred_prompt_cache(request_id, request)

            # Native DSV4 extension snapshots contain only immutable deltas
            # beginning at the restored periodic checkpoint. Preserve the
            # fetched table until store_cache() appends those deltas; detaching
            # it here would make the store start at token zero and reject the
            # first non-zero interval. Every other path retains the normal
            # pre-store release, and the post-store finally block releases the
            # preserved table as well.
            _extracted_for_store = (
                getattr(request, "_extracted_cache", None)
                if request is not None
                else None
            )
            _preserve_dsv4_extension_table = bool(
                self._uses_dsv4_cache
                and not _skip_cache_store
                and getattr(request, "block_table", None) is not None
                and isinstance(_extracted_for_store, list)
                and _extracted_for_store
                and all(
                    isinstance(layer, dict)
                    and bool(layer.get("dsv4_block_records"))
                    for layer in _extracted_for_store
                )
            )

            # Always clean up paged cache tracking entries unless a DSV4
            # delta extension still owns the table for the imminent append.
            if self.block_aware_cache is not None:
                if not _preserve_dsv4_extension_table:
                    _entry = self.block_aware_cache._request_tables.pop(
                        request_id, None
                    )
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
            # For thinking models (gen_prompt_len > 0), the extracted SSM
            # state includes gen_prompt + output tokens, placing it at
            # P+gpl+output instead of P (the KV block boundary). Injecting
            # that contaminated state causes garbled output, so the branch
            # below queues a clean prompt-only re-derive instead of storing
            # post-generation state directly.
            if (
                self._is_hybrid
                and not self._uses_dsv4_cache
                and not self._uses_zaya_cache
                and self.config.enable_prefix_cache
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
                    companion_tokens = (
                        all_tokens[:-1] if len(all_tokens) > 1 else list(all_tokens)
                    )
                    companion_len = len(companion_tokens)

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
                        #
                        # 2026-04-30 release-gate audit caught a real RAM
                        # leak via this path: 30 short unique prompts in
                        # burst grew Nemotron-Omni hybrid RSS by +2964 MB
                        # because each enqueue triggered a re-derive that
                        # stored a fresh ~10 MB SSM companion entry in
                        # `_ssm_state_cache` (LRU cap 50 → ~500 MB worst
                        # case) PLUS held onto the original token list +
                        # request_id in the queue PLUS the in-flight
                        # `clean_cache` from `_prefill_for_prompt_only_cache`
                        # whose Metal buffers don't always release in time.
                        # Three guards added below close most of the gap:
                        #   1. Skip enqueue only when the cache key is shorter
                        #      than SSM_REDERIVE_MIN_TOKENS. That threshold is
                        #      intentionally 1: short repeated prompts are
                        #      still valid cache candidates, and an older
                        #      64-token floor caused avoidable full-prefill
                        #      repeats on mid-chat turns.
                        #   2. Skip enqueue if we've stored ≥ max_entries
                        #      worth of companions already — the LRU is
                        #      already saturated, the next eviction would
                        #      drop a useful entry to store one we're
                        #      probably going to evict before it's read.
                        #   3. Drop the queue cap from 20 to 8 so the
                        #      worst-case footprint shrinks 60%.
                        if companion_len < SSM_REDERIVE_MIN_TOKENS:
                            logger.debug(
                                "SSM companion: skipping re-derive for "
                                f"{request_id} (cache_key_len={companion_len} < "
                                f"{SSM_REDERIVE_MIN_TOKENS})"
                            )
                        elif (
                            self._ssm_state_cache is not None
                            and len(getattr(self._ssm_state_cache, "_store", {}))
                                >= self._ssm_state_cache.max_entries
                        ):
                            logger.debug(
                                "SSM companion: skipping re-derive for "
                                f"{request_id} (companion cache saturated, "
                                f"{self._ssm_state_cache.max_entries} entries)"
                            )
                        elif (
                            self._ssm_state_cache is not None
                            and self._ssm_state_cache.has_complete(
                                companion_tokens, companion_len
                            )
                        ):
                            # A complete companion already sits at this exact
                            # key (typical after a cache HIT, which restored
                            # from it). The deferred clean prefill would
                            # recompute byte-identical state and its full
                            # prompt-length prefill starves the next
                            # request's TTFT (the #92 MLLM fix, 1e8602b40 —
                            # this text path never had the probe).
                            logger.info(
                                "SSM companion: skipping re-derive for %s: "
                                "complete companion already stored at "
                                "%d-token key",
                                request_id,
                                companion_len,
                            )
                        else:
                            if not hasattr(self, "_ssm_rederive_queue"):
                                self._ssm_rederive_queue = []
                            # Cap queue at SSM_REDERIVE_QUEUE_CAP. Oldest
                            # entries are the least useful (newer prompts
                            # are more likely to be re-requested). Was 20.
                            if len(self._ssm_rederive_queue) >= SSM_REDERIVE_QUEUE_CAP:
                                self._ssm_rederive_queue.pop(0)
                            self._ssm_rederive_queue.append(
                                (list(companion_tokens), companion_len, request_id)
                            )
                            self._ensure_ssm_rederive_idle_task()
                            logger.info(
                                f"SSM companion: queued deferred re-derive for "
                                f"{request_id} (gpl={_gpl}, {companion_len} cache-key "
                                f"tokens, will run during next idle period)"
                            )
                    elif prompt_len > 0:
                        # gpl=0 (non-thinking) hybrid SSM path. Mirror the
                        # gpl>0 defer-only pattern above — DO NOT extract
                        # post-output SSM layers from `_extracted_cache`
                        # and DO NOT do an immediate `is_complete=False`
                        # store.
                        #
                        # Earlier fix attempted to mark contaminated state
                        # as `is_complete=False` so the rejection check
                        # would force re-prefill until async re-derive
                        # replaced it. That path was correct for
                        # correctness but caused a Metal-buffer RAM leak
                        # on Nemotron-Omni JANGTQ2: each request stored
                        # ~10 MB of `MambaCache.from_state(...)` arrays
                        # backing post-output Metal command buffers AND
                        # queued an async re-derive. The contaminated
                        # entries piled up in `_ssm_state_cache._store`
                        # (LRU cap holds them past the rejection check)
                        # while the async re-derive ALSO populated entries
                        # — doubling resident SSM state per request.
                        #
                        # Fetch path with NO companion entry already
                        # falls back to full prefill (line 2364:
                        # `if _entry is None: ssm_states = None`), so
                        # the immediate-store-with-rejection-flag was
                        # functionally redundant: the next turn re-prefills
                        # either way. Drop the immediate store; just queue
                        # the async re-derive (clean-boundary capture
                        # writes `is_complete=True` directly).
                        if companion_len < SSM_REDERIVE_MIN_TOKENS:
                            logger.debug(
                                "SSM companion (gpl=0): skipping re-derive "
                                f"for {request_id} (cache_key_len={companion_len} "
                                f"< {SSM_REDERIVE_MIN_TOKENS})"
                            )
                        elif (
                            self._ssm_state_cache is not None
                            and len(getattr(self._ssm_state_cache, "_store", {}))
                                >= self._ssm_state_cache.max_entries
                        ):
                            logger.debug(
                                "SSM companion (gpl=0): skipping re-derive "
                                f"for {request_id} (companion cache saturated)"
                            )
                        elif (
                            self._ssm_state_cache is not None
                            and self._ssm_state_cache.has_complete(
                                companion_tokens, companion_len
                            )
                        ):
                            # Same skip as the gpl>0 branch above: a HIT
                            # restored from a complete companion at this key,
                            # so re-deriving it is a full wasted prefill.
                            logger.info(
                                "SSM companion (gpl=0): skipping re-derive "
                                "for %s: complete companion already stored "
                                "at %d-token key",
                                request_id,
                                companion_len,
                            )
                        else:
                            if not hasattr(self, "_ssm_rederive_queue"):
                                self._ssm_rederive_queue = []
                            if len(self._ssm_rederive_queue) >= SSM_REDERIVE_QUEUE_CAP:
                                self._ssm_rederive_queue.pop(0)
                            self._ssm_rederive_queue.append(
                                (list(companion_tokens), companion_len, request_id)
                            )
                            self._ensure_ssm_rederive_idle_task()
                            logger.info(
                                f"SSM companion (gpl=0): queued deferred "
                                f"re-derive for {request_id} ({companion_len} "
                                f"cache-key tokens, runs on next idle tick)"
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
                            cache_key_override = getattr(
                                request, "_extracted_cache_key_tokens", None
                            )
                            prompt_tokens = list(request.prompt_token_ids)
                            cache_data = request._extracted_cache
                            if cache_key_override is not None:
                                store_tokens = list(cache_key_override)
                            else:
                                # Strip generation prompt tokens from cache key.
                                # Original gen_prompt_len prefix cache fix by Jinho Jang
                                # (eric@jangq.ai) — vMLX. This solved 100% cache miss for
                                # all thinking models (Nemotron, Qwen3, DeepSeek-R1, Mistral 4).
                                # Chat templates append assistant role tokens at the end
                                # (e.g., <|im_start|>assistant\n<think>\n) which always
                                # differ on subsequent turns. Including them in the block
                                # hash causes 100% cache misses in multi-turn conversations.
                                gen_prompt_len = getattr(request, "_gen_prompt_len", 0)
                                if gen_prompt_len > 0 and gen_prompt_len < len(
                                    prompt_tokens
                                ):
                                    prompt_tokens = prefix_cache_key_tokens(request)
                                    # Also truncate KV cache data to match shortened key.
                                    # Without this, KV has more tokens than the key,
                                    # causing duplicate KV entries on cache hit → <unk> flood.
                                    # cache_data is extracted state dicts (not raw objects),
                                    # so truncate the tensors within each dict directly.
                                    target = len(prompt_tokens) - 1  # N-1 for re-feed
                                    if target <= 0:
                                        # A reusable paged entry must represent at
                                        # least one cache-key token. Publishing any
                                        # extracted state under an empty/negative
                                        # generation-prefix-stripped key cannot prove
                                        # key/state alignment, even when the layer
                                        # format itself is recognized.
                                        cache_data = None
                                        logger.warning(
                                            "Skipping paged/L2 store for %s: "
                                            "generation-prompt stripping leaves no "
                                            "positive cache boundary (target=%d)",
                                            request_id,
                                            target,
                                        )
                                    if target > 0:
                                        truncated_dicts = []
                                        trunc_ok = True
                                        for sd in cache_data:
                                            if not isinstance(sd, dict):
                                                trunc_ok = False
                                                break
                                            state = sd.get("state")
                                            cls_name = sd.get("class_name", "")
                                            if self._is_dsv4_cache_class_name(cls_name) and state is not None:
                                                try:
                                                    from jang_tools.dsv4.mlx_model import (
                                                        DeepseekV4Cache,
                                                    )

                                                    cache = DeepseekV4Cache(
                                                        sliding_window=int(
                                                            sd.get("sliding_window")
                                                            or 128
                                                        ),
                                                        compress_ratio=sd.get(
                                                            "compress_ratio"
                                                        ),
                                                    )
                                                    cache.state = state
                                                    try:
                                                        cache.meta_state = sd.get(
                                                            "meta_state", ()
                                                        )
                                                    except Exception:
                                                        pass
                                                    current_len = int(
                                                        getattr(cache, "offset", 0) or 0
                                                    )
                                                    if current_len < target:
                                                        logger.warning(
                                                            "Skipping DSV4 paged cache "
                                                            "store for %s: native state "
                                                            "represents %d tokens but "
                                                            "shortened key requires %d",
                                                            request_id,
                                                            current_len,
                                                            target,
                                                        )
                                                        trunc_ok = False
                                                        break
                                                    to_trim = max(0, current_len - target)
                                                    # SAFETY: gen_prompt_len stripping
                                                    # also calls cache.trim() on a
                                                    # reconstructed DeepseekV4Cache. The
                                                    # same SWA RotatingKVCache wrap
                                                    # constraint applies here: if the
                                                    # original prefill exceeded
                                                    # sliding_window, the SWA buffer has
                                                    # wrapped and cannot be safely
                                                    # rewound. Refuse the trim and skip
                                                    # this cache (caller falls through
                                                    # to fresh prefill on next-turn).
                                                    # Symmetric with Fix 1 guard.
                                                    local = getattr(cache, "local", None)
                                                    _swa = int(
                                                        getattr(local, "max_size", 128) or 128
                                                    )
                                                    local_state = getattr(local, "state", None)

                                                    def _uniform_seq_len(value):
                                                        if hasattr(value, "shape"):
                                                            if len(value.shape) < 2:
                                                                return None
                                                            return int(value.shape[-2])
                                                        if isinstance(value, (tuple, list)) and value:
                                                            lengths = [_uniform_seq_len(item) for item in value]
                                                            if any(length is None for length in lengths):
                                                                return None
                                                            unique = set(lengths)
                                                            return lengths[0] if len(unique) == 1 else None
                                                        return None

                                                    local_len = _uniform_seq_len(local_state)
                                                    if local_len != current_len:
                                                        logger.warning(
                                                            "Skipping DSV4 paged cache "
                                                            "store for %s: local K/V "
                                                            "represents %s tokens but "
                                                            "native metadata reports %d",
                                                            request_id,
                                                            local_len,
                                                            current_len,
                                                        )
                                                        trunc_ok = False
                                                        break
                                                    if to_trim > 0 and current_len > _swa:
                                                        logger.info(
                                                            f"DSV4 gen_prompt_len strip "
                                                            f"skipped: current_len="
                                                            f"{current_len}, target="
                                                            f"{target}, sliding_window="
                                                            f"{_swa} → SWA wrapped, "
                                                            f"trim unsafe."
                                                        )
                                                        trunc_ok = False
                                                        break
                                                    if to_trim:
                                                        cache.trim(to_trim)
                                                    if int(getattr(cache, "offset", 0) or 0) != target:
                                                        logger.warning(
                                                            "Skipping DSV4 paged cache "
                                                            "store for %s: trimmed native "
                                                            "state did not reach exact "
                                                            "target=%d",
                                                            request_id,
                                                            target,
                                                        )
                                                        trunc_ok = False
                                                        break
                                                    truncated_dicts.append(
                                                        {
                                                            **sd,
                                                            "state": cache.state,
                                                            "meta_state": cache.meta_state,
                                                        }
                                                    )
                                                    continue
                                                except Exception:
                                                    trunc_ok = False
                                                    break
                                            sub_caches = sd.get("sub_caches")
                                            if cls_name == "CacheList":
                                                # Extracted CacheList layers carry
                                                # state=None plus real nested cache
                                                # payloads in sub_caches. Passing the
                                                # wrapper through unchanged would store
                                                # the longer nested states under a
                                                # shorter key, so walk into them and
                                                # apply the same per-layer rule.
                                                if state is None and not sub_caches:
                                                    truncated_dicts.append(sd)
                                                    continue
                                                aligned_cl = _align_cache_list_state_dict(
                                                    sd, target
                                                )
                                                if aligned_cl is not None:
                                                    truncated_dicts.append(aligned_cl)
                                                    continue
                                                logger.warning(
                                                    "Skipping paged cache store for %s: "
                                                    "cannot align nonempty CacheList "
                                                    "sub-caches to generation-prompt-"
                                                    "stripped target=%d (blocked by %s)",
                                                    request_id,
                                                    target,
                                                    _blocking_cache_list_sub_class(sd),
                                                )
                                                trunc_ok = False
                                                break
                                            if (
                                                (state is None and not sub_caches)
                                                or (
                                                    sd.get("no_state") is True
                                                    and not sub_caches
                                                )
                                            ):
                                                # Proven no-state placeholders do not
                                                # encode a token position and may pass.
                                                truncated_dicts.append(sd)
                                                continue
                                            if (
                                                self._is_hybrid
                                                and not self._uses_dsv4_cache
                                                and not self._uses_zaya_cache
                                                and not self._mixed_attention_cache_model
                                                and cls_name
                                                in {
                                                    "MambaCache",
                                                    "BatchMambaCache",
                                                    "ArraysCache",
                                                }
                                            ):
                                                # Generic hybrid SSM/GDN state is
                                                # cumulative and therefore cannot be
                                                # shortened to the generation-prefix-
                                                # stripped boundary.  This lane stores
                                                # it in the typed external companion;
                                                # store_cache(...,
                                                # store_cumulative_state=False) already
                                                # writes a positional placeholder for
                                                # these layers. Canonicalize the payload
                                                # to an explicit skip while retaining
                                                # its layer slot, so aligned KV blocks
                                                # remain reusable without publishing
                                                # contaminated cumulative state.
                                                truncated_dicts.append(
                                                    {
                                                        **sd,
                                                        "state": None,
                                                        "meta_state": None,
                                                    }
                                                )
                                                continue
                                            if cls_name == "MiniMaxM3SparseCache":
                                                truncated_m3 = (
                                                    _truncate_minimax_m3_state_dict(
                                                        sd, target
                                                    )
                                                )
                                                if truncated_m3 is None:
                                                    logger.info(
                                                        "Skipping paged cache store for %s: "
                                                        "cannot align MiniMax-M3 K/V/idx_keys "
                                                        "to generation-prompt-stripped target=%d",
                                                        request_id,
                                                        target,
                                                    )
                                                    trunc_ok = False
                                                    break
                                                truncated_dicts.append(truncated_m3)
                                                continue
                                            # Same rule as the CacheList sub-caches: one implementation, so
                                            # the flat and nested paths cannot drift into different slicing
                                            # or metadata behaviour.
                                            aligned_flat = _align_attention_state_dict(sd, target)
                                            if aligned_flat is not None:
                                                truncated_dicts.append(aligned_flat)
                                                continue
                                            if _is_recognized_attention_layout(state):
                                                # Layout understood, alignment refused -- most often a
                                                # RotatingKVCache whose ring buffer already wrapped, where
                                                # tensor order is no longer temporal. Expected, not a gap.
                                                logger.info(
                                                    "Skipping paged cache store for %s: recognized %s state could not be aligned to target=%d (prompt_tokens=%d, gen_prompt_len=%d)",
                                                    request_id,
                                                    cls_name or type(state).__name__,
                                                    target,
                                                    len(prompt_tokens),
                                                    gen_prompt_len,
                                                )
                                                trunc_ok = False
                                                break
                                            # Unknown nonempty state cannot be assumed
                                            # token-independent. Fail closed rather
                                            # than publish a longer state under the
                                            # shortened generation-prefix key.
                                            logger.warning(
                                                "Skipping paged cache store for %s: "
                                                "unknown nonempty cache state format "
                                                "class=%s target=%d",
                                                request_id,
                                                cls_name or type(state).__name__,
                                                target,
                                            )
                                            trunc_ok = False
                                            break
                                        if trunc_ok and truncated_dicts:
                                            cache_data = truncated_dicts
                                        else:
                                            # Never publish an original longer cache
                                            # state under the generation-prefix-
                                            # stripped token key.  A failed trim means
                                            # key/state equivalence is unknown for every
                                            # cache family, not only DSV4.  Known mixed-
                                            # SWA paths normally materialize a clean
                                            # prompt-boundary snapshot before this
                                            # fallback, but any unalignable/unknown
                                            # state must still fail closed here.
                                            cache_data = None
                                            if not self._uses_dsv4_cache:
                                                logger.warning(
                                                    "Skipping paged/L2 store for %s: "
                                                    "cache state could not be aligned "
                                                    "to the generation-prompt-stripped "
                                                    "key (%d tokens)",
                                                    request_id,
                                                    target,
                                                )
                                            else:
                                                # A DSV4 prompt snapshot may include the
                                                # assistant generation-prefix suffix even
                                                # though the reusable key intentionally
                                                # excludes it. Once local SWA has wrapped,
                                                # DeepseekV4Cache cannot be rewound without
                                                # also proving CSA/HCA pool equivalence.
                                                # The existing clean re-prefill path remains
                                                # the only valid way to materialize that
                                                # boundary.
                                                logger.warning(
                                                    "Skipping DSV4 paged/L2 store for %s: "
                                                    "native composite state could not be "
                                                    "aligned to the generation-prompt-"
                                                    "stripped key (%d tokens)",
                                                    request_id,
                                                    target,
                                                )
                                # Paged prefix cache entries must be keyed by the
                                # same token count represented by cache_data:
                                # prompt_len - 1. The generator re-feeds the last
                                # prompt token on hit to obtain first-token logits.
                                store_tokens = (
                                    prompt_tokens[:-1]
                                    if len(prompt_tokens) > 1
                                    else prompt_tokens
                                )
                            # Previously this path stored a truncated N-1
                            # cache under a full-N token hash. Exact hits then
                            # reported zero remaining tokens; DSV4BatchGenerator
                            # correctly refused to decode without a kickoff
                            # token and returned an empty/stop response. This
                            # is especially visible with block disk L2 because
                            # full-key stale blocks survive process restarts.
                            if cache_data is not None:
                                _t_store = time.perf_counter()
                                _paged_store_kwargs = {
                                    "cache_type": self._pick_cache_type_for_request(request),
                                }
                                if getattr(request, "_cache_extra_keys", None):
                                    _paged_store_kwargs["cache_extra_keys"] = dict(
                                        request._cache_extra_keys
                                    )
                                if (
                                    self._is_hybrid
                                    and not self._uses_dsv4_cache
                                    and not self._uses_zaya_cache
                                    and not self._mixed_attention_cache_model
                                ):
                                    _paged_store_kwargs["store_cumulative_state"] = False
                                _stored_block_table = self.block_aware_cache.store_cache(
                                    request_id,
                                    store_tokens,
                                    cache_data,
                                    **_paged_store_kwargs,
                                )
                                self._retarget_ssm_rederive_to_paged_boundary(
                                    request_id,
                                    store_tokens,
                                    _stored_block_table,
                                )
                                self._persist_hybrid_ssd_companion(
                                    request, store_tokens, _stored_block_table,
                                )
                                self._dsv4_trace_timing(
                                    "store_cache",
                                    _t_store,
                                    request_id,
                                    tokens=len(store_tokens),
                                    layers=len(cache_data or []),
                                )
                                if cache_key_override is not None:
                                    _coverage_note = (
                                        f"key-aligned cache coverage "
                                        f"{len(store_tokens)} tokens"
                                    )
                                else:
                                    _coverage_note = (
                                        f"cache truncated to "
                                        f"{max(len(prompt_tokens) - 1, 0)} tokens"
                                    )
                                logger.info(
                                    f"Stored paged cache for request {request_id} "
                                    f"({len(store_tokens)} cache-key tokens from "
                                    f"{len(prompt_tokens)} prompt tokens, "
                                    f"{len(request._extracted_cache)} layers, "
                                    f"{_coverage_note})"
                                )
                                _PERSIST.record(request_id, "stored", f"paged {_coverage_note}", retained_tokens=len(store_tokens))
                        except Exception as e:
                            logger.warning(
                                f"Failed to store paged cache for {request_id}: {e}"
                            )
                            _PERSIST.record(request_id, "failed", f"paged: {e}")
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None
                            # store_cache() re-registered this request in the
                            # BlockAwarePrefixCache._request_tables map and holds
                            # the freshly-stored / hash-deduped blocks at
                            # ref_count>=1. The pre-store tracking cleanup at the
                            # top of _cleanup_finished ran BEFORE this store, so
                            # without a second release the completed request's
                            # blocks stay pinned at ref>=1 forever: neither the
                            # block-count LRU nor enforce_byte_budget can reclaim
                            # them (both only evict ref==0 blocks), and the in-RAM
                            # KV mirror ratchets upward with every distinct prompt.
                            # Settle the stored blocks to "cached but free" so
                            # memory pressure can reclaim them while they remain
                            # prefix-reusable. Mirrors the pre-store idiom above.
                            try:
                                _stored_entry = (
                                    self.block_aware_cache._request_tables.pop(
                                        request_id, None
                                    )
                                )
                                self.block_aware_cache.paged_cache.release_request_refs(
                                    _stored_entry.block_table
                                    if _stored_entry
                                    else None
                                )
                                self.block_aware_cache.paged_cache.detach_request(
                                    request_id
                                )
                                # store_cache() cannot evict the just-stored
                                # blocks while this completed request still
                                # pins them. Enforce the configured L1 byte
                                # ceiling again after releasing those refs.
                                if (
                                    self.block_aware_cache.paged_cache.enforces_byte_budget
                                ):
                                    self.block_aware_cache.paged_cache.enforce_byte_budget()
                            except Exception as _rel_e:
                                logger.debug(
                                    "Post-store paged ref release failed for %s: %s",
                                    request_id,
                                    _rel_e,
                                )
                    else:
                        _PERSIST.record(request_id, "skipped", "no extracted cache")
                        logger.info(
                            "Skipping paged cache store for %s: no extracted cache "
                            "(prompt_tokens=%d, gen_prompt_len=%d, block_cache_enabled=%s)",
                            request_id,
                            len(request.prompt_token_ids or []),
                            int(getattr(request, "_gen_prompt_len", 0) or 0),
                            self.block_aware_cache is not None,
                        )
                    # NOTE: Tracking cleanup (pop + detach) moved above the
                    # _skip_cache_store guard so it runs unconditionally.

                elif (
                    self.memory_aware_cache is not None
                ):
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
                            if (
                                getattr(self, "_uses_openpangu_cache", False)
                                or getattr(self, "_uses_glm5_next_cache", False)
                            ):
                                _gpl_store = 0
                            if 0 < _gpl_store < len(prompt_tokens):
                                prompt_tokens = prompt_tokens[:-_gpl_store]
                            prompt_len = len(prompt_tokens)
                            cache_key_override = getattr(
                                request, "_extracted_cache_key_tokens", None
                            )
                            if cache_key_override is not None:
                                cache_to_store = request._extracted_cache
                                cache_key_tokens = list(cache_key_override)
                            else:
                                cache_to_store = self._truncate_cache_to_prompt_length(
                                    request._extracted_cache, prompt_len
                                )
                            if cache_to_store is None:
                                logger.debug(
                                    f"Request {request_id}: cannot truncate cache "
                                    f"to prompt length (hybrid model), skipping store"
                                )
                            else:
                                if cache_key_override is None:
                                    cache_key_tokens = (
                                        prompt_tokens[:-1]
                                        if prompt_len > 1
                                        else list(prompt_tokens)
                                    )
                                cache_type = self._pick_cache_type_for_request(request)
                                from .utils.turboquant_config import (
                                    turboquant_cache_telemetry,
                                )

                                self._last_turboquant_cache = (
                                    turboquant_cache_telemetry(cache_to_store)
                                )
                                if self._last_turboquant_cache.get("object_layers", 0):
                                    logger.info(
                                        "TurboQuant live telemetry: %s",
                                        self._last_turboquant_cache,
                                    )
                                # Prompt disk L2 contract: fetch uses the full
                                # generation-prompt-stripped prompt key, while
                                # the payload is N-1 so the last prompt token is
                                # re-fed before generation. M3's default route
                                # is memory-aware (paged off), so write L2 here
                                # too; DiskCacheManager preserves M3 idx_keys.
                                if (
                                    self.disk_cache is not None
                                    and not self._is_hybrid
                                    and len(cache_key_tokens) == max(prompt_len - 1, 0)
                                ):
                                    try:
                                        _call_with_optional_cache_extra(
                                            self.disk_cache.store,
                                            prompt_tokens,
                                            cache_to_store,
                                            cache_type=cache_type,
                                            cache_extra_keys=getattr(
                                                request,
                                                "_cache_extra_keys",
                                                None,
                                            ),
                                        )
                                    except Exception as de:
                                        logger.debug(
                                            f"Disk cache store failed for "
                                            f"{request_id}: {de}"
                                        )
                                # Quantize for storage efficiency
                                if getattr(self, "_kv_cache_bits", 0):
                                    cache_to_store = self._quantize_cache_for_storage(
                                        cache_to_store
                                    )
                                stored = _call_with_optional_cache_extra(
                                    self.memory_aware_cache.store,
                                    cache_key_tokens,
                                    cache_to_store,
                                    cache_type=cache_type,
                                    cache_extra_keys=getattr(
                                        request,
                                        "_cache_extra_keys",
                                        None,
                                    ),
                                )
                                if stored:
                                    if (
                                        getattr(self, "_uses_openpangu_cache", False)
                                        or getattr(
                                            self, "_uses_glm5_next_cache", False
                                        )
                                    ):
                                        _typed_family = (
                                            "glm5_next"
                                            if getattr(
                                                self,
                                                "_uses_glm5_next_cache",
                                                False,
                                            )
                                            else "openPangu"
                                        )
                                        logger.info(
                                            "Stored %s exact typed cache for "
                                            "%s (%d N-1 key tokens from %d prompt "
                                            "tokens; composite state not truncated)",
                                            _typed_family,
                                            request_id,
                                            len(cache_key_tokens),
                                            prompt_len,
                                        )
                                    else:
                                        logger.info(
                                            f"Stored cache for request {request_id} "
                                            f"({len(cache_key_tokens)} cache-key tokens "
                                            f"from {prompt_len} prompt tokens, "
                                            f"KV truncated to {prompt_len - 1})"
                                        )
                                    _PERSIST.record(request_id, "stored", "memory-aware", retained_tokens=len(cache_key_tokens))
                                else:
                                    logger.warning(
                                        f"Cache store rejected for request {request_id} "
                                        f"({prompt_len} tokens) — entry too large for budget"
                                    )
                                    _PERSIST.record(request_id, "refused", "memory-aware: entry too large for budget")
                                # Disk L2 is written above for memory-aware
                                # configs (including MiniMax-M3 paged-off).
                        except Exception as e:
                            logger.warning(
                                f"Failed to store memory-aware cache for {request_id}: {e}"
                            )
                            _PERSIST.record(request_id, "failed", f"memory-aware: {e}")
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None

                elif (
                    self.prefix_cache is not None
                ):
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
                            cache_key_override = getattr(
                                request, "_extracted_cache_key_tokens", None
                            )
                            if cache_key_override is not None:
                                cache_to_store = request._extracted_cache
                                cache_key_tokens = list(cache_key_override)
                            else:
                                cache_to_store = self._truncate_cache_to_prompt_length(
                                    request._extracted_cache, prompt_len
                                )
                            if cache_to_store is not None:
                                from .utils.turboquant_config import (
                                    turboquant_cache_telemetry,
                                )

                                self._last_turboquant_cache = (
                                    turboquant_cache_telemetry(cache_to_store)
                                )
                                if self._last_turboquant_cache.get("object_layers", 0):
                                    logger.info(
                                        "TurboQuant live telemetry: %s",
                                        self._last_turboquant_cache,
                                    )
                                if cache_key_override is None:
                                    cache_key_tokens = (
                                        prompt_tokens[:-1]
                                        if prompt_len > 1
                                        else list(prompt_tokens)
                                    )
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
                                    cache_key_tokens,
                                    cache_to_store,
                                )
                                logger.debug(
                                    f"Stored cache for request {request_id} "
                                    f"({len(cache_key_tokens)} cache-key tokens "
                                    f"from {prompt_len} prompt tokens, "
                                    f"truncated from {prompt_len + len(request.output_token_ids)})"
                                )
                                _PERSIST.record(request_id, "stored", "legacy prefix cache (L1 only)", retained_tokens=len(cache_key_tokens), durable=False)
                                # NOTE: Disk L2 store is handled by the paged
                                # cache path. Legacy prefix cache is L1-only.
                        except Exception as e:
                            logger.debug(f"Failed to store cache for {request_id}: {e}")
                            _PERSIST.record(request_id, "failed", f"legacy: {e}")
                        finally:
                            # Clear extracted cache reference to help GC
                            request._extracted_cache = None

            # DSV4BatchGenerator retains terminal requests (and therefore the
            # live native SWA/CSA/HCA cache) until remove() is called.  Do this
            # only after every cache consumer above has finished with the
            # response-owned snapshot/extracted state.  Deferring removal to
            # the next insert or generator replacement shifts cache teardown
            # onto the next request's TTFT and looks like an admission hang.
            if (
                request is not None
                and self.batch_generator is not None
                and self.batch_generator.__class__.__name__
                == "DSV4BatchGenerator"
            ):
                uid = self.request_id_to_uid.get(request_id)
                if uid is not None:
                    try:
                        self.batch_generator.remove([uid])
                    except Exception as exc:
                        logger.warning(
                            "Failed to release finished DSV4 generator request "
                            "%s (uid=%s): %s",
                            request_id,
                            uid,
                            exc,
                        )

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
                unregister_generation_logprobs(self.model, uid)
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Track as finished
            self.finished_req_ids.add(request_id)

        # Only clear Metal memory cache when no other requests are actively
        # running. Calling Metal cache cleanup during an active prefill
        # can interfere with in-flight GPU operations and cause crashes.
        if finished_ids and not self.running:
            _clear_start = time.perf_counter()
            clear_mlx_memory_cache(log=logger)
            self._dsv4_trace_timing("clear_mlx_memory_cache", _clear_start)
        self._dsv4_trace_timing(
            "cleanup_finished_total",
            _cleanup_start,
            next(iter(finished_ids)) if finished_ids else None,
            count=len(finished_ids),
        )

    def _cleanup_finished_after_terminal_dispatch(
        self,
        finished_ids: Set[str],
    ) -> None:
        """Release text-request cache buffers after the cleanup frame returns."""
        self._cleanup_finished(finished_ids)
        # _cleanup_finished() publishes the native prompt-boundary state and
        # detaches every persistent request/cache owner, but its own final loop
        # locals still reference the just-stored ``request``/``cache_to_store``
        # graph when the in-function allocator clear runs.  The frame returning
        # here is the first point where those references are actually gone.
        # Reclaim again on the same model worker so SSD-only text serving does
        # not retain one MLX allocator step after every completed request.
        if finished_ids and not self.running:
            try:
                import gc as _gc

                _gc.collect()
            except Exception as gc_error:  # noqa: BLE001
                logger.debug(
                    "Could not collect released text terminal cache refs: %s",
                    gc_error,
                )
            clear_mlx_memory_cache(log=logger)

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption.

        A PERMANENT allocation failure is explicitly not one. Asking for more
        than the device's maximum buffer size cannot be relieved by clearing a
        cache, so routing it into cache-clear + reschedule would re-run the
        identical doomed allocation forever — the same trap
        ``DSV4PrefillMemoryError`` is excluded from by name.
        """
        from .utils.prefill_admission import is_permanent_allocation_error

        if is_permanent_allocation_error(error):
            return False
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
            self.block_aware_cache.clear(force=True)
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
        if self._ssm_state_cache is not None:
            self._ssm_state_cache.clear()

        # The pinned DSV4 exact lm_head cache survives allocator-cache
        # clearing; release it so rescheduled requests retry under genuinely
        # lower memory pressure instead of failing through the same path.
        try:
            from .models.dsv4_lm_head_fastpath import (
                release_all_dsv4_lm_head_caches,
            )

            released = release_all_dsv4_lm_head_caches("cache-error recovery")
            if released:
                logger.warning(
                    "Released %d DSV4 exact lm_head cache(s) during cache "
                    "recovery; the fastpath reinstalls on the next model load",
                    released,
                )
        except Exception:
            pass

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        logger.info("Cache recovery completed")

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # A restart on a request that has ALREADY streamed tokens is
            # noteworthy: those tokens are delivered to the client and cannot be
            # recalled, so the re-run must continue under the ORIGINAL budget
            # (total_output_tokens survives this reset; remaining_output_budget
            # shrinks the re-insert cap accordingly). Surface it loudly.
            request.recovery_restarts += 1
            if request.num_output_tokens > 0:
                logger.warning(
                    "Cache-error reschedule of request %s after %d emitted "
                    "tokens (lifetime %d, restart #%d) — continuing under the "
                    "original max_tokens=%d via remaining budget %d",
                    request_id,
                    request.num_output_tokens,
                    request.total_output_tokens,
                    request.recovery_restarts,
                    request.sampling_params.max_tokens,
                    request.remaining_output_budget,
                )
            # Reset request state — must clear ALL generation state so the
            # retried request starts from scratch with correct token budget.
            # NOTE: total_output_tokens is intentionally NOT reset here.
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
            if ab is None or not (
                ab.has_uid(uid) if hasattr(ab, "has_uid") else uid in ab.uids
            ):
                raise RuntimeError(
                    "uid not in active_batch — cannot get forward logprobs"
                )
            ab_idx = (
                ab.index_of(uid)
                if hasattr(ab, "index_of")
                else ab.uids.index(uid)
            )
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
                        c.keys = mx.array(_np.ascontiguousarray(_k[..., :pre_verify_offset, :])).astype(_kd)
                        c.values = mx.array(_np.ascontiguousarray(_v[..., :pre_verify_offset, :])).astype(_vd)
                    c.offset = pre_verify_offset
                    if hasattr(c, "_idx"):  # RotatingKVCache: sync write pointer
                        c._idx = pre_verify_offset
                new_uids = self.batch_generator.insert(
                    [[correction_token]],
                    max_tokens=[max(1, remaining - 1)],
                    caches=[kv_cache],
                    samplers=[self._request_seeded_sampler(request)],
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
                        c.keys = mx.array(_np.ascontiguousarray(_k[..., :accepted_offset, :])).astype(_kd)
                        c.values = mx.array(_np.ascontiguousarray(_v[..., :accepted_offset, :])).astype(_vd)
                    c.offset = accepted_offset
                    if hasattr(c, "_idx"):  # RotatingKVCache: sync write pointer
                        c._idx = accepted_offset

            # 5. Re-insert with bonus token (next to-be-processed token)
            new_remaining = max(1, remaining - num_accept - 1)
            new_uids = self.batch_generator.insert(
                [[bonus_token]],
                max_tokens=[new_remaining],
                caches=[kv_cache],
                samplers=[self._request_seeded_sampler(request)],
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
                        samplers=[self._request_seeded_sampler(request)],
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

    def step(
        self,
        max_retries: int = 2,
        *,
        defer_finished_cleanup: bool = False,
    ) -> SchedulerOutput:
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
            defer_finished_cleanup: Return terminal outputs before synchronous
                cache persistence. The async engine owns cleanup after dispatch.

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
                    active_batch = getattr(self.batch_generator, "active_batch", None)
                    active_cache = getattr(active_batch, "cache", None)
                    if isinstance(active_cache, list):
                        from .utils.turboquant_config import (
                            turboquant_cache_telemetry,
                        )

                        telemetry = turboquant_cache_telemetry(active_cache)
                        if telemetry.get("object_layers", 0):
                            previous = getattr(self, "_last_turboquant_cache", None)
                            self._last_turboquant_cache = telemetry
                            if (
                                previous is None
                                or telemetry.get("compressed_tokens_total")
                                != previous.get("compressed_tokens_total")
                            ):
                                logger.info("TurboQuant live telemetry: %s", telemetry)
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
                        if not defer_finished_cleanup:
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
                    # A prefill-valve abort is a deterministic capacity
                    # rejection, not corruption: clearing caches and
                    # rescheduling would re-run the doomed prefill in a loop.
                    # Match by class name (mirrors _is_dsv4_cache_class_name) to
                    # avoid an import cycle.
                    #
                    # PrefillAdmissionError is the family-agnostic valve, now
                    # also raised by the hybrid chunked prefill. It has to be
                    # listed here for the same reason DSV4's is: without it the
                    # hybrid decline is retried until max_retries, re-running a
                    # prefill already measured as unservable.
                    if any(
                        cls.__name__
                        in ("DSV4PrefillMemoryError", "PrefillAdmissionError")
                        for cls in type(e).__mro__
                    ):
                        is_cache_error = False
                    if is_gen_type_error:
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
        # idle cleanup never triggers. This timer ensures periodic cleanup
        # to prevent Metal's internal allocator cache from growing unbounded.
        now = time.monotonic()
        if now - self._last_metal_gc_time > self._metal_gc_interval:
            self._last_metal_gc_time = now
            if clear_mlx_memory_cache(log=logger):
                logger.debug("Periodic Metal memory cache cleanup")

        # Deferred SSM re-derive no longer runs here. It was moved off the
        # response path onto the engine loop's post-response idle-task drain
        # (vmlx#245): running it inside step() delayed the finishing
        # request's final chunk and inflated a newly-arrived request's TTFT
        # by an unpreemptable full prefill. See register_idle_task /
        # run_one_idle_task and _drain_one_ssm_rederive.

        return output

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[Request]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        generator_cls = (
            self.batch_generator.__class__.__name__
            if self.batch_generator is not None
            else None
        )
        if generator_cls == "SingleBatchGenerator":
            generator_path = "single_active"
        elif generator_cls == "DSV4BatchGenerator":
            generator_path = "dsv4"
        elif generator_cls == "BatchGenerator":
            generator_path = "batched"
        else:
            generator_path = "none" if generator_cls is None else "custom"
        now = time.time()

        def _request_lifecycle_detail(request: Request) -> Dict[str, Any]:
            schedule_time = getattr(request, "_schedule_time", None)
            detail = {
                "request_id": request.request_id,
                "status": getattr(request.status, "name", str(request.status)),
                "age_seconds": round(max(0.0, now - request.arrival_time), 3),
                "scheduled_age_seconds": (
                    round(max(0.0, time.perf_counter() - schedule_time), 3)
                    if isinstance(schedule_time, (int, float))
                    else None
                ),
                "prompt_tokens": int(getattr(request, "num_prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(request, "num_output_tokens", 0) or 0),
                "cached_tokens": int(getattr(request, "cached_tokens", 0) or 0),
                "max_tokens": int(getattr(request, "max_tokens", 0) or 0),
                "batch_uid": getattr(request, "batch_uid", None),
                "cache_detail": getattr(request, "_cache_detail", None),
                "cache_execution": getattr(request, "_cache_execution", None),
                "finish_reason": getattr(request, "finish_reason", None),
            }
            return {k: v for k, v in detail.items() if v is not None}

        running_ids = list(self.running)
        waiting_ids = [request.request_id for request in self.waiting]
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "waiting_request_ids": waiting_ids,
            "running_request_ids": running_ids,
            "pending_abort_request_ids": sorted(self._pending_aborts),
            "running_requests": [
                _request_lifecycle_detail(request)
                for request in self.running.values()
            ],
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "ewma_ttft_seconds": round(self._ewma_ttft, 3),
            "cache_hit_requests": self._cache_hit_requests,
            "cache_hit_tokens": self._cache_hit_tokens,
            "cache_hit_tokens_by_detail": dict(self._cache_hit_tokens_by_detail),
            "cache_reuse_skips": self._cache_reuse_skips,
            "cache_reuse_skip_tokens": self._cache_reuse_skip_tokens,
            "last_cache_reuse_skip": self._last_cache_reuse_skip,
            "cache_reuse_partial_downgrades": self._cache_reuse_partial_downgrades,
            "cache_reuse_partial_tokens": self._cache_reuse_partial_tokens,
            "last_cache_reuse_partial": self._last_cache_reuse_partial,
            "last_cache_selection": self._last_cache_selection,
            "last_cache_execution": self._last_cache_execution,
            "cache_admission_cost": {
                "prefill_seconds_per_token": round(
                    self._cache_admission_prefill_seconds_per_token,
                    9,
                ),
                "prefill_samples": self._cache_admission_prefill_sample_count,
                "prefill_reference_tokens": (
                    self._cache_admission_prefill_reference_tokens
                ),
                "disk_seconds_per_token": round(
                    self._cache_admission_disk_seconds_per_token,
                    9,
                ),
                "disk_samples": self._cache_admission_disk_sample_count,
                "disk_reference_tokens": (
                    self._cache_admission_disk_reference_tokens
                ),
                "max_comparable_ratio": self._cache_admission_max_comparable_ratio,
            },
            "tq_decoder_warmup": self._tq_decoder_warmup_stats,
            "last_turboquant_cache": getattr(
                self, "_last_turboquant_cache", None
            ),
            "engine_path": generator_path,
            "batch_generator": {
                "class": generator_cls,
                "path": generator_path,
                "single_active_decode": generator_path == "single_active",
                "max_num_seqs": self.config.max_num_seqs,
            },
        }
        if generator_cls == "BatchGenerator":
            try:
                from .patches.mlx_lm_mtp.batch_generator import (
                    native_mtp_stats_snapshot,
                )

                native_mtp_stats = native_mtp_stats_snapshot()
                # Publish when EITHER an engagement or a SKIP was recorded.
                # Gating on last_native_mtp alone meant a model that never
                # engaged published nothing at all, so the reason it was
                # skipped could not reach the UI.
                if (
                    native_mtp_stats.get("last_native_mtp") is not None
                    or native_mtp_stats.get("last_native_mtp_skip") is not None
                ):
                    stats["batch_generator"].update(native_mtp_stats)
            except Exception as exc:
                logger.debug("Native MTP telemetry snapshot unavailable: %s", exc)
        elif generator_cls in {"SingleBatchGenerator", "DSV4BatchGenerator"}:
            stats["batch_generator"].update(
                {
                    "prompt_snapshot_max_bytes": getattr(
                        self.batch_generator, "prompt_snapshot_max_bytes", None
                    ),
                    "prompt_snapshot_last_estimated_bytes": getattr(
                        self.batch_generator,
                        "prompt_snapshot_last_estimated_bytes",
                        0,
                    ),
                    "prompt_snapshot_oversize_skips": getattr(
                        self.batch_generator, "prompt_snapshot_oversize_skips", 0
                    ),
                    "prompt_snapshot_headroom_skips": getattr(
                        self.batch_generator, "prompt_snapshot_headroom_skips", 0
                    ),
                    "prompt_snapshot_last_headroom_bytes": getattr(
                        self.batch_generator, "prompt_snapshot_last_headroom_bytes", 0
                    ),
                }
            )
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
        if base is not None:
            generator_cls = (
                self.batch_generator.__class__.__name__
                if self.batch_generator is not None
                else None
            )
            if generator_cls == "SingleBatchGenerator":
                engine_path = "single_active"
            elif generator_cls == "DSV4BatchGenerator":
                engine_path = "dsv4"
            elif generator_cls == "BatchGenerator":
                engine_path = "batched"
            else:
                engine_path = "none" if generator_cls is None else "custom"
            base = dict(base)
            base["engine_path"] = engine_path
            base["batch_generator"] = {
                "class": generator_cls,
                "path": engine_path,
                "single_active_decode": engine_path == "single_active",
                "max_num_seqs": self.config.max_num_seqs,
            }
            if generator_cls in {"SingleBatchGenerator", "DSV4BatchGenerator"}:
                base["batch_generator"].update(
                    {
                        "prompt_snapshot_max_bytes": getattr(
                            self.batch_generator, "prompt_snapshot_max_bytes", None
                        ),
                        "prompt_snapshot_last_estimated_bytes": getattr(
                            self.batch_generator,
                            "prompt_snapshot_last_estimated_bytes",
                            0,
                        ),
                        "prompt_snapshot_oversize_skips": getattr(
                            self.batch_generator,
                            "prompt_snapshot_oversize_skips",
                            0,
                        ),
                        "prompt_snapshot_headroom_skips": getattr(
                            self.batch_generator,
                            "prompt_snapshot_headroom_skips",
                            0,
                        ),
                        "prompt_snapshot_last_headroom_bytes": getattr(
                            self.batch_generator,
                            "prompt_snapshot_last_headroom_bytes",
                            0,
                        ),
                    }
                )
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
            nbytes = int(getattr(cache, "total_nbytes", 0) or 0)
            if nbytes <= 0 and store is not None:
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
            result = {
                "entries": entries,
                "max_entries": int(max_entries) if max_entries else 0,
                "max_bytes": getattr(cache, "max_bytes", None),
                "max_bytes_mb": (
                    round(cache.max_bytes / (1024 * 1024), 2)
                    if getattr(cache, "max_bytes", None) is not None
                    else None
                ),
                "nbytes": nbytes,
                "nbytes_mb": round(nbytes / (1024 * 1024), 2),
                "ram_enabled": bool(getattr(cache, "ram_enabled", False)),
            }
            disk = getattr(cache, "_disk", None)
            result["disk_enabled"] = disk is not None
            result["storage"] = (
                "ram_and_ssd"
                if result["ram_enabled"] and disk is not None
                else "ram_only"
                if result["ram_enabled"]
                else "ssd_only"
                if disk is not None
                else "disabled"
            )
            if disk is not None:
                try:
                    result["disk"] = disk.stats()
                except Exception as _disk_e:
                    result["disk"] = {
                        "enabled": True,
                        "error": str(_disk_e),
                    }
            return result
        except Exception as _e:
            return {
                "entries": 0,
                "max_entries": 0,
                "nbytes": 0,
                "ram_enabled": False,
                "storage": "disabled",
                "error": str(_e),
            }

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
            self.block_aware_cache.clear(force=True)
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
