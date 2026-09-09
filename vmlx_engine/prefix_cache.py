# SPDX-License-Identifier: Apache-2.0
# Base prefix cache from waybarrios/vllm-mlx. Block-aware prefix cache,
# MLA H=1 head validation, QuantizedKVCache support, hybrid SSM cumulative
# state handling, and gen_prompt_len stripping added by Jinho Jang
# (eric@jangq.ai) for vMLX (github.com/jjang-ai/vmlx).
"""
Prefix Cache Manager for vmlx-engine.

Wraps mlx-lm's LRUPromptCache to provide prefix caching functionality,
allowing reuse of computed KV cache for common prompt prefixes.

This module provides two implementations:
- PrefixCacheManager: Original trie-based LRU cache (for backward compatibility)
- BlockAwarePrefixCache: Block-based cache with PagedCacheManager integration
"""

import copy
import hashlib
import importlib.metadata
import pathlib
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .paged_cache import BlockTable, PagedCacheManager, compute_block_hash
from .cache_key import (
    CACHE_EXTRA_SCOPES_KEY,
    cache_extra_keys_for_token_range,
    canonical_cache_extra_marker,
)

logger = logging.getLogger(__name__)

_CACHE_HASH_DEBUG = os.environ.get("VMLX_CACHE_HASH_DEBUG", "") == "1"

# Bump this when the token->cache-state contract changes for paged prefix
# caches or their block-level L2 disk namespace. 2026-05-03 changed paged
# stores to index truncated N-1 prompt cache state under N-1 prompt-token
# keys so the last prompt token is always re-fed on cache hit. Older block
# disk entries were indexed under full-N keys and can replay unrelated text
# after restart, so every family must use a fresh namespace, not just DSV4.
# 2026-05-05 v3 bump: scheduler.py:1964 DSV4 SWA+CSA+HCA truncation guard
# now refuses to store post-generation DeepseekV4Cache when current_len >
# target_len. Old block-disk entries (v2) were stored without the guard
# and contain post-generation rotated/cumulative state that decodes
# garbage on hit. This bump invalidates ALL families' L2 disk caches,
# forcing a clean rebuild on next run. DSV4 specifically also gets a new
# v7 schema tag below to invalidate v6 entries that were stored before
# the guard landed.
# 2026-05-25 v4 bump: Gemma4/mixed-SWA block records now preserve
# RotatingKVCache tags and meta_state through paged/L2 reconstruction.
# Older v3 records can restore sliding-window layers as plain KVCache,
# causing slow and semantically wrong cache-hit decode.
# 2026-05-25 v5 bump: mixed-SWA same-process cache hits keep resident
# rotating/full-attention block data so immediate repeats cannot hit
# partially written or stale disk-only blocks with crossed layer shapes.
# 2026-06-06 v6 bump: MiMo V2 asymmetric full/SWA KV uses
# num_key_value_heads=4 for full layers and swa_num_key_value_heads=8 for
# rotating SWA layers. Older VLM extraction sliced all layers to the primary
# count before storage, so old L2 blocks can restore invalid 4-head rotating
# caches. Miss them cleanly.
# 2026-07-22 v9 bump: generation prompt side-keys and mixed-SWA clean
# prompt-boundary re-prefill changed the causal boundary for thinking/tool
# turns. Older v8 L2 blocks can partially restore a stale first block and poison
# Laguna/other mixed-SWA tool selection with gibberish before the new clean
# prompt-boundary block is written. Put current builds in a fresh block-disk
# namespace instead of asking users to manually delete old caches.
# 2026-07-22 v10 bump: Qwen native multi-tool continuation now recognizes the
# valid tool-result + follow-up-user history shape.  That changes the rendered
# prompt contract before the first reusable block.  Development builds that
# wrote v9 blocks before the continuation fix can otherwise restore a stale
# prefix and emit malformed native tool markup (live reproduction: a truncated
# ``<_command>`` span instead of the required second function call).  Release
# upgrades must miss those blocks cleanly; users must not need a manual clear.
# 2026-07-22 v11 bump: mixed-SWA RotatingKVCache is now stored as one exact,
# bounded terminal temporal-window checkpoint. Older records distributed slices
# from different logical snapshots across reusable blocks, allowing a long
# Laguna chain to combine offset=381 metadata with only 191/277 physical tokens
# at a 3712/3798-token boundary. Those states are causally incomplete and can
# crash the next decode with "Negative dimensions not allowed". Force old L2
# records to miss; do not pad or reinterpret them.
# 2026-08-23 v12 bump: multimodal payload identity is now introduced at each
# causal media placeholder instead of globally salting the root block. This
# lets unchanged text and earlier image/video state dedupe across connected
# turns while keeping the placeholder block and every descendant isolated by
# exact media bytes. Old global-salt records are safe but unreachable under the
# new hash contract, so isolate them explicitly in packaged builds too.
PAGED_CACHE_SCHEMA_VERSION = (
    "paged_n1_keys_v12_qwen_tool_rotating_media_causal_scopes"
)

# DSV4 v10 delta records remain wire-compatible, but the engine now donates an
# explicitly stamped full-block checkpoint immediately before a partial request
# terminal. Scope RAM/L2 namespaces so older pending-only 256-token blocks cannot
# deduplicate over the upgraded payload and silently preserve the short-prefix
# miss that this policy fixes.
DSV4_APPEND_SAFE_CHECKPOINT_POLICY = "full_block_v1"

_LOOPED_CACHE_LAYOUT = "looped_kv_v1"
_MTP_PREFIX_SNAPSHOT_MAX_ENTRIES = 16


def _resolve_runtime_cache_fingerprint() -> str:
    parts: List[str] = []
    try:
        from . import __version__ as engine_version
    except Exception:
        engine_version = "unknown"
    parts.append(f"vmlx_engine={engine_version}")
    for package in ("jang", "mlx", "mlx-lm", "mlx-vlm"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        except Exception:
            version = "unknown"
        parts.append(f"{package}={version}")
    source_id = _resolve_source_checkout_id()
    if source_id:
        parts.append(f"src={source_id}")
    return "runtime_cache=" + ",".join(parts)


def _resolve_source_checkout_id() -> str:
    """Content id for the engine sources, but ONLY in a source checkout.

    Package versions alone cannot separate two runtimes during development: the
    version string sits still while cache/model math changes underneath it, so
    L2 blocks written by the previous build keep matching and are replayed as if
    valid. That is not theoretical — it was reproduced live: with a stale
    block-cache, a warm hit answered a DIFFERENT question than the identical
    cold request (confidently, deterministically, 6/6), and clearing the cache
    made warm agree with cold again.

    Hashing ~300 source files costs ~180 ms, which no released user should pay,
    and they do not need to: a release bumps the version, which already changes
    the fingerprint. So this only runs when the package sits in a git checkout.
    """
    try:
        package_dir = pathlib.Path(__file__).resolve().parent
        if not (package_dir.parent / ".git").exists():
            return ""
        digest = hashlib.sha256()
        for path in sorted(package_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.name.encode("utf-8", "replace"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:12]
    except Exception:
        # Never let fingerprinting break startup; falling back to the
        # version-only string preserves the previous behaviour.
        return ""


# Resolved exactly once at import. Package versions cannot change inside a
# running process, but importlib.metadata re-reads dist-info from disk on
# every call — under mid-session resource pressure (fd exhaustion / memory
# pressure at very long contexts) those reads start failing, every package
# resolves to "unknown", and the flipped fingerprint rejects every stored
# L2 block as a mismatch (live incident 2026-08-07: 547k-token chain stored
# seconds earlier cold-missed 608k tokens). Freezing the string at import
# makes the process's cache identity immune to runtime resource failures.
_RUNTIME_CACHE_FINGERPRINT = _resolve_runtime_cache_fingerprint()


def runtime_cache_fingerprint() -> str:
    """Fingerprint runtime packages that define cache tensor semantics.

    Disk L2 is intentionally reusable across same-version process restarts.
    It must not, however, silently cross an app/runtime update when a cache
    schema string was not bumped. Keep this compact string inside cache
    namespaces and model keys so old blocks miss cleanly after package drift.
    """
    return _RUNTIME_CACHE_FINGERPRINT


def _looped_cache_identity_parts(model: Any) -> List[str]:
    """Return path-free identity fields for validated looped KV layouts.

    Nanbeige reuses 22 layer modules for two forward loops, but owns 44
    independent prompt-cache slots.  ``num_hidden_layers`` alone therefore
    cannot distinguish its native cache from a stock 22-slot cache. The loader
    joins config.jang_runtime, jang_config.runtime, and native cache facts,
    then stamps this validated contract on the loaded model.
    """

    contract = getattr(model, "_vmlx_looped_cache_contract", None)
    if (
        not isinstance(contract, dict)
        or str(contract.get("cache_layout") or "").strip().lower()
        != _LOOPED_CACHE_LAYOUT
    ):
        return []

    parts = [f"cache_layout={_LOOPED_CACHE_LAYOUT}"]
    values = {
        name: value
        for name in ("num_hidden_layers", "num_loops", "cache_slots")
        if isinstance((value := contract.get(name)), int)
        and not isinstance(value, bool)
        and value > 0
    }
    for name in ("num_loops", "cache_slots"):
        if name in values:
            parts.append(f"{name}={values[name]}")
    if "num_hidden_layers" in values and "num_loops" in values:
        layers = values["num_hidden_layers"]
        loops = values["num_loops"]
        parts.append(f"looped_cache_shape={layers}x{loops}={layers * loops}")
    return parts


def looped_cache_identity_scope(model: Any) -> str:
    """Serialize validated loop-cache identity for persistent namespaces."""

    return ":".join(_looped_cache_identity_parts(model))


def expected_cache_layer_count(model, hybrid_num_layers=None):
    """Number of cache-BEARING layers, for the L2 wrong-model validator.

    Not ``num_hidden_layers``: some hybrids have blocks that own no cache state
    (Nemotron-H is the live example, 52 blocks but 29 cache entries), so a
    validator comparing against the raw count rejects correct records.

    Shared because omitting it is silent. ``BlockDiskStore`` treats
    ``expected_num_layers=None`` as "skip the check", and the MLLM scheduler
    constructed its store without the argument — so the one validator that
    catches a wrong-model record was disarmed on exactly the path whose
    namespace was also the weaker one.
    """
    if hybrid_num_layers:
        return int(hybrid_num_layers)
    if hasattr(model, "make_cache"):
        try:
            cache = model.make_cache() or []
            if len(cache) > 0:
                return len(cache)
        except Exception:
            pass
    for _attr in ("args", "config"):
        _cfg = getattr(model, _attr, None)
        if _cfg:
            _ln = getattr(_cfg, "num_hidden_layers", 0)
            if _ln:
                return int(_ln)
    return None


def build_block_cache_namespace(
    *,
    model,
    model_path: str,
    quant_tag: str,
    tq_native_tag: str,
    smelt_enabled: bool = False,
    smelt_pct=None,
    tq_enabled: bool = False,
    kv_quant_bits=None,
    dsv4_scope: str = "",
    zaya_scope: str = "",
) -> str:
    """The ONE recipe for an L2 block-cache namespace.

    Block hashes are tokens+parent only (``paged_cache.py``), so model identity
    lives ENTIRELY in this namespace. Anything omitted here is a way for one
    model's KV to be served to another.

    It existed twice. The text scheduler's copy grew a ``bundle=`` weight/config
    fingerprint precisely so an in-place bundle replacement could not refault
    stale tensors; the MLLM scheduler's copy never did, and also constructed its
    BlockDiskStore without ``expected_num_layers``, which disarms the
    wrong-model record validator (``block_disk_store.py`` treats None as "skip
    the check"). So re-quantizing a VLM bundle in place -- same path, new
    weights, which is the canonical-swap workflow -- kept the same namespace and
    replayed KV computed by the old weights. The per-record fingerprint does not
    save it: that covers RUNTIME drift, not MODEL drift.

    Callers pass their own dsv4/zaya scopes because those describe cache
    SCHEMA, not identity, and only one family sets each.
    """
    bundle_cache_key = compute_model_cache_key(
        model,
        model_path=model_path,
        smelt_enabled=smelt_enabled,
        smelt_pct=smelt_pct,
        tq_enabled=tq_enabled,
        kv_quant_bits=kv_quant_bits,
    )
    scope = (
        f"{model_path}:quant={quant_tag}"
        f":tq_native={tq_native_tag}"
        f":bundle={bundle_cache_key}"
        f":paged_cache_schema={PAGED_CACHE_SCHEMA_VERSION}"
        f":{runtime_cache_fingerprint()}"
        f"{dsv4_scope}"
        f"{zaya_scope}"
    )
    looped_scope = looped_cache_identity_scope(model)
    return f"{scope}:{looped_scope}" if looped_scope else scope


@lru_cache(maxsize=16)
def _qwen4_native_artifact_digest(files: tuple) -> str:
    digest = hashlib.sha256()
    for name, _size, _mtime in files:
        path = pathlib.Path(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


_QWEN4_GDN_MATH_ABI = "stock-lanes32-v2"
_QWEN4_CHECKPOINT_MATH_ABI = "stock-segments-v2"
_QWEN4_VERIFY_MATH_ABI = "stock-order-sparse-qk-absolute16-pv-v5"


def _qwen4_native_artifact_identity() -> str:
    """Identify native cache math without loading an extension or GPU stream."""
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "native_extensions" / "qsa_kernels" / "mtplx_qsa_kernels"
    )
    try:
        if not any(source.glob("_ext*.so")):
            try:
                distribution = importlib.metadata.distribution("mtplx_qsa_kernels")
            except importlib.metadata.PackageNotFoundError:
                return "unavailable"
            source = pathlib.Path(distribution.locate_file("mtplx_qsa_kernels"))
        paths = sorted(
            p for p in source.iterdir()
            if p.suffix in {".so", ".dylib", ".metallib"}
        )
        if not paths:
            return "unavailable"
        files = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
        return _qwen4_native_artifact_digest(files)
    except OSError:
        # Do not share an unknown artifact with the explicitly absent lane.
        return f"unreadable-process-{os.getpid()}"


def compute_model_cache_key(
    model: Any,
    model_path: Optional[str] = None,
    smelt_enabled: bool = False,
    smelt_pct: Optional[float] = None,
    tq_enabled: bool = False,
    kv_quant_bits: int = 0,
) -> str:
    """
    Compute a stable, content-derived cache key for a loaded model.

    Replaces the legacy `id(model)` key (F6 / Concern #6) which:
    - Changes on JIT reload (cache becomes orphaned, never reachable)
    - Doesn't survive across processes for L2 disk cache lookups
    - Allows trie pollution if a session swaps models in-process

    Mixes in loader fingerprint (A4 Concern #1) so two sessions on the SAME
    model with DIFFERENT loader configs (smelt %, TQ, JANG quant bits) NEVER
    share trie entries — divergent K/V tensors otherwise produce silent
    corruption on cross-session fetch.

    Components:
    - Architecture id: model class + num_hidden_layers + key arch fields
    - Path + mtime: catches in-place edits to config.json / jang_config.json
    - Loader flags: smelt mode, smelt %, TQ enabled, KV quant bits

    Returns the first 32 hex chars of SHA-256 (16 bytes — collision-safe for
    realistic per-process model counts; trie key compares are O(1) hash).

    Falls back to id(model) string if any component is unavailable, so this
    is strictly additive over the previous behaviour.
    """
    parts: List[str] = []

    # 1. Architecture identity (cheap and safe even if path is unknown)
    projection_layout = getattr(model, "_vmlx_attention_projection_layout", None)
    if isinstance(projection_layout, str) and projection_layout:
        parts.append(f"attention_projection_layout={projection_layout}")
    try:
        parts.append(type(model).__module__ + "." + type(model).__name__)
        for attr in ("args", "config"):
            cfg = getattr(model, attr, None)
            if cfg is None:
                continue
            for f in (
                "model_type",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "hidden_size",
                "vocab_size",
                "kv_lora_rank",
            ):
                v = getattr(cfg, f, None)
                if v is not None and not callable(v):
                    parts.append(f"{f}={v}")
            break
    except Exception:
        pass

    # Looped transformers can own more cache slots than their shared module
    # count. Persisted prefix/L2 records must bind that actual runtime layout,
    # not only ``num_hidden_layers``.
    parts.extend(_looped_cache_identity_parts(model))

    # 2. Path + mtime — catches edited config.json / jang_config.json
    if model_path:
        parts.append(f"path={model_path}")
        try:
            for fname in ("config.json", "jang_config.json"):
                p = os.path.join(model_path, fname)
                if os.path.exists(p):
                    # Sub-second config edits share an integer mtime, so mix
                    # fractional mtime + size too (2026-07-10 audit High-2).
                    parts.append(f"{fname}_mtime={os.path.getmtime(p):.6f}")
                    parts.append(f"{fname}_size={os.path.getsize(p)}")
        except Exception:
            pass

        # 2b. Weight-artifact fingerprint — replacing weight shards in place
        # under the same path/config MUST invalidate the L2 disk cache
        # (2026-07-10 audit High-2). Hash the safetensors index, or absent an
        # index each shard's size+mtime. Cheap: no tensor bytes are read.
        try:
            import hashlib as _hl

            wsig = _hl.sha256()
            hashed = False
            index_p = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_p):
                with open(index_p, "rb") as _f:
                    index_bytes = _f.read()
                wsig.update(index_bytes)
                wsig.update(str(os.path.getsize(index_p)).encode())
                # The index normally stays byte-identical when a conversion or
                # download replaces shards in place. Bind the namespace to the
                # referenced files as well, without reading multi-GB tensor
                # payloads. Size plus ns-resolution mtime is stable across
                # process restarts while still changing for a same-path shard
                # rewrite; inode/ctime are intentionally excluded because a
                # faithful copy can change them without changing the bundle.
                try:
                    import json as _json

                    index_data = _json.loads(index_bytes)
                    weight_map = (
                        index_data.get("weight_map")
                        if isinstance(index_data, dict)
                        else None
                    )
                    shard_names = sorted(
                        {
                            str(name)
                            for name in (weight_map or {}).values()
                            if isinstance(name, str) and name
                        }
                    )
                    for shard_name in shard_names:
                        shard_path = os.path.join(model_path, shard_name)
                        try:
                            shard_stat = os.stat(shard_path)
                            shard_identity = (
                                f"{shard_name}:{shard_stat.st_size}:"
                                f"{shard_stat.st_mtime_ns}"
                            )
                        except OSError:
                            shard_identity = f"{shard_name}:missing"
                        wsig.update(b"\0")
                        wsig.update(shard_identity.encode())
                except Exception:
                    # The index bytes still participate in the signature. A
                    # malformed index is rejected later by the loader; cache
                    # identity must not turn that unrelated error into startup
                    # failure here.
                    pass
                hashed = True
            else:
                for fn in sorted(
                    f for f in os.listdir(model_path)
                    if f.endswith(".safetensors")
                ):
                    sp = os.path.join(model_path, fn)
                    wsig.update(
                        f"{fn}:{os.path.getsize(sp)}:"
                        f"{os.path.getmtime(sp):.6f}".encode()
                    )
                    hashed = True
            if hashed:
                parts.append(f"weights={wsig.hexdigest()[:16]}")
        except Exception:
            pass

    # 3. Loader fingerprint — A4 Concern #1
    parts.append(f"paged_cache_schema={PAGED_CACHE_SCHEMA_VERSION}")
    parts.append(runtime_cache_fingerprint())
    parts.append(f"smelt={1 if smelt_enabled else 0}")
    if smelt_enabled and smelt_pct is not None:
        parts.append(f"smelt_pct={float(smelt_pct):.4f}")
    parts.append(f"tq={1 if tq_enabled else 0}")
    if tq_enabled:
        # TurboQuant's decoder emits float32 irrespective of the source cache
        # dtype. codec_config_v2 records the original attention dtype and keys
        # the namespace by every model-owned/Auto codec field.  This also keeps
        # legacy uncalibrated 3-bit L2 records outside the correctness-first
        # 8-bit Auto namespace.
        parts.append("tq_storage_schema=codec_config_v2")
        make_cache = getattr(model, "make_cache", None)
        tq_signature = getattr(
            make_cache, "_vmlx_tq_storage_signature", "legacy_unspecified"
        )
        parts.append(f"tq_storage_signature={tq_signature}")
    parts.append(f"kvq={int(kv_quant_bits or 0)}")

    # Opt-in Qwen4 math paths can differ in floating-point accumulation.
    # Separate persisted state across configurations even within one release.
    if any(p in {"model_type=qwen4_exp", "model_type=qwen4_exp_text"} for p in parts):
        for flag in (
            "VMLX_QWEN4_GDN_BLOCKED_PREFILL",
            "VMLX_QWEN4_VERIFY_SDPA",
            "VMLX_QWEN4_PREFILL_DIRECT",
            "VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS",
            "VMLX_QWEN4_ALIGNED_MOE_PREFILL",
        ):
            value = os.environ.get(flag, "0")
            enabled = (
                value.lower() in {"1", "true", "yes", "on"}
                if flag in {"VMLX_QWEN4_GDN_BLOCKED_PREFILL", "VMLX_QWEN4_VERIFY_SDPA"}
                else value == "1"
            )
            parts.append(f"{flag}={int(enabled)}")
            if enabled and flag == "VMLX_QWEN4_ALIGNED_MOE_PREFILL":
                from vmlx_engine.metal.qwen4_aligned_moe_prefill import MATH_ABI
                parts.append("qwen4_aligned_moe_math=" + MATH_ABI)
            if enabled and flag == "VMLX_QWEN4_GDN_BLOCKED_PREFILL":
                parts.append("qwen4_gdn_math=" + _QWEN4_GDN_MATH_ABI)
                parts.append(
                    "qwen4_gdn_block_t="
                    + os.environ.get("VMLX_QWEN4_GDN_BLOCKED_PREFILL_TB", "auto")
                )
            if enabled and flag == "VMLX_QWEN4_VERIFY_SDPA":
                parts.append("qwen4_verify_math=" + _QWEN4_VERIFY_MATH_ABI)
            if enabled and flag == "VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS":
                parts.append("qwen4_checkpoint_math=" + _QWEN4_CHECKPOINT_MATH_ABI)
            if enabled and flag == "VMLX_QWEN4_PREFILL_DIRECT":
                # A source build is optional: distinguish absent, rebuilt and
                # installed artifacts before reusing persisted floating-point
                # state. This is read-only and never imports a Metal module.
                parts.append("qwen4_qsa_native=" + _qwen4_native_artifact_identity())

    # DSV4 cache correctness depends on runtime cache shape. Keep these in
    # the model key so L1/L2 prefix cache entries never cross between SWA-only
    # and tri-mode SWA+CSA/HCA, or between different DSV4 composite-cache
    # schema versions.
    if any(p == "model_type=deepseek_v4" for p in parts):
        parts.append(f"dsv4_long_ctx={os.environ.get('DSV4_LONG_CTX', '0')}")
        parts.append(f"dsv4_pool_quant={os.environ.get('DSV4_POOL_QUANT', '')}")
        activation_qat = str(
            os.environ.get("DSV4_ACTIVATION_QAT", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Activation-QAT changes the values written into attention KV and
        # compressor/indexer pools. Never restore an L1/L2 state produced by
        # the opposite graph even when model weights and prompt tokens match.
        parts.append(f"dsv4_activation_qat={1 if activation_qat else 0}")
        # v5: DSV4 runtime uses PR #1195 flat-pool CSA/HCA masks plus
        # chunked prefill. Bump the schema so old blocks captured by the
        # pre-v5 L*topk expansion path can never replay into the fixed
        # runtime.
        #
        # v4: paged/block L2 stores DSV4 cache data under N-1 prompt-token
        # keys so the last prompt token is always re-fed on prefix hits. This
        # intentionally invalidates older v2 disk blocks that were keyed by
        # the full prompt despite holding truncated cache state.
        parts.append("dsv4_cache_schema=deepseek_v4_v10_delta")
        parts.append(
            "dsv4_append_safe_checkpoint="
            f"{DSV4_APPEND_SAFE_CHECKPOINT_POLICY}"
        )

    if not parts:
        # Defensive: fall back to identity
        return f"id_{id(model):x}"

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]
    return digest


def is_hybrid_ssm_cache(prompt_cache: Optional[List[Any]]) -> bool:
    """
    Detect if a cache list contains any cumulative SSM layer (MambaCache /
    ArraysCache / BatchMambaCache). Used to short-circuit `mode=longer` trim
    paths (F2): cumulative SSM state cannot be rewound by `cache.trim(N)`,
    so a longer-prefix match must downgrade to a miss for hybrid models.

    Inline structural detection — no model_inspector dependency. Walks layer
    objects looking for either:
    - Class name in the cumulative cache set, OR
    - `.cache` attribute that is a list (MambaCache/ArraysCache shape)
      AND no positional `.keys/.values` attributes
    """
    if not prompt_cache:
        return False
    cumulative_cls = {"MambaCache", "BatchMambaCache", "ArraysCache"}
    for layer in prompt_cache:
        cls = type(layer).__name__
        if cls in cumulative_cls:
            return True
        if isinstance(layer, dict) and layer.get("class_name", "") in cumulative_cls:
            return True
        # Structural fallback for nested CacheList wrappers
        sub = getattr(layer, "caches", None)
        if isinstance(sub, (list, tuple)):
            for s in sub:
                if type(s).__name__ in cumulative_cls:
                    return True
    return False


@dataclass
class CacheEntry:
    """Entry in the prefix cache."""

    prompt_cache: List[Any]  # The cached KV state
    count: int  # Reference count for sharing
    cache_type: str = "assistant"  # "system" | "user" | "assistant" — eviction priority
    nbytes: int = 0  # Estimated memory footprint, computed at store time


# Type priority for LRU eviction. Lower-priority types are evicted first.
# Order: assistant (least sticky) → user → system (pinned, evicted last).
# This mirrors mlx-lm 0.31.2 LRUPromptCache.CacheOrder so cross-session
# system prompts stay cached across users.
_CACHE_TYPE_PRIORITY: Tuple[str, ...] = ("assistant", "user", "system")


@dataclass
class PrefixCacheStats:
    """Statistics for prefix cache performance."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


class PrefixCacheManager:
    """
    Manages prefix caching for vmlx-engine using a trie-based LRU cache.

    This implementation is inspired by mlx-lm's LRUPromptCache but adapted
    for vmlx-engine's batching architecture.

    The cache stores KV states keyed by token sequences, allowing:
    - Exact match: Full prompt found in cache
    - Shorter match: Partial prefix found, process remaining tokens
    - Longer match: Cached prefix longer than request, trim excess

    Example:
        cache_manager = PrefixCacheManager(model, max_entries=100)

        # Check for cached prefix
        cache, remaining_tokens = cache_manager.fetch_cache(tokens)
        if cache:
            # Use cached KV, only process remaining_tokens
            pass

        # After generation, store cache for reuse
        cache_manager.store_cache(full_tokens, prompt_cache)
    """

    def __init__(
        self,
        model: Any,
        max_entries: int = 100,
        max_bytes: Optional[int] = None,
        model_path: Optional[str] = None,
        smelt_enabled: bool = False,
        smelt_pct: Optional[float] = None,
        tq_enabled: bool = False,
        kv_quant_bits: int = 0,
    ):
        """
        Initialize the prefix cache manager.

        Args:
            model: The MLX model (used for cache key identification)
            max_entries: Maximum number of cached entries before LRU eviction
            max_bytes: Optional global byte budget. When set, eviction also
                triggers when total cached bytes exceed this. None = unlimited.
            model_path, smelt_enabled, smelt_pct, tq_enabled, kv_quant_bits:
                Loader fingerprint inputs (F6 + A4 Concern #1). Two sessions
                with divergent loader configs get different model keys and
                never share trie entries — prevents silent K/V corruption.
        """
        self.model = model
        # Content-derived stable key (replaces fragile id(model) — F6).
        # Survives JIT reload, prevents cross-config trie pollution, and
        # mixes loader fingerprint so smelt/TQ/JANG variants don't collide.
        self.model_key = compute_model_cache_key(
            model,
            model_path=model_path,
            smelt_enabled=smelt_enabled,
            smelt_pct=smelt_pct,
            tq_enabled=tq_enabled,
            kv_quant_bits=kv_quant_bits,
        )
        self.max_size = max_entries
        self.max_bytes: Optional[int] = max_bytes

        # Trie-based cache: nested dicts with token keys
        # Structure: {model_key: {token1: {token2: {..., "cache": CacheEntry}}}}
        self._cache: Dict[Any, Dict] = {}

        # Per-cache-type LRU tracking. Each type has its own OrderedDict;
        # eviction pops from the lowest-priority non-empty type first
        # (assistant → user → system). System entries are evicted last so
        # shared system prompts persist across users/sessions.
        # Key shape: (model_key, tuple(tokens)) → True (presence)
        self._lru_by_type: Dict[str, "OrderedDict[Tuple[Any, tuple], bool]"] = {
            t: OrderedDict() for t in _CACHE_TYPE_PRIORITY
        }

        # Per-type byte counters and total
        self._n_bytes: int = 0
        self._n_bytes_by_type: Dict[str, int] = {t: 0 for t in _CACHE_TYPE_PRIORITY}

        # Statistics
        self.stats = PrefixCacheStats()

    def _scoped_model_key(self, cache_extra_keys: Optional[Any] = None) -> Any:
        from .cache_key import canonical_cache_extra_marker

        marker = canonical_cache_extra_marker(cache_extra_keys)
        if marker is None:
            return self.model_key
        return (self.model_key, "__extra__", marker)

    def _search(
        self,
        tokens: List[int],
        scoped_model_key: Optional[Any] = None,
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]], int]:
        """
        Search for cached prefix matching tokens.

        Returns:
            Tuple of (exact, shorter, longer, common_prefix_len)
            - exact: Tokens if exact match found
            - shorter: Tokens of shorter cached prefix
            - longer: Tokens of longer cached prefix
            - common_prefix_len: Length of common prefix with longer match
        """
        model_key = self.model_key if scoped_model_key is None else scoped_model_key
        if model_key not in self._cache:
            return None, None, None, 0

        current = self._cache[model_key]
        path = []

        # Traverse trie following token sequence
        for i, tok in enumerate(tokens):
            if tok not in current:
                # No match for this token
                # Check if we have a shorter prefix with cache
                if "cache" in current:
                    return None, list(path), None, 0
                return None, None, None, 0

            path.append(tok)
            current = current[tok]

        # Reached end of tokens
        if "cache" in current:
            # Exact match
            return list(tokens), None, None, 0

        # Check for longer cached prefix
        # BFS to find shortest extension with cache
        from collections import deque
        queue = deque([(current, list(path))])
        while queue:
            node, node_path = queue.popleft()
            if "cache" in node:
                return None, None, node_path, len(tokens)
            for tok, child in node.items():
                if tok != "cache":
                    queue.append((child, node_path + [tok]))

        return None, None, None, 0

    def fetch_cache(
        self,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> Tuple[Optional[List[Any]], List[int]]:
        """
        Find cached prefix for the given tokens.

        Args:
            tokens: Input token sequence

        Returns:
            Tuple of (cache, remaining_tokens)
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        self.stats.total_queries += 1
        tokens_tuple = tuple(tokens)
        scoped_model_key = self._scoped_model_key(cache_extra_keys)

        exact, shorter, longer, common_len = self._search(
            tokens,
            scoped_model_key=scoped_model_key,
        )

        if exact:
            # Exact match - return full cache
            cache_entry = self._get_cache_entry(exact, scoped_model_key)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                self._touch_lru(tokens_tuple, scoped_model_key)
                # Return reference directly — MLX arrays are immutable,
                # so sharing the cache is safe (no mutation possible).
                return cache_entry.prompt_cache, []

        if shorter:
            # Shorter prefix cached - return cache and remaining tokens
            cache_entry = self._get_cache_entry(shorter, scoped_model_key)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(shorter)
                self._touch_lru(tuple(shorter), scoped_model_key)
                remaining = tokens[len(shorter) :]
                # Return reference directly — MLX arrays are immutable,
                # so sharing the cache is safe (no mutation possible).
                return cache_entry.prompt_cache, remaining

        if longer:
            # Longer prefix cached - trim to match and return
            cache_entry = self._get_cache_entry(longer, scoped_model_key)
            if cache_entry:
                # Check if cache supports trimming
                prompt_cache = cache_entry.prompt_cache
                # F2 (R-003): hybrid SSM models cannot be trimmed — cumulative
                # SSM state at position L can't be rewound to L-k without
                # re-running the model. The auto-switch at scheduler.py:322 saves
                # the default-config user, but explicit `--no-memory-aware-cache`
                # users on hybrid models would otherwise hit silent corruption.
                # Downgrade to miss instead.
                if is_hybrid_ssm_cache(prompt_cache):
                    logger.debug(
                        "PrefixCacheManager: hybrid SSM longer-match downgraded "
                        "to miss (R-003 — cumulative state cannot be rewound)"
                    )
                elif self._can_trim_cache(prompt_cache):
                    trim_amount = len(longer) - len(tokens)
                    # Deep copy IS needed here: _trim_cache mutates the cache
                    # objects' offset/keys/values refs via .trim(), which would
                    # corrupt the stored entry. (Unlike the read-only fetch paths
                    # above where sharing is safe.)
                    trimmed_cache = self._trim_cache(
                        copy.deepcopy(prompt_cache), trim_amount
                    )
                    self.stats.hits += 1
                    self.stats.tokens_saved += len(tokens)
                    return trimmed_cache, []

        # No cache hit
        self.stats.misses += 1
        return None, tokens

    def store_cache(
        self,
        tokens: List[int],
        prompt_cache: List[Any],
        cache_type: str = "assistant",
        cache_extra_keys: Optional[Any] = None,
    ) -> None:
        """
        Store computed cache for future reuse.

        Args:
            tokens: Token sequence that was processed
            prompt_cache: The computed KV cache to store
            cache_type: One of "system", "user", or "assistant". Controls
                LRU eviction priority — system entries are pinned and evicted
                last so shared system prompts persist across users/sessions.
                Defaults to "assistant" for backward compatibility.
        """
        if not tokens:
            return
        if cache_type not in self._lru_by_type:
            cache_type = "assistant"

        tokens_tuple = tuple(tokens)
        scoped_model_key = self._scoped_model_key(cache_extra_keys)

        # Build trie path
        if scoped_model_key not in self._cache:
            self._cache[scoped_model_key] = {}

        current = self._cache[scoped_model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # Estimate bytes for this entry. Reuse memory_cache helper to support
        # KVCache, QuantizedKVCache, MambaCache, ArraysCache, CacheList, and
        # extracted-state dicts.
        try:
            from .memory_cache import estimate_kv_cache_memory  # local import: optional
            entry_nbytes = estimate_kv_cache_memory(prompt_cache)
        except Exception:
            entry_nbytes = 0

        key = (scoped_model_key, tokens_tuple)

        # Replace existing entry: drop old byte counters before re-tracking
        if "cache" in current:
            old: CacheEntry = current["cache"]
            self._n_bytes -= old.nbytes
            self._n_bytes_by_type[old.cache_type] = max(
                0, self._n_bytes_by_type[old.cache_type] - old.nbytes
            )
            self._remove_from_all_lru(key)
            current["cache"] = CacheEntry(
                prompt_cache=prompt_cache,
                count=old.count + 1,
                cache_type=cache_type,
                nbytes=entry_nbytes,
            )
        else:
            current["cache"] = CacheEntry(
                prompt_cache=prompt_cache,
                count=1,
                cache_type=cache_type,
                nbytes=entry_nbytes,
            )

        # Track bytes
        self._n_bytes += entry_nbytes
        self._n_bytes_by_type[cache_type] += entry_nbytes

        # Push to type-specific LRU (most recently used at the end)
        type_lru = self._lru_by_type[cache_type]
        type_lru[key] = True
        type_lru.move_to_end(key)

        # Evict if over entry count
        while self._total_lru_size() > self.max_size:
            if not self._evict_lru():
                break

        # Evict if over byte budget
        if self.max_bytes is not None:
            while self._n_bytes > self.max_bytes:
                if not self._evict_lru():
                    break

    def trim_to(self, n_bytes: int) -> int:
        """
        Evict entries until total cached bytes <= n_bytes.

        Used by the scheduler to reserve room for in-flight requests when
        operating with a global byte budget. Returns the number of entries
        evicted. n_bytes <= 0 clears the cache.
        """
        evicted = 0
        if n_bytes <= 0:
            count = self._total_lru_size()
            self.clear()
            return count
        while self._n_bytes > n_bytes:
            if not self._evict_lru():
                break
            evicted += 1
        return evicted

    def _total_lru_size(self) -> int:
        return sum(len(d) for d in self._lru_by_type.values())

    def _remove_from_all_lru(self, key: Tuple[Any, tuple]) -> None:
        for d in self._lru_by_type.values():
            if key in d:
                del d[key]

    def _get_cache_entry(
        self,
        tokens: List[int],
        scoped_model_key: Optional[Any] = None,
    ) -> Optional[CacheEntry]:
        """Get cache entry for given tokens."""
        model_key = self.model_key if scoped_model_key is None else scoped_model_key
        if model_key not in self._cache:
            return None

        current = self._cache[model_key]
        for tok in tokens:
            if tok not in current:
                return None
            current = current[tok]

        return current.get("cache")

    def _touch_lru(
        self,
        tokens_tuple: tuple,
        scoped_model_key: Optional[Any] = None,
    ) -> None:
        """Move entry to end of its type's LRU (most recently used). O(1)."""
        model_key = self.model_key if scoped_model_key is None else scoped_model_key
        key = (model_key, tokens_tuple)
        # Find the type that owns this key (entries live in exactly one bucket)
        for t in _CACHE_TYPE_PRIORITY:
            d = self._lru_by_type[t]
            if key in d:
                d.move_to_end(key)
                return
        # New entry — default to assistant bucket. Real type assigned at store.
        self._lru_by_type["assistant"][key] = True

    def _evict_lru(self) -> bool:
        """Evict the least recently used entry from the lowest-priority
        non-empty bucket. Returns True if an entry was evicted, False if all
        buckets were empty. Eviction order: assistant → user → system.
        """
        for t in _CACHE_TYPE_PRIORITY:
            d = self._lru_by_type[t]
            if d:
                (model_key, tokens_tuple), _ = d.popitem(last=False)
                # Untrack bytes BEFORE deleting the entry (lookup needs it)
                entry = self._get_cache_entry(
                    list(tokens_tuple),
                    scoped_model_key=model_key,
                )
                if entry is not None:
                    self._n_bytes -= entry.nbytes
                    self._n_bytes_by_type[entry.cache_type] = max(
                        0, self._n_bytes_by_type[entry.cache_type] - entry.nbytes
                    )
                self._delete_cache(model_key, list(tokens_tuple))
                self.stats.evictions += 1
                return True
        return False

    def _delete_cache(self, model_key: Any, tokens: List[int]) -> None:
        """Delete cache entry and clean up empty trie branches."""
        if model_key not in self._cache:
            return

        # Navigate to entry
        path = [(self._cache[model_key], None)]
        current = self._cache[model_key]

        for tok in tokens:
            if tok not in current:
                return
            path.append((current[tok], tok))
            current = current[tok]

        # Delete cache entry
        if "cache" in current:
            del current["cache"]

        # Clean up empty branches (bottom-up)
        for i in range(len(path) - 1, 0, -1):
            node, tok = path[i]
            parent, _ = path[i - 1]
            if not node:  # Empty dict
                del parent[tok]

    def _can_trim_cache(self, prompt_cache: List[Any]) -> bool:
        """Check if cache can be trimmed."""
        if not prompt_cache:
            return False
        # Check if first cache layer has is_trimmable method
        first_cache = prompt_cache[0]
        if hasattr(first_cache, "is_trimmable"):
            return first_cache.is_trimmable()
        return hasattr(first_cache, "trim")

    def _trim_cache(self, prompt_cache: List[Any], num_tokens: int) -> List[Any]:
        """Trim cache by removing num_tokens from the end."""
        for cache in prompt_cache:
            if hasattr(cache, "trim"):
                cache.trim(num_tokens)
        return prompt_cache

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including per-type byte usage."""
        d = self.stats.to_dict()
        d["nbytes"] = self._n_bytes
        d["nbytes_by_type"] = dict(self._n_bytes_by_type)
        d["entries_by_type"] = {
            t: len(self._lru_by_type[t]) for t in _CACHE_TYPE_PRIORITY
        }
        d["max_bytes"] = self.max_bytes
        return d

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PrefixCacheStats()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        for d in self._lru_by_type.values():
            d.clear()
        self._n_bytes = 0
        for t in self._n_bytes_by_type:
            self._n_bytes_by_type[t] = 0
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return self._total_lru_size()


# =============================================================================
# Block-Aware Prefix Cache (uses PagedCacheManager)
# =============================================================================



def _is_dsv4_cache_class(class_name: str) -> bool:
    return (class_name or "") in {"DeepseekV4Cache", "PoolQuantizedV4Cache"}


def _is_minimax_m3_cache_class(class_name: str) -> bool:
    """MiniMax-M3 MSA sparse layer cache (KVCache + append-only idx_keys).

    Unlike DSV4/ZAYA composite caches, M3's state is FULLY positional: keys,
    values, and idx_keys all grow append-only in lockstep on the sequence axis,
    so every block carries a complete, independently-sliceable payload. It is
    serialized as the 4-tuple ("minimax_m3", keys_slice, values_slice,
    idx_keys_slice) and reconstructed via restore_minimax_m3_sparse().
    """
    return (class_name or "") == "MiniMaxM3SparseCache"


def _model_is_minimax_m3(model: Any) -> bool:
    """Identify the M3 architecture, not merely its reusable cache container.

    Qwen4Exp QSA deliberately reuses ``MiniMaxM3SparseCache`` for its positional
    K/V/index payload.  That makes the payload serializer compatible, but it
    does *not* make Qwen's blocks subject to MiniMax-M3's prefill-matrix shape
    discriminator.  Classifying by the cache object alone salted a Qwen clean
    media boundary produced at 5,440 tokens differently from the 5,445-token
    warm request that was supposed to consume it, forcing a block-zero miss.
    """
    seen: set[int] = set()
    pending = [model]
    # Real wrapper chains are a few levels deep. A probe that keeps yielding
    # fresh objects on every unwrap (auto-creating proxies, test doubles) would
    # otherwise walk forever — the full suite hung here inside Scheduler.__init__
    # because MagicMock materializes a new child for each _model/model/
    # language_model access, so the frontier never drained.
    while pending and len(seen) < 32:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        if isinstance(config, dict):
            model_type = config.get("model_type") or ""
            text_config = config.get("text_config")
            if not model_type and isinstance(text_config, dict):
                model_type = text_config.get("model_type") or ""
        else:
            model_type = getattr(config, "model_type", "") if config else ""
            text_config = getattr(config, "text_config", None) if config else None
            if not model_type and text_config is not None:
                model_type = (
                    text_config.get("model_type", "")
                    if isinstance(text_config, dict)
                    else getattr(text_config, "model_type", "")
                )
        normalized = str(model_type or "").lower().replace("-", "_")
        if "minimax_m3" in normalized or "minimaxm3" in normalized:
            return True
        # An explicit non-M3 model type is authoritative. This is what keeps
        # qwen4_exp from inheriting M3 policy even when a wrapper, test module,
        # or reusable cache class happens to contain MiniMax naming.
        if normalized:
            pending.extend(
                getattr(candidate, attr, None)
                for attr in ("_model", "model", "language_model")
            )
            continue

        identity = (
            f"{type(candidate).__module__}.{type(candidate).__name__}"
        ).lower()
        if "minimax_m3" in identity or "minimaxm3" in identity:
            return True

        pending.extend(
            getattr(candidate, attr, None)
            for attr in ("_model", "model", "language_model")
        )
    return False


def _cache_data_has_minimax_m3(cache_data) -> bool:
    """Return whether extracted states contain MiniMax-M3 sparse cache data."""
    return any(
        isinstance(layer_state, dict)
        and _is_minimax_m3_cache_class(layer_state.get("class_name", ""))
        for layer_state in (cache_data or [])
    )


def _cache_data_has_dsv4(cache_data) -> bool:
    """Return True when extracted cache states include DSV4 composite layers."""
    try:
        for layer_state in cache_data or []:
            if not isinstance(layer_state, dict):
                continue
            if _is_dsv4_cache_class(layer_state.get("class_name", "")):
                return True
    except Exception:
        return False
    return False


def _cache_data_has_dsv4_deltas(cache_data) -> bool:
    """Return True for generator-captured DSV4 block-delta transports."""
    try:
        return bool(cache_data) and all(
            isinstance(layer_state, dict)
            and bool(layer_state.get("dsv4_block_records"))
            and bool(layer_state.get("dsv4_record_intervals"))
            for layer_state in cache_data
        )
    except Exception:
        return False


def _block_has_complete_dsv4_delta_interval(
    cache_data,
    *,
    start_token: int,
    end_token: int,
    expected_layers: Optional[int] = None,
    expected_transport: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Return whether one stored page is a complete native DSV4 interval.

    Non-anchor pages legitimately carry ``rotating_kv_pending`` for the two
    ratio-zero SWA layers. Every composite layer must still carry a native
    delta record with geometry matching this exact page. Generic
    ``deepseek_v4_pending``/``deepseek_v4`` pages are not interchangeable:
    appending native deltas to them creates a mixed chain which the restore
    contract correctly rejects.
    """
    entries = list(cache_data or ())
    if not entries or (
        expected_layers is not None and len(entries) != int(expected_layers)
    ):
        return False
    wanted_start = int(start_token)
    wanted_end = int(end_token)
    saw_delta = False
    if expected_transport is not None and len(expected_transport) != len(entries):
        return False
    for layer_index, entry in enumerate(entries):
        if not isinstance(entry, (tuple, list)) or not entry:
            return False
        expected = (
            expected_transport[layer_index]
            if expected_transport is not None
            else None
        )
        expected_class = (
            str(expected.get("class_name", ""))
            if isinstance(expected, dict)
            else ""
        )
        tag = entry[0]
        if tag == "deepseek_v4_delta_v1":
            if (
                len(entry) < 4
                or not isinstance(entry[1], dict)
                or entry[1].get("schema") != "deepseek_v4_block_delta_v1"
                or not isinstance(entry[2], str)
                or not isinstance(entry[3], dict)
                or (
                    expected_class
                    and entry[2] != expected_class
                )
            ):
                return False
            try:
                if (
                    int(entry[1].get("start_token", -1)) != wanted_start
                    or int(entry[1].get("end_token", -1)) != wanted_end
                ):
                    return False
            except (TypeError, ValueError):
                return False
            if isinstance(expected, dict):
                expected_meta = (
                    expected.get("compress_ratio"),
                    expected.get("sliding_window"),
                    bool(expected.get("pool_quant")),
                )
                actual_meta = (
                    entry[3].get("compress_ratio"),
                    entry[3].get("sliding_window"),
                    bool(entry[3].get("pool_quant")),
                )
                if actual_meta != expected_meta:
                    return False
            saw_delta = True
        elif tag == "rotating_kv":
            if (
                len(entry) < 7
                or (expected_class and "RotatingKVCache" not in expected_class)
            ):
                return False
            try:
                if int(entry[5]) != wanted_end:
                    return False
            except (TypeError, ValueError):
                return False
        elif tag == "rotating_kv_pending":
            if (
                len(entry) < 2
                or (expected_class and "RotatingKVCache" not in expected_class)
            ):
                return False
        else:
            return False
    return saw_delta


def _dsv4_delta_record_for_interval(
    layer_state: Dict[str, Any], start_idx: int, end_idx: int
):
    records = tuple(layer_state.get("dsv4_block_records") or ())
    intervals = tuple(layer_state.get("dsv4_record_intervals") or ())
    if len(records) != len(intervals):
        raise ValueError("DSV4 block records and interval map differ in length")
    wanted = (int(start_idx), int(end_idx))
    for interval, record in zip(intervals, records):
        if tuple(map(int, interval)) == wanted:
            return record
    raise ValueError(f"DSV4 block transport has no interval {wanted}")



def _block_payload_needs_native_residency(cache_data) -> bool:
    """Return True for path-dependent native cache records.

    These payloads are not interchangeable with ordinary positional KV.  When
    an explicit frugal override is active, keep a successful reconstruction in
    L1 long enough for same-process native reuse, then let the normal byte
    budget evict it.  Nested ``cache_list`` records are traversed without
    iterating array leaves.
    """
    native_tags = {
        "deepseek_v4",
        "deepseek_v4_pending",
        "deepseek_v4_delta_v1",
        "zaya_cca",
        "rotating_kv",
        "rotating_kv_pending",
    }

    def _has_native(value) -> bool:
        if not isinstance(value, (tuple, list)):
            return False
        if value and isinstance(value[0], str) and value[0] in native_tags:
            return True
        return any(_has_native(item) for item in value)

    try:
        return _has_native(cache_data)
    except Exception:
        return False


def _dsv4_delta_anchor_kind(entry) -> Optional[Tuple[bool, bool]]:
    """Return ``(periodic, terminal)`` for one anchored DSV4 delta entry."""
    if (
        not isinstance(entry, (tuple, list))
        or len(entry) < 2
        or entry[0] != "deepseek_v4_delta_v1"
        or not isinstance(entry[1], dict)
    ):
        return None
    anchor = entry[1].get("anchor")
    if not isinstance(anchor, dict):
        return None
    try:
        if int(anchor.get("tokens", -1)) != int(entry[1].get("end_token", -2)):
            return None
    except (TypeError, ValueError):
        return None
    return bool(anchor.get("periodic")), bool(anchor.get("terminal"))


def _dsv4_delta_anchor_is_append_safe(entry) -> bool:
    """Return True only for an explicitly stamped full-block DSV4 anchor."""
    if (
        not isinstance(entry, (tuple, list))
        or len(entry) < 2
        or entry[0] != "deepseek_v4_delta_v1"
        or not isinstance(entry[1], dict)
    ):
        return False
    record = entry[1]
    anchor = record.get("anchor")
    if not isinstance(anchor, dict) or anchor.get("append_safe") is not True:
        return False
    try:
        start = int(record.get("start_token", -1))
        end = int(record.get("end_token", -1))
        block_size = int(record.get("block_size", 0))
        anchor_tokens = int(anchor.get("tokens", -1))
    except (TypeError, ValueError):
        return False
    return bool(
        block_size > 0
        and start >= 0
        and end - start == block_size
        and start % block_size == 0
        and end % block_size == 0
        and anchor_tokens == end
        and _dsv4_delta_anchor_kind(entry) is not None
    )


# DSV4's generation rail is 3 tokens (<|Assistant|> + the think/answer marker).
# 4 leaves one token of margin without admitting a materially changed suffix.
_DSV4_DEFAULT_TERMINAL_ANCHOR_TAIL = 4


def _dsv4_terminal_anchor_tail_budget() -> int:
    """How many trailing tokens may follow a restored DSV4 terminal anchor.

    Defaults to DSV4's generation-rail length so the visible-answer pass can
    restore the anchor its own first pass just stored, instead of re-prefilling
    the whole tail to change the one rail token that differs.

    Proven byte-identical before being made the default: the same 127k query
    hashed to 9884a471bb9bc9a5 with the budget both on and off, while
    cached_tokens differed (127,472 vs 127,232), so the restore path was
    genuinely taken and still produced the same reasoning and content.
    Set VMLX_DSV4_TERMINAL_ANCHOR_TAIL=0 to restore the old <= 1 behaviour.
    """
    raw = os.environ.get("VMLX_DSV4_TERMINAL_ANCHOR_TAIL", "").strip()
    if not raw:
        return _DSV4_DEFAULT_TERMINAL_ANCHOR_TAIL
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _DSV4_DEFAULT_TERMINAL_ANCHOR_TAIL


def _block_has_complete_dsv4_delta_anchor(
    cache_data, *, expected_layers: Optional[int] = None, periodic_only: bool = False
) -> bool:
    """Validate an all-layer native DSV4 checkpoint in one block payload."""
    entries = list(cache_data or ())
    if expected_layers is not None and len(entries) != int(expected_layers):
        return False
    saw_delta = False
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or not entry:
            return False
        tag = entry[0]
        if tag == "deepseek_v4_delta_v1":
            saw_delta = True
            kind = _dsv4_delta_anchor_kind(entry)
            if kind is None or (periodic_only and not kind[0]):
                return False
        elif tag == "rotating_kv":
            # The two ratio-zero SWA layers use the existing exact rotating
            # window record. Periodicity is established by the composite
            # layer records at the same token boundary.
            if len(entry) < 7:
                return False
        else:
            return False
    return saw_delta


def _block_has_complete_dsv4_append_safe_anchor(
    cache_data,
    *,
    target_tokens: int,
    expected_layers: Optional[int] = None,
) -> bool:
    """Validate an all-layer aligned checkpoint donated for suffix growth."""
    entries = list(cache_data or ())
    if expected_layers is not None and len(entries) != int(expected_layers):
        return False
    saw_delta = False
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or not entry:
            return False
        tag = entry[0]
        if tag == "deepseek_v4_delta_v1":
            saw_delta = True
            if not _dsv4_delta_anchor_is_append_safe(entry):
                return False
        elif tag == "rotating_kv":
            if len(entry) < 7:
                return False
            try:
                expected = int(target_tokens)
                max_size = int(entry[3])
                offset = int(entry[5])
                idx = int(entry[6])
                key_shape = tuple(getattr(entry[1], "shape", ()))
                value_shape = tuple(getattr(entry[2], "shape", ()))
                seq_axis = (
                    2
                    if len(key_shape) == 4
                    else 1
                    if len(key_shape) == 3
                    else -1
                )
                expected_rows = min(expected, max_size)
            except (TypeError, ValueError, IndexError):
                return False
            if (
                expected <= 0
                or max_size <= 0
                or seq_axis < 0
                or key_shape != value_shape
                or int(key_shape[seq_axis]) != expected_rows
                or offset != expected
                or idx != expected_rows
            ):
                return False
        else:
            return False
    return saw_delta


def _to_numpy_tree(obj):
    """Convert nested MLX-array state to numpy, preserving tuple/list shape.

    MLX >= 0.32 removed the PEP-3118 buffer shim for bfloat16, so
    ``np.array(bf16)`` raises "'bfloat16' is not a valid PEP 3118 buffer
    format string" and the ENTIRE store silently fails (observed live:
    blocks_on_disk stayed 0 on every model after the 0.32.1 upgrade).  Cast
    through float32 — value-exact for bf16, and the loaders already restore
    the original dtype via astype.
    """
    import numpy as np

    if obj is None:
        return None
    if hasattr(obj, "__array__"):
        if hasattr(obj, "dtype") and "bfloat16" in str(getattr(obj, "dtype", "")):
            import mlx.core as mx

            return np.array(obj.astype(mx.float32))
        return np.array(obj)
    if isinstance(obj, tuple):
        return tuple(_to_numpy_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_to_numpy_tree(x) for x in obj]
    return obj


def _readonly_numpy_buffer_view(obj):
    """Expose a materialized MLX buffer to NumPy without a full-size copy.

    ``BlockAwarePrefixCache.store_cache`` keeps the model-owned cache state
    alive while it cuts independent per-block payloads. Positional source K/V
    tensors therefore only need a read-only CPU view at this stage. The block
    slicer materializes each exact payload independently; the block-disk store
    then queues evaluated MLX payloads as immutable NumPy memoryviews and copies
    only caller-owned NumPy inputs.

    This distinction is material on macOS unified memory.  Measured on the
    shipping MLX stack, ``np.array`` of one 64 MiB F16 tensor raised process RSS
    by 128 MiB and left that anonymous malloc high-water resident, while
    ``np.asarray`` raised it by 0 KiB.  A NumPy ``memoryview`` base keeps the MLX
    buffer alive even when the temporary producer reference is a BF16-to-F32
    bridge array.  Marking the view read-only protects the live cache from
    accidental NumPy mutation until the independent per-block payload owns its
    lifetime.
    """

    import numpy as np

    view = np.asarray(obj)
    view.setflags(write=False)
    return view


_DSV4_SHARED_POOL_KEYS = ("compressor_pool", "indexer_pool")


def _copy_dsv4_delta_record(record):
    """Copy a DSV4 block-delta record without duplicating pool payloads.

    Pool arrays (q8 segment tuples / bf16 rows) are immutable by contract:
    the live cache only appends, and compaction/slicing rebinds list entries
    to NEW arrays without mutating shared buffers. Deep-copying them doubled
    resident pool bytes during reconstruct (originals in the paged blocks +
    copies in the live cache) — linear in context, and it OOM'd Metal at
    ~183k tokens on a 128GB box (95GB weights leave ~12GB headroom). Share
    the arrays; copy only the bounded mutable remainder (anchor local
    window, metadata).
    """
    if not isinstance(record, dict):
        return _copy_mlx_tree(record)
    copied = {}
    for key, value in record.items():
        if key in _DSV4_SHARED_POOL_KEYS and isinstance(value, dict):
            shared = dict(value)
            segments = shared.get("segments")
            if isinstance(segments, list):
                shared["segments"] = list(segments)
            copied[key] = shared
        else:
            copied[key] = _copy_mlx_tree(value)
    return copied


def _wrap_cumulative_meta(meta, arrays):
    """Carry the original per-array dtypes beside a cumulative meta_state.

    The numpy store bridge casts bf16 state through fp32 (np.array(bf16)
    raises under MLX >= 0.32) and, unlike the KV branches, recorded no
    orig_dtype — so restart/L2-path hybrid restores handed the model FLOAT32
    conv/SSM states, which promote the whole residual stream to f32 and
    silently drop the bf16 QMM fast path for the rest of the request. The
    wrapper is JSON-safe for the block-disk meta and unwraps to a no-op on
    records stored before it existed.
    """
    dts = []
    for a in arrays:
        dt = str(getattr(a, "dtype", "") or "")
        dts.append(dt.replace("mlx.core.", ""))
    return {"m": meta, "dt": dts}


def _unwrap_cumulative_meta(meta):
    if isinstance(meta, dict) and "m" in meta:
        return meta.get("m", ""), meta.get("dt") or None
    return meta, None


def _cast_cumulative_state_dtypes(state, dtypes):
    if not dtypes or not isinstance(state, (list, tuple)):
        return state
    out = []
    for idx, a in enumerate(state):
        dt = dtypes[idx] if idx < len(dtypes) else None
        if a is not None and dt and hasattr(a, "astype"):
            target = getattr(mx, dt, None)
            if target is not None and str(getattr(a, "dtype", "")) != str(target):
                try:
                    a = a.astype(target)
                except Exception:
                    pass
        out.append(a)
    return out


def _copy_mlx_tree(obj):
    """Materialize independent MLX-array copies in a nested cache state tree."""
    if obj is None or not HAS_MLX:
        return obj
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        copied = obj * 1
        mx.eval(copied)
        return copied
    if isinstance(obj, tuple):
        return tuple(_copy_mlx_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_copy_mlx_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {key: _copy_mlx_tree(value) for key, value in obj.items()}
    return obj


def _is_zaya_cca_cache_list_state(layer_state: Dict[str, Any]) -> bool:
    """Return True for ZAYA's CacheList(KVCache, ArraysCache) CCA layer."""
    if layer_state.get("class_name") != "CacheList":
        return False
    subs = layer_state.get("sub_caches")
    if not isinstance(subs, (list, tuple)) or len(subs) < 2:
        return False
    first = subs[0] if isinstance(subs[0], dict) else {}
    second = subs[1] if isinstance(subs[1], dict) else {}
    first_cls = str(first.get("class_name", ""))
    second_cls = str(second.get("class_name", ""))
    if first_cls not in {"KVCache", "QuantizedKVCache"}:
        return False
    if second_cls != "ArraysCache":
        return False
    second_state = second.get("state")
    return isinstance(second_state, (list, tuple)) and len(second_state) >= 2


def _cache_data_has_zaya_cca(cache_data) -> bool:
    """Return True when extracted cache states include typed ZAYA CCA layers."""
    try:
        for layer_state in cache_data or []:
            if isinstance(layer_state, dict) and _is_zaya_cca_cache_list_state(
                layer_state
            ):
                return True
    except Exception:
        return False
    return False


def _cache_data_has_rotating_kv(cache_data) -> bool:
    """Return True when extracted cache states include rotating-window KV."""
    try:
        for layer_state in cache_data or []:
            if not isinstance(layer_state, dict):
                continue
            if "Rotating" in str(layer_state.get("class_name", "")):
                return True
    except Exception:
        return False
    return False


def _rotating_kv_layer_count(cache_data) -> int:
    """Count live rotating-window layers in an extracted cache snapshot."""
    try:
        return sum(
            1
            for layer_state in cache_data or []
            if isinstance(layer_state, dict)
            and "Rotating" in str(layer_state.get("class_name", ""))
        )
    except Exception:
        return 0


def _rotating_terminal_window(
    keys,
    values,
    meta_state,
    *,
    expected_offset: Optional[int] = None,
    concatenate=None,
):
    """Return the exact bounded state needed to resume a rotating KV cache.

    ``RotatingKVCache`` is not an append-only positional cache.  Its physical
    tensors are a path-dependent ring/window snapshot, and a multi-token
    prefill may temporarily retain ``max_size + chunk_size - 1`` tokens.  A
    terminal paged-cache checkpoint must therefore store one complete temporal
    window, not token slices spread across independently reusable blocks.

    The returned tuple is ``(keys, values, max_size, keep, offset, idx)``.  The
    tensors are in temporal order and bounded to ``min(offset, max_size)``;
    ``idx`` is the exact state mlx-lm would have after trimming the prefill
    buffer immediately before the next single-token decode.
    """
    if not isinstance(meta_state, (tuple, list)) or len(meta_state) < 4:
        raise ValueError("RotatingKVCache terminal snapshot is missing meta_state")
    try:
        keep, max_size, offset, idx_state = map(int, meta_state[:4])
    except (TypeError, ValueError) as exc:
        raise ValueError("RotatingKVCache terminal metadata is not integral") from exc
    if max_size <= 0 or keep < 0 or keep > max_size:
        raise ValueError(
            f"invalid RotatingKVCache window metadata keep={keep} max_size={max_size}"
        )
    if offset < 0:
        raise ValueError(f"invalid RotatingKVCache offset={offset}")
    if expected_offset is not None and offset != int(expected_offset):
        raise ValueError(
            "RotatingKVCache snapshot boundary mismatch: "
            f"offset={offset} expected={int(expected_offset)}"
        )

    ndim = len(keys.shape)
    if ndim == 4:
        seq_axis = 2
    elif ndim == 3:
        seq_axis = 1
    else:
        raise ValueError(f"unsupported RotatingKVCache rank={ndim}")
    if len(values.shape) != ndim:
        raise ValueError("RotatingKVCache key/value rank mismatch")
    seq_len = int(keys.shape[seq_axis])
    if int(values.shape[seq_axis]) != seq_len:
        raise ValueError("RotatingKVCache key/value sequence mismatch")
    if idx_state < 0 or idx_state > seq_len:
        raise ValueError(
            f"invalid RotatingKVCache idx={idx_state} physical_len={seq_len}"
        )
    if concatenate is None:
        concatenate = mx.concatenate

    def _slice(value, start=None, stop=None):
        index = [slice(None)] * ndim
        index[seq_axis] = slice(start, stop)
        return value[tuple(index)]

    def _cat(parts):
        return concatenate(parts, axis=seq_axis)

    # Match mlx-lm RotatingKVCache._temporal_order exactly, generalized to the
    # 3-D compatibility shape also accepted by the block-cache layer.
    if idx_state == seq_len:
        temporal_keys, temporal_values = keys, values
    elif idx_state < offset:
        temporal_keys = _cat(
            [_slice(keys, 0, keep), _slice(keys, idx_state, None), _slice(keys, keep, idx_state)]
        )
        temporal_values = _cat(
            [_slice(values, 0, keep), _slice(values, idx_state, None), _slice(values, keep, idx_state)]
        )
    else:
        temporal_keys = _slice(keys, 0, idx_state)
        temporal_values = _slice(values, 0, idx_state)

    temporal_len = int(temporal_keys.shape[seq_axis])
    required_len = min(offset, max_size)
    if temporal_len < required_len:
        raise ValueError(
            "incomplete RotatingKVCache terminal window: "
            f"physical={temporal_len} required={required_len} offset={offset}"
        )
    if temporal_len > required_len:
        if required_len == 0:
            temporal_keys = _slice(temporal_keys, 0, 0)
            temporal_values = _slice(temporal_values, 0, 0)
        elif keep:
            recent = required_len - keep
            if recent < 0:
                raise ValueError(
                    f"RotatingKVCache keep={keep} exceeds retained length={required_len}"
                )
            temporal_keys = _cat(
                [_slice(temporal_keys, 0, keep), _slice(temporal_keys, -recent, None)]
            ) if recent else _slice(temporal_keys, 0, keep)
            temporal_values = _cat(
                [_slice(temporal_values, 0, keep), _slice(temporal_values, -recent, None)]
            ) if recent else _slice(temporal_values, 0, keep)
        else:
            temporal_keys = _slice(temporal_keys, -required_len, None)
            temporal_values = _slice(temporal_values, -required_len, None)

    physical_len = int(temporal_keys.shape[seq_axis])
    if physical_len != required_len:
        raise ValueError(
            "RotatingKVCache terminal normalization failed: "
            f"physical={physical_len} required={required_len}"
        )
    return temporal_keys, temporal_values, max_size, keep, offset, physical_len


def _rotating_previous_block_window(
    keys,
    values,
    meta_state,
    *,
    target_offset: int,
    block_size: int,
    concatenate=None,
):
    """Derive a bounded changed-tail rotating-KV resume checkpoint.

    mlx-lm's multi-token ``RotatingKVCache._update_concat`` retains
    ``max_size + S - 1`` entries: the previous temporal window minus the one
    token that the first appended token replaces, followed by the ``S`` new
    entries.  That overhang is sufficient to materialize the block boundary
    immediately before a terminal partial/full block.  Persisting that exact
    boundary lets a changed-tail request reuse the shared paged prefix instead
    of finding only ``rotating_kv_pending`` markers and falling back cold.

    A saturated cache can be short by exactly one oldest token.  That token is
    unobservable on resume because the mandatory first uncached token replaces
    its ring slot before attention.  In that single case we duplicate the
    oldest available entry into the disposable slot; any larger history gap is
    rejected.

    Keep the checkpoint fan-out deliberately bounded to the two block
    boundaries before the terminal record.  A final partial block can make the
    longest shared prefix end *two* boundaries behind the final offset (for
    example, offset 1026 with 64-token blocks and a changed 961..1024 block
    matches through token 960).  The old one-block limit therefore stored an
    exact 1024 checkpoint but left 960 as ``rotating_kv_pending``.  Allowing two
    boundaries covers that common changed-tail shape while adding at most one
    more rotating window per stored prompt; older boundaries remain clean
    misses rather than multiplying a max-size SWA window into every page.

    The distance bound is only an allocation bound.  The retained-history
    checks below remain authoritative and reject a checkpoint unless the live
    mlx-lm concat overhang can reproduce the target temporal window exactly.
    """
    if not isinstance(meta_state, (tuple, list)) or len(meta_state) < 4:
        raise ValueError("RotatingKVCache resume snapshot is missing meta_state")
    try:
        keep, max_size, offset, idx_state = map(int, meta_state[:4])
        target_offset = int(target_offset)
        block_size = int(block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("RotatingKVCache resume metadata is not integral") from exc
    if max_size <= 0 or keep < 0 or keep > max_size:
        raise ValueError(
            f"invalid RotatingKVCache window metadata keep={keep} max_size={max_size}"
        )
    distance = offset - target_offset
    max_checkpoint_distance = 2 * block_size
    if (
        target_offset <= 0
        or distance <= 0
        or block_size <= 0
        or distance > max_checkpoint_distance
    ):
        raise ValueError(
            "RotatingKVCache resume boundary is outside the bounded tail: "
            f"target={target_offset} offset={offset} block_size={block_size} "
            f"max_distance={max_checkpoint_distance}"
        )

    ndim = len(keys.shape)
    if ndim == 4:
        seq_axis = 2
    elif ndim == 3:
        seq_axis = 1
    else:
        raise ValueError(f"unsupported RotatingKVCache rank={ndim}")
    if len(values.shape) != ndim:
        raise ValueError("RotatingKVCache key/value rank mismatch")
    seq_len = int(keys.shape[seq_axis])
    if int(values.shape[seq_axis]) != seq_len:
        raise ValueError("RotatingKVCache key/value sequence mismatch")
    if idx_state < 0 or idx_state > seq_len:
        raise ValueError(
            f"invalid RotatingKVCache idx={idx_state} physical_len={seq_len}"
        )
    if concatenate is None:
        concatenate = mx.concatenate

    def _slice(value, start=None, stop=None):
        index = [slice(None)] * ndim
        index[seq_axis] = slice(start, stop)
        return value[tuple(index)]

    def _cat(parts):
        return concatenate(parts, axis=seq_axis)

    if idx_state == seq_len:
        temporal_keys, temporal_values = keys, values
    elif idx_state < offset:
        temporal_keys = _cat(
            [_slice(keys, 0, keep), _slice(keys, idx_state, None), _slice(keys, keep, idx_state)]
        )
        temporal_values = _cat(
            [_slice(values, 0, keep), _slice(values, idx_state, None), _slice(values, keep, idx_state)]
        )
    else:
        temporal_keys = _slice(keys, 0, idx_state)
        temporal_values = _slice(values, 0, idx_state)

    temporal_len = int(temporal_keys.shape[seq_axis])
    sink_len = min(keep, target_offset)
    if sink_len and temporal_len < sink_len:
        raise ValueError("RotatingKVCache resume snapshot is missing sink tokens")
    sink_keys = _slice(temporal_keys, 0, sink_len)
    sink_values = _slice(temporal_values, 0, sink_len)
    recent_keys = _slice(temporal_keys, keep, None)
    recent_values = _slice(temporal_values, keep, None)
    recent_len = int(recent_keys.shape[seq_axis])
    recent_start = offset - recent_len

    required_len = min(target_offset, max_size)
    required_recent = required_len - sink_len
    desired_recent_start = (
        sink_len if target_offset < max_size else target_offset - required_recent
    )
    desired_recent_end = target_offset
    if desired_recent_end > offset:
        raise ValueError("RotatingKVCache resume boundary exceeds live offset")

    missing_oldest = max(recent_start - desired_recent_start, 0)
    if missing_oldest > 1 or (missing_oldest and target_offset < max_size):
        raise ValueError(
            "RotatingKVCache resume history is incomplete: "
            f"available_start={recent_start} required_start={desired_recent_start}"
        )
    available_start = max(desired_recent_start, recent_start)
    start_idx = available_start - recent_start
    end_idx = desired_recent_end - recent_start
    if start_idx < 0 or end_idx < start_idx or end_idx > recent_len:
        raise ValueError(
            "RotatingKVCache resume slice is outside the retained temporal window"
        )
    selected_keys = _slice(recent_keys, start_idx, end_idx)
    selected_values = _slice(recent_values, start_idx, end_idx)
    selected_len = int(selected_keys.shape[seq_axis])
    if missing_oldest:
        if selected_len <= 0:
            raise ValueError("RotatingKVCache resume snapshot has no disposable source")
        selected_keys = _cat([_slice(selected_keys, 0, 1), selected_keys])
        selected_values = _cat([_slice(selected_values, 0, 1), selected_values])

    if sink_len:
        resume_keys = _cat([sink_keys, selected_keys])
        resume_values = _cat([sink_values, selected_values])
    else:
        resume_keys, resume_values = selected_keys, selected_values
    resume_len = int(resume_keys.shape[seq_axis])
    if resume_len != required_len:
        raise ValueError(
            "RotatingKVCache resume normalization failed: "
            f"physical={resume_len} required={required_len}"
        )
    return (
        resume_keys,
        resume_values,
        max_size,
        keep,
        target_offset,
        resume_len,
    )


def _block_has_complete_rotating_terminal(
    cache_data,
    *,
    target_tokens: int,
    expected_layers: int,
) -> bool:
    """Whether a block already carries every exact rotating terminal layer."""
    if not cache_data or expected_layers <= 0:
        return False
    complete = 0
    for entry in cache_data:
        if not isinstance(entry, (tuple, list)) or not entry or entry[0] != "rotating_kv":
            continue
        if len(entry) < 7:
            continue
        try:
            max_size = int(entry[3])
            offset = int(entry[5])
            keys = entry[1]
            ndim = len(keys.shape)
            seq_axis = 2 if ndim == 4 else 1 if ndim == 3 else -1
            physical_len = int(keys.shape[seq_axis]) if seq_axis >= 0 else -1
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        if offset == int(target_tokens) and physical_len == min(offset, max_size):
            complete += 1
    return complete == int(expected_layers)


def _dsv4_cache_meta(layer_state: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    cr = layer_state.get("compress_ratio")
    sw = layer_state.get("sliding_window")
    if cr is not None:
        try:
            meta["compress_ratio"] = int(cr)
        except Exception:
            meta["compress_ratio"] = cr
    if sw is not None:
        try:
            meta["sliding_window"] = int(sw)
        except Exception:
            meta["sliding_window"] = sw
    if "pool_quant" in layer_state:
        meta["pool_quant"] = layer_state.get("pool_quant")
    if "pool_storage_schema" in layer_state:
        meta["pool_storage_schema"] = layer_state.get("pool_storage_schema")
    if "local_quant_meta" in layer_state:
        try:
            meta["local_quant_meta"] = [
                str(x) for x in layer_state.get("local_quant_meta") or []
            ]
        except Exception:
            meta["local_quant_meta"] = layer_state.get("local_quant_meta")
    return meta


def _mx_from_np_slice(view):
    """Import a numpy slice into MLX without dragging its base buffer.

    mx.array() on a NON-CONTIGUOUS numpy view imports a buffer sized like the
    view's BASE array (measured: 30.6MB of Metal for a 0.26MB 64-token block
    slice of a 42.7MB full-layer mirror). Per-block extraction over a long
    prompt turns that into O(blocks x full_layer) live Metal — ~140GB on an
    11k-token qwen hybrid store, enough to abort the serve process and twice
    panic the machine. A host-side contiguous copy of just the slice keeps
    the import at slice size.
    """
    import numpy as np

    return mx.array(np.ascontiguousarray(view))


def _positional_layer_slice_bounds(
    layer_state, class_name, start_idx, end_idx, seq_len, existing_tokens=0,
):
    """Map global token positions to a layer-local KV slice range."""
    layer_offset = 0
    if seq_len > 0 and "Rotating" in str(class_name):
        meta = layer_state.get("meta_state", ()) if isinstance(layer_state, dict) else ()
        if meta and len(meta) >= 3:
            try:
                absolute_offset = int(meta[2])
            except (TypeError, ValueError):
                absolute_offset = 0
            if absolute_offset > seq_len:
                layer_offset = absolute_offset - seq_len
    elif seq_len > 0 and existing_tokens > seq_len:
        layer_offset = existing_tokens

    local_start = start_idx - layer_offset
    local_end = end_idx - layer_offset
    actual_end = min(local_end, seq_len)
    slice_start = max(local_start, 0)
    return slice_start, actual_end


def _numpy_block_slice(
    cache_data,
    np_sources,
    start_idx,
    end_idx,
    is_last_block,
    existing_tokens=0,
    store_cumulative_state=True,
    rotating_resume_block_size=0,
):
    """Create per-block cache_data using NumPy slicing.

    Positional slicing stays on the CPU. The final per-block arrays are moved
    back to MLX only when their original dtype must be restored (notably bf16,
    which NumPy snapshots represent as float32). The disk writer would perform
    the same NumPy-to-MLX conversion later; restoring here prevents an L2 record
    from silently changing attention math and decode speed.

    Args:
        cache_data: Full layer-state dicts from _extract_cache_states.
        np_sources: dict mapping layer_idx → (np_keys, np_values).
        start_idx, end_idx: Token range for this block.
        is_last_block: Whether this is the last block in the sequence.

    Returns:
        List of tuples per layer in the same format as _extract_block_tensor_slice
        but with numpy arrays instead of MLX arrays.
    """
    import numpy as np

    block_slices = []
    for idx, layer_state in enumerate(cache_data):
        if "state" not in layer_state:
            continue
        cls = layer_state.get("class_name", "")

        if layer_state.get("dsv4_block_records"):
            record = _dsv4_delta_record_for_interval(
                layer_state, start_idx, end_idx
            )
            if isinstance(record, dict):
                block_slices.append(
                    (
                        "deepseek_v4_delta_v1",
                        record,
                        cls,
                        _dsv4_cache_meta(layer_state),
                    )
                )
            elif isinstance(record, (tuple, list)) and record:
                block_slices.append(tuple(record))
            else:
                raise ValueError("invalid DSV4 block transport record")
            continue

        if _is_dsv4_cache_class(cls):
            state = layer_state.get("state")
            if is_last_block and state is not None:
                block_slices.append((
                    "deepseek_v4",
                    _to_numpy_tree(state),
                    layer_state.get("meta_state", ""),
                    cls,
                    _dsv4_cache_meta(layer_state),
                ))
            else:
                block_slices.append((
                    "deepseek_v4_pending",
                    cls,
                    _dsv4_cache_meta(layer_state),
                ))
            continue

        if cls == "ZayaNoStateCache" or layer_state.get("no_state"):
            block_slices.append(("no_state", cls or "ZayaNoStateCache"))
            continue

        if _is_zaya_cca_cache_list_state(layer_state):
            zsrc = np_sources.get(idx) if isinstance(np_sources, dict) else None
            if not isinstance(zsrc, dict) or zsrc.get("type") != "zaya_cca":
                block_slices.append(("skip",))
                continue

            np_k, np_v, orig_dtype = zsrc["kv"]
            ndim = np_k.ndim
            seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
            if seq_dim < 0:
                block_slices.append(("skip",))
                continue
            seq_len = np_k.shape[seq_dim]
            actual_end = min(end_idx, seq_len)
            if start_idx >= actual_end:
                kv_entry = ("skip",)
            elif ndim == 4:
                ks = _mx_from_np_slice(np_k[:, :, start_idx:actual_end, :])
                vs = _mx_from_np_slice(np_v[:, :, start_idx:actual_end, :])
                if ks.dtype != orig_dtype:
                    ks = ks.astype(orig_dtype)
                    vs = vs.astype(orig_dtype)
                kv_entry = ("kv", ks, vs)
            else:
                ks = _mx_from_np_slice(np_k[:, start_idx:actual_end, :])
                vs = _mx_from_np_slice(np_v[:, start_idx:actual_end, :])
                if ks.dtype != orig_dtype:
                    ks = ks.astype(orig_dtype)
                    vs = vs.astype(orig_dtype)
                kv_entry = ("kv", ks, vs)

            cca_state = zsrc.get("cca_state") if is_last_block else None
            if cca_state is not None:
                def _to_mx_tree(x):
                    if x is None:
                        return None
                    if isinstance(x, tuple):
                        return tuple(_to_mx_tree(v) for v in x)
                    if isinstance(x, list):
                        return [_to_mx_tree(v) for v in x]
                    return _mx_from_np_slice(x)

                cca_state = _to_mx_tree(cca_state)
            block_slices.append((
                "zaya_cca",
                kv_entry,
                cca_state,
                zsrc.get("cca_meta", ""),
                zsrc.get("cache_meta", {}),
            ))
            continue

        m3src = np_sources.get(idx) if isinstance(np_sources, dict) else None
        if isinstance(m3src, dict) and m3src.get("type") == "minimax_m3":
            np_k, np_v, np_idx, orig_dtype = m3src["kv"]
            seq_len = np_k.shape[2]
            actual_end = min(end_idx, seq_len)
            if start_idx >= actual_end:
                block_slices.append(("skip",))
                continue
            ks = _mx_from_np_slice(np_k[:, :, start_idx:actual_end, :])
            vs = _mx_from_np_slice(np_v[:, :, start_idx:actual_end, :])
            idxs = None
            if np_idx is not None:
                idxs = _mx_from_np_slice(np_idx[:, :, start_idx:actual_end, :])
            if orig_dtype is not None and ks.dtype != orig_dtype:
                ks = ks.astype(orig_dtype)
                vs = vs.astype(orig_dtype)
                if idxs is not None:
                    idxs = idxs.astype(orig_dtype)
            block_slices.append(("minimax_m3", ks, vs, idxs))
            continue

        # CacheList (MoE): not yet supported in numpy path
        if cls == "CacheList":
            block_slices.append(("skip",))
            continue

        if idx in np_sources:
            np_k, np_v, *source_meta = np_sources[idx]
            orig_dtype = source_meta[0] if source_meta else None
            ndim = np_k.ndim
            if ndim == 4:
                seq_len = np_k.shape[2]
            elif ndim == 3:
                seq_len = np_k.shape[1]
            else:
                block_slices.append(("skip",))
                continue
            if "Rotating" in cls:
                # A rotating-window state is one terminal checkpoint, not a
                # collection of immutable token pages.  Reusing slices written
                # from snapshots with different logical offsets produced the
                # live Laguna mosaic (old offset metadata + an incomplete new
                # physical window).  Keep old non-terminal blocks shareable,
                # but use mlx-lm's bounded concat overhang to checkpoint the
                # immediately preceding boundary when it is exact.
                try:
                    if is_last_block:
                        terminal = _rotating_terminal_window(
                            np_k,
                            np_v,
                            layer_state.get("meta_state", ()),
                            expected_offset=end_idx,
                            concatenate=np.concatenate,
                        )
                    else:
                        terminal = _rotating_previous_block_window(
                            np_k,
                            np_v,
                            layer_state.get("meta_state", ()),
                            target_offset=end_idx,
                            block_size=rotating_resume_block_size,
                            concatenate=np.concatenate,
                        )
                except ValueError as exc:
                    if is_last_block:
                        # A terminal boundary the snapshot cannot cut (e.g. a
                        # warm-media store whose extracted ring already rolled
                        # past the store key) must stay VISIBLE.  A bare
                        # ("skip",) here published chains whose matched
                        # boundary passed the fetch-side terminal guard —
                        # skip entries carry no family information — and then
                        # reconstructed to an empty cache.  A pending marker
                        # makes the fetch lanes normalize the candidate back
                        # to the newest exact rotating anchor instead.
                        logger.warning(
                            "Layer %s (%s): %s — storing terminal boundary as "
                            "rotating_kv_pending so fetches walk back to an "
                            "earlier exact anchor",
                            idx,
                            cls,
                            exc,
                        )
                    block_slices.append(("rotating_kv_pending", cls))
                    continue
                tk, tv, max_size, keep, offset, idx_state = terminal
                tk = _mx_from_np_slice(tk)
                tv = _mx_from_np_slice(tv)
                if orig_dtype is not None and tk.dtype != orig_dtype:
                    tk = tk.astype(orig_dtype)
                    tv = tv.astype(orig_dtype)
                block_slices.append((
                    "rotating_kv",
                    tk,
                    tv,
                    max_size,
                    keep,
                    offset,
                    idx_state,
                ))
                continue
            slice_start, actual_end = _positional_layer_slice_bounds(
                layer_state, cls, start_idx, end_idx, seq_len, existing_tokens,
            )
            if actual_end <= 0 or slice_start >= actual_end:
                block_slices.append(("skip",))
                continue
            if ndim == 4:
                ks = np_k[:, :, slice_start:actual_end, :]
                vs = np_v[:, :, slice_start:actual_end, :]
            else:
                ks = np_k[:, slice_start:actual_end, :]
                vs = np_v[:, slice_start:actual_end, :]
            # NumPy has no native bfloat16 dtype. Preserve the model-owned
            # attention dtype rather than persisting the fp32 bridge arrays.
            # The import MUST go through _mx_from_np_slice: ks/vs are
            # non-contiguous views of the full-prompt mirror, and a bare
            # mx.array() import allocates a base-buffer-sized Metal buffer
            # per block — retained in pending_disk_writes for the whole
            # store loop, this reached ~159GB live Metal on an 11k-token
            # hybrid store (the disk-lane half of the machine-killer).
            if orig_dtype is not None:
                ks = _mx_from_np_slice(ks)
                vs = _mx_from_np_slice(vs)
                if ks.dtype != orig_dtype:
                    ks = ks.astype(orig_dtype)
                    vs = vs.astype(orig_dtype)
            block_slices.append(("kv", ks, vs))
        else:
            # Cumulative / non-positional layer
            state = layer_state.get("state")
            if is_last_block and store_cumulative_state and state is not None:
                meta = layer_state.get("meta_state", "")
                # Convert cumulative state to numpy
                if isinstance(state, (list, tuple)):
                    np_state = []
                    for s in state:
                        if hasattr(s, '__array__'):
                            if hasattr(s, "dtype") and "bfloat16" in str(
                                getattr(s, "dtype", "")
                            ):
                                # MLX >= 0.32: np.array(bf16) raises; cast
                                # through fp32 (value-exact for bf16). The
                                # original dtype travels in the wrapped meta
                                # so the restore casts back.
                                s = s.astype(mx.float32)
                            np_state.append(np.array(s))
                        else:
                            np_state.append(s)
                    block_slices.append(
                        (
                            "cumulative",
                            np_state,
                            _wrap_cumulative_meta(meta, state),
                            cls,
                        )
                    )
                else:
                    block_slices.append(("skip",))
            else:
                # Windowed dots3 latent layers checkpoint recent NON-terminal
                # boundaries exactly (ledger row 152) so divergent-prefix
                # restores keep their sliding state instead of collapsing to
                # cold prefill.
                checkpoint = (
                    _dots3_window_boundary_checkpoint(layer_state, end_idx)
                    if store_cumulative_state
                    else None
                )
                if checkpoint is not None:
                    tag, sliced, cmeta, ccls = checkpoint
                    block_slices.append(
                        (tag, _copy_mlx_tree(sliced), cmeta, ccls)
                    )
                else:
                    block_slices.append(("skip",))

    return block_slices if block_slices else None


def _entry_has_native_tq(entry) -> bool:
    """Return whether one typed block entry contains packed TQ attention KV."""
    if not isinstance(entry, (tuple, list)) or not entry:
        return False
    if entry[0] == "turboquant_kv":
        return True
    if entry[0] == "cache_list" and len(entry) > 1:
        return any(_entry_has_native_tq(sub) for sub in entry[1] or [])
    return False


def _dots3_window_boundary_checkpoint(layer_state, boundary: int):
    """Boundary checkpoint for a windowed dots3 latent layer (ledger row 152).

    dots3 sliding layers store CUMULATIVE state in the terminal block only,
    which makes every divergent-prefix (non-terminal-boundary) restore
    impossible — the partial span reconstructs without sliding state and the
    generator must fall back to cold prefill. The cache deliberately retains
    ``retain_overhang`` extra masked-out keys, so for block boundaries B
    within that overhang of the snapshot offset T the exact window state at B
    (keys B-window+1..B) is still physically present and can be checkpointed
    into the block itself. Returns a ("cumulative", [latent, k_pe, idx_k],
    meta, cls) entry with offset rewritten to B, or None when the boundary is
    not exactly reconstructible (caller falls back to ("skip",)).

    Returns LAZY slices in the input arrays' own representation (MX in both
    live store paths — bf16 cannot round-trip numpy directly). Callers wrap
    the result in _copy_mlx_tree, which materializes independent copies the
    same way the proven terminal-cumulative branch does; the store flow's
    mx.synchronize() has already run by the time extraction slices.
    """
    cls = str(layer_state.get("class_name") or "")
    if "Dots3LatentCache" not in cls:
        return None
    meta = layer_state.get("meta_state") or ()
    if not isinstance(meta, (tuple, list)) or len(meta) < 2:
        return None
    try:
        total = int(meta[0])
        window = int(meta[1]) if str(meta[1]) else 0
    except (TypeError, ValueError):
        return None
    if window <= 0 or boundary <= 0 or boundary >= total:
        return None
    state = layer_state.get("state")
    if not isinstance(state, (tuple, list)) or len(state) != 3:
        return None
    latent, k_pe, idx_k = state
    if latent is None or not getattr(latent, "size", 0):
        return None
    # A boundary before the window fills needs ALL of its keys, not window-1.
    keep = min(window - 1, boundary)
    phys = int(latent.shape[2])
    drop = total - boundary
    if phys - drop < keep:
        return None  # boundary older than the retained overhang
    lo, hi = phys - drop - keep, phys - drop
    # Absent streams MUST be size-0 arrays, never None: the cache's own
    # ``state`` property uses the same convention, the restore setter's
    # ``_real()`` maps them back to None, and a None that gets dropped in
    # transport turns the 3-tuple into a 2-tuple that the setter misparses
    # as the POSITIONAL presentation (measured live: sliding latent replaced
    # by the 64-dim rope stream, concat shape blowup on the next chunk).
    # float32 empty: a size-0 bf16 array cannot cross the numpy buffer
    # protocol on the disk-serialization path, and the restore setter maps
    # ANY size-0 array to None regardless of dtype.
    empty = mx.zeros((0,), dtype=mx.float32)
    latent_s = latent[:, :, lo:hi]
    k_pe_s = (
        k_pe[:, :, lo:hi]
        if k_pe is not None and getattr(k_pe, "size", 0)
        else empty
    )
    idx_s = empty
    if idx_k is not None and getattr(idx_k, "size", 0):
        ilen = int(idx_k.shape[1])
        if ilen == total:
            idx_s = idx_k[:, :boundary]
        elif ilen - drop >= keep:
            idx_s = idx_k[:, ilen - drop - keep : ilen - drop]
    new_meta = (str(boundary),) + tuple(str(m) for m in meta[1:])
    return ("cumulative", [latent_s, k_pe_s, idx_s], new_meta, cls)


def _block_needs_cumulative_update(cache_data) -> bool:
    """Check if a block's cache_data is missing cumulative SSM state.

    Returns True if the block has "skip" entries (SSM layers stored as
    non-last) but no "cumulative" entries. This indicates the block was
    originally stored at a non-last position and needs cumulative state
    before it can be used as the last block in a sequence.

    Checks both top-level entries and sub-slices inside CacheList entries
    (for MoE hybrid models where SSM sub-caches are nested).

    Used by store_cache() to detect when block reuse would lose SSM state
    for hybrid models (KVCache + MambaCache/ArraysCache).
    """
    if not cache_data:
        return False
    has_skip = False
    has_cumulative = False
    for entry in cache_data:
        if not isinstance(entry, (tuple, list)) or len(entry) == 0:
            continue
        tag = entry[0]
        if tag == "skip":
            has_skip = True
        elif tag == "deepseek_v4_pending":
            has_skip = True
        elif tag == "deepseek_v4_delta_v1":
            if _dsv4_delta_anchor_kind(entry) is None:
                has_skip = True
            else:
                has_cumulative = True
        elif tag in ("cumulative", "deepseek_v4"):
            has_cumulative = True
        elif tag == "zaya_cca":
            # Non-terminal ZAYA blocks carry KV pages only. The exact CCA
            # conv_state/prev_hs payload lives on the terminal prompt block.
            # Reusing a non-terminal block as the final block would restore
            # standard KV without the path-dependent CCA state.
            if len(entry) > 2 and entry[2] is not None:
                has_cumulative = True
            else:
                has_skip = True
        elif tag == "cache_list" and len(entry) > 1:
            # Recurse into CacheList sub-slices for MoE hybrid models
            for sub in entry[1]:
                if isinstance(sub, (tuple, list)) and len(sub) > 0:
                    if sub[0] == "skip":
                        has_skip = True
                    elif sub[0] == "deepseek_v4_pending":
                        has_skip = True
                    elif sub[0] in ("cumulative", "deepseek_v4"):
                        has_cumulative = True
    return has_skip and not has_cumulative


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: Optional[List[Any]]  # Legacy full-cache reference, if needed
    last_access: float
    cache_type: str = "assistant"  # "system" | "user" | "assistant" — segment ownership


class BlockAwarePrefixCache:
    """
    Prefix cache that uses PagedCacheManager for block-based storage.

    Features:
    - Block-level prefix sharing (64 tokens per block)
    - Copy-on-Write for efficient forking
    - Hash-based deduplication across requests
    - Reference counting for memory efficiency

    This is the recommended cache for production use when memory
    efficiency for concurrent requests is important.

    Example:
        paged_manager = PagedCacheManager(block_size=64, max_blocks=1000)
        cache = BlockAwarePrefixCache(model, paged_manager)

        # Check for cached prefix
        block_table, remaining_tokens = cache.fetch_cache(request_id, tokens)

        # After generation, store cache
        cache.store_cache(request_id, tokens, kv_cache_data)

        # Clean up when request completes
        cache.release_cache(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
        model_path: Optional[str] = None,
        smelt_enabled: bool = False,
        smelt_pct: Optional[float] = None,
        tq_enabled: bool = False,
        kv_quant_bits: int = 0,
        uses_dsv4_cache: Optional[bool] = None,
        uses_zaya_cache: Optional[bool] = None,
        mixed_attention_cache_model: Optional[bool] = None,
    ):
        """
        Initialize block-aware prefix cache.

        Args:
            model: The MLX model (used for identification)
            paged_cache_manager: The PagedCacheManager instance for block management
            model_path, smelt_enabled, smelt_pct, tq_enabled, kv_quant_bits:
                Loader fingerprint inputs (F6 + A4 Concern #1). Mixed into
                paged-cache content hash so divergent loader configs never
                collide on shared blocks.
            uses_dsv4_cache, uses_zaya_cache, mixed_attention_cache_model:
                Observed runtime-cache contracts from the owning scheduler.
                ``False`` proves that the corresponding path-dependent payload
                cannot occur and avoids loading every SSD block merely to scan
                for its terminal tag. ``None`` is conservative and preserves
                the payload-driven validation used by standalone callers.
        """
        self.model = model
        # Content-derived stable key (replaces id(model)). Includes loader
        # fingerprint so two sessions with different smelt/TQ/JANG settings
        # never share blocks (would otherwise corrupt K/V routing).
        self.model_key = compute_model_cache_key(
            model,
            model_path=model_path,
            smelt_enabled=smelt_enabled,
            smelt_pct=smelt_pct,
            tq_enabled=tq_enabled,
            kv_quant_bits=kv_quant_bits,
        )
        self.paged_cache = paged_cache_manager
        self.block_size = paged_cache_manager.block_size
        # These validators exist to reject incomplete native/path-dependent
        # checkpoints. In SSD-only mode they must deserialize payloads to inspect
        # their tags. Running all three against a runtime-proven ordinary KV or
        # KV+external-SSM layout loads the whole chain once here and once again
        # during reconstruction (measured on LFM2: 508 physical reads for 254
        # logical blocks). Only an explicit False from instantiated runtime
        # detection may skip a validator; unknown standalone/test callers retain
        # the former fail-closed behavior.
        self._validate_dsv4_terminal = uses_dsv4_cache is not False
        self._validate_zaya_terminal = uses_zaya_cache is not False
        self._validate_rotating_terminal = mixed_attention_cache_model is not False
        self._native_terminal_validation_source = (
            "instantiated_runtime_cache"
            if all(
                value is not None
                for value in (
                    uses_dsv4_cache,
                    uses_zaya_cache,
                    mixed_attention_cache_model,
                )
            )
            else "conservative_payload_probe"
        )
        # Exact native-MTP prompt-history snapshots are opaque, memory-only
        # sidecars of ordinary full-block backbone entries.  They are never
        # trusted without the matching live block hash and are discarded with
        # that hash; SSD persistence requires a future versioned tensor schema.
        self._mtp_prefix_snapshots: OrderedDict[Any, tuple[int, Any]] = (
            OrderedDict()
        )
        self._mtp_prefix_snapshot_lock = threading.RLock()
        self._strict_block_disk_write_fence = os.environ.get(
            "VMLX_STRICT_BLOCK_DISK_WRITE_FENCE",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            self._native_block_disk_admission_timeout = max(
                0.0,
                float(
                    os.environ.get(
                        "VMLX_NATIVE_BLOCK_DISK_ADMISSION_TIMEOUT_SECONDS",
                        "30",
                    )
                ),
            )
        except (TypeError, ValueError):
            self._native_block_disk_admission_timeout = 30.0
        # Ordinary (non-path-dependent) families used a hard 0.0 here, meaning a
        # block whose payload did not fit the pending-write byte budget AT THAT
        # INSTANT was discarded rather than waiting for the background writer to
        # drain. That is not a rare safety valve: MEASURED on the box 2026-08-12
        # with Nanbeige4.2-3B (a 3B model, so small payloads), an 8.9k-token
        # prompt plus follow-ups produced 143 disk writes and 128
        # byte_budget_drops — 47% of the blocks the engine had already paid to
        # prefill were thrown away, with only a repeated WARNING to show for it.
        # Bigger models have bigger per-block payloads and drop more.
        #
        # The wait happens on the model-owning thread, so ordinary paged-RAM
        # write-through stays short. The longer window above is required for
        # path-dependent records AND SSD-only operation: a disk-only request has
        # no permitted RAM fallback, so a best-effort one-second publication can
        # only throw away work the engine already paid to prefill.
        # O(n) chained prefix-index hashing. Default OFF: it changes the
        # in-memory index KEYS (not any persisted record), and the win is a
        # lock-hold reduction that still wants a live long-conversation A/B
        # before it becomes the default.
        self._chained_prefix_index_hash = os.environ.get(
            "VMLX_CHAINED_PREFIX_INDEX_HASH", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        # 1s, up from 0.25s: measured live on dots3-note at ~17k context, a
        # multiturn store burst filled the 1GB pending-write budget and the
        # burst TAIL blocks (6 of 220) timed out at 0.25s — one dropped
        # ancestor then poisons every descendant for the session ("cannot
        # publish block whose parent ancestry is unavailable" cascade), L1
        # drops the pinned blocks, and the conversation re-prefills every
        # turn. The wait blocks the model-owning thread, so it must stay
        # sub-second per block (test-pinned); it only engages when the
        # budget is actually full, and the writer drains continuously, so
        # 1s per tail block rides out a burst the 0.25s window could not.
        try:
            self._block_disk_admission_timeout = max(
                0.0,
                float(
                    os.environ.get(
                        "VMLX_BLOCK_DISK_ADMISSION_TIMEOUT_SECONDS",
                        "1.0",
                    )
                ),
            )
        except (TypeError, ValueError):
            self._block_disk_admission_timeout = 1.0

        # Hash table for quick prefix lookup
        # Maps hash(tokens[:block_size*n]) -> (tokens, block_ids)
        self._prefix_index: Dict[str, Tuple[List[int], List[int]]] = {}

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Per-cache-type LRU buckets for block eviction priority. Tracks
        # request_id sets per type so we can prefer evicting assistant entries
        # (and the blocks they uniquely reference) before user/system. The
        # paged_cache itself uses ref-count eviction; this layer adds a
        # priority signal so the higher-level scheduler can call
        # `release_low_priority(n)` when under block pressure.
        self._entries_by_type: Dict[str, "OrderedDict[str, bool]"] = {
            t: OrderedDict() for t in _CACHE_TYPE_PRIORITY
        }

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        # Per-reconstruction source accounting. Fetch can discover a matching
        # in-process block chain before frugal mode lazily reads its payloads
        # from L2 on the worker, so fetch-time disk counters alone are not
        # sufficient for truthful cache_detail telemetry.
        self._last_reconstruct_disk_blocks = 0
        self._last_reconstruct_tq_blocks = 0
        # Per-request credit lets a hybrid consumer roll back a KV lookup that
        # cannot actually be used without matching path-dependent companion
        # state.  Lookup success and execution success are not equivalent for
        # SSM/GDN hybrids.
        self._hit_credits: Dict[str, int] = {}

        # fetch_cache() match telemetry (task #23: cached_tokens alone cannot
        # say WHICH branch produced it). Kept on self rather than widening
        # fetch_cache()'s return tuple: ~80 call sites across
        # scheduler.py/mllm_scheduler.py/mllm_batch_generator.py and the test
        # suite unpack `block_table, remaining = fetch_cache(...)`, so a
        # return-shape change would be a breaking edit to every one of them
        # for a debug-only signal. Bounded ring so a long-running server does
        # not accumulate one entry per request forever.
        self._fetch_telemetry: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._last_fetch_telemetry: Optional[Dict[str, Any]] = None

        # Lazy-cached expected KV head count for validation
        self._n_kv_heads: Optional[int] = None
        # Lazy-cached set of ALL valid KV head counts (for mixed-head models
        # like Gemma 4 where sliding_attention layers use num_key_value_heads
        # and full_attention layers use num_global_key_value_heads).
        self._allowed_n_kv_heads: Optional[set] = None

        # Expected layer count is derived from the model's
        # cache contract. Hybrid models such as Nemotron-H have no-cache MoE
        # blocks (52 transformer blocks, 29 Mamba/attention cache entries), so
        # validating against raw num_hidden_layers rejects legitimate L2 blocks.
        # Used by reconstruct_cache to hard-reject blocks whose layer count
        # diverges (canonical "wrong-model L2 entry" signal).
        self._expected_num_layers: Optional[int] = None
        # Payload compatibility and prompt-shape identity are separate axes.
        # Qwen4Exp QSA uses MiniMaxM3SparseCache as a three-tensor container,
        # but only an actual MiniMax-M3 architecture needs M3's matrix-shape
        # discriminator.  The store serializer still detects the payload class
        # independently through _cache_data_has_minimax_m3().
        self._uses_minimax_m3_cache = _model_is_minimax_m3(model)
        if model is not None and hasattr(model, "make_cache"):
            try:
                _cache = model.make_cache() or []
                if len(_cache) > 0:
                    self._expected_num_layers = len(_cache)
            except Exception:
                self._expected_num_layers = None
        for _attr in ("args", "config"):
            if self._expected_num_layers is not None:
                break
            _cfg = getattr(model, _attr, None)
            if _cfg is not None:
                _ln = getattr(_cfg, "num_hidden_layers", 0)
                if _ln:
                    self._expected_num_layers = int(_ln)
                    break

    _FETCH_TELEMETRY_MAX_ENTRIES = 128

    @staticmethod
    def _fetch_telemetry_source(blocks: Optional[List[Any]]) -> str:
        """Which tier actually supplied the winning blocks.

        Two independent disk-origin signals, either one sufficient:
        ``cache_data_from_disk`` is set by ``PagedCacheManager._promote_from_disk()``
        whenever this fetch had to read the block from L2 -- true regardless of
        whether the promoted payload stays resident afterward (frugal/disk-only
        promotions set ``cache_data_transient`` instead of evicting it, so a
        ``cache_data is None`` check alone misses them). ``cache_data is None``
        alone still catches the complementary case: a metadata-only block whose
        hash chain is resident but whose payload was never mirrored into RAM at
        store time (paged_frugal skip) and has not been promoted by this
        request. Mirrors the same signals server.py's ``/health`` cache
        snapshot already reads to tell a resident block from a disk-only one
        (see ``_cache_telemetry_snapshot``), rather than inferring source from
        a disk-hit counter delta.
        """
        if not blocks:
            return "unknown"
        for block in blocks:
            if getattr(block, "cache_data_from_disk", False):
                return "disk_l2"
            if getattr(block, "cache_data", None) is None:
                return "disk_l2"
        return "memory_l1"

    @classmethod
    def prefix_key_for_blocks(cls, blocks: Optional[List[Any]]) -> Optional[str]:
        """Short identity of a block chain: the terminal block's chained
        content hash (12 hex chars), or None when the chain has no hash."""
        key = cls._fetch_telemetry_cache_key(blocks)
        return key[:12] if key else None

    def prefix_key_for_block_ids(self, block_ids: Optional[Any]) -> Optional[str]:
        """Same identity, from a stored block table's physical block ids."""
        if not isinstance(block_ids, (list, tuple)) or not block_ids:
            return None
        try:
            block = self.paged_cache.blocks[int(block_ids[-1])]
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return self.prefix_key_for_blocks([block])

    # A later request that shares only a PREFIX of a stored chain hits an
    # interior block, so its terminal key is not the chain's terminal key.
    # The store line lists every block key for chains up to this many
    # blocks (4k tokens at 64) so such a partial restore can still be bound
    # to its publication by identity; longer chains log the terminal key only.
    BLOCK_KEYS_LOG_MAX_BLOCKS = 64

    def block_keys_for_block_ids(self, block_ids: Optional[Any]) -> Optional[str]:
        """Comma-joined short keys of every block in a stored chain, or None
        when the chain is longer than ``BLOCK_KEYS_LOG_MAX_BLOCKS``."""
        if not isinstance(block_ids, (list, tuple)) or not block_ids:
            return None
        if len(block_ids) > self.BLOCK_KEYS_LOG_MAX_BLOCKS:
            return None
        keys = []
        for bid in block_ids:
            try:
                block = self.paged_cache.blocks[int(bid)]
            except (AttributeError, IndexError, TypeError, ValueError):
                return None
            key = self.prefix_key_for_blocks([block])
            if key is None:
                return None
            keys.append(key)
        return ",".join(keys)

    @staticmethod
    def _fetch_telemetry_cache_key(blocks: Optional[List[Any]]) -> Optional[str]:
        if not blocks:
            return None
        terminal_hash = getattr(blocks[-1], "block_hash", None)
        if terminal_hash is None:
            return None
        try:
            return terminal_hash.hex()
        except AttributeError:
            return str(terminal_hash)

    # request_id suffixes the server layer appends for a SECOND, internal
    # fetch_cache() call issued for the same user-facing request (e.g. the
    # bounded ":visible-answer" retry when reasoning content exhausts budget
    # with no visible answer yet -- server.py's answer_kwargs["request_id"]).
    # A naive reader of last_fetch_telemetry right after a response can land
    # on this internal call instead of the base request's own fetch; expose
    # is_internal_continuation/base_request_id so a caller can filter without
    # string-matching request_id itself.
    _FETCH_TELEMETRY_CONTINUATION_SUFFIXES = (":visible-answer",)

    def _record_fetch_telemetry(
        self,
        *,
        request_id: str,
        match_kind: str,
        origin: Optional[str],
        logical_restored_tokens: int,
        native_companion_boundary: Optional[int] = None,
        source: str = "unknown",
        cache_key: Optional[str] = None,
        dsv4_delta_applied: bool = False,
        rotating_swa_normalized: bool = False,
        miss_reason: Optional[str] = None,
        attempted_tokens: Optional[int] = None,
    ) -> None:
        """Record one fetch_cache() outcome as a pure side effect.

        Must never influence fetch_cache()'s control flow or return values --
        any failure here is swallowed so telemetry can never turn an
        observability add-on into a request-affecting change.
        """
        try:
            base_request_id = request_id
            is_continuation = False
            for suffix in self._FETCH_TELEMETRY_CONTINUATION_SUFFIXES:
                if request_id.endswith(suffix):
                    base_request_id = request_id[: -len(suffix)]
                    is_continuation = True
                    break
            record: Dict[str, Any] = {
                "request_id": request_id,
                "base_request_id": base_request_id,
                "is_internal_continuation": is_continuation,
                "cache_key": cache_key,
                "match_kind": match_kind,
                "origin": origin,
                "logical_restored_tokens": int(logical_restored_tokens),
                "native_companion_boundary": (
                    int(native_companion_boundary)
                    if native_companion_boundary is not None
                    else None
                ),
                "source": source,
                "dsv4_delta_applied": bool(dsv4_delta_applied),
                "rotating_swa_normalized": bool(rotating_swa_normalized),
                "miss_reason": miss_reason,
                "attempted_tokens": (
                    int(attempted_tokens) if attempted_tokens is not None else None
                ),
                "timestamp": time.time(),
            }
            self._fetch_telemetry[request_id] = record
            self._fetch_telemetry.move_to_end(request_id)
            while len(self._fetch_telemetry) > self._FETCH_TELEMETRY_MAX_ENTRIES:
                self._fetch_telemetry.popitem(last=False)
            self._last_fetch_telemetry = record
        except Exception:
            logger.debug(
                "fetch_cache telemetry recording failed for %s",
                request_id,
                exc_info=True,
            )

    def record_fetch_bypass(
        self,
        request_id: str,
        *,
        attempted_tokens: int,
    ) -> None:
        """Record an intentional per-request fetch bypass.

        ``skip_prefix_cache`` deliberately avoids calling ``fetch_cache()``.
        Without a record at that owning branch, the process-wide health
        snapshot keeps exposing the preceding request's hit or miss as though
        it belonged to the bypassed request.  This method is telemetry-only:
        it does not consult, mutate, fetch, store, or account cache state.
        """

        self._record_fetch_telemetry(
            request_id=request_id,
            match_kind="request_bypass",
            origin=None,
            logical_restored_tokens=0,
            source="none",
            miss_reason="skip_prefix_cache",
            attempted_tokens=attempted_tokens,
        )

    def _write_admission_timeout_for_store(
        self,
        *,
        disk_only: bool,
        path_dependent: bool,
    ) -> float:
        """Choose lossless SSD admission when disk is the only cache tier."""

        if disk_only or path_dependent:
            return float(self._native_block_disk_admission_timeout)
        return float(self._block_disk_admission_timeout)

    @staticmethod
    def _write_block_immediately_for_store(
        *,
        disk_only: bool,
        minimax_m3: bool,
        native_tq: bool,
    ) -> bool:
        """Whether one frozen page must be queued before extracting the next."""

        return bool(disk_only or minimax_m3 or native_tq)

    def _shape_scoped_cache_extra_keys(
        self,
        tokens: List[int],
        cache_extra_keys: Optional[Any],
        *,
        stored_prompt_boundary: bool = False,
    ) -> Optional[Any]:
        """Bind native M3 blocks to the prefill matrix shape that produced them.

        Identical token prefixes can produce slightly different K/V/index
        tensors when MLX evaluates them inside different prompt-length matrix
        shapes. Ordinary attention tolerates that numerical drift, but
        MiniMax-M3's Lightning Indexer can select a different sparse block and
        change greedy decoding. Exact requests and same-length partial-prefix
        requests still share blocks; cross-length M3 aliases do not.
        """
        if not self._uses_minimax_m3_cache:
            return cache_extra_keys

        # M3 stores the prompt-boundary cache under N-1 token IDs because the
        # final prompt token is re-fed to obtain first-token logits. Lookup
        # receives the corresponding full cache-key prompt. Normalize both
        # sides to the matrix shape that originally produced the cache.
        prefill_shape_tokens = len(tokens) + (1 if stored_prompt_boundary else 0)
        discriminator = {
            "schema": "minimax_m3_prefill_shape_v1",
            "cache_key_tokens": prefill_shape_tokens,
        }
        if cache_extra_keys is None:
            return {"__vmlx_native_cache_shape__": discriminator}
        # Keep causal per-token scopes at the top level. Nesting the complete
        # request discriminator under ``__vmlx_request_extra_keys__`` would
        # hide its media boundary from the block-hash resolver and regress to
        # salting every M3 media block from token zero.
        if (
            isinstance(cache_extra_keys, dict)
            and CACHE_EXTRA_SCOPES_KEY in cache_extra_keys
        ):
            return {
                "__vmlx_native_cache_shape__": discriminator,
                **cache_extra_keys,
            }
        return {
            "__vmlx_native_cache_shape__": discriminator,
            "__vmlx_request_extra_keys__": cache_extra_keys,
        }

    def _get_n_kv_heads(self) -> int:
        """Get expected KV head count from model config (cached).

        Only returns num_key_value_heads or num_kv_heads — never falls back
        to num_attention_heads, which is wrong for GQA models (e.g. 32 attn
        heads but 8 KV heads). Returns 0 if unknown (skips validation).

        For MLA models (kv_lora_rank > 0): returns 1 because MLA stores
        compressed latents with H=1, not per-head KV. The config's
        num_key_value_heads is the pre-compression head count (32), but
        the actual cache tensor has shape (B, 1, T, kv_lora_rank).

        Original MLA prefix cache integration by Jinho Jang (eric@jangq.ai).
        This fix required tracing through the full paged cache block hash →
        store → reconstruct → validate pipeline to find the head count
        mismatch that caused 100% cache misses for all MLA models.
        """
        if self._n_kv_heads is not None:
            return self._n_kv_heads
        n_kv = 0
        try:
            # Check model, language_model (VLM wrapper), and model.model (nested)
            candidates = [self.model]
            lm = getattr(self.model, 'language_model', None)
            if lm is not None:
                candidates.append(lm)
            mm = getattr(self.model, 'model', None)
            if mm is not None and mm is not self.model:
                candidates.append(mm)
            # TWO-PASS scan: MLA detection MUST come first. Most MLA models
            # store compressed latent cache with H=1, but Ling/Bailing stores
            # full expanded KV heads in MLAAttention.update_and_fetch().
            # Do not slice those blocks down to one head.
            for model_obj in candidates:
                for attr in ('args', 'config', 'text_config'):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    model_type = str(getattr(cfg, 'model_type', '') or '').lower()
                    if not model_type:
                        tc = getattr(cfg, 'text_config', None)
                        if tc is not None:
                            model_type = str(
                                getattr(tc, 'model_type', '') or ''
                            ).lower()
                    kv_lora_rank = getattr(cfg, 'kv_lora_rank', 0)
                    if not kv_lora_rank:
                        tc = getattr(cfg, 'text_config', None)
                        if tc is not None:
                            kv_lora_rank = getattr(tc, 'kv_lora_rank', 0)
                    if kv_lora_rank and kv_lora_rank > 0:
                        if model_type in ('bailing_hybrid', 'bailing_moe_v2_5'):
                            n_kv = int(getattr(cfg, 'num_attention_heads', 0) or 0)
                            if not n_kv:
                                tc = getattr(cfg, 'text_config', None)
                                if tc is not None:
                                    n_kv = int(
                                        getattr(tc, 'num_attention_heads', 0) or 0
                                    )
                        else:
                            n_kv = 1  # MLA compressed latent, single "head"
                        break
                if n_kv:
                    break
            # Pass 2: if no MLA found, get num_key_value_heads.
            # Gemma 4 VLM stores num_key_value_heads inside
            # model.config.text_config (not directly on model.config),
            # so we also check cfg.text_config as a nested fallback.
            if not n_kv:
                for model_obj in candidates:
                    for attr in ('args', 'config', 'text_config'):
                        cfg = getattr(model_obj, attr, None)
                        if cfg is None:
                            continue
                        n_kv = (
                            getattr(cfg, 'num_key_value_heads', 0)
                            or getattr(cfg, 'num_kv_heads', 0)
                        )
                        if not n_kv:
                            # Nested text_config (VLM wrappers like Gemma 4
                            # whose ModelConfig wraps TextConfig)
                            tc = getattr(cfg, 'text_config', None)
                            if tc is not None:
                                n_kv = (
                                    getattr(tc, 'num_key_value_heads', 0)
                                    or getattr(tc, 'num_kv_heads', 0)
                                )
                        if n_kv:
                            break
                    if n_kv:
                        break
        except Exception:
            pass
        if not isinstance(n_kv, int):
            n_kv = 0
        self._n_kv_heads = n_kv
        return n_kv

    def _get_allowed_n_kv_heads(self) -> set:
        """Get the set of valid KV head counts across ALL layers.

        For uniform-head models (Llama, Qwen, etc.), returns {num_key_value_heads}.
        For mixed-head models (Gemma 4 with interleaved sliding_attention +
        full_attention, where sliding uses num_key_value_heads=16 and full uses
        num_global_key_value_heads=4), returns {4, 16}.

        Used to validate reconstructed paged-cache KV shapes without forcing
        false-positive cache misses on layers that legally have a different
        head count than the primary num_key_value_heads.

        Returns the empty set if no valid counts could be determined (in which
        case head-count validation is skipped — better than forcing false misses).
        """
        if self._allowed_n_kv_heads is not None:
            return self._allowed_n_kv_heads

        allowed: set = set()
        primary = self._get_n_kv_heads()
        if primary > 0:
            allowed.add(primary)

        # Scan model config tree for mixed-head architectures.
        # Gemma 4: num_global_key_value_heads on full_attention layers.
        # Future-proof for other mixed-head variants by walking the same
        # config candidates _get_n_kv_heads uses.
        try:
            candidates = [self.model]
            lm = getattr(self.model, 'language_model', None)
            if lm is not None:
                candidates.append(lm)
            mm = getattr(self.model, 'model', None)
            if mm is not None and mm is not self.model:
                candidates.append(mm)
            for model_obj in candidates:
                for attr in ('args', 'config', 'text_config'):
                    cfg = getattr(model_obj, attr, None)
                    if cfg is None:
                        continue
                    # Mixed full/SWA KV heads. Gemma 4 exposes global KV
                    # heads for full-attention layers; MiMo V2 exposes
                    # swa_num_key_value_heads for rotating SWA layers.
                    for field in (
                        'num_global_key_value_heads',
                        'global_num_key_value_heads',
                        'swa_num_key_value_heads',
                        'num_swa_key_value_heads',
                        'sliding_num_key_value_heads',
                        'local_num_key_value_heads',
                    ):
                        val = getattr(cfg, field, None)
                        if val is None:
                            tc = getattr(cfg, 'text_config', None)
                            if tc is not None:
                                val = getattr(tc, field, None)
                        if isinstance(val, int) and val > 0:
                            allowed.add(val)
        except Exception:
            pass

        self._allowed_n_kv_heads = allowed
        return allowed

    def _touch_disk_chain_access(
        self,
        tokens: List[int],
        num_tokens: int,
        cache_extra_keys: Optional[Any] = None,
    ) -> None:
        """Refresh L2 LRU access for a chain served from paged RAM.

        A paged/L1 hit never reads the disk store, so the chain's L2 rows
        keep their store-time access order (the L1-masking class). Under a
        bounded L2 the global LRU then ranks the HOT conversation's disk
        backing as cold as stale data and evicts it — measured live: two
        filler bursts swept a fully readable 382-block recent chain in the
        paged lane while the disk-only lane (whose reads DO touch L2) kept
        it. Best-effort: non-blocking enqueues, drops under queue pressure
        are harmless (LRU freshness, not correctness).
        """
        disk_store = getattr(self.paged_cache, "_disk_store", None)
        if disk_store is None:
            return
        touch = getattr(disk_store, "_queue_access_update", None)
        if not callable(touch):
            return
        total = max(0, int(num_tokens))
        num_full = total // self.block_size
        if num_full <= 0 and total <= 0:
            return
        try:
            from .paged_cache import compute_block_hash as _compute_chain_hash

            parent_hash = None
            for idx in range(num_full):
                start = idx * self.block_size
                parent_hash = _compute_chain_hash(
                    parent_hash,
                    tokens[start : start + self.block_size],
                    extra_keys=cache_extra_keys_for_token_range(
                        cache_extra_keys, start, start + self.block_size
                    ),
                )
                touch(parent_hash.hex())
            # The TERMINAL PARTIAL was the one row this walk used to skip, and
            # it is the row that matters most: it is by definition a leaf, and
            # the global L2 trim evicts leaves oldest-first. So on an L1-hot
            # chain every full block got its timestamp refreshed each warm
            # turn while the partial kept its store-time stamp -- making the
            # block that completes the chain the FIRST casualty of any budget
            # pressure. Costs up to block_size-1 tokens of reuse for plain KV,
            # and a whole-prompt re-prefill for DSV4 composite chains whose
            # CSA/HCA state lives only in that terminal block.
            tail_len = total - num_full * self.block_size
            if tail_len > 0:
                start = num_full * self.block_size
                tail_hash = _compute_chain_hash(
                    parent_hash,
                    tokens[start:total],
                    extra_keys=cache_extra_keys_for_token_range(
                        cache_extra_keys, start, total
                    ),
                )
                touch(tail_hash.hex())
        except Exception:
            logger.debug(
                "L2 access touch for a RAM-served chain failed (best-effort)",
                exc_info=True,
            )

    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """
        Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing

        Ref ownership:
            get_computed_blocks() increments request refs through touch();
            fallback prefix-index matching validates and pins refs atomically
            under the paged-cache lock.
            Every returned table is registered in _request_tables so the
            scheduler's normal completion cleanup can release those refs.
        """
        if not tokens:
            self._record_fetch_telemetry(
                request_id=request_id,
                match_kind="empty_request",
                origin=None,
                logical_restored_tokens=0,
            )
            return None, tokens

        cache_extra_keys = self._shape_scoped_cache_extra_keys(
            tokens,
            cache_extra_keys,
        )
        _fetch_origin = "chain_block_hit"

        # Primary path: chain-hash lookup. This includes the parent block hash
        # and therefore the full prefix context. Do not use the legacy
        # find_shared_prefix() block-content hash here: it keys only on the
        # current block's token bytes and can replay a repeated 64-token chunk
        # under the wrong history/position, corrupting decoded text.
        cached_blocks, num_cached = self.paged_cache.get_computed_blocks(
            tokens,
            extra_keys=cache_extra_keys,
        )
        # MLLM/DSV4 paths store blocks with N-1 tokens (truncated for re-feed).
        # Always try the N-1 key as well, not only on total miss. After process
        # restart the in-memory partial-size index is empty, so the full-N lookup
        # can restore the full 64-token blocks but miss the terminal partial
        # block (e.g. full prompt remaining=41 while stored N-1 terminal=40).
        # For DSV4 that terminal partial carries the only deepseek_v4 composite
        # CSA/HCA state; accepting the shorter hit yields "2 layers, expected
        # 43" or a forced full prefill. Prefer whichever lookup restores more
        # cached tokens, and release request refs from the loser.
        if len(tokens) > 1:
            alt_blocks, alt_num_cached = self.paged_cache.get_computed_blocks(
                tokens[:-1],
                extra_keys=cache_extra_keys,
            )
            if alt_num_cached > num_cached:
                if cached_blocks:
                    try:
                        self.paged_cache.release_request_refs(
                            BlockTable(
                                request_id=request_id,
                                block_ids=[b.block_id for b in cached_blocks],
                                num_tokens=num_cached,
                            )
                        )
                    except Exception:
                        pass
                cached_blocks, num_cached = alt_blocks, alt_num_cached
                _fetch_origin = "n_minus_1_partial_index"
            elif alt_blocks:
                try:
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id=request_id,
                            block_ids=[b.block_id for b in alt_blocks],
                            num_tokens=alt_num_cached,
                        )
                    )
                except Exception:
                    pass

        # get_computed_blocks() is intentionally block-chain-first. For prompts
        # that extend a previously cached terminal partial block by more than
        # one block, it can stop at the last full block and never consider the
        # longer exact partial prefix. The prefix index stores those terminal
        # partial boundaries, so compare before accepting the shorter hit.
        best_match = self._find_best_prefix_match(
            tokens,
            cache_extra_keys=cache_extra_keys,
        )
        if cached_blocks and best_match:
            if len(best_match[0]) > num_cached:
                try:
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id=request_id,
                            block_ids=[b.block_id for b in cached_blocks],
                            num_tokens=num_cached,
                        )
                    )
                except Exception:
                    pass
                logger.info(
                    "Prefix index exact-partial match for %s extends paged hit "
                    "from %d to %d tokens",
                    request_id,
                    num_cached,
                    len(best_match[0]),
                )
                cached_blocks = []
                num_cached = 0
            else:
                # _find_best_prefix_match() already owns one ref per indexed
                # block. The authoritative chain-hash candidate wins ties and
                # longer matches, so release the unused fallback ownership.
                self._release_pinned_prefix_match(request_id, best_match)
                best_match = None
        if cached_blocks:
            _pre_normalization_cached_blocks = list(cached_blocks)
            _pre_normalization_num_cached = num_cached
            _disk_store = getattr(self.paged_cache, "_disk_store", None)
            # Metadata-only/frugal blocks keep their immutable payload solely
            # in L2. Share one request-local read-through cache across DSV4
            # candidate validators without attaching payloads to the blocks;
            # worker reconstruction remains the sole owner of live hydration.
            _validation_payload_cache: Dict[int, Any] = {}
            dsv4_delta_match = (
                self._normalize_dsv4_delta_candidate(
                    request_id=request_id,
                    blocks=cached_blocks,
                    matched_tokens=num_cached,
                    request_tokens=tokens,
                    disk_store=_disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
                if self._validate_dsv4_terminal
                else None
            )
            dsv4_matched_tokens: Optional[int] = None
            dsv4_replayed_tokens = 0
            if dsv4_delta_match is not None:
                kept_blocks, checkpoint_tokens, dsv4_replayed_tokens = (
                    dsv4_delta_match
                )
                dsv4_matched_tokens = int(num_cached)
                if not kept_blocks or checkpoint_tokens <= 0:
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id=request_id,
                            block_ids=[block.block_id for block in cached_blocks],
                            num_tokens=sum(
                                int(getattr(block, "token_count", 0) or 0)
                                for block in cached_blocks
                            ),
                        )
                    )
                    self._misses += 1
                    self._record_fetch_telemetry(
                        request_id=request_id,
                        match_kind="miss",
                        origin=_fetch_origin,
                        logical_restored_tokens=0,
                        source=self._fetch_telemetry_source(
                            _pre_normalization_cached_blocks
                        ),
                        dsv4_delta_applied=True,
                        miss_reason="dsv4_delta_no_safe_checkpoint",
                        attempted_tokens=_pre_normalization_num_cached,
                    )
                    return None, tokens
                dropped_blocks = cached_blocks[len(kept_blocks) :]
                if dropped_blocks:
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id=request_id,
                            block_ids=[block.block_id for block in dropped_blocks],
                            num_tokens=sum(
                                int(getattr(block, "token_count", 0) or 0)
                                for block in dropped_blocks
                            ),
                        )
                    )
                cached_blocks = kept_blocks
                num_cached = int(checkpoint_tokens)

            # DSV4 v10 delta chains deliberately carry pending rotating-SWA
            # markers between exact periodic anchors.  Normalize such a match
            # to its latest safe anchor before applying the generic mixed-SWA
            # terminal guard; non-DSV4 chains still take the unchanged guard on
            # their original matched boundary.
            _rotating_normalized = False
            if (
                self._validate_rotating_terminal
                and self._rotating_l2_chain_missing_terminal_state(
                    cached_blocks,
                    target_tokens=num_cached,
                    disk_store=_disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
            ):
                _rot_anchor = self._normalize_rotating_candidate(
                    cached_blocks,
                    target_tokens=num_cached,
                    disk_store=_disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
                if _rot_anchor is not None:
                    _kept_rot, _rot_tokens = _rot_anchor
                    _dropped_rot = cached_blocks[len(_kept_rot):]
                    if _dropped_rot:
                        self.paged_cache.release_request_refs(
                            BlockTable(
                                request_id=request_id,
                                block_ids=[b.block_id for b in _dropped_rot],
                                num_tokens=sum(
                                    int(getattr(b, "token_count", 0) or 0)
                                    for b in _dropped_rot
                                ),
                            )
                        )
                    logger.info(
                        "Trimming mixed-SWA paged prefix for %s from %d to %d "
                        "tokens: the matched boundary carried rotating_kv_pending "
                        "markers, so the candidate is normalized to its newest "
                        "exact RotatingKV anchor instead of going cold.",
                        request_id,
                        num_cached,
                        _rot_tokens,
                    )
                    cached_blocks = _kept_rot
                    num_cached = int(_rot_tokens)
                    _rotating_normalized = True
                else:
                    _reject_table = BlockTable(
                        request_id=request_id,
                        block_ids=[cb.block_id for cb in cached_blocks],
                        num_tokens=sum(
                            getattr(cb, "token_count", 0) for cb in cached_blocks
                        ),
                    )
                    self.paged_cache.release_request_refs(_reject_table)
                    logger.info(
                        "Ignoring mixed-SWA paged prefix candidate for %s at %d "
                        "tokens: the matched boundary has rotating_kv_pending "
                        "markers instead of an exact RotatingKV checkpoint, and "
                        "no earlier block in the chain carries one either. "
                        "Prefilling cleanly before recording any hit credit.",
                        request_id,
                        num_cached,
                    )
                    self._misses += 1
                    self._record_fetch_telemetry(
                        request_id=request_id,
                        match_kind="miss",
                        origin=_fetch_origin,
                        logical_restored_tokens=0,
                        source=self._fetch_telemetry_source(cached_blocks),
                        dsv4_delta_applied=dsv4_matched_tokens is not None,
                        miss_reason="rotating_swa_no_anchor",
                        attempted_tokens=num_cached,
                    )
                    return None, tokens

            if (
                self._validate_dsv4_terminal
                and self._dsv4_l2_chain_missing_terminal_state(
                    cached_blocks,
                    _disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
            ):
                _reject_table = BlockTable(
                    request_id=request_id,
                    block_ids=[cb.block_id for cb in cached_blocks],
                    num_tokens=sum(getattr(cb, "token_count", 0) for cb in cached_blocks),
                )
                self.paged_cache.release_request_refs(_reject_table)
                logger.warning(
                    "Ignoring DSV4 paged prefix hit for %s: restored blocks "
                    "contain DeepseekV4Cache pending markers but no terminal "
                    "deepseek_v4 composite state. A non-terminal DSV4 block "
                    "only carries SWA/local fragments; CSA/HCA pool state "
                    "lives in the terminal block, so using this prefix would "
                    "reconstruct an incomplete cache.",
                    request_id,
                )
                self._misses += 1
                self._record_fetch_telemetry(
                    request_id=request_id,
                    match_kind="miss",
                    origin=_fetch_origin,
                    logical_restored_tokens=0,
                    source=self._fetch_telemetry_source(cached_blocks),
                    dsv4_delta_applied=dsv4_matched_tokens is not None,
                    miss_reason="dsv4_l2_missing_terminal_state",
                    attempted_tokens=num_cached,
                )
                return None, tokens

            if (
                self._validate_zaya_terminal
                and self._zaya_l2_chain_missing_terminal_state(
                    cached_blocks,
                    _disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
            ):
                _reject_table = BlockTable(
                    request_id=request_id,
                    block_ids=[cb.block_id for cb in cached_blocks],
                    num_tokens=sum(getattr(cb, "token_count", 0) for cb in cached_blocks),
                )
                self.paged_cache.release_request_refs(_reject_table)
                logger.warning(
                    "Ignoring ZAYA paged prefix hit for %s: restored blocks "
                    "contain typed zaya_cca KV pages but no terminal CCA "
                    "conv_state/prev_hs payload. ZAYA CCA state is "
                    "path-dependent, so this prefix must re-prefill cleanly.",
                    request_id,
                )
                self._misses += 1
                self._record_fetch_telemetry(
                    request_id=request_id,
                    match_kind="miss",
                    origin=_fetch_origin,
                    logical_restored_tokens=0,
                    source=self._fetch_telemetry_source(cached_blocks),
                    dsv4_delta_applied=dsv4_matched_tokens is not None,
                    miss_reason="zaya_l2_missing_terminal_state",
                    attempted_tokens=num_cached,
                )
                return None, tokens

            # get_computed_blocks() holds request refs for each returned
            # block via touch(); the prefix-index path below uses
            # increment_ref directly because it starts from block IDs.
            block_table = self.paged_cache.create_block_table(request_id)
            for cb in cached_blocks:
                block_table.block_ids.append(cb.block_id)
                block_table.num_tokens += cb.token_count

            if dsv4_matched_tokens is not None:
                block_table.matched_tokens = dsv4_matched_tokens
                block_table.checkpoint_tokens = block_table.num_tokens
                block_table.replayed_tokens = int(dsv4_replayed_tokens)

            remaining = tokens[block_table.num_tokens:]
            self._hits += 1
            self._tokens_saved += block_table.num_tokens
            self._hit_credits[request_id] = block_table.num_tokens
            # prefix_key = the terminal block's chained content hash: the same
            # value the store line logs for the publication this hit came
            # from, so a restore can be bound to its publisher by identity,
            # not by token count (S5 audit: length is eligibility only).
            logger.info(
                "Paged cache hit for %s: %d blocks, checkpoint_tokens=%d%s prefix_key=%s",
                request_id,
                len(cached_blocks),
                block_table.num_tokens,
                (
                    f", matched_tokens={dsv4_matched_tokens}, "
                    f"replayed_tokens={dsv4_replayed_tokens}"
                    if dsv4_matched_tokens is not None
                    else ""
                ),
                self.prefix_key_for_blocks(cached_blocks),
            )
            # fetch_cache() owns one request ref for every returned block.  The
            # scheduler releases completed hits through _request_tables before
            # storing the refreshed prompt snapshot.  Without registering the
            # fetched table here, successful L1/L2 hits retain one ref forever;
            # a small block pool then fills with non-evictable blocks after the
            # first agent/tool iteration.
            self._request_tables[request_id] = BlockCacheEntry(
                block_table=block_table,
                cache_data=None,
                last_access=time.time(),
                cache_type="assistant",
            )
            self._touch_disk_chain_access(
                tokens,
                block_table.num_tokens,
                cache_extra_keys,
            )
            self._record_fetch_telemetry(
                request_id=request_id,
                match_kind=(
                    "rotating_swa_anchor_normalized"
                    if _rotating_normalized
                    else "dsv4_delta_checkpoint"
                    if dsv4_matched_tokens is not None
                    else _fetch_origin
                ),
                origin=_fetch_origin,
                logical_restored_tokens=block_table.num_tokens,
                native_companion_boundary=(
                    block_table.num_tokens
                    if _rotating_normalized or dsv4_matched_tokens is not None
                    else None
                ),
                source=self._fetch_telemetry_source(cached_blocks),
                cache_key=self._fetch_telemetry_cache_key(cached_blocks),
                dsv4_delta_applied=dsv4_matched_tokens is not None,
                rotating_swa_normalized=_rotating_normalized,
            )
            return block_table, remaining

        # Try prefix index for longer matches
        if best_match:
            matched_tokens, matched_block_ids = best_match
            matched_blocks = [
                self.paged_cache.allocated_blocks.get(block_id)
                for block_id in matched_block_ids
            ]
            matched_blocks = [b for b in matched_blocks if b is not None]
            pinned_table = BlockTable(
                request_id=request_id,
                block_ids=list(matched_block_ids),
                num_tokens=sum(
                    int(getattr(block, "token_count", 0) or 0)
                    for block in matched_blocks
                ),
            )
            _disk_store = getattr(self.paged_cache, "_disk_store", None)
            _validation_payload_cache: Dict[int, Any] = {}
            dsv4_delta_match = (
                self._normalize_dsv4_delta_candidate(
                    request_id=request_id,
                    blocks=matched_blocks,
                    matched_tokens=len(matched_tokens),
                    request_tokens=tokens,
                    disk_store=_disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
                if self._validate_dsv4_terminal
                else None
            )
            dsv4_matched_tokens: Optional[int] = None
            dsv4_replayed_tokens = 0
            if dsv4_delta_match is not None:
                kept_blocks, checkpoint_tokens, dsv4_replayed_tokens = (
                    dsv4_delta_match
                )
                dsv4_matched_tokens = len(matched_tokens)
                if not kept_blocks or checkpoint_tokens <= 0:
                    self.paged_cache.release_request_refs(pinned_table)
                    self._misses += 1
                    self._record_fetch_telemetry(
                        request_id=request_id,
                        match_kind="miss",
                        origin="prefix_index_match",
                        logical_restored_tokens=0,
                        source=self._fetch_telemetry_source(matched_blocks),
                        dsv4_delta_applied=True,
                        miss_reason="dsv4_delta_no_safe_checkpoint",
                        attempted_tokens=len(matched_tokens),
                    )
                    return None, tokens
                dropped_blocks = matched_blocks[len(kept_blocks) :]
                if dropped_blocks:
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id=request_id,
                            block_ids=[block.block_id for block in dropped_blocks],
                            num_tokens=sum(
                                int(getattr(block, "token_count", 0) or 0)
                                for block in dropped_blocks
                            ),
                        )
                    )
                matched_blocks = kept_blocks
                pinned_table.block_ids = [block.block_id for block in kept_blocks]
                pinned_table.num_tokens = int(checkpoint_tokens)
                pinned_table.matched_tokens = dsv4_matched_tokens
                pinned_table.checkpoint_tokens = int(checkpoint_tokens)
                pinned_table.replayed_tokens = int(dsv4_replayed_tokens)

            _rotating_normalized = False
            if (
                self._validate_rotating_terminal
                and self._rotating_l2_chain_missing_terminal_state(
                    matched_blocks,
                    target_tokens=pinned_table.num_tokens,
                    disk_store=_disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
            ):
                # SAME normalization as the paged lane above. A fix applied to
                # only one of these two lanes is inert in the other -- that has
                # been the failure mode here repeatedly, so both walk back to
                # the newest exact anchor before giving up.
                _rot_anchor = self._normalize_rotating_candidate(
                    matched_blocks,
                    target_tokens=pinned_table.num_tokens,
                    disk_store=_disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
                if _rot_anchor is not None:
                    _kept_rot, _rot_tokens = _rot_anchor
                    _dropped_rot = matched_blocks[len(_kept_rot):]
                    if _dropped_rot:
                        self.paged_cache.release_request_refs(
                            BlockTable(
                                request_id=request_id,
                                block_ids=[b.block_id for b in _dropped_rot],
                                num_tokens=sum(
                                    int(getattr(b, "token_count", 0) or 0)
                                    for b in _dropped_rot
                                ),
                            )
                        )
                    logger.info(
                        "Trimming mixed-SWA prefix-index candidate for %s from "
                        "%d to %d tokens: normalized to its newest exact "
                        "RotatingKV anchor instead of going cold.",
                        request_id,
                        pinned_table.num_tokens,
                        _rot_tokens,
                    )
                    matched_blocks = _kept_rot
                    pinned_table.block_ids = [b.block_id for b in _kept_rot]
                    pinned_table.num_tokens = int(_rot_tokens)
                    pinned_table.checkpoint_tokens = int(_rot_tokens)
                    _rotating_normalized = True
                else:
                    self.paged_cache.release_request_refs(pinned_table)
                    logger.info(
                        "Ignoring mixed-SWA prefix-index candidate for %s at %d "
                        "tokens: the matched boundary has no exact RotatingKV "
                        "checkpoint, and no earlier block in the chain has one.",
                        request_id,
                        pinned_table.num_tokens,
                    )
                    self._misses += 1
                    self._record_fetch_telemetry(
                        request_id=request_id,
                        match_kind="miss",
                        origin="prefix_index_match",
                        logical_restored_tokens=0,
                        source=self._fetch_telemetry_source(matched_blocks),
                        dsv4_delta_applied=dsv4_matched_tokens is not None,
                        miss_reason="rotating_swa_no_anchor",
                        attempted_tokens=pinned_table.num_tokens,
                    )
                    return None, tokens
            if (
                self._validate_dsv4_terminal
                and self._dsv4_l2_chain_missing_terminal_state(
                    matched_blocks,
                    _disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
            ):
                self.paged_cache.release_request_refs(pinned_table)
                logger.warning(
                    "Ignoring DSV4 prefix-index hit for %s: matched blocks "
                    "contain DeepseekV4Cache pending markers but no terminal "
                    "deepseek_v4 composite state.",
                    request_id,
                )
                self._misses += 1
                self._record_fetch_telemetry(
                    request_id=request_id,
                    match_kind="miss",
                    origin="prefix_index_match",
                    logical_restored_tokens=0,
                    source=self._fetch_telemetry_source(matched_blocks),
                    dsv4_delta_applied=dsv4_matched_tokens is not None,
                    miss_reason="dsv4_l2_missing_terminal_state",
                    attempted_tokens=pinned_table.num_tokens,
                )
                return None, tokens
            if (
                self._validate_zaya_terminal
                and self._zaya_l2_chain_missing_terminal_state(
                    matched_blocks,
                    _disk_store,
                    validation_payload_cache=_validation_payload_cache,
                )
            ):
                self.paged_cache.release_request_refs(pinned_table)
                logger.warning(
                    "Ignoring ZAYA prefix-index hit for %s: matched blocks "
                    "contain zaya_cca KV pages but no terminal CCA state.",
                    request_id,
                )
                self._misses += 1
                self._record_fetch_telemetry(
                    request_id=request_id,
                    match_kind="miss",
                    origin="prefix_index_match",
                    logical_restored_tokens=0,
                    source=self._fetch_telemetry_source(matched_blocks),
                    dsv4_delta_applied=dsv4_matched_tokens is not None,
                    miss_reason="zaya_l2_missing_terminal_state",
                    attempted_tokens=pinned_table.num_tokens,
                )
                return None, tokens

            # The index lookup pinned this exact chain before releasing the
            # paged-cache lock, so no second increment is needed here.
            block_table = pinned_table

            consumed_tokens = int(block_table.num_tokens)
            remaining = tokens[consumed_tokens:]
            self._hits += 1
            self._tokens_saved += consumed_tokens
            self._hit_credits[request_id] = consumed_tokens

            logger.info(
                "Prefix index hit for %s: matched_tokens=%d "
                "checkpoint_tokens=%d replayed_tokens=%d",
                request_id,
                (
                    dsv4_matched_tokens
                    if dsv4_matched_tokens is not None
                    else len(matched_tokens)
                ),
                consumed_tokens,
                dsv4_replayed_tokens,
            )

            # Prefix-index hits acquire refs with increment_ref(), so they need
            # the same completion ownership registration as chain-hash hits.
            self._request_tables[request_id] = BlockCacheEntry(
                block_table=block_table,
                cache_data=None,
                last_access=time.time(),
                cache_type="assistant",
            )
            self._touch_disk_chain_access(
                tokens,
                block_table.num_tokens,
                cache_extra_keys,
            )
            self._record_fetch_telemetry(
                request_id=request_id,
                match_kind=(
                    "rotating_swa_anchor_normalized"
                    if _rotating_normalized
                    else "dsv4_delta_checkpoint"
                    if dsv4_matched_tokens is not None
                    else "prefix_index_match"
                ),
                origin="prefix_index_match",
                logical_restored_tokens=block_table.num_tokens,
                native_companion_boundary=(
                    block_table.num_tokens
                    if _rotating_normalized or dsv4_matched_tokens is not None
                    else None
                ),
                source=self._fetch_telemetry_source(matched_blocks),
                cache_key=self._fetch_telemetry_cache_key(matched_blocks),
                dsv4_delta_applied=dsv4_matched_tokens is not None,
                rotating_swa_normalized=_rotating_normalized,
            )
            return block_table, remaining

        # No cache hit
        self._misses += 1
        logger.debug(f"Cache miss for {request_id}")
        self._record_fetch_telemetry(
            request_id=request_id,
            match_kind="miss",
            origin=None,
            logical_restored_tokens=0,
            miss_reason="no_candidate",
        )
        return None, tokens

    def _release_pinned_prefix_match(
        self,
        request_id: str,
        match: Tuple[List[int], List[int]],
    ) -> None:
        """Release ownership acquired by an unused prefix-index candidate."""
        matched_tokens, block_ids = match
        self.paged_cache.release_request_refs(
            BlockTable(
                request_id=request_id,
                block_ids=list(block_ids),
                num_tokens=len(matched_tokens),
            )
        )

    def _normalize_dsv4_delta_candidate(
        self,
        *,
        request_id: str,
        blocks: List[Any],
        matched_tokens: int,
        request_tokens: List[int],
        disk_store: Optional[Any],
        validation_payload_cache: Optional[Dict[int, Any]] = None,
    ) -> Optional[Tuple[List[Any], int, int]]:
        """Map a longest DSV4 token match to its latest safe checkpoint.

        Returns ``None`` for non-delta payloads. For a native delta chain the
        tuple is ``(kept_blocks, checkpoint_tokens, replayed_tokens)``. An
        empty kept list means the chain is DSV4 delta data but has no safe
        checkpoint and must be treated as a cache miss.

        A request that needs at most the conventional final N-1 kickoff token
        may restore an exact terminal anchor. Any materially changed suffix
        must restore either the latest 2K-periodic anchor or an explicitly
        stamped full 256-token checkpoint and replay the already-matched tail.
        A prior request terminal remains ineligible unless its full-block state
        independently carries that append-safe stamp.
        """
        boundaries: List[Tuple[int, int, bool, bool, bool]] = []
        cumulative = 0
        saw_delta = False
        expected_layers = getattr(self, "_expected_num_layers", None)
        for index, block in enumerate(blocks or ()):
            token_count = int(getattr(block, "token_count", 0) or 0)
            if token_count <= 0:
                return ([], 0, max(0, int(matched_tokens)))
            cumulative += token_count
            entries = list(
                self._iter_terminal_check_entries(
                    block,
                    disk_store,
                    validation_payload_cache=validation_payload_cache,
                )
            )
            delta_entries = [
                entry
                for entry in entries
                if isinstance(entry, (tuple, list))
                and entry
                and entry[0] == "deepseek_v4_delta_v1"
            ]
            if not delta_entries:
                if saw_delta:
                    return ([], 0, max(0, int(matched_tokens)))
                continue
            saw_delta = True
            interval_start = cumulative - token_count
            try:
                for entry in delta_entries:
                    record = entry[1] if len(entry) > 1 else None
                    if (
                        not isinstance(record, dict)
                        or int(record.get("start_token", -1)) != interval_start
                        or int(record.get("end_token", -1)) != cumulative
                    ):
                        return ([], 0, max(0, int(matched_tokens)))
            except (TypeError, ValueError):
                return ([], 0, max(0, int(matched_tokens)))
            complete = _block_has_complete_dsv4_delta_anchor(
                entries,
                expected_layers=expected_layers,
            )
            periodic = complete and _block_has_complete_dsv4_delta_anchor(
                entries,
                expected_layers=expected_layers,
                periodic_only=True,
            )
            terminal = complete and any(
                (_dsv4_delta_anchor_kind(entry) or (False, False))[1]
                for entry in delta_entries
            )
            append_safe = complete and _block_has_complete_dsv4_append_safe_anchor(
                entries,
                target_tokens=cumulative,
                expected_layers=expected_layers,
            )
            boundaries.append(
                (index + 1, cumulative, periodic, terminal, append_safe)
            )

        if not saw_delta:
            return None

        matched = max(0, min(int(matched_tokens), len(request_tokens)))
        # A terminal anchor is the exact state after `matched` tokens, and the
        # chain hash already proves those tokens are a prefix of this request,
        # so appending the remainder is sound for any suffix length. The <= 1
        # bound was sized for a single N-1 kickoff token and silently excludes
        # DSV4's real generation rail, which is 3 tokens.
        #
        # Measured cost of that exclusion at 127k: the answer pass matches
        # 127,444 but falls back to the block-aligned 127,232 checkpoint and
        # re-prefills 215 tokens for 5.8s — 53% of the whole decode span — to
        # change ONE token (<think> -> </think>). Both passes pay it.
        #
        # The budget is a POLICY THROTTLE, not a correctness boundary. Nothing
        # in the restore path makes a 4-token suffix sound and a 5-token suffix
        # unsound: `terminal` already requires a complete record (see the
        # boundary scan above), a terminal payload is built by the same
        # export_block_delta as a periodic one, and selection can never admit a
        # boundary ahead of the chain-hash-verified `matched`. Restoring at a
        # captured boundary and moving forward never rewinds path-dependent
        # state — it is the periodic FALLBACK that re-derives tokens under
        # fresh chunk boundaries and so drifts further from the original pass.
        # The bound exists only to keep the widened path's blast radius small.
        # VMLX_DSV4_TERMINAL_ANCHOR_TAIL sets the permitted suffix length;
        # 0 restores the historical <= 1.
        _terminal_tail_budget = _dsv4_terminal_anchor_tail_budget()
        allow_terminal = len(request_tokens) - matched <= max(1, _terminal_tail_budget)
        selected: Optional[Tuple[int, int, bool, bool, bool]] = None
        for boundary in boundaries:
            _, boundary_tokens, periodic, terminal, append_safe = boundary
            if boundary_tokens > matched:
                break
            if periodic or append_safe or (allow_terminal and terminal):
                selected = boundary
        if selected is None:
            logger.info(
                "DSV4 native delta match for %s reached %d tokens but has no "
                "safe %s anchor; prefilling cleanly",
                request_id,
                matched,
                (
                    "terminal/periodic/aligned"
                    if allow_terminal
                    else "periodic/aligned"
                ),
            )
            return ([], 0, matched)

        keep_count, checkpoint, periodic, terminal, append_safe = selected
        replayed = max(0, matched - checkpoint)
        anchor_kind = (
            "periodic"
            if periodic
            else "aligned"
            if append_safe
            else "terminal"
        )
        logger.info(
            "DSV4 native delta match for %s: matched_tokens=%d "
            "checkpoint_tokens=%d replayed_tokens=%d anchor=%s",
            request_id,
            matched,
            checkpoint,
            replayed,
            anchor_kind,
        )
        return (list(blocks[:keep_count]), checkpoint, replayed)

    @staticmethod
    def _iter_terminal_check_entries(
        block: Any,
        disk_store: Optional[Any] = None,
        *,
        validation_payload_cache: Optional[Dict[int, Any]] = None,
    ):
        cache_key = id(block)
        if (
            validation_payload_cache is not None
            and cache_key in validation_payload_cache
        ):
            cache_data = validation_payload_cache[cache_key]
        else:
            cache_data = getattr(block, "cache_data", None)
            if cache_data is None and disk_store is not None:
                block_hash = getattr(block, "block_hash", None)
                if block_hash is not None:
                    try:
                        metadata_reader = getattr(
                            disk_store,
                            "read_block_validation_entries",
                            None,
                        )
                        if callable(metadata_reader):
                            cache_data = metadata_reader(block_hash)
                        if cache_data is None:
                            cache_data = disk_store.read_block(block_hash)
                    except Exception:
                        cache_data = None
            if validation_payload_cache is not None:
                validation_payload_cache[cache_key] = cache_data
        for entry in cache_data or []:
            yield entry

    @staticmethod
    def _dsv4_l2_chain_missing_terminal_state(
        cached_blocks: List[Any],
        disk_store: Optional[Any] = None,
        *,
        validation_payload_cache: Optional[Dict[int, Any]] = None,
    ) -> bool:
        """True when an L2 hit restored only non-terminal DSV4 blocks.

        DSV4 block L2 intentionally writes cheap ``deepseek_v4_pending``
        markers for non-terminal blocks and stores the full SWA+CSA/HCA
        composite state only on the terminal block. A chain hit that stops
        before that terminal block is not a usable prefix cache: rebuilding it
        yields a few SWA layers and misses every DeepseekV4Cache layer.
        Treat it as a miss so the scheduler does a full prefill instead of
        running with incomplete compressor/indexer pool state.
        """
        saw_pending = False
        saw_terminal = False
        for block in cached_blocks or []:
            for entry in BlockAwarePrefixCache._iter_terminal_check_entries(
                block,
                disk_store,
                validation_payload_cache=validation_payload_cache,
            ):
                if not isinstance(entry, (tuple, list)) or not entry:
                    continue
                tag = entry[0]
                if tag == "deepseek_v4_pending":
                    saw_pending = True
                elif tag == "deepseek_v4":
                    saw_terminal = True
        return saw_pending and not saw_terminal

    @staticmethod
    def _zaya_l2_chain_missing_terminal_state(
        cached_blocks: List[Any],
        disk_store: Optional[Any] = None,
        *,
        validation_payload_cache: Optional[Dict[int, Any]] = None,
    ) -> bool:
        """True when a ZAYA hit has KV pages but lacks terminal CCA state."""
        saw_zaya = False
        saw_terminal = False
        for block in cached_blocks or []:
            for entry in BlockAwarePrefixCache._iter_terminal_check_entries(
                block,
                disk_store,
                validation_payload_cache=validation_payload_cache,
            ):
                if not isinstance(entry, (tuple, list)) or not entry:
                    continue
                if entry[0] != "zaya_cca":
                    continue
                saw_zaya = True
                if len(entry) > 2 and entry[2] is not None:
                    saw_terminal = True
        return saw_zaya and not saw_terminal

    @staticmethod
    def _normalize_rotating_candidate(
        cached_blocks: List[Any],
        *,
        target_tokens: int,
        disk_store: Optional[Any] = None,
        validation_payload_cache: Optional[Dict[int, Any]] = None,
    ) -> Optional[Tuple[List[Any], int]]:
        """Trim a mixed-SWA candidate back to its newest exact RotatingKV anchor.

        The store keeps exact ``rotating_kv`` records only for a stored prompt's
        terminal cluster (its last two full-block boundaries plus the terminal
        record) -- older boundaries are deliberately ``rotating_kv_pending``,
        because materialising a full SWA window per page would cost a max-size
        ring for every block. Extending that fan-out is not an option either:
        ``_rotating_previous_block_window`` can only rebuild boundaries that are
        still inside the live ring's concat overhang, so far ones fail the
        retained-history checks no matter how large a budget is allowed.

        A multi-turn follow-up therefore matches FAR from any terminal and lands
        on a pending marker, and the whole candidate is discarded -- going cold
        even though earlier turns left exact anchors in the very same chain.

        Walking back to the newest anchor recovers the shared history at zero
        storage cost. Mirrors ``_normalize_dsv4_delta_candidate``, which solves
        the identical problem for DSV4 delta chains.

        Returns ``(kept_blocks, anchor_tokens)`` or ``None`` when no prefix of
        the candidate ends on a reconstructable boundary.
        """
        if not cached_blocks or int(target_tokens or 0) <= 0:
            return None
        counts = [
            int(getattr(block, "token_count", 0) or 0) for block in cached_blocks
        ]
        # Longest first, and never the full candidate: the caller only reaches
        # here because the full candidate already failed the terminal check.
        for end in range(len(cached_blocks) - 1, 0, -1):
            anchor_tokens = sum(counts[:end])
            if anchor_tokens <= 0:
                break
            prefix = cached_blocks[:end]
            if not BlockAwarePrefixCache._rotating_l2_chain_missing_terminal_state(
                prefix,
                target_tokens=anchor_tokens,
                disk_store=disk_store,
                validation_payload_cache=validation_payload_cache,
            ):
                return prefix, anchor_tokens
        return None

    @staticmethod
    def _rotating_l2_chain_missing_terminal_state(
        cached_blocks: List[Any],
        *,
        target_tokens: int,
        disk_store: Optional[Any] = None,
        validation_payload_cache: Optional[Dict[int, Any]] = None,
    ) -> bool:
        """True when a matched mixed-SWA boundary cannot be reconstructed.

        RotatingKV state is path-dependent: an interior block is durable as a
        token/hash chain node, but ``rotating_kv_pending`` deliberately carries
        no ring-buffer payload.  Selecting that boundary as a hit causes an
        avoidable SSD read and a later 12/48-layer reconstruction failure for
        Laguna before the scheduler rolls the hit credit back to zero.

        Reject the candidate before crediting it.  A terminal block with exact
        ``rotating_kv`` entries remains eligible; malformed offsets or window
        shapes are still rejected by the normal reconstruction validator.
        """
        if not cached_blocks or int(target_tokens or 0) <= 0:
            return False

        terminal_entries = list(
            BlockAwarePrefixCache._iter_terminal_check_entries(
                cached_blocks[-1],
                disk_store,
                validation_payload_cache=validation_payload_cache,
            )
        )
        rotating_entries = [
            entry
            for entry in terminal_entries
            if isinstance(entry, (tuple, list))
            and entry
            and entry[0] == "rotating_kv"
        ]
        saw_pending = any(
            isinstance(entry, (tuple, list))
            and entry
            and entry[0] == "rotating_kv_pending"
            for entry in terminal_entries
        )
        if saw_pending:
            return True
        if not rotating_entries:
            # Blind-spot repair for chains written by older builds: a warm
            # mixed-SWA media store used to cut every terminal rotating layer
            # to a bare ("skip",) when the extracted post-generation ring had
            # rolled past the store key (the cutters now emit
            # rotating_kv_pending).  Such a terminal block carries NO rotating
            # marker at all, so this guard mistook the chain for a
            # non-rotating family, the full boundary was credited, and
            # reconstruction later found no exact window and returned an
            # empty-handed None — the live gemma4 2638-token silent
            # re-prefill.  Rotating families stamp every non-terminal block
            # with rotating_kv or rotating_kv_pending, so when the terminal
            # block contains skip entries one probe of the previous block
            # decides the family.  Dense-family terminals carry no skip
            # entries, so they never pay the probe read.
            if len(cached_blocks) >= 2 and any(
                isinstance(entry, (tuple, list))
                and entry
                and entry[0] == "skip"
                for entry in terminal_entries
            ):
                for entry in BlockAwarePrefixCache._iter_terminal_check_entries(
                    cached_blocks[-2],
                    disk_store,
                    validation_payload_cache=validation_payload_cache,
                ):
                    if (
                        isinstance(entry, (tuple, list))
                        and entry
                        and entry[0] in ("rotating_kv", "rotating_kv_pending")
                    ):
                        return True
            return False
        return not _block_has_complete_rotating_terminal(
            terminal_entries,
            target_tokens=int(target_tokens),
            expected_layers=len(rotating_entries),
        )

    @staticmethod
    def _wait_native_write_fence_blocks(
        disk_store: Any,
        fence_id: str,
        target_hashes: set[bytes],
        *,
        timeout: Optional[float],
    ) -> set[bytes]:
        wait_for_fence_blocks = getattr(
            disk_store,
            "wait_for_write_fence_blocks",
            None,
        )
        if not callable(wait_for_fence_blocks) or not target_hashes:
            return set()
        try:
            return set(
                wait_for_fence_blocks(
                    fence_id,
                    list(target_hashes),
                    timeout=timeout,
                    allow_partial=True,
                )
            )
        except TypeError:
            # Compatibility for a non-BlockDiskStore test/third-party backend.
            # It cannot expose partial terminal retention, but still preserves
            # the former all-or-nothing safe fallback contract.
            return set(
                wait_for_fence_blocks(
                    fence_id,
                    list(target_hashes),
                    timeout=5.0 if timeout is None else timeout,
                )
            )

    def _release_durable_native_holds(
        self,
        durable_hashes: set[bytes],
        disk_only_fallbacks: Dict[bytes, Tuple[Any, Any]],
        native_paged_holds: Dict[bytes, Any],
    ) -> None:
        """Release only exact native hashes proven retained after eviction."""

        released_paged = False
        for block_hash in set(durable_hashes):
            fallback = disk_only_fallbacks.pop(block_hash, None)
            if fallback is not None:
                block, fallback_payload = fallback
                current_hash = getattr(block, "block_hash", block_hash)
                if (
                    fallback_payload is not None
                    and current_hash == block_hash
                    and getattr(block, "cache_data", None) is fallback_payload
                ):
                    release_when_unreferenced = getattr(
                        self.paged_cache,
                        "release_resident_payload_when_unreferenced",
                        None,
                    )
                    if callable(release_when_unreferenced):
                        release_when_unreferenced(block)
                    else:
                        # Compatibility backends without the ref-aware helper
                        # may release only an unreferenced fallback. An active
                        # payload remains safe in RAM and becomes normally
                        # evictable after its reference lifecycle completes.
                        if int(getattr(block, "ref_count", 0) or 0) <= 0:
                            release_payload = getattr(
                                self.paged_cache,
                                "release_resident_payload",
                                None,
                            )
                            if callable(release_payload):
                                release_payload(block)
                            else:
                                block.cache_data = None
                                block.cache_data_from_disk = False
                                block.keep_resident = False
                        else:
                            make_evictable = getattr(
                                self.paged_cache,
                                "make_resident_payload_evictable",
                                None,
                            )
                            if callable(make_evictable):
                                make_evictable(block)

            block = native_paged_holds.pop(block_hash, None)
            if block is not None:
                current_hash = getattr(block, "block_hash", block_hash)
                if current_hash == block_hash:
                    self.paged_cache.make_resident_payload_evictable(block)
                    released_paged = True

        if released_paged:
            enforce_budget = getattr(self.paged_cache, "enforce_byte_budget", None)
            if callable(enforce_budget):
                enforce_budget()

    def _schedule_eventual_native_fence_release(
        self,
        request_id: str,
        disk_store: Any,
        fence_id: str,
        disk_only_fallbacks: Dict[bytes, Tuple[Any, Any]],
        native_paged_holds: Dict[bytes, Any],
    ) -> None:
        """Release timeout-retained RAM only when the writer later settles."""

        target_hashes = set(disk_only_fallbacks) | set(native_paged_holds)
        if not target_hashes:
            return

        def _finish() -> None:
            try:
                durable_hashes = self._wait_native_write_fence_blocks(
                    disk_store,
                    fence_id,
                    target_hashes,
                    timeout=None,
                )
                self._release_durable_native_holds(
                    durable_hashes,
                    disk_only_fallbacks,
                    native_paged_holds,
                )
                if durable_hashes:
                    logger.info(
                        "Native block-disk fence eventually released %d RAM "
                        "hold(s) for %s",
                        len(durable_hashes),
                        request_id,
                    )
            except Exception as wait_error:
                logger.warning(
                    "Native block-disk eventual fence wait failed for %s: %s",
                    request_id,
                    wait_error,
                )

        threading.Thread(
            target=_finish,
            daemon=True,
            name=f"native-cache-fence-{str(fence_id)[-8:]}",
        ).start()

    def _settle_native_write_fence(
        self,
        request_id: str,
        write_fence: Dict[str, Any],
    ) -> None:
        """Release native RAM holds only after post-eviction L2 retention.

        DSV4 and other path-dependent native records cannot use raw index
        readability as their durability boundary: the background writer commits
        each row before applying the aggregate SSD budget.  Seal the request
        fence, wait for its post-eviction terminal state, and verify every exact
        hash under the disk store's mutation guard before releasing temporary
        publication state. Paged-RAM native holds remain fail-closed. SSD-only
        failures instead retire the nondurable suffix: retaining it would create
        a hidden, unbounded RAM cache in direct conflict with the selected tier.
        """
        disk_store = write_fence.get("disk_store")
        fence_id = write_fence.get("fence_id")
        disk_only_fallbacks = dict(
            write_fence.get("disk_only_fallbacks") or {}
        )
        native_paged_holds = dict(
            write_fence.get("native_paged_holds") or {}
        )
        target_hashes = set(disk_only_fallbacks) | set(native_paged_holds)
        durable_hashes: set[bytes] = set()
        eventual_wait_allowed = False
        fence_wait_timeout = (
            float(
                getattr(
                    self,
                    "_native_block_disk_admission_timeout",
                    30.0,
                )
            )
            if disk_only_fallbacks
            else 5.0
        )

        sealed = False
        if disk_store is not None and fence_id is not None:
            try:
                sealed = bool(
                    disk_store.seal_write_fence(
                        fence_id,
                        producer_aborted=False,
                    )
                )
                write_fence["seal_attempted"] = True
                write_fence["sealed"] = sealed
            except Exception as fence_error:
                logger.warning(
                    "Block disk: could not seal native write fence %s for %s: %s",
                    fence_id,
                    request_id,
                    fence_error,
                )

        if sealed and target_hashes:
            if not callable(
                getattr(disk_store, "wait_for_write_fence_blocks", None)
            ):
                logger.warning(
                    "Native block-disk fence wait is unavailable for %s; "
                    "preserving paged native holds and retiring SSD-only "
                    "nondurable blocks",
                    request_id,
                )
            else:
                try:
                    durable_hashes = self._wait_native_write_fence_blocks(
                        disk_store,
                        fence_id,
                        target_hashes,
                        timeout=fence_wait_timeout,
                    )
                    eventual_wait_allowed = True
                except Exception as wait_error:
                    logger.warning(
                        "Native block-disk fence wait failed for %s: %s",
                        request_id,
                        wait_error,
                    )

        self._release_durable_native_holds(
            durable_hashes,
            disk_only_fallbacks,
            native_paged_holds,
        )

        discarded_disk_only = 0
        if disk_only_fallbacks:
            discarded_disk_only = self._discard_nondurable_disk_only_blocks(
                disk_only_fallbacks
            )
            disk_only_fallbacks.clear()
            logger.error(
                "Block-disk-only post-eviction fence discarded %d nondurable "
                "block(s); persistent RAM fallback remains 0 bytes",
                discarded_disk_only,
            )
        elif write_fence.get("disk_only_fallbacks"):
            logger.info(
                "Block-disk-only post-eviction fence committed %d block(s); "
                "persistent RAM KV payloads=0",
                len(durable_hashes),
            )

        pending_native = len(native_paged_holds)
        if pending_native:
            logger.warning(
                "Native paged cache retained %d post-eviction-pending RAM "
                "block(s); those records are not eligible for eviction",
                pending_native,
            )

        if (
            sealed
            and eventual_wait_allowed
            and native_paged_holds
        ):
            self._schedule_eventual_native_fence_release(
                request_id,
                disk_store,
                fence_id,
                {},
                native_paged_holds,
            )

    def _discard_nondurable_disk_only_blocks(
        self,
        pending: dict[bytes, tuple[Any, Any]],
    ) -> int:
        """Invalidate an SSD-only suffix without ever installing its payload."""

        retired_ids: set[int] = set()
        discard = getattr(
            self.paged_cache,
            "discard_nondurable_cache_blocks",
            None,
        )
        if callable(discard):
            try:
                retired_ids.update(int(value) for value in discard(pending))
            except Exception:
                logger.exception(
                    "Paged cache failed to retire nondurable SSD-only blocks"
                )

        # Compatibility/fail-safe path for a third-party or test manager. Even
        # if its lookup map cannot be edited here, never leave a tensor payload
        # behind under a zero-RAM policy; reset_hash makes the prefix-index
        # validator reject the stale numeric block id.
        for block_hash, (block, _payload) in pending.items():
            if block is None:
                continue
            block_id = getattr(block, "block_id", None)
            current_hash = getattr(block, "block_hash", None)
            if current_hash == block_hash and block_id is not None:
                retired_ids.add(int(block_id))
            release = getattr(self.paged_cache, "release_resident_payload", None)
            if callable(release):
                try:
                    release(block)
                except Exception:
                    block.cache_data = None
            else:
                block.cache_data = None
                block.cache_data_from_disk = False
                block.keep_resident = False
            if current_hash == block_hash:
                reset_hash = getattr(block, "reset_hash", None)
                if callable(reset_hash):
                    reset_hash()

        if retired_ids:
            lock = getattr(self.paged_cache, "_lock", None)

            def _prune_index() -> None:
                for prefix_hash, entry in list(self._prefix_index.items()):
                    block_ids = entry[1] if len(entry) > 1 else ()
                    if any(int(block_id) in retired_ids for block_id in block_ids):
                        del self._prefix_index[prefix_hash]

            if lock is None:
                _prune_index()
            else:
                with lock:
                    _prune_index()
        return len(retired_ids)

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
        cache_type: str = "assistant",
        cache_extra_keys: Optional[Any] = None,
        store_cumulative_state: bool = True,
    ) -> Optional[BlockTable]:
        """Store cache data and deterministically terminate any begun L2 fence."""
        cache_extra_keys = self._shape_scoped_cache_extra_keys(
            tokens,
            cache_extra_keys,
            stored_prompt_boundary=True,
        )
        write_fence: Dict[str, Any] = {}
        producer_aborted = False
        try:
            result = self._store_cache_impl(
                request_id,
                tokens,
                cache_data,
                cache_type=cache_type,
                cache_extra_keys=cache_extra_keys,
                store_cumulative_state=store_cumulative_state,
                _write_fence=write_fence,
            )
            self._settle_native_write_fence(request_id, write_fence)
            # SSD-only publication waits for the writer fence above, so every
            # MLX-backed NumPy view has been released before this point. The
            # per-block allocations were evaluated on this model-owning thread;
            # clear their allocator pages here as well. Calling clear_cache()
            # earlier in _store_cache_impl cannot release buffers still owned by
            # the writer queue and produced the measured +202 MiB quiet RSS
            # ratchet on Muse despite every retained cache tier reporting 0 B.
            if write_fence.get("disk_only_fallbacks") and HAS_MLX:
                try:
                    import gc as _gc

                    _gc.collect()
                    mx.clear_cache()
                except Exception as clear_error:  # noqa: BLE001
                    logger.debug(
                        "Could not clear settled SSD-only MLX writer buffers: %s",
                        clear_error,
                    )
            return result
        except BaseException:
            producer_aborted = True
            raise
        finally:
            disk_store = write_fence.get("disk_store")
            fence_id = write_fence.get("fence_id")
            if (
                disk_store is not None
                and fence_id is not None
                and not write_fence.get("seal_attempted")
            ):
                try:
                    sealed = disk_store.seal_write_fence(
                        fence_id,
                        producer_aborted=producer_aborted,
                    )
                    write_fence["seal_attempted"] = True
                    write_fence["sealed"] = bool(sealed)
                except Exception as fence_error:
                    sealed = False
                    logger.warning(
                        "Block disk: could not terminate request write fence %s "
                        "for %s: %s",
                        fence_id,
                        request_id,
                        fence_error,
                    )
                if not sealed:
                    logger.warning(
                        "Block disk: could not seal request write fence %s for %s",
                        fence_id,
                        request_id,
                    )

    def _store_cache_impl(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
        cache_type: str = "assistant",
        cache_extra_keys: Optional[Any] = None,
        store_cumulative_state: bool = True,
        *,
        _write_fence: Dict[str, Any],
    ) -> Optional[BlockTable]:
        """
        Store computed cache for future reuse.

        This method stores actual tensor data (not references) when cache_data
        contains extracted states from mlx-lm's KVCache.state property.

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: The computed KV cache to store. Can be:
                - List of KVCache objects (legacy, stores references)
                - List of dicts with 'state': (keys, values) tensors (new, stores slices)
            cache_type: One of "system" | "user" | "assistant". Tagged on the
                BlockCacheEntry for stats / future eviction prioritization.
                Block-level eviction itself is governed by paged-cache reference
                counts; this tag does not change current eviction order.
            store_cumulative_state: Store path-dependent cumulative state in
                the terminal block. Generic hybrid SSM/GDN schedulers set this
                false because their separately keyed typed companion store is
                authoritative; keeping a second copy in block L2 wastes space
                and is never accepted without the companion hit.

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = (
            cache_data
            and isinstance(cache_data, list)
            and len(cache_data) > 0
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )
        has_dsv4_cache_data = _cache_data_has_dsv4(cache_data) if is_tensor_data else False
        has_dsv4_delta_cache_data = (
            _cache_data_has_dsv4_deltas(cache_data) if is_tensor_data else False
        )
        has_zaya_cca_cache_data = (
            _cache_data_has_zaya_cca(cache_data) if is_tensor_data else False
        )
        has_rotating_kv_cache_data = (
            _cache_data_has_rotating_kv(cache_data) if is_tensor_data else False
        )
        has_minimax_m3_cache_data = (
            _cache_data_has_minimax_m3(cache_data) if is_tensor_data else False
        )
        has_native_path_dependent_cache_data = bool(
            has_dsv4_cache_data
            or has_dsv4_delta_cache_data
            or has_zaya_cca_cache_data
            or has_rotating_kv_cache_data
        )
        rotating_kv_layer_count = (
            _rotating_kv_layer_count(cache_data) if has_rotating_kv_cache_data else 0
        )
        if has_dsv4_delta_cache_data:
            # Fail LOUD and NAMED on a block-size mismatch. Native DSV4 delta
            # records are cut at a fixed 256; a cutter at any other size asks
            # for interval (0, N), never finds it, and every store of every
            # request dies with a generic "no interval" ValueError that names
            # no cause. The scheduler reconciles this at construction, so
            # reaching here means something bypassed that — say so.
            try:
                from .utils.dsv4_batch_generator import DSV4_NATIVE_BLOCK_SIZE
            except Exception:
                DSV4_NATIVE_BLOCK_SIZE = 256
            if int(self.block_size) != int(DSV4_NATIVE_BLOCK_SIZE):
                raise ValueError(
                    "DSV4 native delta records require a "
                    f"{DSV4_NATIVE_BLOCK_SIZE}-token paged block, but this "
                    f"cache is configured with block_size={self.block_size}. "
                    "Every store would abort on a missing interval. The "
                    "scheduler normally reconciles this at construction; a "
                    "launcher that builds the cache directly must pass "
                    f"paged_cache_block_size={DSV4_NATIVE_BLOCK_SIZE}."
                )

        disk_store = self.paged_cache._disk_store  # May be None
        _disk_only = bool(getattr(self.paged_cache, "disk_only", False))

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        if has_dsv4_delta_cache_data and block_table.block_ids:
            # A resumed delta transport begins at the restored checkpoint and
            # depends on every retained parent page already being native. The
            # older shadow-rekey implementation donated generic pending/full
            # composite pages; retaining one here would publish a mixed chain
            # that cannot reconstruct. Refuse it before partial-table
            # realignment mutates the still-valid generic entry.
            parent_start = 0
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                token_count = int(getattr(block, "token_count", 0) or 0)
                parent_end = parent_start + token_count
                interval_stamp = getattr(
                    block,
                    "dsv4_native_interval",
                    None,
                )
                if interval_stamp == (parent_start, parent_end):
                    parent_start = parent_end
                    continue
                payload = getattr(block, "cache_data", None)
                if payload is None and disk_store is not None and block is not None:
                    try:
                        payload = disk_store.read_block(block.block_hash)
                    except Exception:
                        payload = None
                if not _block_has_complete_dsv4_delta_interval(
                    payload,
                    start_token=parent_start,
                    end_token=parent_end,
                    expected_layers=len(cache_data),
                    expected_transport=cache_data,
                ):
                    raise ValueError(
                        "cannot append native DSV4 deltas to a non-native "
                        f"parent interval [{parent_start}, {parent_end})"
                    )
                block.dsv4_native_interval = (parent_start, parent_end)
                parent_start = parent_end

        # Determine tokens we need to cache (not already in block_table)
        existing_tokens = block_table.num_tokens
        if existing_tokens % self.block_size:
            # A cached terminal partial is a valid fetch boundary, but it must
            # not become the parent of a new full-size block at a shifted token
            # offset (for example 62 + 64 with block_size=64). Rebuild from the
            # last complete block boundary so the durable chain remains
            # discoverable after the in-memory prefix index is lost.
            keep_ids: List[int] = []
            aligned_tokens = 0
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                token_count = int(getattr(block, "token_count", 0) or 0)
                if token_count != self.block_size:
                    break
                keep_ids.append(block_id)
                aligned_tokens += token_count
            trailing_ids = block_table.block_ids[len(keep_ids) :]
            for block_id in trailing_ids:
                self.paged_cache.decrement_ref(block_id)
            logger.info(
                "Realigning extended paged prefix for %s from %d to %d "
                "tokens before store (%d partial block(s) released)",
                request_id,
                existing_tokens,
                aligned_tokens,
                len(trailing_ids),
            )
            block_table.block_ids = keep_ids
            block_table.num_tokens = aligned_tokens
            existing_tokens = aligned_tokens
        new_tokens = tokens[existing_tokens:]

        if (
            not new_tokens
            and has_rotating_kv_cache_data
            and block_table.block_ids
        ):
            terminal_block_id = block_table.block_ids[-1]
            terminal_block = self.paged_cache.allocated_blocks.get(terminal_block_id)
            if not _block_has_complete_rotating_terminal(
                getattr(terminal_block, "cache_data", None),
                target_tokens=existing_tokens,
                expected_layers=rotating_kv_layer_count,
            ):
                # This exact token block may have first been created as an
                # interior block of a longer prefix.  Its ordinary KV payload
                # is reusable, but it has no rotating state for this causal
                # boundary. Rebuild only the terminal block under the same
                # chain hash, mirroring the cumulative-state promotion path.
                terminal_count = int(
                    getattr(terminal_block, "token_count", 0) or 0
                )
                if terminal_count <= 0 or terminal_count > existing_tokens:
                    logger.warning(
                        "Cannot promote rotating terminal block for %s: "
                        "invalid token_count=%s existing_tokens=%s",
                        request_id,
                        terminal_count,
                        existing_tokens,
                    )
                    return block_table
                self.paged_cache.decrement_ref(terminal_block_id)
                block_table.block_ids.pop()
                existing_tokens -= terminal_count
                block_table.num_tokens = existing_tokens
                new_tokens = tokens[existing_tokens:]
                logger.info(
                    "Promoting interior rotating block to terminal checkpoint "
                    "for %s at %s tokens",
                    request_id,
                    len(tokens),
                )

        if not new_tokens:
            # All tokens already cached
            return block_table

        # Allocate blocks for new tokens
        num_new_blocks = (len(new_tokens) + self.block_size - 1) // self.block_size

        # For disk write-through, compute chain hashes over the full token sequence.
        # Reconstruct parent_hash from existing blocks (if any).
        from .paged_cache import compute_block_hash as _compute_chain_hash
        parent_hash = None
        if existing_tokens > 0:
            # Recompute chain hash up to where the existing blocks end
            num_existing_full = existing_tokens // self.block_size
            for eb_idx in range(num_existing_full):
                eb_start = eb_idx * self.block_size
                eb_end = eb_start + self.block_size
                parent_hash = _compute_chain_hash(
                    parent_hash,
                    tokens[eb_start:eb_end],
                    extra_keys=cache_extra_keys_for_token_range(
                        cache_extra_keys, eb_start, eb_end
                    ),
                )

        disk_write_fence_id: Optional[str] = None
        if disk_store is not None:
            begin_write_fence = getattr(disk_store, "begin_write_fence", None)
            if callable(begin_write_fence):
                try:
                    begin_kwargs = {
                        "strict_reconcile": self._strict_block_disk_write_fence,
                        "admission_timeout": self._write_admission_timeout_for_store(
                            disk_only=_disk_only,
                            path_dependent=has_native_path_dependent_cache_data,
                        ),
                    }
                    try:
                        begun_fence = begin_write_fence(
                            request_id,
                            **begin_kwargs,
                        )
                    except TypeError as fence_type_error:
                        if "admission_timeout" not in str(fence_type_error):
                            raise
                        # Preserve the legacy backend contract. The in-tree
                        # BlockDiskStore owns bounded native admission; a test
                        # or third-party store without that extension remains
                        # fail-closed through the RAM fallback.
                        begun_fence = begin_write_fence(
                            request_id,
                            strict_reconcile=self._strict_block_disk_write_fence,
                        )
                    disk_write_fence_id = str(begun_fence)
                    _write_fence["disk_store"] = disk_store
                    _write_fence["fence_id"] = disk_write_fence_id
                except Exception as fence_error:
                    # The write still proceeds, but WITHOUT a fence: the guards
                    # below treat a None fence id as "no tracking", so nothing
                    # settles it and a native payload can stay keep_resident
                    # with no drain path. That is a real degradation of the
                    # commit-before-eviction guarantee, not a routine hiccup —
                    # the most common cause is the >64 unfinished-fence cap, so
                    # a burst of these means the L2 writer is falling behind.
                    #
                    # ERROR, not WARNING: at WARNING this was easy to miss while
                    # the tier quietly stopped being durable.
                    logger.error(
                        "Block disk: could not begin request write fence for "
                        "%s: %s — writes proceed UNFENCED; native payloads may "
                        "stay resident with no drain path.",
                        request_id,
                        fence_error,
                    )

        disk_write_chain_open = True

        def _record_unadmitted_native_writes(count: int) -> None:
            if (
                not has_native_path_dependent_cache_data
                or disk_store is None
                or disk_write_fence_id is None
                or int(count or 0) <= 0
            ):
                return
            record_unadmitted = getattr(
                disk_store,
                "record_write_fence_unadmitted",
                None,
            )
            if callable(record_unadmitted):
                record_unadmitted(disk_write_fence_id, int(count))

        def _write_block_to_disk(
            block_hash: bytes,
            block_data: List[Tuple[Any, ...]],
            token_count: int,
            block_parent_hash: Optional[bytes],
        ) -> bool:
            nonlocal disk_write_chain_open
            if disk_store is None:
                return False
            if has_native_path_dependent_cache_data and not disk_write_chain_open:
                _record_unadmitted_native_writes(1)
                return False
            fence_kwargs: Dict[str, Any] = {}
            if disk_write_fence_id is not None:
                fence_kwargs = {
                    "request_id": request_id,
                    "fence_id": disk_write_fence_id,
                }
            if block_hash in dsv4_disk_replacements:
                fence_kwargs["replace_existing"] = True
            admitted = bool(
                disk_store.write_block_async(
                    block_hash,
                    block_data,
                    token_count,
                    parent_hash=block_parent_hash,
                    **fence_kwargs,
                )
            )
            if has_native_path_dependent_cache_data and not admitted:
                disk_write_chain_open = False
            return admitted

        # Paged RAM and block-disk L2 are separate cache tiers.  The manager
        # defaults ordinary Paged On sessions to a resident L1 write-through
        # mirror, while disk-only sessions and an explicit
        # VMLX_PAGED_FRUGAL=1 override suppress that mirror.
        _paged_frugal = bool(
            getattr(self.paged_cache, "paged_frugal", _disk_only)
        )
        logger.debug(
            f"Block disk write-through: disk_store={'present' if disk_store else 'None'}, "
            f"is_tensor_data={is_tensor_data}, new_tokens={len(new_tokens)}, "
            f"num_new_blocks={num_new_blocks}, frugal={_paged_frugal}, "
            f"cumulative={'block' if store_cumulative_state else 'external-companion'}"
        )

        # Expose source KV arrays to numpy for safe slicing.
        # MLX has a Metal command buffer bug: any evaluation of lazy slices
        # whose source was previously evaluated triggers fatal Metal
        # assertions ("addCompletedHandler after commit") or kernel panics
        # ("completeMemory() prepare count underflow").  Read-only zero-copy
        # numpy views of the synchronized full source arrays let both
        # _extract_block_tensor_slice (for block cache_data) and
        # _numpy_block_slice (for disk writes) do per-block slicing in numpy
        # space. The slicer still materializes each exact block independently
        # before async enqueue, so the writer never aliases mutable model cache
        # state.
        np_sources: dict = {}  # layer_idx → (np_keys, np_values)
        if is_tensor_data and HAS_MLX:
            import numpy as np
            # Synchronize all Metal streams before converting arrays to numpy.
            # mlx_lm's BatchGenerator.next() does NOT synchronize after
            # inference — in-place KV/SSM cache updates (keys[...] = new_k)
            # create lazy side-effect operations that may still be pending on
            # the Metal command queue.  Without this sync, np.array() on cache
            # arrays (especially cumulative SSM state in _numpy_block_slice)
            # triggers "addCompletedHandler after commit" or segfault.
            # MLLMBatchGenerator.next() already syncs its dedicated _stream,
            # but the stock BatchGenerator and the default stream are not
            # covered — a bare mx.synchronize() handles both.
            mx.synchronize()
            for idx, layer_state in enumerate(cache_data):
                cls = layer_state.get("class_name", "")
                if _is_zaya_cca_cache_list_state(layer_state):
                    try:
                        subs = layer_state["sub_caches"]
                        kv_state = subs[0].get("state")
                        keys, values = kv_state
                        if isinstance(keys, (tuple, list)):
                            # Keep generic quantized CacheList on the MLX
                            # path for now. ZAYA runtime TQ-KV needs its own
                            # typed partial codec before disk writes use it.
                            continue
                        k_np, v_np = keys, values
                        if hasattr(k_np, "dtype") and "bfloat16" in str(k_np.dtype):
                            k_np = k_np.astype(mx.float32)
                            v_np = v_np.astype(mx.float32)
                            mx.eval(k_np, v_np)
                        cca_state = subs[1].get("state")
                        cca_np = _to_numpy_tree(cca_state)
                        np_sources[idx] = {
                            "type": "zaya_cca",
                            "kv": (
                                _readonly_numpy_buffer_view(k_np),
                                _readonly_numpy_buffer_view(v_np),
                                keys.dtype,
                            ),
                            "cca_state": cca_np,
                            "cca_meta": subs[1].get("meta_state", ""),
                            "cache_meta": {
                                "schema": "zaya_cca_v1",
                                "cca_class_name": subs[1].get(
                                    "class_name", "ArraysCache"
                                ),
                            },
                        }
                    except Exception as _npe:
                        logger.debug(f"np_sources skip ZAYA layer {idx}: {_npe}")
                    continue
                if _is_minimax_m3_cache_class(cls):
                    # Never create a full-prompt FP32 NumPy mirror for M3.
                    # A live 6K post-tool cleanup was killed while holding that
                    # mirror plus two payloads for every extracted block. M3
                    # state is fully positional, so the block loop below slices
                    # one independent (K, V, idx_keys) payload and immediately
                    # uses the same payload for L1 and L2 serialization.
                    continue
                state = layer_state.get("state")
                if state is None:
                    continue
                if self._is_positional_cache(state, cls):
                    try:
                        keys, values = state
                        if isinstance(keys, (tuple, list)):
                            # Quantized KV (QuantizedKVCache / TurboQuantKVCache).
                            # state = ((w_packed, scales, biases), (w_packed, scales, biases))
                            # Dequantize to float16 for the numpy extraction path so the
                            # block disk cache can serialize per-block slices. Without
                            # this, np_sources stays empty for quantized layers and the
                            # block disk write-through silently drops everything.
                            #
                            # Trade-off: disk cache entries are ~4x larger for q8 (vs.
                            # storing packed bytes), but storage works.
                            if len(keys) < 2 or len(values) < 2:
                                logger.debug(
                                    f"np_sources skip layer {idx}: quantized state "
                                    f"arity keys={len(keys)} values={len(values)}"
                                )
                                continue
                            meta = layer_state.get("meta_state", ())
                            # meta_state typically ends with (group_size, bits)
                            g_size, q_bits = 64, 8
                            if isinstance(meta, (tuple, list)) and len(meta) >= 2:
                                try:
                                    g_size = int(meta[-2])
                                    q_bits = int(meta[-1])
                                except (ValueError, TypeError):
                                    pass
                            k_w, k_s = keys[0], keys[1]
                            k_b = keys[2] if len(keys) >= 3 else mx.zeros_like(k_s)
                            v_w, v_s = values[0], values[1]
                            v_b = values[2] if len(values) >= 3 else mx.zeros_like(v_s)
                            k_dq = mx.dequantize(
                                k_w, k_s, k_b, group_size=g_size, bits=q_bits,
                            )
                            v_dq = mx.dequantize(
                                v_w, v_s, v_b, group_size=g_size, bits=q_bits,
                            )
                            original_dtype = k_dq.dtype
                            if 'bfloat16' in str(original_dtype):
                                # Use float32 (not float16) for the numpy round-trip.
                                # fp16 has only 5 exponent bits vs bf16's 8 — casting
                                # down silently clips/loses precision for many
                                # attention KV values. On Gemma 4 JANG this caused
                                # "step-by-step" word loops on the 3rd multi-turn
                                # request because the cached KV drift pushed the
                                # sampler into a degenerate rep_pen-proof basin.
                                k_dq = k_dq.astype(mx.float32)
                                v_dq = v_dq.astype(mx.float32)
                            mx.eval(k_dq, v_dq)
                            np_sources[idx] = (
                                _readonly_numpy_buffer_view(k_dq),
                                _readonly_numpy_buffer_view(v_dq),
                                original_dtype,
                            )
                            continue
                        if hasattr(keys, 'shape'):
                            k_np, v_np = keys, values
                            # numpy doesn't support bfloat16 — cast through fp32
                            # (NOT fp16, which silently clips). See the Gemma 4
                            # multi-turn word-loop note above.
                            if hasattr(k_np, 'dtype') and 'bfloat16' in str(k_np.dtype):
                                k_np = k_np.astype(mx.float32)
                                v_np = v_np.astype(mx.float32)
                                mx.eval(k_np, v_np)
                            np_sources[idx] = (
                                _readonly_numpy_buffer_view(k_np),
                                _readonly_numpy_buffer_view(v_np),
                                keys.dtype,
                            )
                    except Exception as _npe:
                        logger.debug(f"np_sources skip layer {idx}: {_npe}")

        if disk_store is not None:
            logger.debug(
                f"Block disk: np_sources has {len(np_sources)} positional layers "
                f"(out of {len(cache_data)} total)"
            )

        pending_disk_writes: list = []
        disk_only_fallbacks: dict = {}
        native_paged_holds: dict = {}
        dsv4_disk_replacements: set[bytes] = set()

        for i in range(num_new_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]

            # Token range in the original sequence (accounting for existing tokens)
            global_start = existing_tokens + start_idx
            global_end = existing_tokens + end_idx

            # Compute chain hash for this block
            block_chain_hash = _compute_chain_hash(
                parent_hash,
                block_tokens,
                extra_keys=cache_extra_keys_for_token_range(
                    cache_extra_keys, global_start, global_end
                ),
            )

            if _CACHE_HASH_DEBUG:
                logger.info(
                    "cache-hash-debug STORE pos=%d hash=%s parent=%s "
                    "tok[:4]=%s tok[-4:]=%s n=%d extra=%r",
                    (global_start // self.block_size),
                    block_chain_hash.hex()[:16],
                    parent_hash.hex()[:16] if parent_hash else None,
                    list(block_tokens[:4]),
                    list(block_tokens[-4:]),
                    len(block_tokens),
                    cache_extra_keys,
                )

            # Check if this block already exists via chain hash (deduplication)
            # IMPORTANT: lookup + ref bump must be atomic under _lock to prevent
            # the block from being freed between get_block and increment_ref.
            is_last = (i == num_new_blocks - 1)
            reused = False
            preloaded_dsv4_block_id = None
            preloaded_dsv4_payload = None
            preloaded_dsv4_l2_readable = None
            if has_dsv4_delta_cache_data and disk_store is not None:
                preloaded_dsv4_needs_payload = False
                with self.paged_cache._lock:
                    candidate = (
                        self.paged_cache.cached_block_hash_to_block.get_block(
                            block_chain_hash
                        )
                    )
                    if candidate is not None:
                        preloaded_dsv4_block_id = candidate.block_id
                        preloaded_dsv4_needs_payload = (
                            candidate.cache_data is None
                            and getattr(
                                candidate,
                                "dsv4_native_interval",
                                None,
                            )
                            != (global_start, global_end)
                        )
                try:
                    has_record = getattr(
                        disk_store,
                        "has_block_record",
                        None,
                    )
                    if not callable(has_record):
                        has_record = getattr(disk_store, "has_block", None)
                    preloaded_dsv4_l2_readable = bool(
                        callable(has_record)
                        and has_record(block_chain_hash)
                    )
                    if preloaded_dsv4_l2_readable and (
                        preloaded_dsv4_block_id is None
                        or preloaded_dsv4_needs_payload
                    ):
                        preloaded_dsv4_payload = disk_store.read_block(
                            block_chain_hash
                        )
                except Exception:
                    preloaded_dsv4_l2_readable = False
                    preloaded_dsv4_payload = None
                    if preloaded_dsv4_block_id is None:
                        dsv4_disk_replacements.add(block_chain_hash)
                if (
                    preloaded_dsv4_block_id is None
                    and preloaded_dsv4_l2_readable
                    and not _block_has_complete_dsv4_delta_interval(
                        preloaded_dsv4_payload,
                        start_token=global_start,
                        end_token=global_end,
                        expected_layers=len(cache_data),
                        expected_transport=cache_data,
                    )
                ):
                    # Restart/recycling can leave an old generic payload only
                    # in L2. The token hash still collides, so explicitly
                    # replace that representation even though no L1 candidate
                    # exists to trigger the normal promotion branch below.
                    dsv4_disk_replacements.add(block_chain_hash)
            with self.paged_cache._lock:
                existing_block = self.paged_cache.cached_block_hash_to_block.get_block(
                    block_chain_hash
                )
                if (
                    existing_block is not None
                    and has_dsv4_delta_cache_data
                    and disk_store is not None
                ):
                    if (
                        existing_block.block_id != preloaded_dsv4_block_id
                        or preloaded_dsv4_l2_readable is not True
                    ):
                        # The candidate changed after the off-lock probe, or
                        # its stamped native payload was evicted from L2.
                        # Retire it from hash discovery and publish a fresh
                        # native copy rather than trusting stale metadata.
                        retired_blocks = (
                            self.paged_cache.cached_block_hash_to_block.pop_all(
                                block_chain_hash
                            )
                        )
                        for retired_block in retired_blocks:
                            retired_block.suppress_l2_republish = True
                        existing_block = None
                if (
                    existing_block is not None
                    and existing_block.cache_data is None
                    and disk_store is not None
                ):
                    # Global disk-budget eviction deletes payload files while
                    # the L1 hash index survives. Reusing such a block skips
                    # the disk write below and leaves the chain permanently
                    # unrestorable: every repeat of the prompt pays a full
                    # prefill (observed as a double full prefill per request
                    # in disk-only mode). Verify the L2 payload is still
                    # readable before honoring the dedup; fall through to a
                    # fresh allocation + rewrite when it is gone. get_block
                    # prefers the newest duplicate, so the rewritten block
                    # shadows the stale entry.
                    if not has_dsv4_delta_cache_data:
                        has_block = getattr(disk_store, "has_block", None)
                        try:
                            if callable(has_block) and not bool(
                                has_block(block_chain_hash)
                            ):
                                existing_block = None
                        except Exception:
                            pass
                if existing_block is not None and has_dsv4_delta_cache_data:
                    # Native shadow re-key may collide with an older generic
                    # DSV4 store at the exact same token-chain hash. Hash
                    # identity proves token identity, not payload-contract
                    # identity. Reuse only a complete native interval;
                    # otherwise allocate a richer duplicate below. The hash
                    # map deliberately prefers the newest live duplicate.
                    native_interval = getattr(
                        existing_block,
                        "dsv4_native_interval",
                        None,
                    )
                    existing_payload = existing_block.cache_data
                    if (
                        existing_payload is None
                        and existing_block.block_id == preloaded_dsv4_block_id
                    ):
                        existing_payload = preloaded_dsv4_payload
                    native_payload = native_interval == (
                        global_start,
                        global_end,
                    ) or _block_has_complete_dsv4_delta_interval(
                        existing_payload,
                        start_token=global_start,
                        end_token=global_end,
                        expected_layers=len(cache_data),
                        expected_transport=cache_data,
                    )
                    if not native_payload:
                        # Preserve active request-table refs, but retire this
                        # representation from content-hash discovery. If the
                        # new native duplicate is later evicted, the generic
                        # block must not silently resurface as the preferred
                        # match for the same token chain.
                        retired_blocks = (
                            self.paged_cache.cached_block_hash_to_block.pop_all(
                                block_chain_hash
                            )
                        )
                        for retired_block in retired_blocks:
                            retired_block.suppress_l2_republish = True
                        if disk_store is not None:
                            dsv4_disk_replacements.add(block_chain_hash)
                        existing_block = None
                    else:
                        existing_block.dsv4_native_interval = (
                            global_start,
                            global_end,
                        )
                if existing_block:
                    # If this is the last block in the new sequence and the
                    # existing block was stored as non-last (SSM layers tagged
                    # "skip" without cumulative state), skip reuse so we can
                    # allocate a new block with proper cumulative SSM state.
                    # Without this, hybrid SSM models get "Reconstructed 10
                    # layers but expected 40" because cumulative entries are
                    # never stored.
                    if (
                        is_last
                        and is_tensor_data
                        and (
                            (
                                store_cumulative_state
                                and _block_needs_cumulative_update(
                                    existing_block.cache_data
                                )
                            )
                            or (
                                (has_dsv4_cache_data or has_zaya_cca_cache_data)
                                and existing_block.cache_data is None
                            )
                            or (
                                has_rotating_kv_cache_data
                                and not _block_has_complete_rotating_terminal(
                                    existing_block.cache_data,
                                    target_tokens=global_end,
                                    expected_layers=rotating_kv_layer_count,
                                )
                            )
                        )
                    ):
                        pass  # Fall through to new block allocation
                    else:
                        # issue #198 (1A): revive through touch() so a
                        # cached-but-free block is removed from the free
                        # queue AND restored to allocated_blocks atomically
                        # (RLock is reentrant). Bare ref_count += 1 left
                        # revived blocks missing from allocated_blocks.
                        self.paged_cache.touch([existing_block])
                        if existing_block.ref_count == 2:
                            self.paged_cache.stats.shared_blocks += 1
                        reused = True
            if reused:
                block_table.block_ids.append(existing_block.block_id)
                block_table.num_tokens += len(block_tokens)
                parent_hash = block_chain_hash
                if disk_store is not None:
                    logger.debug(
                        f"Block disk: block {existing_block.block_id} reused from L1 "
                        f"(skip disk write)"
                    )
                continue

            # Also check legacy hash for non-tensor blocks stored before chain hashing.
            #
            # Tensor KV state must NEVER use this content-only fallback.
            # Repeated 64-token chunks have identical text but different hidden
            # state because real KV values depend on the full prefix. DSV4 made
            # this obvious through CSA/HCA state, but the same invariant applies
            # to ordinary attention, RoPE, hybrid SSM companions, and TQ-KV.
            # Chain hashes above already cover the safe exact-prefix case.
            existing_block = None
            if not is_tensor_data:
                # find_cached_block already holds _lock internally, but we need
                # to do the ref bump under the same lock to avoid the same race.
                with self.paged_cache._lock:
                    hash_value = self.paged_cache.compute_block_hash(block_tokens)
                    if hash_value in self.paged_cache.hash_to_block:
                        bid = self.paged_cache.hash_to_block[hash_value]
                        if bid in self.paged_cache.allocated_blocks:
                            existing_block = self.paged_cache.allocated_blocks[bid]
                            # Same cumulative state check as chain hash path:
                            # skip reuse if last block needs cumulative SSM state
                            if is_last and is_tensor_data and _block_needs_cumulative_update(
                                existing_block.cache_data
                            ):
                                existing_block = None  # Fall through to allocation
                            else:
                                # issue #198 (1A): atomic revive (see above)
                                self.paged_cache.touch([existing_block])
                                if existing_block.ref_count == 2:
                                    self.paged_cache.stats.shared_blocks += 1
                                self.paged_cache.stats.cache_hits += 1
                                # Set chain hash on the existing block if it doesn't have one
                                if existing_block.block_hash is None:
                                    existing_block.block_hash = block_chain_hash
                                    existing_block.parent_hash = parent_hash
                                    self.paged_cache.cached_block_hash_to_block.insert(
                                        block_chain_hash, existing_block
                                    )
                    if existing_block is None:
                        self.paged_cache.stats.cache_misses += 1
            else:
                with self.paged_cache._lock:
                    self.paged_cache.stats.cache_misses += 1
            if existing_block:
                block_table.block_ids.append(existing_block.block_id)
                block_table.num_tokens += len(block_tokens)
                parent_hash = block_chain_hash
                continue

            # Allocate new block
            block = self.paged_cache.allocate_block()
            if not block:
                # First free low-priority entries (assistant -> user -> system)
                # so their request refs drop and blocks can enter the free LRU queue.
                self.release_low_priority(1)

                # Then run block-level LRU eviction under memory pressure.
                if not self.paged_cache.handle_memory_pressure(1):
                    if block_table.num_tokens > 0:
                        logger.warning(
                            "Paged cache capacity reached for %s; stored partial "
                            "prefix %s/%s tokens (%s blocks). Increase Max Cache "
                            "Blocks or lower Block Size to cache more of this prompt.",
                            request_id,
                            block_table.num_tokens,
                            len(tokens),
                            len(block_table.block_ids),
                        )
                    else:
                        logger.warning(
                            "Paged cache capacity reached for %s before any prefix "
                            "blocks fit. Increase Max Cache Blocks or lower Block "
                            "Size to enable prefix reuse for this prompt.",
                            request_id,
                        )
                    break
                block = self.paged_cache.allocate_block()
                if not block:
                    if block_table.num_tokens > 0:
                        logger.warning(
                            "Paged cache allocation stopped for %s; stored partial "
                            "prefix %s/%s tokens (%s blocks).",
                            request_id,
                            block_table.num_tokens,
                            len(tokens),
                            len(block_table.block_ids),
                        )
                    break

            # Store block data
            block.token_count = len(block_tokens)
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)

            # Set chain hash on the block (for L1 dedup and L2 disk addressing)
            block.block_hash = block_chain_hash
            block.parent_hash = parent_hash

            # Extract and store actual tensor slices for this block
            if is_tensor_data and HAS_MLX:
                # is_last already computed above (for cumulative state checks)
                block_kv_data = self._extract_block_tensor_slice(
                    cache_data,
                    global_start,
                    global_end,
                    is_last_block=is_last,
                    np_sources=np_sources if np_sources else None,
                    existing_tokens=existing_tokens,
                    store_cumulative_state=store_cumulative_state,
                )
                if block_kv_data:
                    if has_dsv4_delta_cache_data:
                        block.dsv4_native_interval = (
                            global_start,
                            global_end,
                        )
                    # DSV4, ZAYA CCA, and mixed-SWA rotating KV are not normal
                    # per-block KV payloads.
                    # DSV4 terminal blocks carry SWA+CSA/HCA composite state;
                    # ZAYA terminal blocks carry CCA conv_state + prev_hs.
                    # Mixed-SWA carries per-layer rotating-window metadata.
                    # If frugal mode drops those in-RAM records, an immediate
                    # same-process repeat can hit the block table before the
                    # async L2 write is readable and reconstruct as None. Keep
                    # native path-dependent records resident; L2 still gets the
                    # write-through copy for restart restore.
                    keep_in_ram = (
                        not _disk_only
                        and (
                            has_dsv4_cache_data
                            or has_dsv4_delta_cache_data
                            or has_zaya_cca_cache_data
                            or has_rotating_kv_cache_data
                        )
                    )
                    if _disk_only:
                        # Track only metadata until the fence confirms that the
                        # SSD record survived aggregate eviction. A zero-RAM
                        # cache must never hold a payload fallback: failure means
                        # this block is invalidated and later re-prefilled.
                        disk_only_fallbacks[block_chain_hash] = (
                            block,
                            None,
                        )
                    if _paged_frugal and not keep_in_ram:
                        # Disk has it — skip the in-RAM duplicate. L1 lookup
                        # will fall through to L2 disk + _promote_from_disk
                        # which lazily re-creates cache_data on hit.
                        # Keep token_count/block_hash set above so disk
                        # promotion can verify the block.
                        logger.debug(
                            f"Frugal: skipped in-RAM mirror for block "
                            f"{block.block_id} (tokens [{global_start}:{global_end}], "
                            f"disk-only)"
                        )
                    else:
                        block.cache_data = block_kv_data
                        block.cache_data_from_disk = False
                        # Native composite state (DSV4/ZAYA/rotating-SWA) must
                        # survive until its async L2 write is post-eviction
                        # retained; flag it so the byte ceiling never evicts the
                        # RAM mirror out from under an immediate repeat.
                        block.keep_resident = keep_in_ram
                        if keep_in_ram and disk_store is not None:
                            native_paged_holds[block_chain_hash] = block
                        if self.paged_cache.max_resident_bytes > 0:
                            self.paged_cache._note_resident(
                                block,
                                self.paged_cache.estimate_block_nbytes(block_kv_data),
                            )
                        logger.debug(
                            f"Stored tensor slice for block {block.block_id}: "
                            f"tokens [{global_start}:{global_end}], {len(block_kv_data)} layers"
                            f"{' (includes cumulative states)' if is_last and store_cumulative_state else ''}"
                        )

                    # Ordinary caches use independent numpy slices for L2.
                    # M3 deliberately reuses its already-independent typed L1
                    # block and serializes it immediately: retaining a second
                    # disk payload for every block caused unbounded cleanup
                    # growth and a live SIGKILL on a 6K tool continuation.
                    if disk_store is not None:
                        if has_minimax_m3_cache_data or has_dsv4_delta_cache_data:
                            np_block = block_kv_data
                        else:
                            np_block = _numpy_block_slice(
                                cache_data, np_sources or {},
                                global_start, global_end, is_last, existing_tokens,
                                store_cumulative_state,
                                self.block_size,
                            )
                        if np_block:
                            # The numpy mirror is authoritative for ordinary KV
                            # and cumulative/path-dependent state, but it cannot
                            # represent native TurboQuant metadata: its generic
                            # branch emits ("kv", ...) and silently discards the
                            # per-layer seed/bit policy.  The main-thread block
                            # extractor above has already created independent,
                            # fully packed TQ entries, so splice only those back
                            # into the otherwise numpy-safe disk payload.
                            for _idx, _entry in enumerate(block_kv_data):
                                if _idx < len(np_block) and _entry_has_native_tq(
                                    _entry
                                ):
                                    np_block[_idx] = _entry
                            # Count non-skip entries. DSV4 intentionally logs
                            # only two plain KV layers plus native composite
                            # markers/state for the remaining layers; showing
                            # those tags keeps support logs from looking like
                            # the CSA/HCA cache was dropped.
                            _tag_counts = {}
                            for _entry in np_block:
                                if not isinstance(_entry, (tuple, list)) or not _entry:
                                    continue
                                _tag = _entry[0]
                                if _tag == "skip":
                                    continue
                                _tag_counts[_tag] = _tag_counts.get(_tag, 0) + 1
                            _kv_count = _tag_counts.get("kv", 0)
                            _extra_tags = ", ".join(
                                f"{_tag}={_count}"
                                for _tag, _count in sorted(_tag_counts.items())
                                if _tag != "kv"
                            )
                            _typed_layer_count = sum(_tag_counts.values())
                            _layer_summary = (
                                f"total_typed={_typed_layer_count}, "
                                f"standard_kv={_kv_count}"
                            )
                            if _extra_tags:
                                _layer_summary += f", {_extra_tags}"
                            _has_native_tq = any(
                                _entry_has_native_tq(_entry)
                                for _entry in np_block
                            )
                            if self._write_block_immediately_for_store(
                                disk_only=_disk_only,
                                minimax_m3=has_minimax_m3_cache_data,
                                native_tq=_has_native_tq,
                            ):
                                # Native TQ entries contain lazy MLX encode graphs.
                                # SSD-only plain KV has the same boundedness need:
                                # deferring every independent page duplicates the
                                # whole prompt in pending_disk_writes before queue
                                # admission even begins. Materialize/freeze one page
                                # now; the disk store still performs encoding,
                                # publication and indexing off-thread.
                                logger.debug(
                                    f"Block disk: writing bounded "
                                    f"{'MiniMax-M3' if has_minimax_m3_cache_data else 'TQ' if _has_native_tq else 'SSD-only'} block "
                                    f"{block.block_id} ({_layer_summary}, "
                                    f"{len(block_tokens)} tokens)"
                                )
                                _write_block_to_disk(
                                    block_chain_hash,
                                    np_block,
                                    len(block_tokens),
                                    parent_hash,
                                )
                            else:
                                logger.debug(
                                    f"Block disk: queuing write for block "
                                    f"{block.block_id} ({_layer_summary}, "
                                    f"{len(block_tokens)} tokens)"
                                )
                                pending_disk_writes.append(
                                    (
                                        block_chain_hash,
                                        np_block,
                                        len(block_tokens),
                                        parent_hash,
                                    )
                                )
                        else:
                            logger.warning(
                                f"Block disk: _numpy_block_slice returned empty "
                                f"for block {block.block_id} "
                                f"(global [{global_start}:{global_end}], is_last={is_last})"
                            )

            # Register in hash caches under lock (both chain hash and legacy)
            with self.paged_cache._lock:
                self.paged_cache.cached_block_hash_to_block.insert(
                    block_chain_hash, block
                )
            self.paged_cache.register_block_hash(block, block_tokens)

            parent_hash = block_chain_hash

        # Release every full-cache numpy VIEW immediately after extraction.
        # These views no longer duplicate F16/F32 source storage, but their
        # memoryview bases deliberately keep source/bridge MLX buffers alive.
        # Pending disk writes below already own immutable per-block payload
        # buffers, so the full-buffer lifetimes can end before writer admission.
        if np_sources:
            np_sources.clear()
        np_sources = None
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

        # Write deferred disk blocks (all numpy — no MLX/Metal ops).
        if pending_disk_writes and disk_store is not None:
            logger.debug(
                f"Block disk: writing {len(pending_disk_writes)} blocks to SSD"
            )
            for write_index, (
                block_hash,
                block_data,
                tok_count,
                block_parent_hash,
            ) in enumerate(pending_disk_writes):
                try:
                    admitted = _write_block_to_disk(
                        block_hash,
                        block_data,
                        tok_count,
                        block_parent_hash,
                    )
                    if has_native_path_dependent_cache_data and not admitted:
                        remaining = len(pending_disk_writes) - write_index - 1
                        _record_unadmitted_native_writes(remaining)
                        break
                except Exception as _wbe:
                    logger.warning(
                        f"Block disk: write_block_async failed: {_wbe}"
                    )
                    if has_native_path_dependent_cache_data:
                        remaining = len(pending_disk_writes) - write_index
                        _record_unadmitted_native_writes(remaining)
                        break

        # The lifecycle wrapper seals the request fence after every write has
        # been admitted, waits for post-eviction completion, and only then
        # releases these native RAM holds.  Keeping the payload references in the
        # fence context closes the commit-before-budget-eviction race without
        # broadening the synchronous barrier to ordinary KV blocks.
        if _disk_only and disk_only_fallbacks:
            _write_fence["disk_only_fallbacks"] = disk_only_fallbacks
        if native_paged_holds:
            _write_fence["native_paged_holds"] = native_paged_holds

        # Update prefix index
        self._update_prefix_index(
            tokens,
            block_table.block_ids,
            cache_extra_keys=cache_extra_keys,
        )

        # Store entry for request (for legacy compatibility)
        norm_type = cache_type if cache_type in ("system", "user", "assistant") else "assistant"
        # Tensor-backed paged cache uses block.cache_data as the authoritative
        # in-memory payload and block-disk L2 as the durable spill tier. Keeping
        # the original full cache_data object here pins a second full KV copy
        # after the request has finished, which can push large native
        # RotatingKVCache models over the Metal working-set guard. Non-tensor
        # callers keep the legacy reference for get_cache_for_generation().
        entry_cache_data = None if is_tensor_data else cache_data
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=entry_cache_data,
            last_access=time.time(),
            cache_type=norm_type,
        )
        if is_tensor_data:
            cache_data = None
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass
            if HAS_MLX:
                try:
                    mx.clear_cache()
                except Exception:
                    pass

        # Track in per-type bucket for priority-aware eviction (F1 backport).
        # Drop old entry from any other bucket (request may have been re-stored
        # with a different role on a follow-up turn).
        for _t, _d in self._entries_by_type.items():
            if request_id in _d and _t != norm_type:
                del _d[request_id]
        self._entries_by_type[norm_type][request_id] = True
        self._entries_by_type[norm_type].move_to_end(request_id)

        blocks_with_data = sum(
            1
            for bid in block_table.block_ids
            if self.paged_cache.allocated_blocks.get(bid)
            and self.paged_cache.allocated_blocks[bid].cache_data is not None
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_with_data} with tensor data), "
            f"{block_table.num_tokens} tokens"
        )

        # Hold the RAM byte ceiling: evict free (ref==0) cached blocks — disk-L2
        # write-through first — so the in-RAM block mirror doesn't ratchet upward
        # with distinct prefixes. No-op when the ceiling is disabled. The blocks
        # just stored for this request are ref_count>=1, so they are never the
        # ones evicted here.
        if self.paged_cache.enforces_byte_budget:
            self.paged_cache.enforce_byte_budget()

        return block_table

    # Preserve source-introspection guards that validate the bounded M3/TQ
    # implementation rather than this lifecycle wrapper.
    store_cache.__wrapped__ = _store_cache_impl

    @staticmethod
    def _is_positional_cache(state_tuple, class_name: str = "") -> bool:
        """
        Determine if a cache layer's state is position-indexed or cumulative.

        Positional (sliceable by token position):
            KVCache, RotatingKVCache, QuantizedKVCache
        Cumulative (represents all processed tokens):
            MambaCache, ArraysCache

        Args:
            state_tuple: Cache state (keys, values) or arrays list
            class_name: Cache class name for disambiguation
        """
        if not state_tuple:
            return False

        # Class name is most reliable
        if class_name:
            positional = {"KVCache", "BatchKVCache", "RotatingKVCache",
                          "BatchRotatingKVCache", "QuantizedKVCache",
                          "TurboQuantKVCache"}
            cumulative = {"MambaCache", "BatchMambaCache", "ArraysCache"}
            if any(cls in class_name for cls in positional):
                return True
            if any(cls in class_name for cls in cumulative):
                return False

        # Structure-based fallback
        if isinstance(state_tuple, (tuple, list)) and len(state_tuple) == 2:
            first = state_tuple[0]
            if hasattr(first, "shape") and len(first.shape) in (3, 4):
                return True
            # QuantizedKVCache: state is ((data, scales, zeros), (data, scales, zeros))
            # first element is a tuple of arrays, not an array itself
            if isinstance(first, tuple) and len(first) >= 2:
                if hasattr(first[0], "shape") and len(first[0].shape) in (3, 4):
                    return True
        return False

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_last_block: bool = False,
        np_sources: Optional[dict] = None,
        existing_tokens: int = 0,
        store_cumulative_state: bool = True,
    ) -> Optional[List[Tuple]]:
        """
        Extract tensor slices for a single block from cache data.

        Handles both positional caches (KVCache - attention layers) and
        cumulative caches (MambaCache - SSM/hybrid layers).

        For KVCache layers: slices the KV tensors by token position.
        For MambaCache layers: stores the full cumulative state (only in
        the last block, since it represents all processed tokens).

        IMPORTANT — Metal safety:
        When np_sources is provided, positional (KV) layers are sliced in
        numpy space and wrapped with mx.array() to produce materialized MLX
        arrays with their own Metal buffers.  This avoids creating lazy MLX
        slices of already-evaluated parent arrays, which is fundamentally
        unsafe: evaluating such slices (whether via mx.eval or np.array)
        corrupts Metal command buffer state, causing either
        "addCompletedHandler after commit" assertions or IOGPUFamily
        "completeMemory() prepare count underflow" kernel panics.

        Args:
            cache_data: List of layer states from _extract_cache_states
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence
            is_last_block: Whether this is the last block in the sequence
            np_sources: Optional dict of layer_idx → (np_keys, np_values)
                        pre-converted numpy arrays from the parent KV cache.
                        When present, slicing is done in numpy space.
            store_cumulative_state: Whether terminal blocks own cumulative
                        state. False when a typed external companion store is
                        authoritative for hybrid SSM/GDN layers.

        Returns:
            List of tuples per layer. Each tuple is either:
            - ("kv", keys_slice, values_slice) for positional layers
            - ("cumulative", state_list) for cumulative/SSM layers
            - ("skip",) for cumulative layers in non-last blocks
        """
        if not HAS_MLX or not cache_data:
            return None

        block_slices = []
        for layer_idx, layer_state in enumerate(cache_data):
            if "state" not in layer_state:
                continue

            class_name = layer_state.get("class_name", "")

            if layer_state.get("dsv4_block_records"):
                record = _dsv4_delta_record_for_interval(
                    layer_state, start_idx, end_idx
                )
                if isinstance(record, dict):
                    block_slices.append(
                        (
                            "deepseek_v4_delta_v1",
                            record,
                            class_name,
                            _dsv4_cache_meta(layer_state),
                        )
                    )
                elif isinstance(record, (tuple, list)) and record:
                    block_slices.append(tuple(record))
                else:
                    raise ValueError("invalid DSV4 block transport record")
                continue

            if _is_dsv4_cache_class(class_name):
                state = layer_state["state"]
                if is_last_block and state is not None:
                    meta = layer_state.get("meta_state", "")
                    block_slices.append((
                        "deepseek_v4",
                        state,
                        meta,
                        class_name,
                        _dsv4_cache_meta(layer_state),
                    ))
                else:
                    block_slices.append((
                        "deepseek_v4_pending",
                        class_name,
                        _dsv4_cache_meta(layer_state),
                    ))
                continue

            if class_name == "ZayaNoStateCache" or layer_state.get("no_state"):
                block_slices.append(("no_state", class_name or "ZayaNoStateCache"))
                continue

            if _is_zaya_cca_cache_list_state(layer_state):
                subs = layer_state["sub_caches"]
                kv_sub = subs[0]
                state_sub = subs[1]
                kv_entry = ("skip",)
                try:
                    keys, values = kv_sub.get("state")
                    if isinstance(keys, (tuple, list)):
                        first_k = keys[0]
                        seq_len = first_k.shape[-2]
                        actual_end = min(end_idx, seq_len)
                        if start_idx < actual_end:
                            keys_slice = tuple(
                                t[..., start_idx:actual_end, :] for t in keys
                            )
                            values_slice = tuple(
                                t[..., start_idx:actual_end, :] for t in values
                            )
                            kv_entry = (
                                "quantized_kv",
                                keys_slice,
                                values_slice,
                                kv_sub.get("meta_state", ()),
                            )
                    else:
                        ndim = len(keys.shape)
                        seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
                        if seq_dim >= 0:
                            seq_len = keys.shape[seq_dim]
                            actual_end = min(end_idx, seq_len)
                            if start_idx < actual_end:
                                if ndim == 4:
                                    ks = keys[:, :, start_idx:actual_end, :]
                                    vs = values[:, :, start_idx:actual_end, :]
                                else:
                                    ks = keys[:, start_idx:actual_end, :]
                                    vs = values[:, start_idx:actual_end, :]
                                kv_entry = ("kv", ks, vs)
                except Exception:
                    kv_entry = ("skip",)

                cca_state = (
                    _copy_mlx_tree(state_sub.get("state"))
                    if is_last_block
                    else None
                )
                block_slices.append((
                    "zaya_cca",
                    kv_entry,
                    cca_state,
                    state_sub.get("meta_state", ""),
                    {
                        "schema": "zaya_cca_v1",
                        "cca_class_name": state_sub.get("class_name", "ArraysCache"),
                    },
                ))
                continue

            if _is_minimax_m3_cache_class(class_name):
                # MSA sparse layer: positional (keys, values, idx_keys), all on
                # the seq axis (dim 2). Slice all three to this block's range.
                # Prefer numpy sources (zero Metal ops); fall back to MLX slices.
                m3src = (
                    np_sources.get(layer_idx)
                    if isinstance(np_sources, dict)
                    else None
                )
                try:
                    if isinstance(m3src, dict) and m3src.get("type") == "minimax_m3":
                        np_k, np_v, np_idx, orig_dtype = m3src["kv"]
                        seq_len = np_k.shape[2]
                        actual_end = min(end_idx, seq_len)
                        if start_idx >= actual_end:
                            block_slices.append(("skip",))
                            continue
                        ks = _mx_from_np_slice(np_k[:, :, start_idx:actual_end, :])
                        vs = _mx_from_np_slice(np_v[:, :, start_idx:actual_end, :])
                        idxs = (
                            _mx_from_np_slice(np_idx[:, :, start_idx:actual_end, :])
                            if np_idx is not None
                            else None
                        )
                        if ks.dtype != orig_dtype:
                            ks = ks.astype(orig_dtype)
                            vs = vs.astype(orig_dtype)
                            if idxs is not None:
                                idxs = idxs.astype(orig_dtype)
                    else:
                        m3_state = layer_state["state"]
                        keys, values, idx_keys = m3_state
                        seq_len = keys.shape[2]
                        actual_end = min(end_idx, seq_len)
                        if start_idx >= actual_end:
                            block_slices.append(("skip",))
                            continue
                        ks = keys[:, :, start_idx:actual_end, :]
                        vs = values[:, :, start_idx:actual_end, :]
                        idxs = (
                            idx_keys[:, :, start_idx:actual_end, :]
                            if idx_keys is not None
                            else None
                        )
                    block_slices.append(("minimax_m3", ks, vs, idxs))
                except Exception as e:
                    logger.warning(
                        f"Layer {layer_idx} (MiniMaxM3SparseCache): "
                        f"failed to slice MSA cache: {e}"
                    )
                    block_slices.append(("skip",))
                continue

            # CacheList (MoE models): extract slices from each sub-cache
            if class_name == "CacheList" and "sub_caches" in layer_state:
                sub_slices = []
                for sub in layer_state["sub_caches"]:
                    sub_state = sub.get("state")
                    sub_cls = sub.get("class_name", "")
                    if sub_state is None:
                        sub_slices.append(("skip",))
                        continue
                    if self._is_positional_cache(sub_state, sub_cls):
                        try:
                            keys, values = sub_state
                            if isinstance(keys, (tuple, list)):
                                first_k = keys[0]
                                seq_len = first_k.shape[-2]
                                actual_end = min(end_idx, seq_len)
                                if start_idx >= actual_end:
                                    sub_slices.append(("skip",))
                                    continue
                                keys_slice = tuple(t[..., start_idx:actual_end, :] for t in keys)
                                values_slice = tuple(t[..., start_idx:actual_end, :] for t in values)
                                meta = sub.get("meta_state", ())
                                sub_slices.append(("quantized_kv", keys_slice, values_slice, meta))
                            else:
                                ndim = len(keys.shape)
                                seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
                                if seq_dim < 0:
                                    sub_slices.append(("skip",))
                                    continue
                                seq_len = keys.shape[seq_dim]
                                actual_end = min(end_idx, seq_len)
                                if start_idx >= actual_end:
                                    sub_slices.append(("skip",))
                                    continue
                                if ndim == 4:
                                    ks = keys[:, :, start_idx:actual_end, :]
                                    vs = values[:, :, start_idx:actual_end, :]
                                else:
                                    ks = keys[:, start_idx:actual_end, :]
                                    vs = values[:, start_idx:actual_end, :]
                                if sub_cls == "TurboQuantKVCache" and isinstance(
                                    sub.get("tq_config"), dict
                                ):
                                    from .tq_disk_store import encode_tq_block

                                    sub_slices.append(
                                        encode_tq_block(ks, vs, sub["tq_config"])
                                    )
                                else:
                                    sub_slices.append(("kv", ks, vs))
                        except Exception:
                            sub_slices.append(("skip",))
                    else:
                        if is_last_block and store_cumulative_state:
                            meta = sub.get("meta_state", "")
                            sub_slices.append((
                                "cumulative",
                                _copy_mlx_tree(sub_state),
                                meta,
                                sub_cls,
                            ))
                        else:
                            sub_slices.append(("skip",))
                block_slices.append(("cache_list", sub_slices))
                continue

            state = layer_state["state"]

            # Detect if this is a positional (KVCache) or cumulative (MambaCache) layer
            if self._is_positional_cache(state, class_name):
                try:
                    keys, values = state

                    # QuantizedKVCache: keys/values are tuples or lists of (data, scales, zeros)
                    if isinstance(keys, (tuple, list)):
                        # Use first component to detect shape
                        first_k = keys[0]
                        seq_len = first_k.shape[-2]  # seq axis is always -2
                        actual_end = min(end_idx, seq_len)
                        if start_idx >= actual_end:
                            block_slices.append(("skip",))
                            continue

                        keys_slice = tuple(
                            t[..., start_idx:actual_end, :] for t in keys
                        )
                        values_slice = tuple(
                            t[..., start_idx:actual_end, :] for t in values
                        )
                        meta = layer_state.get("meta_state", ())
                        block_slices.append(("quantized_kv", keys_slice, values_slice, meta))
                        continue

                    ndim = len(keys.shape)

                    # Handle both 3D (n_kv_heads, seq, dim) and
                    # 4D (batch, n_kv_heads, seq, dim) tensors
                    if ndim == 4:
                        seq_dim = 2
                    elif ndim == 3:
                        seq_dim = 1
                    else:
                        block_slices.append(("skip",))
                        continue

                    if "Rotating" in class_name:
                        try:
                            if np_sources is not None and layer_idx in np_sources:
                                import numpy as np

                                np_k, np_v, orig_dtype = np_sources[layer_idx]
                                if is_last_block:
                                    terminal = _rotating_terminal_window(
                                        np_k,
                                        np_v,
                                        layer_state.get("meta_state", ()),
                                        expected_offset=end_idx,
                                        concatenate=np.concatenate,
                                    )
                                else:
                                    terminal = _rotating_previous_block_window(
                                        np_k,
                                        np_v,
                                        layer_state.get("meta_state", ()),
                                        target_offset=end_idx,
                                        block_size=self.block_size,
                                        concatenate=np.concatenate,
                                    )
                                tk, tv, max_size, keep, offset, idx_state = terminal
                                tk = _mx_from_np_slice(tk)
                                tv = _mx_from_np_slice(tv)
                                if tk.dtype != orig_dtype:
                                    tk = tk.astype(orig_dtype)
                                    tv = tv.astype(orig_dtype)
                            else:
                                if is_last_block:
                                    terminal = _rotating_terminal_window(
                                        keys,
                                        values,
                                        layer_state.get("meta_state", ()),
                                        expected_offset=end_idx,
                                        concatenate=mx.concatenate,
                                    )
                                else:
                                    terminal = _rotating_previous_block_window(
                                        keys,
                                        values,
                                        layer_state.get("meta_state", ()),
                                        target_offset=end_idx,
                                        block_size=self.block_size,
                                        concatenate=mx.concatenate,
                                    )
                                tk, tv, max_size, keep, offset, idx_state = terminal
                                tk = _copy_mlx_tree(tk)
                                tv = _copy_mlx_tree(tv)
                            block_slices.append((
                                "rotating_kv",
                                tk,
                                tv,
                                max_size,
                                keep,
                                offset,
                                idx_state,
                            ))
                        except ValueError as exc:
                            if is_last_block:
                                # Same contract as the numpy cutter above: a
                                # terminal boundary the snapshot cannot cut is
                                # stored as a PENDING marker, never a silent
                                # ("skip",), so the fetch-side terminal guard
                                # can see the family and walk the candidate
                                # back to the newest exact anchor instead of
                                # crediting a hit that reconstructs to nothing.
                                logger.warning(
                                    "Layer %s (%s): %s — storing terminal "
                                    "boundary as rotating_kv_pending so fetches "
                                    "walk back to an earlier exact anchor",
                                    layer_idx,
                                    class_name,
                                    exc,
                                )
                            block_slices.append((
                                "rotating_kv_pending",
                                class_name,
                            ))
                        continue

                    seq_len = keys.shape[seq_dim]
                    slice_start, actual_end = _positional_layer_slice_bounds(
                        layer_state,
                        class_name,
                        start_idx,
                        end_idx,
                        seq_len,
                        existing_tokens,
                    )
                    if actual_end <= 0 or slice_start >= actual_end:
                        block_slices.append(("skip",))
                        continue

                    # When numpy sources are available, slice in numpy space
                    # and wrap with mx.array() to get materialized MLX arrays.
                    # This avoids creating lazy MLX slices of evaluated parent
                    # arrays, which corrupts Metal command buffer state.
                    if np_sources is not None and layer_idx in np_sources:
                        np_k, np_v, orig_dtype = np_sources[layer_idx]
                        if ndim == 4:
                            ks = _mx_from_np_slice(np_k[:, :, slice_start:actual_end, :])
                            vs = _mx_from_np_slice(np_v[:, :, slice_start:actual_end, :])
                        else:
                            ks = _mx_from_np_slice(np_k[:, slice_start:actual_end, :])
                            vs = _mx_from_np_slice(np_v[:, slice_start:actual_end, :])
                        # Restore original dtype (e.g. bfloat16 → float16 → bfloat16)
                        if ks.dtype != orig_dtype:
                            ks = ks.astype(orig_dtype)
                            vs = vs.astype(orig_dtype)
                    elif ndim == 4:
                        ks = keys[:, :, slice_start:actual_end, :]
                        vs = values[:, :, slice_start:actual_end, :]
                    else:  # ndim == 3
                        ks = keys[:, slice_start:actual_end, :]
                        vs = values[:, slice_start:actual_end, :]

                    # Native TQ paged storage: encode each independent block with
                    # the layer's exact seed/bit policy. The cumulative SSM/GDN
                    # companion state remains on its typed full-precision path.
                    if class_name == "TurboQuantKVCache" and isinstance(
                        layer_state.get("tq_config"), dict
                    ):
                        from .tq_disk_store import encode_tq_block

                        block_slices.append(
                            encode_tq_block(ks, vs, layer_state["tq_config"])
                        )
                    else:
                        block_slices.append(("kv", ks, vs))
                except Exception as e:
                    logger.warning(
                        f"Layer {layer_idx} ({class_name}): "
                        f"failed to slice positional cache: {e}"
                    )
                    block_slices.append(("skip",))
            else:
                # Cumulative cache (MambaCache, ArraysCache, etc.)
                # State is not position-indexed — it represents ALL tokens processed
                # Only store in the last block (it encompasses all prior tokens)
                if is_last_block and store_cumulative_state and state is not None:
                    meta = layer_state.get("meta_state", "")
                    block_slices.append((
                        "cumulative",
                        _copy_mlx_tree(state),
                        meta,
                        class_name,
                    ))
                else:
                    # Windowed dots3 latent layers can checkpoint recent
                    # NON-terminal boundaries exactly (ledger row 152) —
                    # without this every divergent-prefix restore collapses
                    # to cold prefill.
                    checkpoint = (
                        _dots3_window_boundary_checkpoint(layer_state, end_idx)
                        if store_cumulative_state
                        else None
                    )
                    if checkpoint is not None:
                        tag, sliced, cmeta, ccls = checkpoint
                        block_slices.append(
                            (tag, _copy_mlx_tree(sliced), cmeta, ccls)
                        )
                    else:
                        block_slices.append(("skip",))

        if block_slices:
            tag_counts: dict = {}
            for bs in block_slices:
                t = bs[0] if isinstance(bs, (tuple, list)) else "?"
                tag_counts[t] = tag_counts.get(t, 0) + 1
            tag_str = ", ".join(f"{k}={v}" for k, v in tag_counts.items())
            logger.debug(
                f"Block tensor slice: {len(block_slices)}/{len(cache_data)} layers "
                f"({tag_str}), tokens [{start_idx}:{end_idx}], is_last={is_last_block}"
            )
        return block_slices if block_slices else None

    def get_cache_for_generation(
        self,
        request_id: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Get cache data for generation, applying COW if needed.

        Args:
            request_id: Request identifier

        Returns:
            Tuple of (cache_data, was_copied)
        """
        entry = self._request_tables.get(request_id)
        if not entry:
            return None, False

        # Get blocks with COW
        blocks, was_copied = self.paged_cache.get_blocks_for_generation(
            entry.block_table
        )
        if entry.cache_data is None:
            cache_data = self.reconstruct_cache(entry.block_table)
            entry.last_access = time.time()
            return cache_data, was_copied

        if was_copied:
            # Deep copy cache data for modified blocks
            cache_data = copy.deepcopy(entry.cache_data)
        else:
            cache_data = entry.cache_data

        entry.last_access = time.time()
        return cache_data, was_copied

    def release_cache(self, request_id: str) -> None:
        """
        Release cache blocks for a completed request.

        Args:
            request_id: Request identifier
        """
        self._hit_credits.pop(request_id, None)
        entry = self._request_tables.pop(request_id, None)
        if entry:
            # Drop from per-type LRU bucket so eviction priority stays accurate
            for _d in self._entries_by_type.values():
                if request_id in _d:
                    del _d[request_id]
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")

    def detach_request(self, request_id: str) -> None:
        """Drop the per-request block_table entry WITHOUT freeing blocks.

        Used by the SSM-companion-miss fast path: the request will do a
        full prefill and the in-memory block_table is no longer useful for
        it, but the underlying KV blocks must stay cache-resident so that
        future cross-session requests with the same prompt prefix can hit
        them. Mirrors PagedCacheManager.detach_request semantics.

        Without this, scheduler.py:2381 was calling release_cache which
        free_block()s every block — turning the SSM-companion miss into
        a cache poisoning event that defeats cross-session reuse.
        """
        self._hit_credits.pop(request_id, None)
        entry = self._request_tables.pop(request_id, None)
        if entry:
            for _d in self._entries_by_type.values():
                if request_id in _d:
                    del _d[request_id]
            self.paged_cache.detach_request(request_id)
            logger.debug(f"Detached request {request_id} (blocks kept cached)")

    def adjust_cache_hit_credit(
        self,
        request_id: str,
        *,
        accepted_tokens: int,
    ) -> bool:
        """Reconcile lookup credit with the prefix the consumer actually used.

        Hybrid MLLM lookup can find attention-KV blocks and only then discover
        that the matching SSM/GDN companion is absent.  In that case generation
        full-prefills from token zero, so reporting a hit or saved tokens is
        accounting fiction.  A shorter valid companion may also trim the usable
        prefix; retain only that smaller token credit.

        Returns True when an outstanding hit credit was adjusted.
        """

        credited_tokens = self._hit_credits.get(request_id)
        if credited_tokens is None:
            return False
        try:
            accepted = max(0, min(int(accepted_tokens or 0), credited_tokens))
        except (TypeError, ValueError):
            accepted = 0
        if accepted >= credited_tokens:
            return False

        self._tokens_saved = max(
            0,
            self._tokens_saved - (credited_tokens - accepted),
        )
        if accepted == 0:
            self._hits = max(0, self._hits - 1)
            self._misses += 1
            self._hit_credits.pop(request_id, None)
        else:
            self._hit_credits[request_id] = accepted
        logger.info(
            "Adjusted prefix-cache hit credit for %s: credited=%d accepted=%d "
            "(hybrid companion boundary)",
            request_id,
            credited_tokens,
            accepted,
        )
        return True

    def finalize_cache_hit_credit(self, request_id: str) -> None:
        """Drop reconciliation state once a request's cache decision is final."""

        self._hit_credits.pop(request_id, None)

    def trim_block_table(
        self, request_id: str, target_tokens: int
    ) -> Optional["BlockTable"]:
        """Shrink a live block_table down to ``target_tokens`` by releasing
        trailing blocks, keeping only the block-aligned prefix.

        Used by the vmlx#91 SSM resume path: when the exact SSM companion
        cache misses but a shorter checkpoint is a valid prefix, we trim
        the KV cache to match the checkpoint so KV + SSM stay aligned, and
        the scheduler prefills only the tail delta.

        Args:
            request_id: Request whose block_table to trim (must be held in
                the paged cache — look up via get_block_table).
            target_tokens: Desired cached-token count. The actual result is
                block-aligned (floor to nearest multiple of block_size), so
                it may be slightly less than requested.

        Returns:
            Trimmed BlockTable if trim succeeded and >0 blocks retained,
            None if target_tokens == 0 (caller should release entirely) or
            if the request has no active block_table.

        Safety:
            * Trailing blocks have their ref_count decremented via
              ``paged_cache.decrement_ref`` so they rejoin the free pool
              for future reuse. No ref-count leak.
            * Block IDs retained are exactly the prefix, preserving
              content-hash chain integrity for future sibling requests.
        """
        block_table = self.paged_cache.get_block_table(request_id)
        if block_table is None or not block_table.block_ids:
            return None
        if target_tokens <= 0:
            return None

        # Block-align (floor): never exceed target_tokens.
        block_size = self.block_size
        kept_blocks_count = target_tokens // block_size
        if kept_blocks_count <= 0:
            return None
        kept_blocks_count = min(kept_blocks_count, len(block_table.block_ids))
        if kept_blocks_count == len(block_table.block_ids):
            # Nothing to trim — already at or below target.
            return block_table

        # Release the trailing block refs so they return to the free pool.
        to_release = block_table.block_ids[kept_blocks_count:]
        for block_id in to_release:
            self.paged_cache.decrement_ref(block_id)

        # Truncate the block_table in-place.
        block_table.block_ids = block_table.block_ids[:kept_blocks_count]
        block_table.num_tokens = kept_blocks_count * block_size

        logger.debug(
            f"vmlx#91: trimmed block_table for {request_id} to "
            f"{kept_blocks_count} blocks ({block_table.num_tokens} tokens); "
            f"released {len(to_release)} trailing blocks"
        )
        return block_table

    @staticmethod
    def _block_has_terminal_state(block: Any, tag: str) -> bool:
        """Return True if a block carries a terminal path-dependent payload."""
        for entry in getattr(block, "cache_data", None) or []:
            if not isinstance(entry, (tuple, list)) or not entry:
                continue
            if entry[0] != tag:
                continue
            if tag == "zaya_cca":
                return len(entry) > 2 and entry[2] is not None
            return True
        return False

    def trim_block_table_to_terminal_state(
        self,
        request_id: str,
        target_tokens: int,
        tag: str,
    ) -> Optional["BlockTable"]:
        """Shrink a block table only to a native terminal-state checkpoint.

        Path-dependent cache families cannot use plain block-aligned KV
        trimming: a non-terminal DSV4/ZAYA block carries only partial state.
        This helper scans the request's prefix blocks and keeps the longest
        prefix at or below ``target_tokens`` whose last block contains the
        family-specific terminal payload (for example ``deepseek_v4``).
        """
        block_table = self.paged_cache.get_block_table(request_id)
        if block_table is None or not block_table.block_ids or target_tokens <= 0:
            return None

        best_index: Optional[int] = None
        best_tokens = 0
        running_tokens = 0
        for idx, block_id in enumerate(block_table.block_ids):
            block = self.paged_cache.allocated_blocks.get(block_id)
            if block is None:
                return None
            running_tokens += int(getattr(block, "token_count", 0) or 0)
            if running_tokens > target_tokens:
                break
            if self._block_has_terminal_state(block, tag):
                best_index = idx
                best_tokens = running_tokens

        if best_index is None or best_tokens <= 0:
            return None

        kept_count = best_index + 1
        if kept_count == len(block_table.block_ids):
            block_table.num_tokens = best_tokens
            return block_table

        to_release = block_table.block_ids[kept_count:]
        for block_id in to_release:
            self.paged_cache.decrement_ref(block_id)

        block_table.block_ids = block_table.block_ids[:kept_count]
        block_table.num_tokens = best_tokens
        logger.debug(
            "Trimmed path-dependent block_table for %s to terminal %s "
            "checkpoint: %d blocks, %d tokens; released %d trailing blocks",
            request_id,
            tag,
            kept_count,
            best_tokens,
            len(to_release),
        )
        return block_table

    def release_low_priority(self, n_entries: int = 1) -> int:
        """
        Release up to n_entries from the lowest-priority non-empty bucket
        (assistant → user → system). Used by the scheduler when under block
        pressure to evict ephemeral assistant entries before pinned system
        prompts. Returns the number of entries actually released.

        Block-level reclamation still happens through paged_cache ref counts
        — this just biases WHICH request_table entries get released first.
        """
        released = 0
        for t in _CACHE_TYPE_PRIORITY:
            d = self._entries_by_type[t]
            while d and released < n_entries:
                rid, _ = d.popitem(last=False)
                if rid in self._request_tables:
                    self.release_cache(rid)
                    released += 1
            if released >= n_entries:
                break
        return released

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """
        Fork cache from one request to another (COW).

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable, or None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        # Create new entry with reference to same cache data
        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            cache_data=source_entry.cache_data,  # Shared reference
            last_access=time.time(),
        )

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    @staticmethod
    def _reconstruct_memo_signature(block_table: Any) -> Optional[tuple]:
        """Identity of a reconstruction: same blocks and length, same state."""
        try:
            ids = tuple(int(b) for b in (block_table.block_ids or ()))
        except Exception:
            return None
        if not ids:
            return None
        return (ids, int(getattr(block_table, "num_tokens", 0) or 0))

    def _reconstruct_memo_fits_in_headroom(self, reconstructed_caches: Any) -> bool:
        """Is there room to retain a SECOND copy of this reconstruction?

        The memo turns a repeated reconstruction into a lookup (measured on the
        text path: 3071ms -> 0.015ms at 413 blocks), but it pays for that with a
        full ``deepcopy``. On a compact DSV4 delta record that is cheap; on a
        mixed-SWA L1 at 86k tokens it is ~4.6GB, which would roughly DOUBLE
        resident KV at exactly the depth where memory is already tightest — and
        an OOM costs far more than the reconstruction it was avoiding.

        So the memo is opportunistic: keep it only while the copy comfortably
        fits in the REMAINING headroom, otherwise decline and let the second pass
        reconstruct. Unknown readings decline too — the safe default here is the
        slower one, because the fast path's failure mode is an OOM.

        MEASURE AGAINST HEADROOM, NOT AGAINST TOTAL ACTIVE. This first compared
        ``active + copy`` to a percentage of the whole working set, and ``active``
        includes the resident MODEL. Measured live on DSV4-Flash:

            copy 0.54GB on top of active 95.44GB would pass the 70%
            working-set budget (75.26GB)

        active alone was already 20GB past the ceiling, so the sum exceeded it no
        matter how small the copy was — the memo could never be retained once a
        large model was loaded, which is exactly when it matters. It declined on
        every turn while the answer pass re-read the whole prefix from L2: 7.5s at
        83k tokens, 21.7s at 166k, both reproducible to ~0.3s across runs.

        The real question is whether the copy fits in what is left
        (``max_ws - active``, ~12GB in that reading, for a 0.54GB copy), and how
        much of that remainder we are willing to spend.
        """
        try:
            from .utils.memory_limits import get_effective_metal_working_set_bytes

            active, max_ws = get_effective_metal_working_set_bytes(mx)
        except Exception:  # noqa: BLE001
            return False
        if max_ws <= 0 or active <= 0:
            return False
        try:
            from .memory_cache import estimate_kv_cache_memory

            copy_bytes = int(estimate_kv_cache_memory(reconstructed_caches) or 0)
        except Exception:  # noqa: BLE001
            return False
        if copy_bytes <= 0:
            return False
        headroom = int(max_ws) - int(active)
        if headroom <= 0:
            logger.info(
                "Reconstruct memo DECLINED: no working-set headroom "
                "(active %.2fGB of %.2fGB); the second pass will re-read from L2.",
                active / (1024**3),
                max_ws / (1024**3),
            )
            return False
        try:
            # Percent of the REMAINING headroom the copy may occupy. Named for
            # the working set for backwards compatibility, but it was measured
            # against the wrong quantity before and never admitted anything.
            budget_pct = float(
                os.environ.get("VMLX_RECONSTRUCT_MEMO_MAX_WS_PCT", "50")
            )
        except (TypeError, ValueError):
            budget_pct = 50.0
        ceiling = int(headroom * (budget_pct / 100.0))
        if copy_bytes > ceiling:
            # INFO, not debug: this is a SILENT CAP. When it fires the second
            # pass re-reads the whole prefix from L2 and the user sees a
            # multi-second stall with nothing in the log to explain it — 21.3s
            # at 166k tokens on DSV4-Flash, measured. A guard that declines
            # invisibly also makes any A/B of the memo compare stock to stock.
            logger.info(
                "Reconstruct memo DECLINED: copy %.2fGB exceeds %.0f%% of the "
                "%.2fGB working-set headroom (%.2fGB allowed, active %.2fGB of "
                "%.2fGB); the second pass will re-read from L2. Raise "
                "VMLX_RECONSTRUCT_MEMO_MAX_WS_PCT to allow it.",
                copy_bytes / (1024**3),
                budget_pct,
                headroom / (1024**3),
                ceiling / (1024**3),
                active / (1024**3),
                max_ws / (1024**3),
            )
            return False
        return True

    def arm_reconstruct_memo(self, enabled: bool = True) -> None:
        """Ask the next reconstruction to retain a pristine copy for reuse.

        Set by the scheduler when the request it is admitting will be followed
        immediately by a second pass over the same prompt. Kill switch:
        VMLX_DSV4_RECONSTRUCT_MEMO=0 restores the plain double replay.

        An SSD-only manager is an explicit no-retained-cache contract. The
        memo is a full deep copy of reconstructed KV/native state, so keeping
        it there would be an unreported in-RAM cache even though the paged
        manager correctly reports zero resident bytes.
        """
        memo_disabled = os.environ.get(
            "VMLX_DSV4_RECONSTRUCT_MEMO", "1"
        ).strip().lower() in {
            "0",
            "off",
            "false",
            "no",
        }
        if (
            not enabled
            or memo_disabled
            or bool(getattr(self.paged_cache, "disk_only", False))
        ):
            self._reconstruct_memo_arm = False
            self._reconstruct_memo = None
            return
        self._reconstruct_memo_arm = True

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from stored block data.

        Handles both positional caches (KVCache - attention layers) and
        cumulative caches (MambaCache - SSM/hybrid layers):
        - KVCache: concatenates tensor slices from all blocks along seq axis
        - MambaCache: restores full cumulative state from the last block

        Uses mlx_lm's cache classes and from_state() for proper reconstruction.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed cache objects (one per layer),
            or None if reconstruction fails
        """
        self._last_reconstruct_disk_blocks = 0
        self._last_reconstruct_tq_blocks = 0
        if not block_table or not block_table.block_ids:
            return None

        # Reasoning families that never close their think rail run every reply
        # as two requests over the SAME prompt, so this replay used to run twice
        # back to back on an identical block table — measured at ~4.2ms per
        # block with no disk tier at all (it is delta-replay compute, not I/O),
        # which is ~3s per turn at 100k and the dominant term of the
        # reasoning-to-content stall. When the caller knows a second pass is
        # imminent it asks us to keep a pristine copy, and that pass returns it
        # instead of replaying. Single use, and any non-matching reconstruct
        # drops it so a stale composite is never held.
        memo_signature = self._reconstruct_memo_signature(block_table)
        memo = getattr(self, "_reconstruct_memo", None)
        if memo is not None:
            self._reconstruct_memo = None
            if memo_signature is not None and memo[0] == memo_signature:
                logger.info(
                    "Reconstruct memo HIT (%d tokens, %d blocks) — second pass "
                    "skips the replay",
                    (memo_signature[1] if len(memo_signature) > 1 else 0),
                    len(memo_signature[0]) if memo_signature else 0,
                )
                self._last_reconstruct_memo_hit = True
                return memo[1]
            # A retained copy that does not match is the expensive case: the
            # first pass paid the deepcopy AND the second pass still replays.
            # It was silent, so a stall could not be attributed without adding
            # instrumentation after the fact — say which half differs.
            held_ids, held_tokens = (memo[0] + ((), 0))[:2] if memo[0] else ((), 0)
            want_ids, want_tokens = (
                (memo_signature + ((), 0))[:2] if memo_signature else ((), 0)
            )
            logger.info(
                "Reconstruct memo MISS: held %d blocks/%s tokens, asked for "
                "%d blocks/%s tokens (%s) — second pass will replay",
                len(held_ids),
                held_tokens,
                len(want_ids),
                want_tokens,
                "same blocks, different length"
                if held_ids == want_ids
                else "different blocks",
            )
        self._last_reconstruct_memo_hit = False

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        disk_backed_block_ids: set[int] = set()
        l2_readable_block_ids: set[int] = set()
        native_resident_block_ids: set[int] = set()
        reconstruction_succeeded = False
        try:
            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning(f"Block {block_id} not found in allocated blocks")
                    # issue #198 (1B): report contiguous resident prefix so
                    # partial paged-cache eviction is visible (full partial
                    # reuse deferred pending live cache-pressure verification).
                    logger.info(
                        "issue #198 (1B): partial paged eviction — %d/%d blocks "
                        "resident before block %s; falling back to cold prefill",
                        len(all_block_data), len(block_table.block_ids), block_id,
                    )
                    return None

                block_data = block.cache_data
                transient_selective_read = False
                if block_data is None:
                    # Frugal mode (or post-eviction): in-RAM mirror skipped,
                    # but the block was written to L2 disk during _store. Pull
                    # it back lazily so the in-session reconstruct path still
                    # succeeds. Without this, every prefix-cache hit on a
                    # frugal-stored prompt would silently fall through to a
                    # full prefill, defeating the cache.
                    _disk = getattr(self.paged_cache, "_disk_store", None)
                    if _disk is not None and block.block_hash is not None:
                        try:
                            _selective_reader = getattr(
                                _disk,
                                "read_block_for_reconstruction",
                                None,
                            )
                            if (
                                bool(getattr(self.paged_cache, "disk_only", False))
                                and self._validate_rotating_terminal
                                and callable(_selective_reader)
                            ):
                                _disk_data = _selective_reader(
                                    block.block_hash,
                                    rotating_target_offset=block_table.num_tokens,
                                )
                                # This result may contain explicit pending
                                # markers in place of stale rotating windows.
                                # It is request-local reconstruction input, not
                                # a complete shared block payload.
                                transient_selective_read = _disk_data is not None
                            else:
                                _disk_data = _disk.read_block(block.block_hash)
                        except Exception as _re:
                            logger.warning(
                                f"Block {block_id} disk read failed: {_re}"
                            )
                            _disk_data = None
                        if _disk_data is not None:
                            disk_backed_block_ids.add(block_id)
                            block_data = _disk_data
                            if transient_selective_read:
                                logger.debug(
                                    "Block %s loaded as request-local L2 "
                                    "reconstruction input (hash=%s)",
                                    block_id,
                                    block.block_hash.hex()[:12]
                                    if hasattr(block.block_hash, "hex")
                                    else block.block_hash,
                                )
                            else:
                                block.cache_data = _disk_data
                                block.cache_data_from_disk = True
                                block.cache_data_transient = True
                                self.paged_cache._note_resident(
                                    block,
                                    self.paged_cache.estimate_block_nbytes(_disk_data),
                                )
                                self.paged_cache.transient_disk_promotions += 1
                                self.paged_cache.transient_disk_peak_bytes = max(
                                    self.paged_cache.transient_disk_peak_bytes,
                                    self.paged_cache.resident_bytes,
                                )
                                logger.debug(
                                    f"Block {block_id} rehydrated from L2 disk "
                                    f"(hash={block.block_hash.hex()[:12] if hasattr(block.block_hash, 'hex') else block.block_hash})"
                                )
                    if block_data is None:
                        logger.debug(f"Block {block_id} has no tensor data stored")
                        # issue #198 (1B): partial eviction diagnostic
                        logger.info(
                            "issue #198 (1B): partial paged eviction — %d/%d blocks "
                            "resident before block %s (no tensor data); cold prefill",
                            len(all_block_data), len(block_table.block_ids), block_id,
                        )
                        return None

                if transient_selective_read:
                    # Already accounted as a disk-backed reconstruction input;
                    # deliberately leave the shared CacheBlock empty.
                    pass
                elif getattr(block, "cache_data_from_disk", False):
                    disk_backed_block_ids.add(block_id)
                elif block.block_hash is not None:
                    disk_store = getattr(self.paged_cache, "_disk_store", None)
                    if disk_store is not None:
                        try:
                            has_block = getattr(disk_store, "has_block", None)
                            if callable(has_block):
                                if has_block(block.block_hash):
                                    l2_readable_block_ids.add(block_id)
                            elif disk_store.read_block(block.block_hash) is not None:
                                l2_readable_block_ids.add(block_id)
                        except Exception:
                            pass
                all_block_data.append(block_data)
                if _block_payload_needs_native_residency(block_data):
                    native_resident_block_ids.add(block_id)

            if not all_block_data:
                return None

            # Diagnostic only, off by default. Reconstructing a long chain has
            # aborted the PROCESS on weight-heavy boxes with an uncatchable
            # Metal command-buffer OOM (MiniMax-M2.7-JANG_K, 451 blocks). The
            # finished KV demonstrably fits — 253,952 B/token x 28,864 tokens =
            # 7.33GB against 27.5GB free — so the peak lives somewhere in the
            # staging above, and the failure kills the process before any
            # Python-side accounting can report it. Set
            # VMLX_TRACE_RECONSTRUCT_MEMORY=1 to have each phase print active
            # Metal memory, which is the only way to see the spike at all.
            _trace_memory = os.environ.get(
                "VMLX_TRACE_RECONSTRUCT_MEMORY", ""
            ).strip().lower() in {"1", "true", "yes", "on"}

            def _memory_probe(phase: str) -> None:
                if not _trace_memory:
                    return
                try:
                    from .utils.memory_limits import (
                        get_effective_metal_working_set_bytes,
                    )

                    active, max_ws = get_effective_metal_working_set_bytes(mx)
                    logger.info(
                        "reconstruct-memory[%s]: active=%.2fGB free=%.2fGB "
                        "blocks=%d",
                        phase,
                        active / (1024**3),
                        (max_ws - active) / (1024**3),
                        len(all_block_data),
                    )
                except Exception:
                    pass

            _memory_probe("blocks_staged")

            # Get number of layers from first block
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # Validate every block before
            # reconstructing tensors. A single corrupt block whose
            # entries carry nonsense shapes used to cascade through
            # ``mx.concatenate(...)`` then ``cache.state = state`` then
            # blow up on the next forward pass with a 555 GB
            # ``[metal::malloc]`` request. Validate up-front; on any
            # failure, force cache miss and let the scheduler re-prefill.
            try:
                from .cache_record_validator import reject_or_warn as _reject_or_warn
            except Exception:
                _reject_or_warn = None
            if _reject_or_warn is not None:
                _expected_layers = getattr(self, "_expected_num_layers", None)
                if _expected_layers is None:
                    _expected_layers = num_layers  # at least guard layer-count drift across blocks
                for _bi, _block_data in enumerate(all_block_data):
                    if not _reject_or_warn(
                        _block_data,
                        expected_num_layers=_expected_layers,
                        source=f"reconstruct[block={_bi}]",
                    ):
                        return None

            # Import cache classes
            try:
                from mlx_lm.models.cache import KVCache, RotatingKVCache
            except ImportError:
                from mlx_lm.models.cache import KVCache
                RotatingKVCache = None
            try:
                from mlx_lm.models.cache import MambaCache
                has_mamba = True
            except ImportError:
                has_mamba = False
            has_rotating = RotatingKVCache is not None

            # Reconstruct each layer
            reconstructed_caches = []
            reconstructed_indices: set = set()  # tracks which layer_idx values were rebuilt
            kv_count = 0
            cumulative_count = 0
            tq_native_entry_count = 0
            # TQ codec decode graphs are kept lazy per layer and evaluated in
            # one batch before returning: 16 per-layer eval sync points cost
            # ~90ms each on an 8k restore (vmlx#91), a single combined eval
            # lets Metal overlap the layers.
            deferred_tq_eval: list = []

            # vmlx#91: pre-scan native TQ layers (and CacheList TQ sub-caches)
            # so compatible layers decode as ONE stacked cross-layer graph
            # instead of L independent per-layer codec graphs. Output is
            # bit-identical to decode_tq_blocks per layer; layers that cannot
            # co-batch fall back inside decode_tq_layer_groups. Everything
            # stays lazy — the single deferred eval below is the only sync.
            tq_predecoded: dict = {}
            tq_group_candidates: dict = {}
            for _pre_layer_idx in range(num_layers):
                _pre_top: list = []
                _pre_subs: dict = {}
                _pre_blocked = False
                for _pre_block_data in all_block_data:
                    if _pre_layer_idx >= len(_pre_block_data):
                        continue
                    _pre_entry = _pre_block_data[_pre_layer_idx]
                    if not isinstance(_pre_entry, (tuple, list)) or not _pre_entry:
                        continue
                    if _pre_entry[0] == "turboquant_kv":
                        _pre_top.append(tuple(_pre_entry))
                    elif _pre_entry[0] == "zaya_cca":
                        # ZAYA branch outranks the TQ branch in the loop below;
                        # don't waste a batched decode on entries it ignores.
                        _pre_blocked = True
                    elif _pre_entry[0] == "cache_list" and len(_pre_entry) > 1:
                        _pre_sub_list = _pre_entry[1]
                        if isinstance(_pre_sub_list, (tuple, list)):
                            for _pre_sub_idx, _pre_sub in enumerate(_pre_sub_list):
                                if (
                                    isinstance(_pre_sub, (tuple, list))
                                    and _pre_sub
                                    and _pre_sub[0] == "turboquant_kv"
                                ):
                                    _pre_subs.setdefault(
                                        _pre_sub_idx, []
                                    ).append(tuple(_pre_sub))
                if _pre_blocked:
                    continue
                if _pre_top:
                    tq_group_candidates[("layer", _pre_layer_idx)] = _pre_top
                elif _pre_subs:
                    for _pre_sub_idx, _pre_sub_entries in _pre_subs.items():
                        tq_group_candidates[
                            ("sub", _pre_layer_idx, _pre_sub_idx)
                        ] = _pre_sub_entries
            if len(tq_group_candidates) >= 2:
                from .tq_disk_store import decode_tq_layer_groups

                tq_predecoded = decode_tq_layer_groups(tq_group_candidates)

            for layer_idx in range(num_layers):
                if _trace_memory and layer_idx % 8 == 0:
                    _memory_probe(f"layer_{layer_idx}/{num_layers}")
                # Collect this layer's data from all blocks
                layer_entries = []
                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        layer_entries.append(block_data[layer_idx])

                if not layer_entries:
                    continue

                # Check the type tag from _extract_block_tensor_slice
                # Collect entries by type, find best cumulative entry
                best_cumulative = None
                best_dsv4 = None
                dsv4_delta_entries = []
                kv_slices_keys = []
                kv_slices_values = []
                rotating_entries = []
                rotating_terminal = None
                quantized_kv_slices_keys = []  # list of tuples of (data, scales, zeros)
                quantized_kv_slices_values = []
                quantized_meta = None
                tq_block_entries = []

                cache_list_entries = []  # sub-slice lists from CacheList blocks
                zaya_cca_entries = []
                m3_slices_keys = []      # MiniMax-M3 MSA sparse layer
                m3_slices_values = []
                m3_slices_idx = []
                m3_has_idx = False

                for entry in layer_entries:
                    if not isinstance(entry, (tuple, list)):
                        continue
                    tag = entry[0]
                    if tag == "kv":
                        kv_slices_keys.append(entry[1])
                        kv_slices_values.append(entry[2])
                    elif tag == "quantized_kv":
                        quantized_kv_slices_keys.append(entry[1])  # tuple of 3
                        quantized_kv_slices_values.append(entry[2])  # tuple of 3
                        if len(entry) > 3 and quantized_meta is None:
                            quantized_meta = entry[3]
                    elif tag == "turboquant_kv":
                        tq_block_entries.append(tuple(entry))
                        tq_native_entry_count += 1
                    elif tag == "rotating_kv":
                        rotating_entries.append(entry)
                        # v11 rotating records are exact terminal windows.  A
                        # reused earlier terminal can legitimately appear in
                        # the middle of a longer block chain; only the record
                        # whose logical offset equals this table boundary is
                        # valid for reconstruction.
                        if len(entry) >= 7:
                            try:
                                if int(entry[5]) == int(block_table.num_tokens):
                                    rotating_terminal = entry
                            except (TypeError, ValueError):
                                pass
                    elif tag == "rotating_kv_pending":
                        pass
                    elif tag == "cumulative":
                        best_cumulative = entry  # Last cumulative entry wins
                    elif tag == "deepseek_v4":
                        best_dsv4 = entry  # Last DSV4 composite state wins
                    elif tag == "deepseek_v4_delta_v1":
                        dsv4_delta_entries.append(entry)
                    elif tag == "deepseek_v4_pending":
                        pass
                    elif tag == "no_state":
                        best_cumulative = entry
                    elif tag == "zaya_cca":
                        zaya_cca_entries.append(entry)
                    elif tag == "cache_list":
                        cache_list_entries.append(entry[1])  # list of sub-slices
                    elif tag == "minimax_m3":
                        m3_slices_keys.append(entry[1])
                        m3_slices_values.append(entry[2])
                        m3_idx_slice = entry[3] if len(entry) > 3 else None
                        m3_slices_idx.append(m3_idx_slice)
                        if m3_idx_slice is not None:
                            m3_has_idx = True
                    # "skip" entries are ignored

                if zaya_cca_entries:
                    sub_kv_keys = []
                    sub_kv_vals = []
                    sub_qkv_keys = []
                    sub_qkv_vals = []
                    sub_qkv_meta = None
                    terminal_cca_state = None
                    terminal_cca_meta = ""
                    for zentry in zaya_cca_entries:
                        if len(zentry) < 2:
                            continue
                        kv_entry = zentry[1]
                        if isinstance(kv_entry, (tuple, list)) and kv_entry:
                            if kv_entry[0] == "kv":
                                sub_kv_keys.append(kv_entry[1])
                                sub_kv_vals.append(kv_entry[2])
                            elif kv_entry[0] == "quantized_kv":
                                sub_qkv_keys.append(kv_entry[1])
                                sub_qkv_vals.append(kv_entry[2])
                                if len(kv_entry) > 3 and sub_qkv_meta is None:
                                    sub_qkv_meta = kv_entry[3]
                        if len(zentry) > 2 and zentry[2] is not None:
                            terminal_cca_state = _copy_mlx_tree(zentry[2])
                            terminal_cca_meta = zentry[3] if len(zentry) > 3 else ""

                    if terminal_cca_state is None:
                        logger.warning(
                            "Cannot reconstruct ZAYA CCA layer %s: typed "
                            "prefix chain has KV pages but no terminal "
                            "conv_state/prev_hs payload",
                            layer_idx,
                        )
                        return None

                    try:
                        from mlx_lm.models.cache import CacheList as CLCache
                    except ImportError:
                        logger.warning("Cannot reconstruct ZAYA CCA CacheList: import failed")
                        return None

                    if sub_qkv_keys:
                        n_comp = len(sub_qkv_keys[0])
                        ck = tuple(
                            mx.concatenate([s[c] for s in sub_qkv_keys], axis=-2)
                            for c in range(n_comp)
                        )
                        cv = tuple(
                            mx.concatenate([s[c] for s in sub_qkv_vals], axis=-2)
                            for c in range(n_comp)
                        )
                        # Single-block chains: mx.concatenate([x]) can return
                        # the same object, aliasing block.cache_data with the
                        # live cache (see the standard-KV guard below); force
                        # fresh buffers.
                        if len(sub_qkv_keys) == 1:
                            ck = tuple(c * 1 for c in ck)
                            cv = tuple(c * 1 for c in cv)
                        mx.eval(*ck, *cv)
                        try:
                            from mlx_lm.models.cache import QuantizedKVCache as QKVCache
                            g_size, q_bits = 64, 8
                            if sub_qkv_meta and len(sub_qkv_meta) >= 3:
                                try:
                                    _, g_size, q_bits = map(int, sub_qkv_meta[:3])
                                except (ValueError, TypeError):
                                    pass
                            kv_cache = QKVCache(group_size=g_size, bits=q_bits)
                            kv_cache.keys = ck
                            kv_cache.values = cv
                            kv_cache.offset = ck[0].shape[-2]
                        except ImportError:
                            logger.warning("Cannot reconstruct ZAYA quantized KV sub-cache")
                            return None
                    elif sub_kv_keys:
                        ndim = len(sub_kv_keys[0].shape)
                        seq_axis = 1 if ndim == 3 else 2
                        ck = mx.concatenate(sub_kv_keys, axis=seq_axis)
                        cv = mx.concatenate(sub_kv_vals, axis=seq_axis)
                        # Same single-block aliasing hazard; the step-padding
                        # re-concat below only fires when offset % step != 0,
                        # so it cannot be relied on for a fresh buffer.
                        if len(sub_kv_keys) == 1:
                            ck = ck * 1
                            cv = cv * 1
                        mx.eval(ck, cv)
                        if ndim == 4:
                            allowed_kv = self._get_allowed_n_kv_heads()
                            if allowed_kv and ck.shape[1] not in allowed_kv:
                                logger.warning(
                                    f"ZAYA CCA layer {layer_idx} KV head mismatch: "
                                    f"got {ck.shape[1]}, expected one of "
                                    f"{sorted(allowed_kv)}"
                                )
                                return None
                        kv_cache = KVCache()
                        offset = ck.shape[seq_axis]
                        step = int(getattr(kv_cache, "step", 0) or 0)
                        if step > 0 and offset > 0 and offset % step:
                            padded_len = ((offset + step - 1) // step) * step
                            pad = padded_len - offset
                            if pad > 0:
                                pad_shape = list(ck.shape)
                                pad_shape[seq_axis] = pad
                                ck = mx.concatenate(
                                    [ck, mx.zeros(tuple(pad_shape), ck.dtype)],
                                    axis=seq_axis,
                                )
                                cv = mx.concatenate(
                                    [cv, mx.zeros(tuple(pad_shape), cv.dtype)],
                                    axis=seq_axis,
                                )
                                mx.eval(ck, cv)
                        kv_cache.keys = ck
                        kv_cache.values = cv
                        kv_cache.offset = offset
                    else:
                        logger.warning(
                            "Cannot reconstruct ZAYA CCA layer %s: no standard KV pages",
                            layer_idx,
                        )
                        return None

                    try:
                        from mlx_lm.models.cache import ArraysCache
                        cca_cache = ArraysCache.from_state(terminal_cca_state, None)
                    except Exception:
                        try:
                            from vmlx_engine.utils.mamba_cache import ArraysCache
                            cca_cache = ArraysCache.from_state(terminal_cca_state, None)
                        except Exception as e:
                            logger.warning(
                                f"Cannot reconstruct ZAYA CCA ArraysCache layer "
                                f"{layer_idx}: {e}"
                            )
                            return None
                    try:
                        cca_cache.meta_state = terminal_cca_meta
                    except Exception:
                        pass
                    cl = CLCache.__new__(CLCache)
                    cl.caches = (kv_cache, cca_cache)
                    reconstructed_caches.append(cl)
                    reconstructed_indices.add(layer_idx)
                    cumulative_count += 1

                elif tq_block_entries:
                    _tq_pre = tq_predecoded.get(("layer", layer_idx))
                    if _tq_pre is not None:
                        concat_keys, concat_values = _tq_pre
                    else:
                        from .tq_disk_store import decode_tq_blocks

                        concat_keys, concat_values = decode_tq_blocks(
                            tq_block_entries
                        )
                    if len(tq_block_entries) == 1:
                        concat_keys = concat_keys * 1
                        concat_values = concat_values * 1
                    deferred_tq_eval.append(concat_keys)
                    deferred_tq_eval.append(concat_values)

                    allowed_kv = self._get_allowed_n_kv_heads()
                    if allowed_kv and concat_keys.shape[1] not in allowed_kv:
                        logger.warning(
                            f"TurboQuant head count mismatch in layer {layer_idx}: "
                            f"got {concat_keys.shape[1]}, expected one of "
                            f"{sorted(allowed_kv)} — forcing cache miss"
                        )
                        return None
                    cache = KVCache()
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = concat_keys.shape[2]
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    kv_count += 1

                elif quantized_kv_slices_keys:
                    # QuantizedKVCache: concatenate each component of the tuple
                    # Each entry is a tuple of 3 arrays (data, scales, zeros)
                    num_components = len(quantized_kv_slices_keys[0])
                    concat_keys = tuple(
                        mx.concatenate([s[i] for s in quantized_kv_slices_keys], axis=-2)
                        for i in range(num_components)
                    )
                    concat_values = tuple(
                        mx.concatenate([s[i] for s in quantized_kv_slices_values], axis=-2)
                        for i in range(num_components)
                    )
                    # Materialize concatenated quantized tensors
                    mx.eval(*concat_keys, *concat_values)
                    # Force independent copies for single-block case (same aliasing issue)
                    if len(quantized_kv_slices_keys) == 1:
                        concat_keys = tuple(k * 1 for k in concat_keys)
                        concat_values = tuple(v * 1 for v in concat_values)
                        mx.eval(*concat_keys, *concat_values)

                    # Validate head count for quantized cache too.
                    # Uses the full allowed set (Gemma 4 mixed-head support).
                    first_k = concat_keys[0]
                    if len(first_k.shape) == 4:
                        allowed_kv = self._get_allowed_n_kv_heads()
                        if allowed_kv and first_k.shape[1] not in allowed_kv:
                            logger.warning(
                                f"Quantized head count mismatch in layer {layer_idx}: "
                                f"got {first_k.shape[1]}, expected one of "
                                f"{sorted(allowed_kv)} — forcing cache miss"
                            )
                            return None

                    try:
                        from mlx_lm.models.cache import QuantizedKVCache as QKVCache
                        # Parse meta_state for group_size and bits
                        g_size, q_bits = 64, 8
                        if quantized_meta and len(quantized_meta) >= 3:
                            try:
                                _, g_size, q_bits = map(int, quantized_meta[:3])
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Layer {layer_idx}: failed to parse quantized meta "
                                    f"{quantized_meta!r}, using defaults "
                                    f"g_size={g_size} bits={q_bits} — "
                                    f"dequantize may produce wrong values"
                                )
                        elif quantized_kv_slices_keys:
                            # Have quantized data but no valid metadata —
                            # this is a corruption risk (wrong dequantize params)
                            logger.warning(
                                f"Layer {layer_idx}: quantized cache block has no "
                                f"metadata (meta={quantized_meta!r}), "
                                f"using defaults g_size={g_size} bits={q_bits} — "
                                f"possible stale disk cache"
                            )
                        cache = QKVCache(group_size=g_size, bits=q_bits)
                        cache.keys = concat_keys
                        cache.values = concat_values
                        cache.offset = concat_keys[0].shape[-2]
                        reconstructed_caches.append(cache)
                        reconstructed_indices.add(layer_idx)
                        kv_count += 1
                    except ImportError:
                        logger.warning("Cannot reconstruct QuantizedKVCache: import failed")
                        return None

                elif kv_slices_keys:
                    # Standard KVCache: concatenate slices
                    # Detect dimensionality: 3D (heads, seq, dim) vs 4D (batch, heads, seq, dim)
                    ndim = len(kv_slices_keys[0].shape)
                    seq_axis = 1 if ndim == 3 else 2
                    concat_keys = mx.concatenate(kv_slices_keys, axis=seq_axis)
                    concat_values = mx.concatenate(kv_slices_values, axis=seq_axis)
                    # Materialize lazy concatenation to avoid accumulating a massive
                    # Metal command buffer that can trigger GPU timeout (SIGTERM)
                    mx.eval(concat_keys, concat_values)
                    # CRITICAL: Force independent copies of the reconstructed arrays.
                    # mx.concatenate([single_array]) can return the same object,
                    # creating aliasing between block.cache_data and the live KVCache.
                    # During generation, the model's forward pass builds a lazy graph
                    # referencing these arrays. On subsequent reuse of the same block,
                    # MLX may reuse/invalidate the underlying buffer, producing garbage.
                    # mx.contiguous() on an already-contiguous, already-evaluated array
                    # is a no-op, so we use slice-and-eval to guarantee a fresh buffer.
                    if len(kv_slices_keys) == 1:
                        concat_keys = concat_keys * 1  # Force new buffer allocation
                        concat_values = concat_values * 1
                        mx.eval(concat_keys, concat_values)

                    # Validate head count against model config.
                    # Catches stale blocks with inflated H from BatchKVCache.merge().
                    #
                    # Mixed-head support: Gemma 4 and similar architectures have
                    # DIFFERENT KV head counts per layer (sliding_attention layers
                    # use num_key_value_heads=16; full_attention layers use
                    # num_global_key_value_heads=4). Checking against a single
                    # primary count falsely forces cache miss on every
                    # global-attention layer. Use the full allowed set instead.
                    allowed_kv = self._get_allowed_n_kv_heads()
                    if allowed_kv:
                        # 4D: (batch, heads, seq, dim), 3D: (heads, seq, dim)
                        head_axis = 1 if ndim == 4 else 0
                        actual_h = concat_keys.shape[head_axis]
                        if actual_h not in allowed_kv:
                            logger.warning(
                                f"Head count mismatch in layer {layer_idx}: "
                                f"got {actual_h}, expected one of {sorted(allowed_kv)} "
                                f"— forcing cache miss"
                            )
                            return None

                    cache = KVCache()
                    _kv_offset = concat_keys.shape[seq_axis]
                    # Pad to the cache's step boundary like the ZAYA and
                    # CacheList restore branches already do: an exactly-sized
                    # restored buffer pays one whole-buffer copy per layer on
                    # the first post-restore append before KVCache's chunked
                    # growth re-establishes slack.
                    _kv_step = int(getattr(KVCache, "step", 256) or 256)
                    if _kv_step > 0 and _kv_offset > 0 and _kv_offset % _kv_step:
                        _kv_pad = (
                            (_kv_offset + _kv_step - 1) // _kv_step
                        ) * _kv_step - _kv_offset
                        _pad_shape = list(concat_keys.shape)
                        _pad_shape[seq_axis] = _kv_pad
                        concat_keys = mx.concatenate(
                            [
                                concat_keys,
                                mx.zeros(tuple(_pad_shape), concat_keys.dtype),
                            ],
                            axis=seq_axis,
                        )
                        _pad_shape_v = list(concat_values.shape)
                        _pad_shape_v[seq_axis] = _kv_pad
                        concat_values = mx.concatenate(
                            [
                                concat_values,
                                mx.zeros(tuple(_pad_shape_v), concat_values.dtype),
                            ],
                            axis=seq_axis,
                        )
                        mx.eval(concat_keys, concat_values)
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = _kv_offset
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    kv_count += 1

                elif m3_slices_keys:
                    # MiniMax-M3 MSA sparse layer: concatenate keys, values, and
                    # idx_keys across blocks (all 4D, seq axis = 2) and rebuild a
                    # MiniMaxM3SparseCache. idx_keys MUST be restored or the
                    # Lightning-indexer block selection diverges on the next step.
                    concat_keys = mx.concatenate(m3_slices_keys, axis=2)
                    concat_values = mx.concatenate(m3_slices_values, axis=2)
                    concat_idx = None
                    if m3_has_idx and all(s is not None for s in m3_slices_idx):
                        concat_idx = mx.concatenate(m3_slices_idx, axis=2)
                    if concat_idx is not None:
                        mx.eval(concat_keys, concat_values, concat_idx)
                    else:
                        mx.eval(concat_keys, concat_values)
                    # Force independent buffers for the single-block case to avoid
                    # aliasing the live cache (see the KVCache note above).
                    if len(m3_slices_keys) == 1:
                        concat_keys = concat_keys * 1
                        concat_values = concat_values * 1
                        if concat_idx is not None:
                            concat_idx = concat_idx * 1
                        if concat_idx is not None:
                            mx.eval(concat_keys, concat_values, concat_idx)
                        else:
                            mx.eval(concat_keys, concat_values)

                    # Validate the real KV head count (idx_keys is not a KV head).
                    allowed_kv = self._get_allowed_n_kv_heads()
                    if allowed_kv and concat_keys.shape[1] not in allowed_kv:
                        logger.warning(
                            f"M3 head count mismatch in layer {layer_idx}: "
                            f"got {concat_keys.shape[1]}, expected one of "
                            f"{sorted(allowed_kv)} — forcing cache miss"
                        )
                        return None

                    try:
                        from vmlx_engine.models.minimax_m3.cache import (
                            restore_minimax_m3_sparse,
                        )
                    except Exception:
                        try:
                            from mlx_lm.models.minimax_m3_vl import (  # vendored namespace
                                restore_minimax_m3_sparse,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Cannot reconstruct M3 sparse layer {layer_idx}: "
                                f"import failed ({e})"
                            )
                            return None
                    cache = restore_minimax_m3_sparse(
                        concat_keys, concat_values, concat_idx
                    )
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    kv_count += 1

                elif rotating_terminal is not None:
                    (
                        _,
                        terminal_keys,
                        terminal_values,
                        max_size,
                        keep,
                        original_offset,
                        original_idx,
                    ) = rotating_terminal[:7]
                    ndim = len(terminal_keys.shape)
                    seq_axis = 1 if ndim == 3 else 2 if ndim == 4 else -1
                    if seq_axis < 0:
                        logger.warning(
                            "Cannot reconstruct RotatingKVCache layer %s: rank=%s",
                            layer_idx,
                            ndim,
                        )
                        return None
                    target_tokens = int(getattr(block_table, "num_tokens", 0) or 0)
                    max_size_int = int(max_size or 0)
                    keep_int = int(keep or 0)
                    original_offset_int = int(original_offset)
                    original_idx_int = int(original_idx)
                    restored_len_int = int(terminal_keys.shape[seq_axis])
                    required_len = min(target_tokens, max_size_int)
                    if (
                        max_size_int <= 0
                        or keep_int < 0
                        or keep_int > max_size_int
                        or original_offset_int != target_tokens
                        or restored_len_int != required_len
                        or original_idx_int != restored_len_int
                    ):
                        logger.warning(
                            "Cannot reconstruct RotatingKVCache layer %s: invalid "
                            "terminal window physical=%s required=%s max_size=%s "
                            "keep=%s target_tokens=%s original_offset=%s idx=%s",
                            layer_idx,
                            restored_len_int,
                            required_len,
                            max_size_int,
                            keep_int,
                            target_tokens,
                            original_offset_int,
                            original_idx_int,
                        )
                        return None
                    # A reconstructed cache is owned by the live request and
                    # RotatingKVCache writes its next decode token in place.
                    # Never install the resident block payload directly: a
                    # disk-promoted hit would otherwise corrupt its new L1
                    # copy, so the next exact paged hit would observe different
                    # logits than the clean L2 restore that populated it.
                    terminal_keys = _copy_mlx_tree(terminal_keys)
                    terminal_values = _copy_mlx_tree(terminal_values)
                    mx.eval(terminal_keys, terminal_values)
                    if has_rotating:
                        cache = RotatingKVCache(max_size=max_size_int, keep=keep_int)
                    else:
                        # A runtime without RotatingKVCache cannot preserve the
                        # typed SWA contract.  Treating it as plain KV would be a
                        # silent semantic change, so reject the hit.
                        logger.warning(
                            "Cannot reconstruct RotatingKVCache layer %s: typed "
                            "runtime class is unavailable",
                            layer_idx,
                        )
                        return None
                    cache.keys = terminal_keys
                    cache.values = terminal_values
                    cache.offset = original_offset_int
                    cache._idx = original_idx_int
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    kv_count += 1

                elif rotating_entries:
                    logger.warning(
                        "Cannot reconstruct RotatingKVCache layer %s: no exact "
                        "terminal window for target_tokens=%s (candidate_offsets=%s)",
                        layer_idx,
                        int(getattr(block_table, "num_tokens", 0) or 0),
                        [entry[5] if len(entry) > 5 else None for entry in rotating_entries],
                    )
                    return None

                elif dsv4_delta_entries:
                    if len(dsv4_delta_entries) != len(all_block_data):
                        logger.warning(
                            "Cannot reconstruct DSV4 delta layer %s: records=%s "
                            "blocks=%s",
                            layer_idx,
                            len(dsv4_delta_entries),
                            len(all_block_data),
                        )
                        return None
                    try:
                        chain_metadata = []
                        for entry in dsv4_delta_entries:
                            if len(entry) < 4 or not isinstance(entry[1], dict):
                                raise ValueError("malformed DSV4 delta wrapper")
                            record = entry[1]
                            wrapper_class = str(entry[2] or "")
                            record_class = str(record.get("class_name") or "")
                            cache_meta = entry[3]
                            if not isinstance(cache_meta, dict):
                                raise ValueError("malformed DSV4 cache metadata")
                            record_ratio = int(record.get("compress_ratio", 0))
                            record_window = int(record.get("sliding_window", 0))
                            meta_ratio = int(cache_meta.get("compress_ratio", 0))
                            meta_window = int(cache_meta.get("sliding_window", 0))
                            pool_quant = cache_meta.get("pool_quant")
                            if (
                                wrapper_class != record_class
                                or record_ratio != meta_ratio
                                or record_window != meta_window
                                or not isinstance(pool_quant, bool)
                                or pool_quant
                                != (wrapper_class == "PoolQuantizedV4Cache")
                            ):
                                raise ValueError(
                                    "DSV4 record/wrapper metadata mismatch"
                                )
                            chain_metadata.append(
                                (
                                    wrapper_class,
                                    record_ratio,
                                    record_window,
                                    pool_quant,
                                    str(cache_meta.get("pool_storage_schema") or ""),
                                )
                            )
                        if len(set(chain_metadata)) != 1:
                            raise ValueError(
                                "DSV4 wrapper metadata changed inside delta chain"
                            )
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Cannot reconstruct DSV4 delta layer %s: %s",
                            layer_idx,
                            exc,
                        )
                        return None
                    class_names = {
                        str(entry[2])
                        for entry in dsv4_delta_entries
                        if len(entry) > 2
                    }
                    if len(class_names) != 1:
                        logger.warning(
                            "Cannot reconstruct DSV4 delta layer %s: cache "
                            "class changed inside chain: %s",
                            layer_idx,
                            sorted(class_names),
                        )
                        return None
                    class_name = next(iter(class_names))
                    try:
                        from jang_tools.dsv4.mlx_model import DeepseekV4Cache

                        cache_cls = DeepseekV4Cache
                        if class_name == "PoolQuantizedV4Cache":
                            from jang_tools.dsv4.pool_quant_cache import (
                                PoolQuantizedV4Cache,
                            )

                            cache_cls = PoolQuantizedV4Cache
                        elif class_name != "DeepseekV4Cache":
                            raise ValueError(
                                f"unsupported DSV4 delta cache class {class_name!r}"
                            )
                        restore_records = [
                            _copy_dsv4_delta_record(entry[1])
                            for entry in dsv4_delta_entries
                        ]
                        # Restore geometry is a property of the chain, not of
                        # this call site. Each record already carries the
                        # block_size and anchor_interval_blocks its writer
                        # used, and the record validator reads them back from
                        # there. Hardcoding them here silently pinned the
                        # writer to one interval: any other spacing still
                        # produced valid records, but restore would then
                        # validate periodic anchors against the wrong modulus
                        # and reject the whole chain. Derive instead, so the
                        # interval stays tunable against a cache budget.
                        geometry = {
                            (
                                int(record.get("block_size") or 0),
                                int(record.get("anchor_interval_blocks") or 0),
                            )
                            for record in restore_records
                            if isinstance(record, dict)
                        }
                        geometry.discard((0, 0))
                        if len(geometry) > 1:
                            raise ValueError(
                                "DSV4 delta chain mixes block geometries: "
                                f"{sorted(geometry)}"
                            )
                        from vmlx_engine.utils.dsv4_batch_generator import (
                            DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS,
                            DSV4_NATIVE_BLOCK_SIZE,
                        )

                        restore_block_size = DSV4_NATIVE_BLOCK_SIZE
                        restore_anchor_blocks = DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS
                        if geometry:
                            recorded_block, recorded_anchor = next(iter(geometry))
                            if recorded_block > 0:
                                restore_block_size = recorded_block
                            if recorded_anchor > 0:
                                restore_anchor_blocks = recorded_anchor
                        restored = cache_cls.restore_anchor_from_deltas(
                            restore_records,
                            target_tokens=int(block_table.num_tokens),
                            block_size=restore_block_size,
                            anchor_interval_blocks=restore_anchor_blocks,
                        )
                        if (
                            int(restored.checkpoint_tokens)
                            != int(block_table.num_tokens)
                            or int(restored.replayed_tokens) != 0
                        ):
                            raise ValueError(
                                "DSV4 restore did not land on the selected "
                                f"checkpoint: restored={restored.checkpoint_tokens} "
                                f"target={block_table.num_tokens} "
                                f"replayed={restored.replayed_tokens}"
                            )
                        cache = restored.cache
                    except Exception as exc:
                        logger.warning(
                            "Cannot reconstruct DSV4 delta layer %s: %s",
                            layer_idx,
                            exc,
                        )
                        return None
                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    cumulative_count += 1

                elif best_dsv4 is not None:
                    _, state, meta, class_name, cache_meta = best_dsv4
                    try:
                        from jang_tools.dsv4.mlx_model import DeepseekV4Cache

                        if not isinstance(cache_meta, dict):
                            cache_meta = {}
                        sliding_window = int(cache_meta.get("sliding_window") or 128)
                        compress_ratio = cache_meta.get("compress_ratio")
                        if compress_ratio is not None:
                            compress_ratio = int(compress_ratio)
                        cache_cls = DeepseekV4Cache
                        if class_name == "PoolQuantizedV4Cache":
                            # The class is part of DSV4's native cache
                            # contract: it controls incremental q4 storage for
                            # newly appended CSA/HCA pool rows. Reconstructing
                            # it as the base cache changes warm-path math.
                            from jang_tools.dsv4.pool_quant_cache import (
                                PoolQuantizedV4Cache,
                            )

                            cache_cls = PoolQuantizedV4Cache

                        # Both DeepseekV4Cache.state and
                        # PoolQuantizedV4Cache.storage_state retain the arrays
                        # they are given. The live one-token kickoff mutates the
                        # local rotating window, so detach the whole composite
                        # tree from its reusable resident block first.
                        state = _copy_mlx_tree(state)
                        cache = cache_cls(
                            sliding_window=sliding_window,
                            compress_ratio=compress_ratio,
                        )
                        local_quant_meta = cache_meta.get("local_quant_meta")
                        pool_storage_schema = cache_meta.get("pool_storage_schema")
                        if pool_storage_schema:
                            if class_name != "PoolQuantizedV4Cache":
                                raise ValueError(
                                    "DSV4 pool storage schema belongs to a non-pool cache"
                                )
                            if local_quant_meta:
                                raise ValueError(
                                    "DSV4 native pool storage cannot be combined with "
                                    "generic local KV quantization"
                                )
                            if (
                                not isinstance(state, (tuple, list))
                                or not state
                                or state[0] != pool_storage_schema
                            ):
                                raise ValueError(
                                    "DSV4 pool storage-state tag does not match cache metadata"
                                )
                            if not hasattr(cache, "storage_state"):
                                raise ValueError(
                                    "installed JANG runtime lacks lossless DSV4 pool restore"
                                )
                            cache.storage_state = state
                            cache.meta_state = meta
                        elif local_quant_meta:
                            # DSV4 q4/q8 storage keeps CSA/HCA pool buffers
                            # native but stores local SWA KV as QuantizedKVCache.
                            # Rebuild that composite explicitly; assigning this
                            # state through DeepseekV4Cache.state would feed a
                            # quantized tuple into RotatingKVCache.state.
                            from mlx_lm.models.cache import QuantizedKVCache

                            local_state, compressor_state, indexer_state = state
                            qlocal = QuantizedKVCache()
                            qlocal.state = local_state
                            qlocal.meta_state = tuple(local_quant_meta)
                            cache.local = qlocal  # type: ignore[attr-defined]
                            cache.compressor_state = dict(
                                zip(
                                    ("buffer_kv", "buffer_gate", "pooled"),
                                    compressor_state,
                                )
                            )
                            cache.indexer_state = dict(
                                zip(
                                    ("buffer_kv", "buffer_gate", "pooled"),
                                    indexer_state,
                                )
                            )
                            cache._vmlx_dsv4_local_meta_state = tuple(meta or ())
                            cache._vmlx_dsv4_local_quant_meta = tuple(local_quant_meta)
                            cache._vmlx_dsv4_sliding_window = sliding_window
                            try:
                                cache._vmlx_dsv4_keep = int(meta[0]) if meta else 0
                            except Exception:
                                cache._vmlx_dsv4_keep = 0
                        else:
                            cache.state = state
                            try:
                                cache.meta_state = meta
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(
                            f"Cannot reconstruct layer {layer_idx} "
                            f"({class_name}) as native DSV4 cache: {e}"
                        )
                        return None

                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    cumulative_count += 1

                elif best_cumulative is not None:
                    if best_cumulative[0] == "no_state":
                        class_name = (
                            best_cumulative[1]
                            if len(best_cumulative) > 1
                            else "ZayaNoStateCache"
                        )
                        if class_name == "ZayaNoStateCache":
                            try:
                                from vmlx_engine.models.zaya import ZayaNoStateCache

                                cache = ZayaNoStateCache()
                            except Exception as e:
                                logger.warning(
                                    f"Cannot reconstruct ZayaNoStateCache layer "
                                    f"{layer_idx}: {e}"
                                )
                                return None
                            reconstructed_caches.append(cache)
                            reconstructed_indices.add(layer_idx)
                            continue
                        logger.warning(
                            f"Cannot reconstruct no_state layer {layer_idx}: "
                            f"unknown class {class_name}"
                        )
                        return None

                    # Cumulative cache (MambaCache/ArraysCache): restore full state
                    _, state, meta, class_name = best_cumulative
                    meta, _state_dtypes = _unwrap_cumulative_meta(meta)
                    state = _cast_cumulative_state_dtypes(
                        _copy_mlx_tree(state), _state_dtypes
                    )

                    # Try class-specific restoration.
                    # ArraysCache.from_state() rejects meta_state (it has no offset),
                    # so try with state-only first, then fall back to state+meta.
                    cache = None
                    try:
                        import mlx_lm.models.cache as cache_mod
                        cache_cls = getattr(cache_mod, class_name, None)
                        if cache_cls is None and class_name == "ArraysCache":
                            try:
                                from vmlx_engine.utils.mamba_cache import ArraysCache
                                cache_cls = ArraysCache
                            except Exception:
                                pass
                        
                        if cache_cls and hasattr(cache_cls, "from_state"):
                            try:
                                cache = cache_cls.from_state(state, meta)
                            except (ValueError, TypeError):
                                # Some cache types (ArraysCache) reject meta_state.
                                # Try state-only reconstruction.
                                cache = cache_cls.from_state(state, None)
                        elif has_mamba and "Mamba" in class_name:
                            cache = MambaCache.from_state(state, meta)
                        elif has_mamba:
                            cache = MambaCache.from_state(state, meta)
                        else:
                            cache = KVCache.from_state(state, meta)
                    except Exception:
                        try:
                            if has_mamba:
                                cache = MambaCache.from_state(state, meta)
                        except Exception:
                            pass
                    if cache is None:
                        logger.warning(
                            f"Cannot reconstruct layer {layer_idx} "
                            f"({class_name}): no suitable cache class"
                        )
                        return None

                    if "Dots3LatentCache" in class_name:
                        # A restored snapshot whose offset disagrees with the
                        # fetched span claims tokens the span does not have
                        # (terminal snapshots are captured at the STORE's
                        # offset, boundary checkpoints at their block's).
                        # Rewind exactly or miss — never restore drifted
                        # window state silently (ledger row 152).
                        _target = int(getattr(block_table, "num_tokens", 0) or 0)
                        _off = int(getattr(cache, "offset", 0) or 0)
                        if _target and _off != _target:
                            _trim = getattr(cache, "trim_to_boundary", None)
                            if not (callable(_trim) and _trim(_target)):
                                logger.info(
                                    "dots3 layer %s: snapshot offset %s not "
                                    "rewindable to span boundary %s — miss",
                                    layer_idx,
                                    _off,
                                    _target,
                                )
                                return None

                    reconstructed_caches.append(cache)
                    reconstructed_indices.add(layer_idx)
                    cumulative_count += 1

                elif cache_list_entries:
                    # CacheList (MoE models): reconstruct each sub-cache
                    # then wrap in CacheList
                    try:
                        from mlx_lm.models.cache import CacheList as CLCache
                    except ImportError:
                        logger.warning("Cannot reconstruct CacheList: import failed")
                        return None

                    # Determine number of sub-caches from first entry
                    num_subs = len(cache_list_entries[0])
                    sub_caches_rebuilt = []
                    for sub_idx in range(num_subs):
                        # Collect this sub-cache's slices across all blocks
                        sub_kv_keys = []
                        sub_kv_vals = []
                        sub_qkv_keys = []  # quantized KV
                        sub_qkv_vals = []
                        sub_qkv_meta = None
                        sub_tq_entries = []
                        sub_cumulative = None
                        for block_entry in cache_list_entries:
                            if sub_idx >= len(block_entry):
                                continue
                            sub = block_entry[sub_idx]
                            if not isinstance(sub, (tuple, list)):
                                continue
                            st = sub[0]
                            if st == "kv":
                                sub_kv_keys.append(sub[1])
                                sub_kv_vals.append(sub[2])
                            elif st == "quantized_kv":
                                sub_qkv_keys.append(sub[1])
                                sub_qkv_vals.append(sub[2])
                                if len(sub) > 3 and sub_qkv_meta is None:
                                    sub_qkv_meta = sub[3]
                            elif st == "turboquant_kv":
                                sub_tq_entries.append(tuple(sub))
                                tq_native_entry_count += 1
                            elif st == "cumulative":
                                sub_cumulative = sub

                        if sub_tq_entries:
                            _tq_pre = tq_predecoded.get(
                                ("sub", layer_idx, sub_idx)
                            )
                            if _tq_pre is not None:
                                ck, cv = _tq_pre
                            else:
                                from .tq_disk_store import decode_tq_blocks

                                ck, cv = decode_tq_blocks(sub_tq_entries)
                            if len(sub_tq_entries) == 1:
                                ck = ck * 1
                                cv = cv * 1
                            deferred_tq_eval.append(ck)
                            deferred_tq_eval.append(cv)
                            allowed_kv = self._get_allowed_n_kv_heads()
                            if allowed_kv and ck.shape[1] not in allowed_kv:
                                logger.warning(
                                    f"CacheList TQ sub {sub_idx} head mismatch: "
                                    f"got {ck.shape[1]}, expected one of "
                                    f"{sorted(allowed_kv)}"
                                )
                                return None
                            sc = KVCache()
                            sc.keys = ck
                            sc.values = cv
                            sc.offset = ck.shape[2]
                            sub_caches_rebuilt.append(sc)
                        elif sub_qkv_keys:
                            # Quantized sub-cache: concatenate each component
                            n_comp = len(sub_qkv_keys[0])
                            ck = tuple(
                                mx.concatenate([s[c] for s in sub_qkv_keys], axis=-2)
                                for c in range(n_comp)
                            )
                            cv = tuple(
                                mx.concatenate([s[c] for s in sub_qkv_vals], axis=-2)
                                for c in range(n_comp)
                            )
                            # Single-block chains: mx.concatenate([x]) can
                            # return the same object, aliasing block.cache_data
                            # with the live cache; force fresh buffers.
                            if len(sub_qkv_keys) == 1:
                                ck = tuple(c * 1 for c in ck)
                                cv = tuple(c * 1 for c in cv)
                            mx.eval(*ck, *cv)
                            try:
                                from mlx_lm.models.cache import QuantizedKVCache as QKVCache
                                g_size, q_bits = 64, 8
                                if sub_qkv_meta and len(sub_qkv_meta) >= 3:
                                    try:
                                        _, g_size, q_bits = map(int, sub_qkv_meta[:3])
                                    except (ValueError, TypeError):
                                        pass
                                sc = QKVCache(group_size=g_size, bits=q_bits)
                                sc.keys = ck
                                sc.values = cv
                                sc.offset = ck[0].shape[-2]
                                sub_caches_rebuilt.append(sc)
                            except ImportError:
                                logger.warning(f"CacheList sub {sub_idx}: QuantizedKVCache import failed")
                                return None
                        elif sub_kv_keys:
                            ndim = len(sub_kv_keys[0].shape)
                            seq_axis = 1 if ndim == 3 else 2
                            ck = mx.concatenate(sub_kv_keys, axis=seq_axis)
                            cv = mx.concatenate(sub_kv_vals, axis=seq_axis)
                            # Same single-block aliasing hazard; the step
                            # padding below only fires when offset % step != 0.
                            if len(sub_kv_keys) == 1:
                                ck = ck * 1
                                cv = cv * 1
                            mx.eval(ck, cv)

                            # Validate head count in sub-cache (mixed-head aware)
                            if ndim == 4:
                                allowed_kv = self._get_allowed_n_kv_heads()
                                if allowed_kv and ck.shape[1] not in allowed_kv:
                                    logger.warning(
                                        f"CacheList sub {sub_idx} head mismatch: "
                                        f"got {ck.shape[1]}, expected one of "
                                        f"{sorted(allowed_kv)}"
                                    )
                                    return None

                            sc = KVCache()
                            offset = ck.shape[seq_axis]
                            step = int(getattr(sc, "step", 0) or 0)
                            if step > 0 and offset > 0 and offset % step:
                                padded_len = ((offset + step - 1) // step) * step
                                pad = padded_len - offset
                                if pad > 0:
                                    pad_shape = list(ck.shape)
                                    pad_shape[seq_axis] = pad
                                    k_pad = mx.zeros(tuple(pad_shape), ck.dtype)
                                    v_pad = mx.zeros(tuple(pad_shape), cv.dtype)
                                    ck = mx.concatenate([ck, k_pad], axis=seq_axis)
                                    cv = mx.concatenate([cv, v_pad], axis=seq_axis)
                                    mx.eval(ck, cv)
                            sc.keys = ck
                            sc.values = cv
                            sc.offset = offset
                            sub_caches_rebuilt.append(sc)
                        elif sub_cumulative is not None:
                            _, sstate, smeta, scls = sub_cumulative
                            smeta, _sub_dtypes = _unwrap_cumulative_meta(smeta)
                            sstate = _cast_cumulative_state_dtypes(
                                _copy_mlx_tree(sstate), _sub_dtypes
                            )
                            try:
                                import mlx_lm.models.cache as cache_mod
                                scls_obj = getattr(cache_mod, scls, None)
                                if scls_obj and hasattr(scls_obj, "from_state"):
                                    sc = scls_obj.from_state(sstate, smeta)
                                elif has_mamba:
                                    sc = MambaCache.from_state(sstate, smeta)
                                else:
                                    sc = KVCache.from_state(sstate, smeta)
                            except Exception:
                                logger.warning(f"CacheList sub {sub_idx}: reconstruction failed")
                                return None
                            sub_caches_rebuilt.append(sc)
                        else:
                            # Skip-only sub-cache — can't reconstruct
                            return None

                    cl = CLCache.__new__(CLCache)
                    cl.caches = tuple(sub_caches_rebuilt)
                    reconstructed_caches.append(cl)
                    reconstructed_indices.add(layer_idx)
                    kv_count += 1

            if not reconstructed_caches:
                return None

            if len(reconstructed_caches) != num_layers:
                # Count how many layers were missed and why
                missed_skip_only = 0
                missed_other = 0
                for li in range(num_layers):
                    if li in reconstructed_indices:
                        continue  # successfully rebuilt — not a miss
                    layer_entries = []
                    for bd in all_block_data:
                        if li < len(bd):
                            layer_entries.append(bd[li])
                    tags = [e[0] if isinstance(e, (tuple, list)) else "?" for e in layer_entries]
                    if all(t == "skip" for t in tags):
                        missed_skip_only += 1
                    else:
                        missed_other += 1

                if missed_other == 0 and missed_skip_only > 0 and reconstructed_caches:
                    # All missing layers are cumulative/SSM with only "skip" entries
                    # (non-last blocks don't store cumulative state).
                    # Return partial reconstruction — the caller's _fix_hybrid_cache
                    # will expand it by inserting fresh SSM caches at missing positions.
                    logger.info(
                        f"Partial hybrid reconstruction: {len(reconstructed_caches)}/{num_layers} layers "
                        f"({missed_skip_only} cumulative layers deferred to _fix_hybrid_cache)"
                    )
                else:
                    logger.warning(
                        f"Reconstructed {len(reconstructed_caches)} layers "
                        f"but expected {num_layers} "
                        f"(skip_only={missed_skip_only}, other={missed_other})"
                    )
                    return None

            logger.debug(
                f"Reconstructed cache: {len(reconstructed_caches)} layers "
                f"({kv_count} KV + {cumulative_count} cumulative), "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            if deferred_tq_eval:
                mx.eval(*deferred_tq_eval)

            self._last_reconstruct_disk_blocks = len(disk_backed_block_ids)
            self._last_reconstruct_tq_blocks = tq_native_entry_count
            reconstruction_succeeded = True
            if getattr(self, "_reconstruct_memo_arm", False) and memo_signature:
                self._reconstruct_memo_arm = False
                if not self._reconstruct_memo_fits_in_headroom(reconstructed_caches):
                    self._reconstruct_memo = None
                else:
                    try:
                        # Copy before the caller decodes into this state, so the
                        # retained composite is the prompt-boundary one the second
                        # pass needs.
                        self._reconstruct_memo = (
                            memo_signature,
                            copy.deepcopy(reconstructed_caches),
                        )
                    except Exception as memo_err:
                        self._reconstruct_memo = None
                        logger.info(
                            "Reconstruct memo FAILED (%s); second pass will replay",
                            memo_err,
                        )
            return reconstructed_caches

        except Exception as e:
            logger.warning(
                "Failed to reconstruct cache: %s",
                e,
                exc_info=True,
            )
            return None
        finally:
            _disk_only = bool(getattr(self.paged_cache, "disk_only", False))
            _paged_frugal = bool(
                getattr(self.paged_cache, "paged_frugal", _disk_only)
            )
            for block_id in disk_backed_block_ids | l2_readable_block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block is not None:
                    keep_native = (
                        bool(getattr(block, "keep_resident", False))
                        or block_id in native_resident_block_ids
                    )
                    if (
                        reconstruction_succeeded
                        and not _disk_only
                        and not bool(
                            getattr(block, "cache_data_transient", False)
                        )
                        and (not _paged_frugal or keep_native)
                    ):
                        # A successful reconstruction in Paged On mode promotes
                        # L2 data into a real RAM tier.  Keep the payload for the
                        # next same-process hit, but make it eligible for normal
                        # byte-budget/LRU eviction. Native path-dependent state
                        # (DSV4/ZAYA/rotating-SWA) retains the same guarantee even
                        # under an explicit frugal override.
                        self.paged_cache.make_resident_payload_evictable(block)
                    else:
                        # Disk-only and explicit generic frugal mode use the
                        # payload only as a transient reconstruction buffer.
                        self.paged_cache.release_resident_payload(block)

    def _find_best_prefix_match(
        self,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Find and pin the best matching prefix-index entry atomically."""
        with self.paged_cache._lock:
            best_match = None
            best_len = 0
            # Try progressively longer prefixes.
            if self._chained_prefix_index_hash:
                candidates = list(
                    self._prefix_index_hash_sequence(
                        tokens,
                        len(tokens) // self.block_size,
                        cache_extra_keys=cache_extra_keys,
                    )
                )
            else:
                candidates = [
                    (
                        n * self.block_size,
                        self._prefix_index_hash(
                            tokens[: n * self.block_size],
                            cache_extra_keys=cache_extra_keys,
                        ),
                    )
                    for n in range(1, len(tokens) // self.block_size + 1)
                    if n * self.block_size <= len(tokens)
                ]
            for prefix_len, prefix_hash in candidates:
                prefix_tokens = tokens[:prefix_len]

                if prefix_hash in self._prefix_index:
                    entry = self._prefix_index[prefix_hash]
                    cached_tokens, block_ids = entry[:2]
                    cached_extra = entry[2] if len(entry) > 2 else None
                    if cached_extra != self._prefix_index_extra_marker(
                        cache_extra_keys, prefix_len
                    ):
                        continue
                    if cached_tokens == prefix_tokens and len(cached_tokens) > best_len:
                        valid = self._prefix_index_blocks_are_current(
                            cached_tokens,
                            block_ids,
                            cache_extra_keys=cache_extra_keys,
                        )
                        if valid:
                            best_match = (cached_tokens, block_ids)
                            best_len = len(cached_tokens)
                        elif self._prefix_index.get(prefix_hash) is entry:
                            # Delete only the entry which was validated. A
                            # concurrent store may have replaced this key with
                            # a fresh chain while the old candidate was stale.
                            del self._prefix_index[prefix_hash]

            # _update_prefix_index() also records the terminal partial prefix for a
            # cached request. A later request can have that exact partial prefix
            # plus a long tail, so the block-aligned loop above will never probe
            # its hash. Scan indexed entries by exact token-prefix equality and
            # blocks that still own the exact expected chain hashes; this preserves
            # chain-hash safety and avoids the legacy content-only block hash path.
            for prefix_hash, entry in list(self._prefix_index.items()):
                cached_tokens, block_ids = entry[:2]
                cached_extra = entry[2] if len(entry) > 2 else None
                cached_len = len(cached_tokens)
                if cached_extra != self._prefix_index_extra_marker(
                    cache_extra_keys, cached_len
                ):
                    continue
                if cached_len <= best_len or cached_len > len(tokens):
                    continue
                if tokens[:cached_len] != cached_tokens:
                    continue

                valid = self._prefix_index_blocks_are_current(
                    cached_tokens,
                    block_ids,
                    cache_extra_keys=cache_extra_keys,
                )
                if valid:
                    best_match = (cached_tokens, block_ids)
                    best_len = cached_len
                elif self._prefix_index.get(prefix_hash) is entry:
                    del self._prefix_index[prefix_hash]

            if best_match is None:
                return None

            # Pin all validated blocks before releasing the lock. Allocation,
            # eviction, hash replacement, and ref-count changes use this same
            # RLock, so the caller receives ownership of the exact chain it
            # validated rather than recycled numeric IDs.
            pinned_ids: List[int] = []
            for block_id in best_match[1]:
                if not self.paged_cache.increment_ref(block_id):
                    self.paged_cache.release_request_refs(
                        BlockTable(
                            request_id="prefix-index-pin-rollback",
                            block_ids=pinned_ids,
                            num_tokens=0,
                        )
                    )
                    return None
                pinned_ids.append(block_id)
            return best_match

    def _prefix_index_blocks_are_current(
        self,
        cached_tokens: List[int],
        block_ids: List[int],
        *,
        cache_extra_keys: Optional[Any] = None,
    ) -> bool:
        """Validate that an index entry still owns its exact paged block chain.

        Numeric block IDs are recycled after paged-cache eviction.  Existence in
        ``allocated_blocks`` therefore does not prove that an ID still contains
        the prefix which originally created the index entry.  Recompute the
        chain hashes and exact per-block token counts before allowing the
        fallback prefix index to bypass the authoritative paged hash lookup.
        """
        if not cached_tokens or not block_ids:
            return False

        expected_block_count = (
            len(cached_tokens) + self.block_size - 1
        ) // self.block_size
        if len(block_ids) != expected_block_count:
            return False

        parent_hash = None
        for ordinal, block_id in enumerate(block_ids):
            start = ordinal * self.block_size
            block_tokens = cached_tokens[start : start + self.block_size]
            if not block_tokens:
                return False

            expected_hash = compute_block_hash(
                parent_hash,
                block_tokens,
                extra_keys=cache_extra_keys_for_token_range(
                    cache_extra_keys, start, start + len(block_tokens)
                ),
            )
            block = self.paged_cache.allocated_blocks.get(block_id)
            if (
                block is None
                or block.block_hash != expected_hash
                or int(getattr(block, "token_count", 0) or 0)
                != len(block_tokens)
            ):
                return False
            parent_hash = expected_hash

        return True

    @staticmethod
    def _prefix_index_extra_marker(
        cache_extra_keys: Optional[Any],
        prefix_len: Optional[int] = None,
    ) -> Optional[str]:
        scoped = (
            cache_extra_keys_for_token_range(
                cache_extra_keys, 0, max(0, int(prefix_len))
            )
            if prefix_len is not None
            else cache_extra_keys
        )
        return canonical_cache_extra_marker(scoped)

    def _prefix_index_hash(
        self,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> str:
        scoped_extra_keys = cache_extra_keys_for_token_range(
            cache_extra_keys, 0, len(tokens)
        )
        if scoped_extra_keys is None:
            return self.paged_cache.compute_block_hash(tokens)
        return compute_block_hash(
            None,
            tokens,
            extra_keys=scoped_extra_keys,
        ).hex()

    def _prefix_index_key(
        self,
        tokens: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> str:
        """The `_prefix_index` key for `tokens` under the ACTIVE scheme.

        The derivation is now conditional on VMLX_CHAINED_PREFIX_INDEX_HASH, so
        anything that needs a key — production code, a probe, a test — must ask
        here rather than call `_prefix_index_hash` directly. Calling the
        from-scratch hash while the chained scheme is active silently produces a
        key nothing stored, which reads as a cache miss rather than an error.
        """
        if not self._chained_prefix_index_hash:
            return self._prefix_index_hash(
                tokens, cache_extra_keys=cache_extra_keys
            )
        key = None
        for _prefix_len, candidate in self._prefix_index_hash_sequence(
            tokens,
            max(1, -(-len(tokens) // self.block_size)),
            cache_extra_keys=cache_extra_keys,
        ):
            key = candidate
        return key if key is not None else self._prefix_index_hash(
            tokens, cache_extra_keys=cache_extra_keys
        )

    def _prefix_index_hash_sequence(
        self,
        tokens: List[int],
        num_blocks: int,
        cache_extra_keys: Optional[Any] = None,
    ):
        """Yield ``(prefix_len, hash)`` for each block-aligned prefix.

        The lookup and the update both walk every block-aligned prefix of the
        prompt. Hashing each prefix FROM SCRATCH makes that quadratic in the
        prompt: block i re-hashes i*block_size tokens, so the total is
        O(len(tokens)^2 / block_size). At 61k tokens with 64-token blocks that
        is ~29 million token-hashes per call — and it runs under
        ``paged_cache._lock``, so every other cache operation waits on it.

        ``compute_block_hash`` already takes a parent hash and is a chain, so
        the same walk can be done incrementally: hash only the new block and
        fold in the previous prefix's hash. That is O(len(tokens)) total.

        The chained values DIFFER from the from-scratch ones. That is safe
        because ``_prefix_index`` is a plain in-memory dict rebuilt per process
        (declared at __init__, never persisted, never compared against an L2
        record) and both the reader and the writer take their keys from here —
        but it is exactly why this is env-gated and default OFF until it has a
        live A/B on a long conversation.
        """
        parent = None
        total = len(tokens)
        for index in range(1, num_blocks + 1):
            start = (index - 1) * self.block_size
            end = min(index * self.block_size, total)
            if start >= end:
                break
            parent = compute_block_hash(
                parent,
                tokens[start:end],
                extra_keys=cache_extra_keys_for_token_range(
                    cache_extra_keys, start, end
                ),
            )
            yield end, parent.hex()

    def _update_prefix_index(
        self,
        tokens: List[int],
        block_ids: List[int],
        cache_extra_keys: Optional[Any] = None,
    ) -> None:
        """Update prefix index with new token sequence."""
        with self.paged_cache._lock:
            # Index block-aligned prefixes.
            if self._chained_prefix_index_hash:
                for i, (prefix_len, prefix_hash) in enumerate(
                    self._prefix_index_hash_sequence(
                        tokens, len(block_ids), cache_extra_keys=cache_extra_keys
                    ),
                    start=1,
                ):
                    self._prefix_index[prefix_hash] = (
                        tokens[:prefix_len],
                        block_ids[:i],
                        self._prefix_index_extra_marker(
                            cache_extra_keys, prefix_len
                        ),
                    )
                return
            for i in range(1, len(block_ids) + 1):
                prefix_len = min(i * self.block_size, len(tokens))
                prefix_tokens = tokens[:prefix_len]
                prefix_hash = self._prefix_index_hash(
                    prefix_tokens,
                    cache_extra_keys=cache_extra_keys,
                )
                self._prefix_index[prefix_hash] = (
                    prefix_tokens,
                    block_ids[:i],
                    self._prefix_index_extra_marker(
                        cache_extra_keys, prefix_len
                    ),
                )

    def _mtp_prefix_snapshot_key(
        self,
        tokens: List[int],
        boundary_tokens: int,
        *,
        extra_keys: Optional[Any] = None,
        extra_key_token_start: Optional[int] = None,
        extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None,
    ) -> Optional[Any]:
        """Return the ordinary block/partial-index key for an MTP sidecar.

        Complex legacy range arguments are intentionally rejected.  Current
        vMLX callers pass the canonical scoped ``cache_extra_keys`` object as
        ``extra_keys``; the same per-block resolver as the backbone store then
        makes media/text identity exact.
        """
        boundary_tokens = int(boundary_tokens)
        if (
            boundary_tokens <= 0
            or boundary_tokens > len(tokens)
            or extra_key_token_start is not None
            or extra_key_ranges is not None
        ):
            return None
        if boundary_tokens % self.block_size != 0:
            # The engine's exact-repeat fast path indexes the predecessor's
            # N-1 terminal partial block without flooring it.  Reuse that same
            # in-memory key; the restore path below still verifies every live
            # block id/hash before trusting the sidecar.
            return (
                "partial_index",
                (
                    self._prefix_index_key(
                        tokens[:boundary_tokens],
                        cache_extra_keys=extra_keys,
                    )
                    if hasattr(self, "_chained_prefix_index_hash")
                    else self._prefix_index_hash(
                        tokens[:boundary_tokens],
                        cache_extra_keys=extra_keys,
                    )
                ),
            )
        parent_hash = None
        for start in range(0, boundary_tokens, self.block_size):
            end = start + self.block_size
            parent_hash = compute_block_hash(
                parent_hash,
                tokens[start:end],
                extra_keys=cache_extra_keys_for_token_range(
                    extra_keys, start, end
                ),
            )
        return ("block_hash", parent_hash)

    def store_mtp_prefix_snapshot(
        self,
        tokens: List[int],
        boundary_tokens: int,
        snapshot: Any,
        *,
        extra_keys: Optional[Any] = None,
        extra_key_token_start: Optional[int] = None,
        extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None,
    ) -> bool:
        """Store one opaque, memory-only MTP history at a full block tip."""
        # Only the predecessor's exact N-1 boundary may publish a partial
        # sidecar.  Arbitrary non-block boundaries are not cache contracts.
        if (
            int(boundary_tokens) % self.block_size != 0
            and int(boundary_tokens) != len(tokens) - 1
        ):
            return False
        tip = self._mtp_prefix_snapshot_key(
            tokens,
            boundary_tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
        if tip is None or snapshot is None:
            return False
        lock = getattr(self, "_mtp_prefix_snapshot_lock", None)
        snapshots = getattr(self, "_mtp_prefix_snapshots", None)
        if lock is None or snapshots is None:
            return False
        with lock:
            snapshots[tip] = (int(boundary_tokens), snapshot)
            snapshots.move_to_end(tip)
            while len(snapshots) > _MTP_PREFIX_SNAPSHOT_MAX_ENTRIES:
                snapshots.popitem(last=False)
        return True

    def restore_mtp_prefix_snapshot(
        self,
        tokens: List[int],
        boundary_tokens: int,
        *,
        extra_keys: Optional[Any] = None,
        extra_key_token_start: Optional[int] = None,
        extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None,
    ) -> Any:
        """Restore an MTP sidecar only while its exact backbone tip is live."""
        tip = self._mtp_prefix_snapshot_key(
            tokens,
            boundary_tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
        if tip is None:
            return None
        key_kind, key_value = tip
        if key_kind == "block_hash":
            block_map = getattr(
                self.paged_cache, "cached_block_hash_to_block", None
            )
            if block_map is None or block_map.get_block(key_value) is None:
                return None
        else:
            prefix_entry = getattr(self, "_prefix_index", {}).get(key_value)
            if prefix_entry is None or len(prefix_entry) < 2:
                return None
            indexed_tokens, block_ids = prefix_entry[:2]
            expected = list(tokens[: int(boundary_tokens)])
            if list(indexed_tokens) != expected:
                return None
            if not self._prefix_index_blocks_are_current(
                expected,
                list(block_ids),
                cache_extra_keys=extra_keys,
            ):
                return None
        lock = getattr(self, "_mtp_prefix_snapshot_lock", None)
        snapshots = getattr(self, "_mtp_prefix_snapshots", None)
        if lock is None or snapshots is None:
            return None
        with lock:
            entry = snapshots.get(tip)
            if entry is None or int(entry[0]) != int(boundary_tokens):
                return None
            snapshots.move_to_end(tip)
            return entry[1]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        paged_stats = self.paged_cache.get_memory_usage()
        disk_store = getattr(self.paged_cache, "_disk_store", None)
        if disk_store is not None and hasattr(disk_store, "get_stats"):
            try:
                disk_stats = disk_store.get_stats()
                # PagedCacheManager's counters only cover blocks promoted into
                # a new block-table slot. In SSD-only mode the hash/index block
                # can remain present with cache_data=None, so reconstruction
                # reads the payload from BlockDiskStore without counting a
                # promotion. Preserve both meanings and make the public
                # disk_hits/disk_misses fields reflect actual SSD reads.
                paged_stats["disk_promotion_hits"] = int(
                    paged_stats.get("disk_hits", 0) or 0
                )
                paged_stats["disk_lookup_misses"] = int(
                    paged_stats.get("disk_misses", 0) or 0
                )
                paged_stats["disk_hits"] = int(
                    disk_stats.get("disk_hits", 0) or 0
                )
                paged_stats["disk_misses"] = int(
                    disk_stats.get("disk_misses", 0) or 0
                )
                paged_stats["disk_counter_source"] = "block_disk_store"
            except Exception:
                pass
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
            "tokens_saved": self._tokens_saved,
            "active_requests": len(self._request_tables),
            "strict_block_disk_write_fence": bool(
                getattr(self, "_strict_block_disk_write_fence", False)
            ),
            "reconstruct_memo_allowed": bool(
                not getattr(self.paged_cache, "disk_only", False)
                and os.environ.get(
                    "VMLX_DSV4_RECONSTRUCT_MEMO", "1"
                ).strip().lower()
                not in {"0", "off", "false", "no"}
            ),
            "reconstruct_memo_resident": bool(
                getattr(self, "_reconstruct_memo", None) is not None
            ),
            "entries_by_type": {
                t: len(self._entries_by_type[t]) for t in _CACHE_TYPE_PRIORITY
            },
            # Server-exact fetch_cache() match provenance (task #23): which
            # branch produced the returned num_cached, not a client-side
            # inference from cached_tokens/latency. Flows into /health and
            # /v1/cache/stats through scheduler.get_cache_stats() ->
            # _cache_telemetry_snapshot(), the same ungated path that already
            # surfaces last_cache_selection/last_cache_execution -- no new
            # debug flag needed since that surface already exposes internal
            # cache-routing detail unconditionally.
            # getattr, not direct reads: several regression fixtures build
            # this cache via __new__ without __init__ (legacy-construction
            # tests for accounting rollback), and get_stats() must keep
            # working there exactly as it did before telemetry existed.
            "last_fetch_telemetry": getattr(self, "_last_fetch_telemetry", None),
            "last_fetch_telemetry_excluding_continuations": next(
                (
                    r
                    for r in reversed(
                        list(
                            (getattr(self, "_fetch_telemetry", None) or {}).values()
                        )
                    )
                    if not r.get("is_internal_continuation")
                ),
                None,
            ),
            "recent_fetch_telemetry": list(
                (getattr(self, "_fetch_telemetry", None) or {}).values()
            )[-20:],
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._hit_credits.clear()
        self._fetch_telemetry = OrderedDict()
        self._last_fetch_telemetry = None
        self.paged_cache.reset_stats()

    def retire_missing_disk_hints(self) -> int:
        """Retire stale lookup metadata after an explicit SSD pool clear.

        Run only on the idle clear path, never per token or ordinary fetch.
        Preserve resident payloads, referenced blocks and pending writes. A
        missing longer entry must not mask a shorter newly durable prefix.
        """
        manager = self.paged_cache
        store = getattr(manager, "_disk_store", None)
        if store is None:
            return 0

        def eligible(block: Any) -> bool:
            return (
                not block.is_null
                and block.block_hash is not None
                and block.cache_data is None
                and block.ref_count == 0
                and not block.durability_write_pending
                and not block.keep_resident
            )

        with manager._lock:
            candidates = [
                (block.block_hash, block)
                for block in manager.allocated_blocks.values()
                if eligible(block)
            ]
        # Avoid filesystem/SQLite work under the paged lock, and check a shared
        # hash only once even when several numeric blocks represent it.
        readable = {
            block_hash: store.has_block_record(block_hash)
            for block_hash in dict.fromkeys(key for key, _ in candidates)
        }
        retired: set[int] = set()
        with manager._lock:
            for block_hash, block in candidates:
                if (
                    not readable[block_hash]
                    and manager.allocated_blocks.get(block.block_id) is block
                    and block.block_hash == block_hash
                    and eligible(block)
                ):
                    retired.update(manager.discard_nondurable_cache_blocks(
                        {block_hash: block}
                    ))
            if retired:
                for key, entry in list(self._prefix_index.items()):
                    if any(block_id in retired for block_id in entry[1]):
                        del self._prefix_index[key]
        if retired:
            logger.info("SSD clear retired %d missing disk-only lookup hints; resident KV preserved", len(retired))
        return len(retired)

    def clear(self, force: bool = False) -> bool:
        """Clear all cached data.

        Returns False without touching anything when the paged pool is still
        serving live requests. The busy check runs FIRST: the index wipe below is
        not reversible, so clearing it and then having the pool refuse would leave
        the index and the block pool describing different worlds.
        """
        if not force and self.paged_cache.blocks_in_use():
            logger.warning(
                "Cannot clear block-aware prefix cache: paged blocks are in use"
            )
            return False
        self._request_tables.clear()
        self._reconstruct_memo_arm = False
        self._reconstruct_memo = None
        with self.paged_cache._lock:
            self._prefix_index.clear()
        lock = getattr(self, "_mtp_prefix_snapshot_lock", None)
        snapshots = getattr(self, "_mtp_prefix_snapshots", None)
        if lock is not None and snapshots is not None:
            with lock:
                snapshots.clear()
        for d in self._entries_by_type.values():
            d.clear()
        self.paged_cache.clear(force=True)
        self._n_kv_heads = None  # Reset cached head count (may change on model switch)
        self.reset_stats()
        return True

    def __len__(self) -> int:
        """Return number of active request entries."""
        return len(self._request_tables)
