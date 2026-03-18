# Continuous Batching Audit & Fix Report

**Date:** 2026-03-18
**Scope:** All model types — dense, MoE, hybrid SSM, JANG, VLM/MLLM
**Files modified:** `vmlx_engine/mllm_batch_generator.py` (synced to bundled Python)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Root Cause: MLLM Path Was Serial](#root-cause-mllm-path-was-serial)
3. [What Was Already Working](#what-was-already-working)
4. [Remaining Gaps (Ordered by Severity)](#remaining-gaps)
5. [KV Cache Quantization Status](#kv-cache-quantization-status)
6. [Cold/Hot Cache & Warmup](#coldhot-cache--warmup)
7. [How to Fix Remaining Gaps](#how-to-fix-remaining-gaps)
8. [All Changes Made in This Session](#all-changes-made-in-this-session)

---

## Architecture Overview

The batching stack has **two parallel paths** selected at startup:

```
LLM path (--continuous-batching, no --is-mllm):
  BatchedEngine._start_llm()
  → AsyncEngineCore → EngineCore → Scheduler → mlx-lm BatchGenerator
  → model.__call__(tokens, cache=BatchKVCache)

MLLM path (--continuous-batching + --is-mllm or auto-detected VLM):
  BatchedEngine._start_mllm()
  → MLLMScheduler → MLLMBatchGenerator
  → model(tokens, pixel_values=..., cache=per_request_cache)  [prefill]
  → language_model(tokens, cache=BatchKVCache)                 [decode]
```

**Engine selection:** Determined by `--continuous-batching` flag (CLI) and `is_mllm_model()`.
- JANG models: forced to LLM text path (`is_jang_model() → return False` in `is_mllm_model()`)
- VLM models: MLLM path (config.json has `vision_config`)
- Everything else: LLM path

**Key classes and their roles:**

| Class | Path | Role |
|-------|------|------|
| `BatchGenerator` (mlx-lm) | LLM | Manages active batch, prefill, decode loop |
| `Batch` (mlx-lm) | LLM | Active batch state with `extend()`, `filter()` |
| `Scheduler` | LLM | Request queue, cache management, wraps BatchGenerator |
| `MLLMBatchGenerator` | MLLM | Vision encoding, prefill, decode for VLMs |
| `MLLMBatch` | MLLM | Active batch state — **had NO extend() before fix** |
| `MLLMScheduler` | MLLM | Request queue, cache management, wraps MLLMBatchGenerator |
| `BatchMambaCache` | Both | Batch-aware SSM cache (extract, merge, filter) |
| `_BatchOffsetSafeCache` | MLLM | Proxy for Qwen3.5 offset int/mx.array safety |
| `HybridSSMStateCache` | MLLM | Companion LRU cache for SSM states at prompt boundary |

---

## Root Cause: MLLM Path Was Serial

### The Bug

In `mllm_batch_generator.py`, `_next()` had this logic:

```python
# Line 1750 (BEFORE fix)
if num_active == 0:
    requests = self.unprocessed_requests[: self.completion_batch_size]
    ...
    new_batch = self._process_prompts(requests)
    self.active_batch = new_batch
```

**New requests could ONLY start when the active batch was completely empty.**

This means: if Request A is generating, Request B must wait until Request A finishes ALL its tokens. Then Request B starts. This is **serial processing, not continuous batching.**

### Why It Existed

The comment explained: "MLLM vision encoding produces per-request KV caches that cannot be safely extended into an active batch's cache (shape mismatch in attention layers)."

This was overly conservative. While vision encoding does produce per-request caches with different shapes, the caches CAN be merged using `BatchKVCache.merge()` and `BatchMambaCache.merge()` — the same mechanism already used for multi-request initial batches. The missing piece was:
1. `MLLMBatch` had no `extend()` method
2. Single-request batches kept raw KVCache (for Qwen3.5 offset compat) with no conversion path
3. `_process_prompts()` didn't support producing batch-aware caches for single requests

### Comparison with LLM Path

mlx-lm's `BatchGenerator._next()` does NOT have this restriction:

```python
# mlx-lm generate.py (LLM path)
num_to_add = self.completion_batch_size - num_active
while num_to_add >= self.prefill_batch_size:
    ...
    self.active_batch.extend(batch)  # CAN extend into active batch!
```

---

## What Was Already Working

| Path | Model Type | Cont. Batching? | Why |
|------|-----------|-----------------|-----|
| LLM | Dense (Llama, Qwen, Mistral) | YES | mlx-lm `Batch.extend()` |
| LLM | MoE (DeepSeek V3, Qwen MoE) | YES | MoE layers are stateless — `gather_qmm` is batch-compatible, no cache needed |
| LLM | JANG (text path) | YES | Same `Scheduler → BatchGenerator` as any LLM |
| LLM | Hybrid SSM (Nemotron-H) | YES | `BatchMambaCache` + `ensure_mamba_support()` patches |
| MLLM | Any (before fix) | **NO — SERIAL** | `if num_active == 0:` gate blocked concurrent decode |

### MoE is NOT a batching problem

MoE layers (`QuantizedSwitchLinear`, `SwitchMLP`, `SwitchGLU`) are **completely stateless**. They have:
- No cache entries (Nemotron-H: `"E"` blocks → no cache; Qwen3.5: MoE is in `SparseMoeBlock` sublayer)
- No per-request routing state — `gather_qmm` selects expert rows from a shared weight tensor
- No sequence-length dependency — expert routing is per-token

The batching challenge is exclusively about **cache management** (KVCache for attention, ArraysCache for SSM/linear attention). MoE expert routing works identically for batch_size=1 and batch_size=N.

### Hybrid SSM support was already comprehensive

`vmlx_engine/utils/mamba_cache.py`:
- `BatchMambaCache`: batch-aware wrapper with `extract()`, `merge()`, `filter()`
- `patch_mlx_lm_for_mamba()`: monkey-patches mlx-lm's `_make_cache` and `_merge_caches`
- `ensure_mamba_support()`: idempotent entry point, called from Scheduler init

`vmlx_engine/mllm_batch_generator.py`:
- `HybridSSMStateCache`: LRU companion cache for SSM states at prompt boundary
- `_fix_hybrid_cache()`: expands KV-only prefix cache to full layer count

---

## Remaining Gaps

### Gap 1: Prefill Stalls Decode for All Active Requests (HIGH)

**Location:** `mllm_batch_generator.py:_next()` line ~1766

When a new request joins an active batch:
1. `mx.synchronize()` is called — full GPU barrier
2. `_process_prompts()` runs the ENTIRE prefill for the new request(s)
3. During this time, ALL existing requests generate ZERO tokens
4. Only after prefill completes does merged decode resume

For a 4096-token prompt, this can be multiple seconds of stall for all active requests.

**Impact:** Users chatting concurrently will see periodic "freezes" whenever a new request joins.

### Gap 2: Serial Per-Request Vision Encoding (HIGH)

**Location:** `mllm_batch_generator.py:_process_prompts()` lines 1468-1572

Multiple image requests are prefilled in a `for` loop:
```python
for i, req in enumerate(requests):
    logits = self._run_vision_encoding(req, cache=req_cache)
```

There is no batched VLM forward pass across multiple pixel_values. Each request's full VLM encoding runs sequentially.

**Impact:** Two simultaneous image requests take 2x the prefill time of one.

### Gap 3: No Chunked Prefill for MLLM (MEDIUM)

**Location:** `mllm_batch_generator.py:_process_prompts()`

`prefill_step_size` is configured in `MLLMBatchGenerator.__init__()` but never used. The VLM forward pass in `_run_vision_encoding()` processes the full token sequence in one shot.

Compare with the LLM path where mlx-lm's `BatchGenerator._process_prompts()` splits into `prefill_step_size` chunks with GPU evaluation + `mx.clear_cache()` between chunks.

**Impact:** Large prompts consume peak GPU memory all at once instead of streaming.

### Gap 4: No Prefix Cache for Image Requests (MEDIUM, BY DESIGN)

**Location:** `mllm_batch_generator.py:_process_prompts()` line 1204

```python
has_images = req.pixel_values is not None
if ... and not has_images:  # Skip cache for image requests
```

Image placeholder tokens are content-independent — same token IDs for same-sized images regardless of actual pixel content. A cache hit would serve KV states from a completely different image's vision encoding.

**This is correct behavior.** Not a bug, but means every image request does full prefill.

### Gap 5: Hard Count Ceiling With No Memory Feedback (LOW)

**Location:** `mllm_scheduler.py:MLLMSchedulerConfig` line 179

`max_num_seqs` is a hard limit (default 16). No dynamic admission based on Metal memory pressure (`mx.metal.device_info()`).

### Gap 6: One-Step Lag for Slot Refill (LOW)

**Location:** `mllm_scheduler.py:step()` ordering

Sequence: `_schedule_waiting()` → `batch_generator.next()` → `_cleanup_finished()`

A request finishing in step N frees its slot in cleanup. A waiting request fills it in step N+1's schedule phase. Standard continuous batching pattern — not a bug.

---

## KV Cache Quantization Status

KV cache quantization is applied **exclusively at the storage boundary**, never during active generation.

### Flow

```
Active generation:  Full-precision KVCache / BatchKVCache
                           | (request finishes)
Store to prefix cache:  _quantize_cache_for_storage() -> QuantizedKVCache
                           | (new request, cache hit)
Fetch from prefix cache: _dequantize_cache() -> KVCache
                           | (merge into batch)
Active generation:  Full-precision BatchKVCache
```

### Key locations

- `mllm_scheduler.py:_quantize_cache_for_storage()` (lines 742-775)
- `mllm_batch_generator.py:_dequantize_cache()` (lines 111-151)
- `scheduler.py:_wrap_make_cache_quantized()` (lines 436-592)

### Interaction with new extend() path

Works correctly. The `_process_prompts()` dequantize step runs before cache merge, so all caches entering `_merge_caches()` or `extend()` are full-precision. The new `_ensure_batch_cache()` helper operates on already-dequantized caches.

---

## Cold/Hot Cache & Warmup

### How It Works

- **Cold:** First request — no prefix cache hit. Full VLM/LLM prefill. After completion, KV cache stored to L1 (in-memory) and optionally L2 (disk).
- **Hot:** Subsequent matching request — prefix cache hit. Skips prefill for cached prefix, only processes remaining tokens.

### Cache Tiers (priority order)

| Tier | Type | Scope | Persistence |
|------|------|-------|-------------|
| L1 Paged | `BlockAwarePrefixCache` + `PagedCacheManager` | vLLM-style blocks, LRU | In-memory, lost on restart |
| L1 Memory-aware | `MemoryAwarePrefixCache` | LRU with auto-sizing to Metal RAM % | In-memory, lost on restart |
| L1 Legacy | `PrefixCacheManager` | Simple trie-based LRU | In-memory, lost on restart |
| L2 Disk | `DiskCacheManager` | Hash-based exact match | On-disk, survives restart |
| L2 Block Disk | `BlockDiskStore` | Block-level persistence for paged cache | On-disk, survives restart |
| SSM Companion | `HybridSSMStateCache` | LRU (max 50) for SSM states | In-memory, lost on restart |
| Vision Embed | `VisionEmbeddingCache` | LRU for pixel preprocessing | In-memory, lost on restart |

### No Pre-Warming API

There is no mechanism to populate the cache without generating tokens. Workaround: send `max_tokens=1` request and discard output.

### Hybrid Model Cache Hits

For hybrid models (Qwen3.5-VL with SSM + attention layers):
- Prefix cache stores ONLY KVCache (attention) layers — SSM state is cumulative
- `HybridSSMStateCache` stores SSM states at prompt boundary
- On cache HIT: if SSM companion also hits -> full prefix skip; if no SSM companion -> forced full prefill
- `_fix_hybrid_cache()` expands KV-only cache to full layer count by inserting fresh ArraysCache at SSM positions

---

## How to Fix Remaining Gaps

### Fix for Gap 1: Chunked Prefill Interleaved with Decode

**Goal:** Instead of stalling decode for the entire prefill duration, process the new prompt in chunks, yielding back to the decode loop between chunks.

**Approach:**

In `_next()`, when `num_active > 0` and new requests arrive:

1. Instead of calling `_process_prompts()` which does full prefill, split the work:
   - Run vision encoding only (produces pixel_values, tokenized input) — this must be done upfront
   - Then process the language model portion in `prefill_step_size` chunks
2. Between each prefill chunk, run one decode step for the active batch
3. After all chunks are done, merge the new request's cache and extend the batch

**Complexity:** HIGH — requires splitting `_process_prompts()` into a multi-step coroutine or state machine. The vision encoding MUST run as one shot (ViT doesn't support chunked inference), but the language model prefill CAN be chunked.

**Files to modify:**
- `mllm_batch_generator.py`: `_process_prompts()`, `_next()`, possibly new `_chunked_prefill()` method

### Fix for Gap 2: Batched Vision Encoding

**Goal:** Process multiple image requests' vision encoding in parallel.

**Approach:**

1. Batch `prepare_inputs()` across requests with compatible image sizes
2. Run the vision encoder (ViT) on a batch of images simultaneously
3. Run the language model prefill for each request separately (different prompt lengths)

**Complexity:** VERY HIGH — mlx-vlm's `prepare_inputs()` and VLM forward pass are not designed for batched pixel_values. Would require upstream changes or significant wrapping.

**Recommendation:** Defer. Serial vision encoding is standard practice (vLLM also does this).

### Fix for Gap 3: Chunked Prefill for MLLM

**Goal:** Process long prompts in `prefill_step_size` chunks with memory cleanup between.

**Approach:**

After vision encoding populates the initial cache, split the remaining language-model-only tokens into chunks:

```python
while remaining_tokens > 0:
    chunk = tokens[:prefill_step_size]
    language_model(chunk, cache=req_cache)
    # force GPU work and free allocator cache
    mx.synchronize()
    mx.clear_cache()
    tokens = tokens[prefill_step_size:]
```

**Complexity:** MEDIUM — the LLM path already does this in mlx-lm. Need to adapt for the MLLM `_run_vision_encoding()` flow where the first chunk includes pixel_values.

**Files to modify:**
- `mllm_batch_generator.py`: `_run_vision_encoding()` or new method after vision pass

### Fix for Gap 5: Memory-Pressure Admission

**Goal:** Dynamically limit batch size based on available Metal GPU memory.

**Approach:**

In `_schedule_waiting()`, check `mx.metal.device_info()` before admitting a new request:

```python
mem_info = mx.metal.device_info()
active_mem = mx.metal.get_active_memory()
if active_mem / mem_info['max_recommended_working_set_size'] > 0.85:
    break  # Don't admit more requests
```

**Complexity:** LOW — simple check in admission logic.

**Files to modify:**
- `mllm_scheduler.py`: `_schedule_waiting()`

---

## All Changes Made in This Session

All changes are in **one file**: `vmlx_engine/mllm_batch_generator.py`
(Also synced to: `panel/bundled-python/python/lib/python3.12/site-packages/vmlx_engine/mllm_batch_generator.py`)

### Change 1: Added `MLLMBatch.extend()` method

**Location:** After `MLLMBatch.filter()` (around line 406)

**What:** New method that merges another `MLLMBatch` into the current one.

**Implementation:**
```python
def extend(self, other: "MLLMBatch") -> None:
    self.uids.extend(other.uids)
    self.request_ids.extend(other.request_ids)
    self.y = mx.concatenate([self.y, other.y])
    self.logprobs.extend(other.logprobs)
    self.max_tokens.extend(other.max_tokens)
    self.num_tokens.extend(other.num_tokens)
    self.requests.extend(other.requests)
    for c, o in zip(self.cache, other.cache):
        c.extend(o)
```

**Why:** Without this, there was no way to add new requests to an active batch. Mirrors mlx-lm's `Batch.extend()`.

---

### Change 2: Added `_ensure_batch_cache()` helper function

**Location:** After `_merge_caches()` (around line 580)

**What:** Converts raw KVCache/ArraysCache to BatchKVCache/BatchMambaCache for a single request.

**Implementation:**
```python
def _ensure_batch_cache(cache: List[Any]) -> List[Any]:
    converted = []
    for c in cache:
        if isinstance(c, BatchKVCache):
            converted.append(c)
        elif isinstance(c, KVCache):
            converted.append(BatchKVCache.merge([c]))
        elif isinstance(c, ArraysCache):
            converted.append(BatchMambaCache.merge([c]))
        elif isinstance(c, CacheList):
            inner = _ensure_batch_cache(list(c.caches))
            converted.append(CacheList(*inner))
        ...
    return converted
```

**Why:** When a batch starts with 1 request, `_process_prompts()` keeps raw KVCache (for Qwen3.5 integer offset compatibility). When a 2nd request joins, we need batch-aware caches before `extend()` can work. This function does that conversion.

---

### Change 3: Added `force_batch_cache` parameter to `_process_prompts()`

**Location:** Line 1153 (method signature) and line 1601 (cache merge condition)

**What:** When `force_batch_cache=True`, even single-request prefills produce batch-aware caches.

**Before:**
```python
if len(per_request_caches) == 1:
    batch_cache = per_request_caches[0]  # Raw KVCache
```

**After:**
```python
if len(per_request_caches) == 1 and not force_batch_cache:
    batch_cache = per_request_caches[0]  # Raw KVCache (single-request optimization)
else:
    batch_cache = _merge_caches(per_request_caches)  # Always batch-aware
```

**Why:** When extending into an active batch, the new request's cache must be batch-aware for `extend()` to work. Without this flag, single new requests would produce raw KVCache that can't be extended.

---

### Change 4: Rewrote `_next()` for true continuous batching (THE CORE FIX)

**Location:** Line 1732

**Before (serial):**
```python
if num_active == 0:
    # Only process new requests when batch is empty
    requests = self.unprocessed_requests[: self.completion_batch_size]
    new_batch = self._process_prompts(requests)
    self.active_batch = new_batch
```

**After (continuous):**
```python
num_to_add = self.completion_batch_size - num_active

if num_to_add > 0 and self.unprocessed_requests:
    requests = self.unprocessed_requests[:num_to_add]

    if num_active == 0:
        # No active batch — create fresh (same as before)
        new_batch = self._process_prompts(requests)
        self.active_batch = new_batch
    else:
        # Active batch — sync GPU, prefill new, extend into active
        mx.synchronize()
        new_batch = self._process_prompts(requests, force_batch_cache=True)
        if new_batch is not None:
            # Convert existing single-request batch cache if needed
            needs_convert = any(
                isinstance(c, KVCache) and not isinstance(c, BatchKVCache)
                for c in batch.cache
            )
            if needs_convert:
                batch.cache = _ensure_batch_cache(batch.cache)
            batch.extend(new_batch)
```

**Why:** This was the root cause. The old `if num_active == 0:` gate meant the MLLM path processed requests **serially**. Request B had to wait until Request A finished all its tokens. Now Request B can join Request A's batch mid-generation.

---

### What Was NOT Changed (and why)

| Component | Why No Change Needed |
|-----------|---------------------|
| MoE layers (SwitchLinear, SwitchMLP) | Stateless — no cache, `gather_qmm` is batch-compatible |
| JANG models | Use LLM text path which already has proper cont. batching |
| `BatchMambaCache` | Already has `extract()`, `merge()`, `filter()`, inherits `extend()` |
| `_step()` | Already wraps caches with `_BatchOffsetSafeCache` for offset safety |
| `HybridSSMStateCache` | SSM companion cache works unchanged |
| `_wrap_batch_caches()` | Already handles mixed BatchKVCache + ArraysCache |
| `_fix_hybrid_cache()` | Hybrid cache expansion works unchanged |
| `mamba_cache.py` | All patches already in place |
| `scheduler.py` (LLM path) | Already has proper continuous batching via mlx-lm |

---

## Verification Checklist

To verify the fix works correctly:

- [ ] Launch with `--continuous-batching --is-mllm`
- [ ] Send two simultaneous `curl` requests
- [ ] Confirm both requests generate tokens concurrently (interleaved SSE events)
- [ ] Confirm logs show second request being scheduled while first is generating
- [ ] Test with Qwen3.5-VL (hybrid SSM+attention model)
- [ ] Test with image requests (verify prefill completes before decode)
- [ ] Test with JANG model on LLM path (verify existing batching still works)
- [ ] Test cache hit path (send same prompt twice, verify "cache HIT" in logs)
- [ ] Test abort mid-batch (verify remaining requests continue)
- [ ] Monitor Metal GPU memory during concurrent requests
