# Continuous Batching Engine

**Branch**: `feature/continuous-batching`
**Base**: `main` (commit `2671c4c`)

Real continuous batching for mlx-swift-lm. Multiple generation requests are batched through a single model forward pass during decode. All model families supported — LLM, VLM, MoE, hybrid SSM, JANG quantized.

## Quick Start

```swift
// Load model (same as before — no changes to loading)
let container = try await LLMModelFactory.shared.loadContainer(
    from: downloader, using: tokenizerLoader,
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit"))

// Create engine
let engine = await container.makeBatchEngine(maxBatchSize: 8)

// Submit requests — same GenerateParameters as existing generate()
let stream = await engine.generate(
    input: lmInput,
    parameters: GenerateParameters(maxTokens: 256, temperature: 0.7))

// Consume — same Generation type as ModelContainer.generate()
for await generation in stream {
    switch generation {
    case .chunk(let text): print(text, terminator: "")
    case .info(let info):  print("\n\(info.summary())")
    case .toolCall:        break
    }
}
```

### Multiple Concurrent Requests

```swift
let engine = await container.makeBatchEngine(maxBatchSize: 8)

// From HTTP handler 1:
Task { for await g in await engine.generate(input: input1, parameters: p1) { ... } }

// From HTTP handler 2 (batched with handler 1):
Task { for await g in await engine.generate(input: input2, parameters: p2) { ... } }
```

### Per-Request Parameters

Each request gets its own temperature, topP, maxTokens, repetition penalty, etc.:

```swift
let greedy  = GenerateParameters(maxTokens: 50, temperature: 0)
let creative = GenerateParameters(maxTokens: 500, temperature: 0.9, topP: 0.95,
                                   repetitionPenalty: 1.2)

let s1 = await engine.generate(input: input, parameters: greedy)
let s2 = await engine.generate(input: input, parameters: creative)
```

### Raw Token Access + Cancellation

```swift
let (requestID, tokenStream) = await engine.submit(input: lmInput, parameters: params)

for await event in tokenStream {
    case .token(let id): feed(id)
    case .info:          break
}

// Cancel mid-generation:
await engine.cancel(requestID)

// Shutdown all:
await engine.shutdown()
```

## API Reference

### BatchEngine (actor)

| Method | Description |
|---|---|
| `init(context:maxBatchSize:memoryPurgeInterval:)` | Create engine from a `ModelContext` |
| `generate(input:parameters:) -> AsyncStream<Generation>` | Submit request, get text chunks back (same type as `ModelContainer.generate()`) |
| `submit(input:parameters:) -> (id, AsyncStream<BatchGeneration>)` | Low-level: raw token IDs + request ID for cancellation |
| `cancel(_:)` | Cancel a specific request by ID |
| `shutdown()` | Stop all requests, close all streams |
| `pendingCount` / `activeCount` / `isRunning` | Engine status |

### ModelContainer Extension

| Method | Description |
|---|---|
| `makeBatchEngine(maxBatchSize:memoryPurgeInterval:)` | Create a `BatchEngine` from the container |

### Types

| Type | Description |
|---|---|
| `BatchGeneration` | `.token(Int)` or `.info(GenerateCompletionInfo)` |
| `BatchRequestID` | Unique request identifier (UUID-based) |

---

## Files

### New (9 files, ~1,300 lines)

| File | Lines | Purpose |
|---|---|---|
| `BatchEngine.swift` | ~500 | Public actor: submit, generate, cancel, scheduling loop, prefill/decode |
| `BatchKVCache.swift` | 174 | Wraps N per-sequence KV caches into `[B,H,maxLen,D]` view for attention layers |
| `BatchArraysCache.swift` | 94 | Wraps N per-sequence MambaCache/SSM states for hybrid models |
| `BatchCacheList.swift` | 85 | Wraps N per-sequence CacheLists (FalconH1, BaichuanM1) |
| `BatchMask.swift` | 69 | Per-sequence causal masks for variable-length batches |
| `BatchScheduler.swift` | 122 | Slot state, phase tracking, per-request sampling |
| `BatchTypes.swift` | 71 | Request ID, generation events |
| `ModelContainerBatch.swift` | 49 | `ModelContainer.makeBatchEngine()` convenience |
| `BATCH_ENGINE.md` | — | This file |

### Modified (11 files, +92 / -50 lines)

| File | Change |
|---|---|
| `RoPEApplication.swift` | +4: `BatchKVCache` type check for per-sequence `[B]`-shaped RoPE offsets |
| `Gemma4Text.swift` | +26: KV-shared layer `sharedOffsetArray` through attention/decoder/model loop |
| `Gemma4.swift` (VLM) | +16: Same KV-shared fix for VLM inner model |
| `Qwen2VL.swift` | -3/+2: `applyRotaryPosition` instead of `cache?.offset` |
| `Qwen25VL.swift` | -3/+2: Same |
| `Pixtral.swift` | -3/+2: Same |
| `Mistral3.swift` (VLM) | -3/+2: Same |
| `FastVLM.swift` | -3/+2: Same |
| `Idefics3.swift` | -3/+2: Same |
| `Qwen3VL.swift` | +20: Custom `BatchKVCache` position ID construction for 3D RoPE |
| `Qwen35.swift` (VLM) | +18: Same for Qwen35 VLM 3D RoPE |

### Tests (1 file)

| File | Tests | Description |
|---|---|---|
| `BatchEngineTests.swift` | 14 | Unit: mask (4), cache (3). Integration: single request, concurrent, different params, cancellation, queue overflow, throughput benchmark, shutdown |

### Unchanged

`KVCache.swift`, `AttentionUtils.swift`, `Evaluate.swift`, `TokenIterator`, `ModelContainer`, all `generate()` APIs, `Package.swift`, `LanguageModel.swift`, all LLM model files except Gemma4Text.

---

## Architecture

```
submit(input, params) ─or─ generate(input, params)
       │
       ▼
  waitQueue
       │
       ▼  admitPendingRequests() ── when activeSlots < maxBatchSize
  BatchSlot:
    cache = model.newCache(parameters:)
    sampler/processor from per-request GenerateParameters
    originalInput preserved (VLM image data)
       │
       ▼  stepPrefill() ── uses model.prepare()
  LLM: chunked prefill → .tokens(remaining) → model(remaining) → sample first token
  VLM: vision tower + maskedScatter + full prefill → .logits → sample first token
  First token yielded, EOS checked
       │
       ▼  stepBatchDecode()
  Build [B, 1] input from all decode slots
  Per-layer cache wrapping:
    KVCacheSimple/RotatingKVCache → BatchKVCache (split/pad/stack)
    ArraysCache/MambaCache        → BatchArraysCache (merge along batch dim)
    CacheList                     → BatchCacheList (wraps sub-caches)
  model([B,1], cache: batchCaches) → logits [B, 1, V]
  After forward: BatchArraysCache.splitBack() writes SSM states back
  Sample per sequence, check EOS before yielding
  Finished → .info(completionInfo) → stream closed
  Slot removed, next request admitted from queue
```

### Cache Wrappers

| Wrapper | Base Class | Passes `as?` Check | Used By |
|---|---|---|---|
| `BatchKVCache` | `BaseKVCache` | `KVCache` | All standard attention layers |
| `BatchArraysCache` | `MambaCache` | `MambaCache` | SSM layers (Qwen3.5, Jamba, LFM2, etc.) |
| `BatchCacheList` | `CacheList` | `CacheList` | Composite layers (FalconH1, BaichuanM1) |

### RoPE

`applyRotaryPosition` detects `BatchKVCache` at runtime → `rope(x, offset: batchCache.offsetArray)` using `[B]`-shaped per-sequence positions. All models that call `applyRotaryPosition` get correct batched RoPE automatically.

VLM models that previously used `cache?.offset` directly (Qwen2VL, Qwen25VL, Qwen3VL, Pixtral, Mistral3, FastVLM, Idefics3/SmolVLM2) have been updated to use `applyRotaryPosition` or custom `BatchKVCache` offset construction for 3D RoPE.

Gemma4 KV-shared layers now pass `sharedOffsetArray: MLXArray?` through the attention chain for correct per-sequence positioning.

---

## Model Compatibility

**All model families supported.** No sequential fallback.

### LLM — Full Batching via BatchKVCache

Llama, Mistral, Qwen2, Qwen3, Qwen3MoE, Phi, Phi3, PhiMoE, DeepseekV3, Gemma, Gemma2, Gemma3, Gemma4 (dense + MoE), Cohere, Starcoder2, GLM4, Granite, Internlm2, Olmo2/3, OpenELM, MiniMax, BailingMoe, MiniCPM, MiMo, Ernie4_5, GPTOSS, Bitnet, NanoChat, SmolLM3, Mistral4, Exaone4, AfMoE, MiMoV2Flash, Apertus, Lille130m, Gemma3n

### LLM — Full Batching via BatchArraysCache (hybrid SSM)

Qwen3.5, Qwen3Next, Jamba, LFM2, LFM2MoE, GraniteMoeHybrid, NemotronH

### LLM — Full Batching via BatchCacheList

FalconH1, BaichuanM1

### VLM — Full Batching

Gemma4 VLM, Gemma3 VLM, Mistral4 VLM, GlmOcr, Qwen2VL, Qwen25VL, Qwen3VL, Pixtral, Mistral3 VLM, Idefics3/SmolVLM2, FastVLM, Qwen35 VLM, Qwen35MoE VLM, LFM2VL

### JANG Quantized

All JANG models inherit their base model's batch support. JANG only affects weight loading.

---

## Throughput Benchmarks

### Swift Unit Test (small 4-layer Llama)

```
Serial (B=1 sequential): 247.9 total tok/s
Batch  (B=4 concurrent): 641.1 total tok/s
Speedup: 2.59x
```

### MLX GPU Benchmark (real model dimensions, 6 attention layers, fp16)

| Model | B=1 tok/s | B=2 tok/s | B=3 tok/s | Speedup |
|---|---|---|---|---|
| Gemma 4 E2B 4-bit (1.5B) | 7,627 | 27,209 | 43,555 | **5.71x** |
| Gemma 4 E2B 8-bit (1.5B) | 14,574 | 27,396 | 41,054 | **2.82x** |
| Gemma 4 E4B 4-bit (4B) | 14,513 | 25,971 | 45,656 | **3.15x** |
| Gemma 4 E4B 8-bit (4B) | 14,837 | 30,671 | 47,608 | **3.21x** |
| Gemma 4 31B JANG (MoE) | 14,594 | 30,078 | 44,573 | **3.05x** |
| Qwen3.5 35B JANG (SSM) | 15,358 | 30,863 | 47,304 | **3.08x** |

GPU is memory-bandwidth-bound at B=1 during decode. Batching fills compute units without proportional latency increase.

---

## Bugs Found and Fixed

| # | Bug | Severity | Fix |
|---|---|---|---|
| 1 | Logit processors crash on 1D tensor | Critical | Range subscript preserves 2D shape |
| 2 | EOS token yielded to caller | Critical | Check EOS before yielding |
| 3 | First token silently consumed | Critical | Yield + EOS check after prefill |
| 4 | VLM images dropped during prefill | Critical | Use `model.prepare()` with full input |
| 5 | SSM state not persisted (MambaCache) | Critical | `BatchArraysCache` with merge/split |
| 6 | CacheList crashes on update() | Critical | `BatchCacheList` wraps sub-caches |
| 7 | Missing unknownTokenId in stop set | Important | Added to init |
| 8 | `[weak self]` drops requests | Important | Removed capture |
| 9 | Processor seeded with partial prompt | Important | Use `originalInput.text.tokens` |
| 10 | 7 VLM models used scalar RoPE offset | Important | Updated to `applyRotaryPosition` |
| 11 | Gemma4 KV-shared layers used scalar RoPE | Important | Added `sharedOffsetArray` parameter |
| 12 | Qwen3VL/Qwen35VLM 3D RoPE used scalar | Important | Custom `BatchKVCache` position ID construction |

---

## Remaining Limitations

1. **RotatingKVCache wrapping** — after cache wraps (offset > maxSize), rotated key order may break batch mask. Very rare (requires sequences exceeding sliding window size).

2. **No multi-turn cache reuse** — each `submit()` creates fresh caches with full re-prefill. Same as `TokenIterator`. Cache persistence is future work.

3. **No KV cache quantization in batch** — `kvBits`/`kvMode` not applied during batched decode. Would need `BatchQuantizedKVCache`.

4. **No compile() in batch** — dynamic batch sizes prevent `compile()` tracing.
