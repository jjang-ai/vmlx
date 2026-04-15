# KV Cache Handling Matrix — vMLX Swift

**Status**: Source of truth for how every model family in the Swift
rewrite builds its per-layer KV cache, which tiers of the multi-tier
coordinator it's compatible with, and what gotchas to watch for.
Updated as models change. See also:

- [`Sources/vMLXLMCommon/Cache/CacheCoordinator.swift`](../Sources/vMLXLMCommon/Cache/CacheCoordinator.swift) — multi-tier coordinator
- [`Sources/vMLXLMCommon/Cache/TQDiskSerializer.swift`](../Sources/vMLXLMCommon/Cache/TQDiskSerializer.swift) — v2 disk format + layer-kind tags
- [`Sources/vMLXLMCommon/Cache/CacheHelpers.swift`](../Sources/vMLXLMCommon/Cache/CacheHelpers.swift) — extract / restore
- [`Sources/vMLXLMCommon/Cache/SSMReDerive.swift`](../Sources/vMLXLMCommon/Cache/SSMReDerive.swift) — prompt-only re-derive
- [`Sources/vMLXLMCommon/KVCache.swift`](../Sources/vMLXLMCommon/KVCache.swift) — all cache types
- [`docs/GEMMA4-NOTES.md`](GEMMA4-NOTES.md) — Gemma 4 specific nuances

## Tiers at a glance

| Tier | Type | What it stores | Gating |
|---|---|---|---|
| L1  | `PagedCacheManager` (in-memory, block-indexed) | KV blocks with LRU | `config.usePagedCache` |
| L1.5 | `MemoryAwarePrefixCache` (byte-budgeted) | Whole-prompt payloads | `config.enableMemoryCache` (default off) |
| L2  | `DiskCache` + `TQDiskSerializer` v2 | Per-prompt full KV + Mamba state on SSD | `config.enableDiskCache` (**default ON** as of 2026-04-14) |
| SSM | `SSMStateCache` (LRU) | Companion Mamba state per hash-prefix | Automatic when `coordinator.isHybrid` |

## Layer-kind support in TQDiskSerializer v2

`TQDiskSerializer.serialize(...)` writes a `__layer_kind_N__` int32
tag per layer so the restore path can dispatch correctly. The
supported kinds are:

| Kind | Handled | Notes |
|---|---|---|
| `.tq`    | ✅ | `TurboQuantKVCache` — compressed 4-bit key + 4-bit value via Hadamard + Lloyd-Max. `restoreCompressed` round-trips. |
| `.kv`    | ✅ | `KVCacheSimple` — direct K/V float16 or bfloat16. |
| `.qkv`   | ✅ | `QuantizedKVCache` — affine q4/q8 with per-group scale. |
| `.mamba` | ✅ | `MambaCache` — Δ + conv + SSM state per layer. Restored via `restoreMambaLayer` into a fresh cache. |
| `.skip`  | — | Placeholder for layers the serializer can't round-trip. Currently written for **`RotatingKVCache`** (silently dropped — see below) and for layers extracted as `nil` by `extractLayerData`. |

**No kind exists for `RotatingKVCache`**. Rotation state is a ring
buffer with `(offset, maxSize, keep, wrapAround)` quadruple metadata
that's not carried in the v2 format.

## The rotating-cache L2 round-trip hazard

A `RotatingKVCache` in a cache list causes `TQDiskSerializer` to
emit a `.skip` tag for that layer. On restore the tag is honoured
(the layer is rebuilt fresh) but the **caller has no signal that
part of the cache is missing** — `diskCache.store()` reported
success, the entry exists on disk, and the next turn's fetch
happily returns partial state.

Because several model families interleave `full_attention` and
`sliding_attention` layers (Gemma 3/4), storing half of the KV
into disk with the other half silently dropped gives a
non-deterministic hit that corrupts sliding-window semantics. This
is a **correctness bug**, not just a performance hazard.

**Fix (2026-04-14)**: `CacheCoordinator.storeAfterGeneration` has a
central guard (`hasRotatingLayer`) that checks every layer for
`RotatingKVCache` — including inside `CacheList` sub-caches — and
**skips the disk and memory tiers entirely** when found. The paged
L1 still works because `CacheHelpers.extractLayerData` already
returns `nil` for rotating layers upstream. SSM companion tier also
still fires on hybrid models.

The skip is intentional until a future format bump adds a
`.rotating` kind with proper offset/wrap serialization. Regression
tests pin the guard in
[`CacheCoordinatorRotatingGuardTests.swift`](../Tests/vMLXTests/CacheCoordinatorRotatingGuardTests.swift).

## Hybrid SSM + thinking-template contamination

For hybrid models (`MambaCache` + `KVCacheSimple` interleaved), the
SSM state is cumulative: after generation it reflects position
`(prompt + generated)`. But cache keys are stripped by
`genPromptLen` (see [`Sources/vMLXLMCommon/Cache/CacheCoordinator.swift`](../Sources/vMLXLMCommon/Cache/CacheCoordinator.swift)
`fetch(..., genPromptLen:)`), so storing the post-gen SSM state
under a prompt-only hash would produce position-mismatch garble on
the next turn.

**Fix**: `shouldSkipSSMStorage(isHybrid:, genPromptLen:)` returns
`true` when `isHybrid && genPromptLen > 0`, and the store path
substitutes `nil` for SSM states in every sub-tier. The L1 paged KV
tier still captures attention state so multi-turn cache reuse works
for the attention path.

**Long-term fix**: [`Sources/vMLXLMCommon/Cache/SSMReDerive.swift`](../Sources/vMLXLMCommon/Cache/SSMReDerive.swift)
implements a prompt-only re-derive pass that runs a fresh forward
on the stripped prompt and stores a clean SSM state. Gated on
`GlobalSettings.enableSSMReDerive` (default `true`).

## Per-model cache matrix

| Model | File | Cache kinds | Hybrid | Rotating | L2 safe? | Notes |
|---|---|---|---|---|---|---|
| Llama / Mistral | `Llama.swift` / `Mistral.swift` | KVCacheSimple | ❌ | ❌ | ✅ | Standard path. |
| Phi / Phi3 / PhiMoE | `Phi*.swift` | KVCacheSimple | ❌ | ❌ | ✅ | |
| Gemma / Gemma2 | `Gemma.swift`, `Gemma2.swift` | KVCacheSimple | ❌ | ❌ | ✅ | |
| **Gemma3 text** | `Gemma3Text.swift` | KVCacheSimple + **RotatingKVCache** | ❌ | ✅ | ⚠️ skipped | `layerTypes` interleaves full / sliding → rotating. Guard fires. |
| **Gemma3n text** | `Gemma3nText.swift` | KVCacheSimple + **RotatingKVCache** | ❌ | ✅ | ⚠️ skipped | Same as Gemma3. |
| **Gemma4 text** | `Gemma4Text.swift` | KVCacheSimple + **RotatingKVCache** | ❌ | ✅ | ⚠️ skipped | `full_attention` → KVCacheSimple or RotatingKVCache(keep=4); `sliding_attention` → RotatingKVCache(window, keep=0). E2B/E4B shared-KV layers omit entries. |
| **Gemma4 VLM** | `vMLXVLM/Models/Gemma4.swift` | Inherits Gemma4Text | ❌ | ✅ | ⚠️ skipped | Same as text. Plus `mediaSalt` mixed into hash. |
| Qwen2 / Qwen2.5 / Qwen3 | `Qwen*.swift` | KVCacheSimple | ❌ | ❌ | ✅ | |
| Qwen3-MoE | `Qwen3MoE.swift` | KVCacheSimple | ❌ | ❌ | ✅ | |
| **Qwen3.5 text** | `Qwen35.swift` | KVCacheSimple + MambaCache | ✅ | ❌ | ✅ (hybrid SSM) | `isLinear` → Mamba. |
| **Qwen3.5 MoE** | `Qwen35MoE.swift` | KVCacheSimple + MambaCache | ✅ | ❌ | ✅ (hybrid SSM) | |
| **Qwen3.5 VL** | `vMLXVLM/Models/Qwen35.swift` | Inherits Qwen35 | ✅ | ❌ | ✅ (hybrid SSM) | |
| **Qwen3-Next** | `Qwen3Next.swift` | KVCacheSimple + MambaCache (gated delta) | ✅ | ❌ | ✅ (hybrid SSM) | |
| **NemotronH** | `NemotronH.swift` | KVCacheSimple + MambaCache | ✅ | ❌ | ✅ (hybrid SSM) | Interleaves attention + mamba + mlp + MoE layers. |
| **Jamba** | `Jamba.swift` | KVCacheSimple + MambaCache | ✅ | ❌ | ✅ (hybrid SSM) | |
| **FalconH1** | `FalconH1.swift` | `CacheList(MambaCache, KVCacheSimple)` | ✅ | ❌ | ✅ (hybrid SSM) | Each layer wraps both sub-caches. |
| **GraniteMoeHybrid** | `GraniteMoeHybrid.swift` | KVCacheSimple + MambaCache | ✅ | ❌ | ✅ (hybrid SSM) | |
| **MiMoV2Flash** | `MiMoV2Flash.swift` | KVCacheSimple + **RotatingKVCache** | ❌ | ✅ | ⚠️ skipped | `isSlidingWindow` layers use RotatingKVCache. |
| **BaichuanM1** | `BaichuanM1.swift` | `CacheList(Mamba, RotatingKVCache)` | ✅ | ✅ | ⚠️ skipped | Hybrid AND sliding — guard fires (CacheList descent). |
| **LFM2** / **LFM2MoE** | `LFM2*.swift` | KVCacheSimple + MambaCache | ✅ | ❌ | ✅ (hybrid SSM) | `fullAttnIdxs` → KV, else Mamba. |
| **Mistral4** | `Mistral4.swift` | KVCacheSimple or **RotatingKVCache** | ❌ | ✅ (when `maxKVSize` set) | ⚠️ skipped | All layers rotating when `parameters.maxKVSize` present. |
| MiniMax / MiniMaxJANGTQ | `MiniMax*.swift` | KVCacheSimple (default newCache) | ❌ | ❌ | ✅ | |
| MiniCPM / Starcoder2 / Cohere / OpenELM / InternLM2 / DeepseekV3 / Granite / MiMo / GLM4 / GLM4MoE / GLM4MoELite / BailingMoe / Bitnet / SmolLM3 / Ernie4.5 | `*.swift` | KVCacheSimple | ❌ | ❌ | ✅ | Standard dense / MoE text models. |

Legend:
- **Hybrid**: cache list contains MambaCache or ArraysCache → `CacheCoordinator.setHybrid(true)` set at load time
- **Rotating**: cache list contains RotatingKVCache (guards disk/memory tiers)
- **L2 safe?**:
  - ✅ — full round-trip, disk cache fires
  - ✅ (hybrid SSM) — disk cache fires, Mamba state bundled via `.mamba` tag; SSM companion skipped on thinking-template turns (genPromptLen>0) unless re-derive is enabled
  - ⚠️ skipped — coordinator skips disk + memory tiers because rotation state can't round-trip; paged L1 still works (rotating layers filtered upstream)

## When to reach for which tier

1. **Interactive chat, short single-turn prompts** — L1 paged is enough. Tens of MB RAM, instant hits.
2. **Long multi-turn chat, <32k tokens/turn** — L1 + L2 disk covers every turn. TurboQuant keeps payloads small.
3. **Long multi-turn chat, >64k tokens or RAM-constrained** — enable L1.5 memory cache (`enableMemoryCache: true`). Whole-prompt hits across session replays.
4. **Hybrid models (Nemotron-H, Qwen3.5-A3B, Jamba)** — coordinator auto-sets `isHybrid` and wires the SSM companion tier. No user action needed.
5. **Thinking models (Qwen3.5, DeepSeek R1 style)** — leave `enableSSMReDerive: true`. Adds one extra prompt-only forward pass at turn-end in exchange for a clean SSM companion store. Disable only if turn-end latency matters more than multi-turn cache reuse.
6. **Sliding-window models (Gemma 3/4, Mistral 4, MiMoV2Flash)** — disk tier is auto-skipped by the guard. Paged L1 + fresh re-prefill on every turn is the current state. A future format bump would enable L2 for these; until then, don't count on disk cache helping.

## Changing the matrix

1. A new model family lands. Add a row to the table and confirm `isHybrid` detection works. Add a `newCache(parameters:)` unit test if the cache mix is non-obvious.
2. TQDiskSerializer gains a new kind tag. Update the "Layer-kind support" section. Bump the format version constant in the serializer.
3. A model switches between dense and sliding (config-driven). Walk both branches through `CacheCoordinatorRotatingGuardTests` to confirm the guard still fires in the sliding case.
