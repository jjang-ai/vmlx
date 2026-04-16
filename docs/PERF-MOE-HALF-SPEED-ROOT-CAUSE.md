# MoE half-speed root cause — typed scalar discipline

**TL;DR.** The recurring "our Swift MoE models run at half the reference's speed" regression is **not** a compile / FFI / KV-quant problem. It is ONE specific, silent pattern: **untyped `MLXArray(<literal>)` scalars inside model forward passes.** Every such scalar promotes the surrounding activation tensor to `Float32` via an implicit `AsType`. At MoE + hybrid-SSM scale this adds hundreds of AsType dispatches per decode token and cuts tok/s roughly in half.

This doc is the authoritative reference. If you are about to edit any file under `Sources/vMLXLLM/Models/` or `Sources/vMLXVLM/Models/`, read this first and run `scripts/lint-typed-mlx-scalars.sh` before committing.

---

## 1. The failure mode

`MLXArray(0)` without a `dtype:` argument constructs a 1-element **Float32** array. When passed to an MLX op that also receives a bfloat16 (or fp16) activation, MLX inserts an implicit `AsType` to upcast the activation, runs the op in fp32, and the consumer of the result later downcasts with another `AsType` to put the tensor back on the model's dtype. That's **two AsType ops per hit**, dispatched to the GPU as separate Metal kernels.

### Why it kills MoE specifically

- MoE models run the FFN once per **active expert** per token. Nemotron-Cascade-30B uses top-8 on 128 experts → up to **8 × FFN dispatches per layer**, with 26+ layers (hybrid Mamba+attention).
- Each FFN contains a `relu2`, a `SwitchGLU` gate/up/down triplet, sometimes a `sigmoid`, sometimes a `softcap`. Any untyped scalar inside any of those fires per active expert per token.
- At ~100 tok/s the engine is already dispatching thousands of small kernels per second. Doubling that with stray AsType ops saturates the CPU-GPU bridge; GPU waits idle for the next command buffer. Decode throughput halves — exactly the symptom.

### Why dense models are less affected

Dense LLMs (Llama 3.2 1B, Gemma dense, etc.) run ONE FFN per layer per token with no expert routing. Even with a couple of untyped scalars the amortized cost per token is small. The bug is still there; it just doesn't dominate. MoE + hybrid is what unmasks it.

---

## 2. Measured impact (M4 Max 128 GB, `RunBench BENCH_SIMPLE`, no coordinator)

Source: reference `/Users/eric/vmlx-swift-lm` main `bf942a8` on **the same hardware as our fork**. `git show cf296b1` reports the single-line identity-weight fix taking Nemotron-Cascade-30B from `45 → 110` tok/s with AsType counter `562 → 161`.

| Model | Our tree | `vmlx-swift-lm` main |
|-------|---:|---:|
| Qwen3.5-35B-A3B-4bit (MoE) | 61 | **98.9** |
| Nemotron-Cascade-2-30B-A3B-JANG_4M (hybrid SSM MoE) | 61 | **132.3** |
| Gemma-4-26B-A4B-it-JANG_4M | 49 | **87.9** |
| Gemma-4-26B-A4B-mxfp4 | — | **99.9** |
| Qwen3.5-9B-JANG_2S | 69 → 98 | **107.1** |

The "61 tok/s is the Swift MoE ceiling" citation you might encounter in older docs is a **stale-doc phantom**, not a Swift limit. Reference has been shipping >100 tok/s on MoE for weeks. Any perf claim citing 61 as a Swift ceiling is wrong and must be corrected at the source, not quoted as evidence of a cap.

The fork's shortfall is not a structural floor (not 2-bit quant, not compile, not HTTP wrapper). It's **stale files** that never received the reference's recent perf commits — see §5.

---

## 3. The exact forbidden patterns

All greppable from model source files.

### 3a. `MLX.maximum(x, MLXArray(0))` — relu / relu2 / clamp_min

```swift
// WRONG — promotes x to fp32 every call
let y = MLX.maximum(x, MLXArray(0))

// RIGHT
let y = MLX.maximum(x, MLXArray(0, dtype: x.dtype))
```

**Seen in (reference fix):** `NemotronH.swift:relu2`, `Gemma3nText.swift:geluTopK`.

### 3b. Scalar subtract / negation

```swift
// WRONG — one AsType + one broadcast dispatch for nothing
let y = MLXArray(0) - scores

// RIGHT — built-in negation, no scalar allocation
let y = -scores
```

**Seen in (reference fix):** `Gemma4Text.swift:407`, `Mistral4.swift:300`, `Gemma4.swift (VLM):251,607`, `Mistral4VLM.swift:219`.

### 3c. `MLX.where` / masked fill with untyped floor

```swift
// WRONG — Float.leastNormalMagnitude is fp32, AsType on every step
scores = MLX.where(causalMask, scores,
                   MLXArray(Float.leastNormalMagnitude))

// RIGHT
scores = MLX.where(causalMask, scores,
                   MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
```

**Seen in (reference fix `adb481f`):** `KVCache.swift:1592,1596,1605` (QuantizedKV attention).

### 3d. `compiledLogitSoftcap(x, MLXArray(cap))`

```swift
// WRONG
let clipped = compiledLogitSoftcap(out, MLXArray(cap))

// RIGHT
let clipped = compiledLogitSoftcap(out, MLXArray(cap, dtype: out.dtype))
```

**Seen in (reference fix):** `Gemma4Text.swift:807`, `Gemma4.swift (VLM):792`, GPT-OSS attention cap.

### 3e. `MLXArray.ones(...)` for identity weights

```swift
// WRONG — every call allocates a fresh fp32 buffer
let identity = MLXArray.ones([groupSize])
let normed = MLXFast.rmsNorm(x, weight: identity, eps: eps)

// RIGHT — dtype matches x so rmsNorm stays in one dtype
let normed = MLXFast.rmsNorm(x,
    weight: MLXArray.ones([groupSize], dtype: x.dtype),
    eps: eps)
```

**Seen in (reference fix `cf296b1`):** `NemotronHRMSNormGated`.
Already applied in our fork at `NemotronH.swift:82`.

### 3f. `timeStepLimit` / `softplus` scalar clips

```swift
// WRONG
let dtClipped = MLX.clip(dt, min: MLXArray(0), max: MLXArray(limitMax))

// RIGHT
let dtClipped = MLX.clip(dt,
    min: MLXArray(0, dtype: dt.dtype),
    max: MLXArray(limitMax, dtype: dt.dtype))
```

Plus `logAddExp(x, MLXArray(0, dtype: x.dtype))` for softplus, instead of `log(1 + exp(x))` which also over-promotes.

**Seen in (reference fix):** `SSM.swift:17-18` in Nemotron Mamba block.

---

## 4. What is NOT the problem

Eric's explicit corrections (2026-04-16 session) — DO NOT repeat these framing errors in any follow-up doc, commit message, or analysis:

- **"Compile is the unbridgeable Swift ceiling" is false.** Reference `HardwareInfo.isCompiledDecodeSupported = true` and has 3+ unconditional `compile(shapeless:true)` sites (`Qwen35-VLM swiglu`, `Qwen35-VLM precise_swiglu`, `GatedDelta.swift:26`). The Python-vs-Swift gap is **how wide** a compiled graph can be, not whether compile is on. Much narrower problem. Stop citing "no compile" as the reason.
- **Do NOT blind-port `setupCompiledDecode` + `CompilableKVCache`.** Reference disabled the whole-model compile path via commit `a8a6a6f` because MLX#3329 crashes with `compiledState.callsToFill[0] index out of range` on M1/M2/M4 Pro. Re-enabling it without re-verifying on every Apple Silicon generation is a regression vector. The micro-fusion compile islands are the safe form. If you want the whole-model variant, it goes behind an opt-in env var, never default.
- **The HTTP / BatchEngine wrapper is NOT the perf killer.** Reference benchmarks its BatchEngine actor against `RunBench` with no measurable overhead (commit `2c9e5c8`). If the wrapper ever looks like the bottleneck, profile — find which hot-path is doing per-step allocation/parsing/metrics and inline it. Do not "rip out HTTP streaming" or propose multi-hour rewrites.
- **"2-bit JANG is structurally slower so 100 tok/s is unreachable" is also false.** This is a soften-the-gap excuse, not a measured ceiling. Any claim like this must be backed by a direct measurement on reference with the exact same model file; absent that, the gap is stale files (§5), not quantization.
- **TurboQuant default-on was a real regression** (TQ reconstruct on every attention-half layer per step) and the default-off fix is correct — but it's only ONE of the perf layers. The rest of this doc is the remaining layer.

---

## 5. Recovery recipe for a regressed fork — the stale-files gap

The dominant reason our fork lags reference is not a single line; it's a set of **stale hot-path files** that never received the reference's recent perf commits. Per `PERF-ISSUES-2026-04-16.md` § "Files that diverge from reference":

- `Gemma4Text.swift` — ref newer by ~33 lines
- `NemotronH.swift` — ref newer by ~222 lines
- `Qwen35.swift` (both LLM and VLM) — ref newer
- `Mistral3.swift` — ref newer
- `TurboQuantKVCache.swift` — ref newer
- `Evaluate.swift` — ref newer
- `BatchEngine.swift` — ref newer by ~145 lines

Reference commits shipped in that window, each perf-relevant:

- `fb46fbd` int4 fused gate+up `gatherQuantizedMM` for decode (MoE)
- `2422e4f` VLM cache-hit remainder rebuild as 2D (VLM correctness that unblocks cache-hit decode)
- `e01fb9d` synchronous disk store + error surfacing
- `5b26831` chunked-prefill axis fix
- `d4e4e45` AsType cascade elimination across MoE families
- `cf55f6d` GatedDelta `_compiledComputeG` (+11 % on Qwen3.5-35B)
- `6b896d4` Qwen35 compiled `swiglu` / `precise_swiglu`
- `c859cc7` Qwen35 VLM fused Metal kernel for gated-delta step
- `bf942a8` SLIDING-1: RotatingKVCache L2 disk persistence + fp16→fp32 SDPA upcast for Gemma 4

**Recovery procedure:**

1. Run `scripts/lint-typed-mlx-scalars.sh` — fastest sanity pass on the typed-scalar layer.
2. **Port the stale files surgically from reference.** Cherry-pick the perf hunks per commit above into each of the files listed — DO NOT blind-overwrite, because our fork carries FlashMoE shims, LatentMoE variants, JANGTQ routing, §15 harmonyActive narrowing, genPromptLen, mediaSalt, and the §20 `finish_reason` fix that reference doesn't have. A blind copy deletes those.
3. Confirm `enableTurboQuant=false` default (done — see `PERF-ISSUES-2026-04-16.md`).
4. Only bench when Eric asks. The stale-file port unblocks ~30–50 % by the reference's own measurement.

**Do not** propose a new fix before steps 1–2. The two together close the gap reference already closed.

---

## 6. Test-suite hooks

- `Tests/vMLXTests/PerfScalarLintTests.swift::testNoUntypedMLXArrayScalarsInHotPath`
- `scripts/lint-typed-mlx-scalars.sh` — wired into `scripts/build-release.sh`
- `SWIFT-NO-REGRESSION-CHECKLIST.md §27` — process gate; every phase / release must grep-verify this holds.
