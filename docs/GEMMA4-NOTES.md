# Gemma 4 — Nuances & Known Quirks (Swift vMLX)

This document is the single source of truth for how the Swift
rewrite handles Gemma 4. Gemma 4 has more edge cases than any other
family in-tree — config variants, chat-template bugs, cache mixing,
MoE routing, fp16 overflow, image-token handling — and several of
them caused past user-reported regressions (v1.3.51 Python bundled
`mlx-vlm` missing `gemma4`, earlier `<image>`-token split failures,
and the `finalLogitSoftcapping` fp16 NaN cascade). Future
refactors should pass a "Gemma 4 smoke pass" before merging.

## Variants

| Variant | Parameters | Config hint | Branch | Text/VLM |
|---|---|---|---|---|
| E2B | ~2B | `enableMoeBlock=false`, `numKvSharedLayers>0` | KV-share fast path | text-only |
| E4B | ~4B | `enableMoeBlock=true`, `numKvSharedLayers>0` | MoE + KV-share | text-only |
| 26B-A4B | active 4-bit MoE | `numExperts=128`, `topK=8` | MoE, no KV-share | text or VLM |
| 31B | dense | `enableMoeBlock=false` | dense, no KV-share | text or VLM |
| JANG-repacked | any of above | `jang_config.json` present | JANGLoader path | inherits |
| mlx-community 4-bit | 8/26/31B | standard quant, per-group scales | standard path | inherits |

Registration:
- LLM: `LLMTypeRegistry["gemma4"]` + `"gemma4_text"` → `Gemma4TextConfiguration` + `Gemma4TextModel`
- VLM: `VLMTypeRegistry["gemma4"]` → `Gemma4Configuration` + `Gemma4.init`
- Guarded by [`Tests/vMLXTests/ModelFactoryRegistrationTests.swift`](../Tests/vMLXTests/ModelFactoryRegistrationTests.swift) — the Swift analogue of the v1.3.51 post-mortem.

## model_type detection

Gemma 4 arrives under several top-level `model_type` values:

1. Pure text file: `"model_type": "gemma4"` or `"gemma4_text"` → LLM factory
2. VLM wrapper: `"model_type": "gemma4"` with `"text_config": { "model_type": "gemma4_text" }` → VLM factory; text decoder fetched via text_config
3. Mistral3-wrapped VLM (some community uploads): `"model_type": "mistral3"` with `"text_config": { "model_type": "gemma4_text" }` → see `CapabilityDetector.resolveModelType` + `VLMModelFactory` fallback
4. JANG-stamped: `jang_config.json` carries `has_vision` and `architecture` → Tier 1 capability stamp wins over `config.json`

`CapabilityDetector` walks these in tier order (JANG → ModelTypeTable lookup → substring heuristic). If a new variant lands with an unknown `model_type`, the bronze tier catches `contains("gemma")` and emits a `family: "gemma"` with the Gemma 4 parser set so reasoning + tool parsing still work; only the loader dispatch falls back to heuristic name matching.

## Chat template fallback

Gemma 4's upstream `chat_template.jinja` uses constructs that swift-jinja can't parse. Three distinct upstream bugs:

1. `multiplicativeBinaryOperator` — Gemma 4 templates multiply ints in loop bounds
2. `SelectExpression` runtime fail on inline-`if`-without-`else`
3. kwarg plumbing mismatch in `evaluateCallExpression` vs `evaluateMacro`

Rather than fork swift-jinja, [`TransformersTokenizerLoader.swift`](../Sources/vMLXEngine/TransformersTokenizerLoader.swift) catches any Jinja error and routes through `ChatTemplateFallback.render(messages:addGenerationPrompt:bosToken:tokenExists:)`. The dispatcher sniffs the tokenizer's vocab for family-specific role tags and picks the matching renderer. For Gemma 4:

- **Native format**: `<|turn>role\n...<turn|>\n` — tokens 105 (`<|turn>`) and 106 (`<turn|>`) are native in every Gemma 4 vocab variant. The dispatcher probes `convertTokenToId("<|turn>")` to confirm.
- **Role mapping**: `assistant → model`. Gemma 4 was trained with a `model` role tag, not `assistant`. Getting this wrong produces immediate EOS (model refuses to speak).
- **System collapse**: System messages get folded into a leading `<|turn>system\n...<turn|>\n` block because the native template doesn't have a dedicated system role.
- **Generation prompt**: appends `<|turn>model\n` when `add_generation_prompt=true`.

21 regression tests in [`Tests/vMLXTests/ChatTemplateReproTests.swift`](../Tests/vMLXTests/ChatTemplateReproTests.swift) pin this behavior.

## Cache handling — mixed sliding + full attention

Gemma 4 interleaves `"full_attention"` and `"sliding_attention"` per-layer via `config.layer_types[i]`. `newCache(parameters:)`:

```
for i in 0 ..< firstKvShared {
    if layerType == "full_attention" {
        if parameters?.maxKVSize != nil {
            RotatingKVCache(maxSize: maxKVSize, keep: 4)
        } else {
            KVCacheSimple()
        }
    } else {  // sliding_attention
        RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
    }
}
```

Note: shared-KV layers (E2B/E4B) don't get cache entries at all — they reuse earlier layer's state via the `sharedKV` parameter in the attention forward.

**L2 disk cache interaction**: the sliding-attention `RotatingKVCache` layers break the v2 TQDiskSerializer round-trip (no `.rotating` kind tag). The central guard in `CacheCoordinator.storeAfterGeneration` detects these layers and **skips disk + memory tiers entirely**. Paged L1 still works because `extractLayerData` returns nil for rotating layers upstream. See [`docs/CACHE-MATRIX.md`](CACHE-MATRIX.md) for the full rationale.

**Until a future format bump**: Gemma 4 users get paged L1 only. Multi-turn cache reuse works, but cross-session persistence doesn't. Multi-turn chat stays warm as long as the model is loaded.

## fp32 SDPA upcast (vmlx #52)

Gemma 4's `finalLogitSoftcapping` is `tanh(logits / cap) * cap`. On fp16 with long contexts, attention scores can exceed the fp16 ±65504 limit during the softmax, feeding an `inf` into softcap → NaN → all-`<pad>` output. Python v1.3.29 fixed it for Python; Swift mirrors the fix at three SDPA call sites:

1. `vMLXLLM/Models/Gemma4Text.swift:313-329` — text-only decoder SDPA
2. `vMLXVLM/Models/Gemma4.swift:567-582` — VLM text-tower SDPA
3. `vMLXVLM/Models/Gemma4.swift:318-329` — vision tower SDPA

All three upcast Q/K/V to fp32 before `scaledDotProductAttention`, then cast the result back to the original dtype. No-op on bf16 models since attention scores rarely overflow bf16 range at normal context lengths.

## KV-share trick (E2B / E4B)

Shared-KV layers:
1. `useKEqV = config.attentionKEqV && !isSliding` → when true, drop the `v_proj` and reuse `K` as `V`
2. `sharedKV` tuple threaded through the forward pass so shared layers reuse their parent's K/V tensors instead of computing fresh ones
3. No cache entry created for shared layers — they pull from the parent layer's cache

This saves ~33% RAM + compute on E2B/E4B at the cost of a small accuracy hit. 26B/31B never set `numKvSharedLayers`, so this code path stays dormant for those variants.

## Per-layer input gating (E2B / E4B only)

E2B/E4B have additional per-layer modulation: `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`. These are optional `@ModuleInfo(key: ...)` fields. When `config.hiddenSizePerLayerInput > 0`, they're instantiated and applied in the forward pass.

## MoE path (E4B / 26B)

Layout is the **sibling router + experts pattern** (not the Qwen / Mistral text-path SwitchGLU pattern). Each layer with `enableMoeBlock=true` has:
- `Gemma4Router` (sibling) — computes top-K indices + weights
- `Gemma4Experts` (sibling) — wraps `switch_glu: SwitchGLU` and calls `switchGLU(xFlat, indicesFlat)` with the router output

**Flash MoE Phase 2b**: the Gemma 4 conformance in `Gemma4Text.swift` and `vMLXVLM/Models/Gemma4.swift` installs a `FlashMoESwitchGLUShim` inside `Gemma4Experts.flashMoeSwitchGLUShim`. Router stays native (Gemma 4 has its own softmax + top-K logic). Swap happens via the `.gemma4` layout in `FlashMoELayer.replaceSwitchGLU(with:)`.

## Vision tower (VLM)

- Patch embedding: 14×14 patches → MultimodalEmbedder projection → text decoder input
- Pooling kernel: default 3×3 → 9 tokens per 3-patch grid block
- Image token: `<|image|>` (not `<image>` — this is the exact bug that broke mlx-vlm's `prepare_inputs` token split on community quants). Swift bypasses `prepare_inputs` and injects image embeddings directly via `imageTokenId` masking.
- Grid position encoding: 2D multidimensional RoPE with separate θ per axis

## Quantization paths

- **JANG repack** (`jang_config.json` present): `LLMModelFactory` merges `jang_config.json` metadata into `config.json` before decoding. `mxtq_bits` is a **per-module-type dict** in JANG (`{attention, routed_expert, shared_expert, ...}`); Swift flattens by reading the `routed_expert` key. A fallback that takes the first key would be safer if future JANG configs use different names.
- **mlx-community 4-bit**: standard safetensors with per-group scales. `sanitizeWeights` strips VLM prefixes (`model.language_model.X → model.X`) and remaps expert naming (`.switch_mlp. → .experts.switch_glu.`) to match the Swift module tree.
- **mxtq packed**: Swift uses the vendored MXTQHadamard path, seeded from `jang_config.mxtq_seed`. PRNG is POSIX `drand48`, matching Python `jang_tools` after the 2026-04-13 fix.

## Known gaps / follow-ups

1. **L2 disk cache**: sliding-attention layers bypass it until TQDiskSerializer gains a `.rotating` kind with proper ring-buffer metadata round-trip (format v3).
2. **gemma3n_text model_type** is not in `ModelTypeTable` — falls to bronze tier. Low-priority; add the entry if E2B/E4B text-only variants land with that exact string.
3. **Chat template fallback is not ideal**: system-role collapse loses nuance vs the native template. Only matters for users with heavy system-prompt customization.
4. **JANG `mxtq_bits` routing key** is hardcoded to `routed_expert`. If future JANG configs rename it, we silently fall back to the scalar default. Add a first-key fallback with a warning log.

## Smoke test before shipping a Gemma 4 refactor

1. `swift build` clean on debug
2. `swift test --filter "ModelFactoryRegistrationTests"` — factory entries still registered
3. `swift test --filter "ChatTemplateReproTests"` — all 21 template cases pass
4. `swift test --filter "CacheCoordinatorRotatingGuardTests"` — rotating-cache guard still fires for the mixed-layer-types config
5. **Live test on Mac Silicon**:
   - Load `mlx-community/gemma-4-e2b-it-4bit` — must produce coherent output (not `<pad>` tokens)
   - Load a Gemma 4 VLM variant with `<|image|>` in the prompt and a real image — image embedding path must work
   - Multi-turn chat, 5 turns — should work with paged L1 (no L2 disk persistence expected)

If any of these fail, **do not release** — this is the same bug class that caused v1.3.51.
