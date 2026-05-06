# vMLX Runtime Examples

Per-model end-to-end Swift examples that exercise the full vMLX runtime stack — TurboQuant KV cache encode/decode, prompt-level disk cache, SSM re-derive, JANGTQ repack, chat encoder + reasoning parser + tool processor.

These mirror what `vmlxctl serve` does internally, but as small standalone executables you can `swift run` to verify a bundle and inspect cache stats turn-by-turn.

## Targets

| Target | Bundle | Architecture | Reasoning | Tool format |
|---|---|---|---|---|
| `DSV4FlashRuntime` | `DeepSeek-V4-Flash-JANGTQ` | MLA + HSA/CSA/SWA tri-mode + mHC | `<think>` | DSML |
| `LagunaRuntime`    | `Laguna-XS.2-JANGTQ` | per-layer head SWA + full + 256-expert MoE | `<think>` (laguna_glm_thinking_v5) | GLM-4 |
| `Mistral3Runtime`  | `Mistral-Medium-3.5-128B-JANGTQ` | dense MLA + Pixtral VL | none | Mistral |

All three take an optional bundle path argument and default to `~/.mlxstudio/models/_bundles/<name>`.

## Build & run

```bash
cd /Users/eric/vmlx/swift

swift run DSV4FlashRuntime ~/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ
swift run LagunaRuntime    ~/.mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ
swift run Mistral3Runtime  ~/.mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ
```

## Cache stack defaults

`RuntimeShared.makeLoadOptions(...)` sets the same flags the production server ships with:

| Layer | Flag | Default | What it does |
|---|---|---|---|
| KV cache encode | `enableTurboQuant` | true | Compresses prefix to TurboQuant encoded form once prefill ends; window stays float for low-latency append. |
| Memory prefix cache | `enablePrefixCache` | true | Multi-turn within-process reuse. |
| L1 disk cache | `enableDiskCache` | true | Whole-prefix shards survive process restart. |
| BlockDiskCache storage | `enableBlockDiskCache` | **false** | Storage primitive exists, but block-level fetch/store is not attached to CacheCoordinator yet. |
| SSM companion cache | `enableSSMCompanion` | true | Hybrid (Mamba + attention) state cache. |
| **SSM re-derive (async)** | `enableSSMReDerive` | **true** | After thinking-mode turn ends, runs a fresh prompt-only forward to align the SSM state to the stripped-prompt cache key — saves a full re-prefill on the next turn. |

Examples keep `enableBlockDiskCache=false` so output reflects the production truth: the live persistent tier today is prompt-level `DiskCache`; `cacheStats().blockDisk.wired` reports `false` until block-level integration lands.

## What the examples verify

For each target:

1. **JANGTQ load** — bundle parses `jang_config.json`, swaps Linear → JANGTQDenseLinear, reads `jangtq_runtime.safetensors` sidecar (codebook + Hadamard signs).
2. **Chat encoder + reasoning parser** — splits `<think>...</think>` from final content; `RuntimeShared.assertNoLeak` asserts no tag leaks.
3. **Tool dispatch** — DSML / GLM-4 / Mistral parser routes tool blocks to `ToolCall` events and strips them from `content`.
4. **TurboQuant KV cache** — exercised on long-context recall test (Mistral 3.5 needle-in-haystack); confirms compress + decode round-trips correctly.
5. **Prefix + disk cache** — multi-turn block prints cache stats per turn; in-process prefix hits and prompt-level disk-cache stats are visible in `cacheStats()`.

## Tips

- Set `VMLX_LOG_CACHE=1` to get per-turn cache trace logs.
- Swift DSV4 uses `DSV4LayerCache` for HSA/CSA pool state. Python's `DSV4_POOL_QUANT` pool-cache variant is not active in Swift.
- The `--bundle` arg is positional `argv[1]`; pass any other path to run against a custom location.
