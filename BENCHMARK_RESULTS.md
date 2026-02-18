# vLLM-MLX Benchmark Results

**Model:** Qwen3-Coder-Next-80B-6bit  
**Date:** 2026-02-04  
**Hardware:** Mac Studio M3 Ultra, 256GB RAM  

## Recommended Startup

```bash
# For small-medium contexts (<10k tokens)
vllm-mlx serve ~/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --continuous-batching \
  --cache-memory-mb 8192 \
  --port 8092

# ⚠️ DO NOT USE: --use-paged-cache (crashes)
# ⚠️ KNOWN ISSUE: Server crashes at ~20k+ context
```

## Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Generation (TPS)** | 50-68 t/s | Consistent |
| **Prompt Processing (PPS)** | 1,100-1,600 t/s | Scales with context |
| **Concurrent Throughput** | ~107 t/s | 3 parallel requests |
| **Max Stable Context** | ~11k tokens | vllm-mlx crashes above this |

## Context Scaling (vLLM-MLX Server)

| Context | Prompt Tokens | Time | PPS | Status |
|---------|--------------|------|-----|--------|
| ~2k | 1,906 | 1.6s | 1,232 t/s | ✅ OK |
| ~11k | 11,406 | 7.3s | 1,578 t/s | ✅ OK |
| ~22k+ | — | — | — | ❌ CRASH |

**Root cause:** vLLM-MLX server has memory/batching bugs at large contexts.

## Context Scaling (MLX Direct) ✅ STABLE

For large contexts, use `mlx_lm.generate` directly:

| Context | Prompt Tokens | Time | PPS | TPS |
|---------|--------------|------|-----|-----|
| ~2k | 1,906 | 1.7s | **1,135 t/s** | ~50 |
| ~11k | 11,406 | 7.3s | **1,569 t/s** | ~50 |
| ~19k | 18,906 | 13.2s | **1,433 t/s** | ~50 |
| ~24k | 23,906 | 16.9s | **1,418 t/s** | ~50 |
| ~49k | 48,906 | 42.0s | **1,163 t/s** | ~50 |

### Multi-Turn Conversation Pattern

As context grows in a multi-turn chat:
- **PPS stays consistent:** 1,100-1,600 t/s regardless of context size
- **TPS stays constant:** ~50 t/s for generation
- **Total time scales linearly:** ~1s per 1k tokens of context
- **50k context:** ~42 seconds to process prompt

**Conclusion:** MLX handles 50k+ contexts fine. vLLM-MLX server crashes above ~20k.

## Generation Speed (TPS)

| Tokens | Time | TPS |
|--------|------|-----|
| 50 | 1.19s | 41.9 t/s |
| 100 | 1.51s | 66.2 t/s |
| 200 | 2.97s | 67.4 t/s |
| 500 | ~7.5s | ~67 t/s |

## Cache Reuse

| Request | Time | Speed | Note |
|---------|------|-------|------|
| 1 (cold) | 1.08s | 18.4 t/s | First request |
| 2-5 (cached) | ~1.00s | ~20 t/s | ~7% speedup |

Cache reuse benefit is modest (7%) — less than expected.

## Concurrent Batching

- 3 parallel requests: **107.7 t/s** throughput
- 5 parallel requests: ~similar

## Known Issues

1. **`--use-paged-cache`** — Crashes immediately, don't use
2. **Large contexts (20k+)** — Server crashes, use MLX direct instead
3. **Process backgrounding** — Server may die if not run with `nohup` or in foreground

## Recommendations

| Use Case | Recommendation |
|----------|---------------|
| Short contexts (<10k) | vLLM-MLX server works fine |
| Long contexts (10k-50k+) | Use MLX direct or LM Studio |
| Production | Consider LM Studio for stability |
| Development/testing | vLLM-MLX OK for quick tests |
