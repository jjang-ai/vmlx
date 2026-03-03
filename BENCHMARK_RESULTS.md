# vLLM-MLX Benchmark Results

**Model:** Qwen3-Coder-Next-80B-6bit
**Date:** 2026-02-04 (original), 2026-03-02 (updated after stability fixes)
**Hardware:** Mac Studio M3 Ultra, 256GB RAM

## Recommended Startup

```bash
# Standard mode with all features (paged cache, prefix cache, continuous batching)
vllm-mlx serve ~/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --continuous-batching \
  --port 8092

# Simple mode (single user, max throughput)
vllm-mlx serve ~/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --port 8092
```

> **Note:** Paged cache is now ON by default. The crashes reported in the original benchmarks
> (Feb 2026) were fixed in v0.2.6 — Metal GPU timeout prevention, memory-aware cache eviction,
> and hybrid model cache reconstruction.

## Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Generation (TPS)** | 50-68 t/s | Consistent across context sizes |
| **Prompt Processing (PPS)** | 1,100-1,600 t/s | Scales with context |
| **Concurrent Throughput** | ~107 t/s | 3 parallel requests |
| **Max Stable Context** | 50k+ tokens | Fixed in v0.2.6 (was ~11k) |
| **Cache Speedup** | 5-8.5x | With paged + prefix cache |

## Context Scaling

| Context | Prompt Tokens | Time | PPS | Status |
|---------|--------------|------|-----|--------|
| ~2k | 1,906 | 1.6s | 1,232 t/s | OK |
| ~11k | 11,406 | 7.3s | 1,578 t/s | OK |
| ~19k | 18,906 | ~13s | ~1,433 t/s | OK |
| ~24k | 23,906 | ~17s | ~1,418 t/s | OK |
| ~49k | 48,906 | ~42s | ~1,163 t/s | OK |

### Multi-Turn Conversation Pattern

As context grows in a multi-turn chat:
- **PPS stays consistent:** 1,100-1,600 t/s regardless of context size
- **TPS stays constant:** ~50 t/s for generation
- **Total time scales linearly:** ~1s per 1k tokens of context

## Generation Speed (TPS)

| Tokens | Time | TPS |
|--------|------|-----|
| 50 | 1.19s | 41.9 t/s |
| 100 | 1.51s | 66.2 t/s |
| 200 | 2.97s | 67.4 t/s |
| 500 | ~7.5s | ~67 t/s |

## Cache Reuse (with Paged + Prefix Cache)

With paged cache and prefix cache enabled (default in v0.2.6+):
- **5-8.5x speedup** on cache hits for shared prompt prefixes
- KV cache quantization (Q4/Q8) reduces cache memory 2-4x

## Concurrent Batching

- 3 parallel requests: **107.7 t/s** throughput
- 5 parallel requests: ~similar

## Previously Known Issues (All Fixed)

1. ~~`--use-paged-cache` crashes~~ - Fixed in v0.2.6 (Metal GPU timeout prevention)
2. ~~Large contexts (20k+) crash~~ - Fixed in v0.2.6 (memory-aware cache, chunked prefill)
3. **Process backgrounding** - Server may die if not run with `nohup` or in foreground (use vMLX panel for managed lifecycle)
