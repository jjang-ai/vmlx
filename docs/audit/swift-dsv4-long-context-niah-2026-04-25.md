# Swift DSV4 Long-Context — 12K NIAH Validation (2026-04-25)

Live test results for §415-§417 long-context Compressor + Indexer
attention forward integration on DSV4-Flash JANGTQ. Mirrors and pins
the parallel Python NIAH validation result.

> **Source**: full live log at
> `/Users/eric/jang/research/SWIFT-DSV4-LONG-CTX-NIAH-2026-04-25.md`.
> Reproduced here so future Swift readers can find it alongside the
> code without crossing repos.

---

## Headline result

**Swift matches Python on long-context retrieval at 12K tokens.**

| | Python (`jang_tools.load_jangtq_model`) | Swift (`vmlxctl bench-direct`) |
|---|---|---|
| **Tokens** | 11,777 (chat-template wrapped) | 11,779 |
| **Output** | `'BLUE-OCTOPUS-42'` | `**BLUE-OCTOPUS-42**` |
| **Wall** | 75 s | 72 s (3.85 load + 67.28 prefill + 0.93 decode) |
| **Decode rate** | n/a in current logs | **21.60 tok/s** |
| **Needle retrieval** | ✅ | ✅ |

Hardware: Mac Studio M3 Ultra, 256 GB unified memory.

## What this validates

1. **§415 — Indexer `@ModuleInfo` wire-up.** Indexer weights load
   correctly at 43-layer 12K context. Bundles ship
   `layers.N.attn.indexer.*` paths; Swift sanitize routes them to
   `model.layers.N.self_attn.indexer.*` and the loader builds
   QuantizedLinear at the right bits per §410.

2. **§416 — `DSV4RoPE.manual(_:positions:)` per-token cos/sin.** The
   Compressor pool stores keys at positions `[0, 4, 8, …, 11772]`
   (≈3000 distinct phases at `compress_ratio=4`, ≈90 phases at
   `compress_ratio=128`). `MLXFast.RoPE(_:offset:)` is per-batch
   (one phase across all axis-2 entries), so passing the position
   array would have collapsed every pool slot to phase-0 → no
   needle recall. The manual cos/sin path correctly rotates each
   slot.

3. **§417 — `DSV4LayerCache` pool accumulation.** Across prefill
   chunks, `accumulateWindows` buffers the tail until it reaches
   `compress_ratio` boundary, then `updatePool` grows the persistent
   `pooled` tensor. Decode reads the cumulative pool. The 12K NIAH
   answer's correctness implies both branches of the prefill chunking
   work correctly.

4. **Indexer top-k path (gather + bool-mask).** At 12K the dense
   `compMask` would be `(B=1, 1, L_q, P)` with `P` up to ~3000.
   Top-k AND'd into the mask reduces effective P each query
   attends to. Without this Swift would either OOM or produce a
   degraded mask. The correct answer implies both paths fire as
   expected (gather for `L==1` decode, bool-mask for `L>1` prefill).

## Operating recipe

To reproduce on your own machine:

```bash
# 1. Build the release binary (or use the notarized DMG).
cd /Users/eric/vmlx/swift
swift build -c release --product vmlxctl

# 2. Long-context flag MUST be set at process start. Compressor +
#    Indexer attention paths only activate when this env var is
#    truthy at vmlxctl spawn time. Without it, every layer falls
#    through to the bare RotatingKVCache path and long-context
#    coherence past sliding_window=128 tokens degrades.
export VMLX_DSV4_LONG_CTX=1

# 3. Run via `bench-direct` (raw decode loop) OR `chat` (chat-template
#    wrap). Both apply the bundle's chat template — DSV4 is an
#    instruct-only model; raw text without template emits EOS
#    immediately. Documented as the "instruct-model trap" in
#    `feedback_instruct_model_check.md`.
.build/release/vmlxctl bench-direct \
  --model /Path/To/DSV4-Flash-JANGTQ \
  --prompt "$(cat your_long_prompt.txt)" \
  --max-tokens 64
```

**Decode rate stability**: 21.60 tok/s at 12K matches the rate measured
at 67-token (21.47) and 146-token (21.12) prompts. Compressor pool
accumulation does not introduce per-token work as `pooled` grows.

**Prefill cost**: 175 prefill tok/s at 12K. About half the no-flag
short-prompt prefill rate, which is expected because ~half the layers
(every other middle layer) now run the extra Compressor + Indexer
compute. Prefill dominates wall time at long context.

## Side-finding — instruct-model trap

Python NIAH attempt 1 fed raw text WITHOUT chat-template wrap → empty
output (model emitted EOS immediately at 12K, needle never reached
the assistant turn). Attempt 2 wrapped in
`<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>` → PASS.

DSV4 is an instruct/reasoning model. Raw text without an explicit
assistant-turn marker leaves no clear handoff for the model to start
generating from. `vmlxctl bench-direct` and `vmlxctl chat` both
auto-wrap; only manual prompt construction needs to handle this.

## What this does NOT validate

- **32K and beyond**: prefill chunking + RotatingKVCache wrap-around
  cycle counts grow with context. Need a 32K NIAH to exercise the
  cross-buffer accumulation path more thoroughly.
- **Multi-turn long context**: the test was a single-turn 12K NIAH.
  Multi-turn chat with growing context is the next interesting
  regression point (cache invalidation + tool-call interleaving).
- **M4 Max**: this run was on Mac Studio M3 Ultra. M4 Max has shown
  divergent short-prompt output in earlier testing (`session 18:50`)
  — likely an unrelated path issue under investigation, but the
  long-context Compressor/Indexer kernels themselves are
  hardware-agnostic.

## Status of §415-§417

| Section | Description | Validation |
|---|---|---|
| §415 | Indexer @ModuleInfo wire-up | ✅ live 12K NIAH |
| §416 | DSV4RoPE.manual per-token cos/sin | ✅ live 12K NIAH (positional needle recall) |
| §417 | DSV4LayerCache pool accumulation | ✅ live 12K NIAH |

Shipped in `2.0.0-beta.15` (DMG SHA256:
`b8ef5dd65c0ea1487726689c3b830c494a5e56d095de3fa2b4f70ffbdcacc725`).

## Next validation

- 32K NIAH (cross-buffer accumulation stress)
- Prefill curves head-to-head Python vs Swift over 1K–32K
- M4 Max DSV4 short-prompt path investigation (separate from
  long-context Compressor work)
- Re-enable `_dsv4Compiled*` fixtures one-by-one and verify each
  under both modes (no-flag + LONG_CTX=1)
