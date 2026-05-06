# /Users/eric/jang docs review — next actions for vMLX (2026-05-05)

User said: "no don't bother with dsv4 i promise u that its fine go check and read ../jang documentation and logs and shit and code and further what to do."

Read the following from /Users/eric/jang:
- `JANG-PLAN.md` — JANG = quantization method + format + inference path. Tier-based mixed-precision. JANG produces .jang.safetensors that stay quantized in GPU memory.
- `JANG-ISSUES.md` — Open issues + fixed items.
- `INTEGRATION.md` — How apps integrate JANG. Uses `mx.load()` mmap directly for v2 format.
- `codex_dsv4_fixkit/` — Codex's DSV4 root-cause analysis + `convert_dsv4_safe.py` rebuild path.
- `docs/plans/2026-04-27-jangfp-design.md` — JANGFP design (successor to MXTQ; DESIGN ONLY, not yet implemented).
- `docs/JANGPRESS.md` — cold-tier MoE memory policy for >RAM bundles. Already shipping in Swift.
- `jangtq_issues_report.md` — earlier DSV4 JANGTQ runtime issues report.

## What's clear from the docs

The DSV4 long-context drift I was hitting is a **known bundle-quality issue** documented in `codex_dsv4_fixkit/docs/JANGTQ2_RECOVERY.md`:

- DSV4 source experts are FP4. JANGTQ2 dequants FP4→fp32→2-bit MXTQ. **Both a codec change AND a bit-rate cut.**
- DSV4 amplifies small numerical errors via mHC, Sinkhorn, hash routing, attention sinks.
- 2-bit ceiling: documented at `JANGTQ_V3_MIN` ~80% on MMLU-50 vs JANGTQ4 affine ~78%. Uniform 2-bit is the floor.
- Fix path: JANGTQ_V3 mixed-bit plan (hash layers 4-bit, smooth-routed 2-bit, control plane 4-bit floor with 6/8 upgrade).
- v3 path requires running `jang_tools.jang_v3.calibrate` + `importance` + `consolidate` + `budget_solver` to produce a new bundle. **Not a vMLX runtime fix.** A converter rebuild on a different bundle.

So: **DSV4 long-context Stanford-loop is a bundle quality issue, not a vMLX bug.** The user is right that I should not keep flailing on it from the vMLX side. The bundle needs a rebuild via codex_dsv4_fixkit's V3 plan.

## Highest-value vMLX-side actions (NOT DSV4 long-context)

### 1. Format v2.0 fast loader path (per `JANG-ISSUES.md` CRITICAL)

> Current JANG format stores qweights as uint8 (packed). The loader repacks uint8→uint32 on EVERY load. This takes 5-10 minutes for large MoE models (47K tensors). Users cannot use models until repack finishes.

**Status check:** does `vmlx_engine/utils/jang_loader.py` already have a v2.0 fast path? If yes, what does it gate on? If no, this is a 1-day win for any v2.0-formatted bundle (DSV4 / MiniMax / Ling / Laguna would all benefit).

### 2. JangPress integration on Python side (per `docs/JANGPRESS.md`)

JangPress is shipping in Swift (`vmlx-swift-lm`) but the Python side ("vMLX Engine") doesn't have an mmap+`madvise(MADV_DONTNEED)` cold-eviction path. Adding this would let Python users load >RAM bundles (Kimi-K2.6 167GB on 128GB Mac).

### 3. Verify Ling cache reconstruction thread-stream bug (still open)

From earlier this session: Ling JANGTQ paged cache hits succeed but reconstruction throws `"There is no Stream(gpu, 1) in current thread"` and falls back to full prefill. Functional, but loses cache speedup. Fix needs same-thread MLX dispatch (analogous to scheduler `_step_executor`).

### 4. Kimi K2.6 mlx_vlm tokenizer wrapper (failed to load in matrix test)

```
File "mlx_vlm/tokenizer_utils.py", line 208, in __init__
    self.tokenmap = [None] * len(tokenizer.vocab)
AttributeError: TikTokenTokenizer has no attribute vocab
```

mlx_vlm's `TokenizerWrapper` assumes `.vocab` exists. Kimi K2.6 uses TikTokenTokenizer which doesn't expose it. Need a workaround.

### 5. Ling-2.6-flash-JANGTQ2-CRACK Chinese-output investigation

User reported CRACK variant responds in Chinese on minimal prompts. CLI tests: same prompts → English. Possible: panel-side `enable_thinking=false` default routing differently than CLI default. Or the abliteration fine-tune on this specific variant.

### 6. JANGTQ_K mixed-bit on DSV4 sidecar fast-load (Codex item #6)

Generic `load_jangtq.py` has `dp_bits` threading (jang 2.5.24+). DSV4 sidecar fast-load in `vmlx_engine/loaders/load_jangtq_dsv4.py` does NOT. If sidecar fast-load is ever used on a mixed-bit DSV4 bundle, down-proj decodes wrong. Currently DSV4 bundles are uniform-bits so latent, but should be fixed.

### 7. SwitchGLU global monkey-patch fragility (Codex item #7)

`load_jangtq.py` replaces `SwitchGLU.__call__` class-wide. Switching from one model family to another in the same process can leave a stale patched class. Need test: load DSV4 → unload → load MiniMax → load Ling. Verify no cross-contamination.

## Recommendation order

Most impactful for users right now, in order:

1. **#3 Ling cache reconstruction thread fix** — affects current users hitting Ling
2. **#4 Kimi K2.6 tokenizer fix** — Kimi currently completely broken
3. **#1 Format v2.0 fast loader** — affects load time for all JANG bundles
4. **#5 Ling CRACK Chinese-output** — affects current user (the report from earlier)
5. **#7 SwitchGLU monkey-patch fragility** — model-swap bug
6. **#2 JangPress Python** — bigger feature, multi-day

Items #6 + DSV4 V3 rebuild are out of vMLX scope (jang-tools + bundle work).

## What I'm NOT going to chase further this session

Per user directive: stop pushing on DSV4 long-context. The drift is bundle-quality (FP4→2-bit MXTQ stack-up at 384+ tokens), documented as known in codex_dsv4_fixkit. Fix is V3 rebuild not vMLX runtime change.
