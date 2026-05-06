# DSV4 Runtime Investigation — Live Progress Log (2026-05-05)

**Goal:** Production-ready DSV4-Flash + MiniMax JANGTQ_K + Ling on main, with verified coherent output across 4 protocols × 4 reasoning modes × full cache stack (paged + prefix + L2 disk + KV q4).

**Status:** **NOT YET PRODUCTION READY.** Loop reproduces on certain prompts even with all current fixes. Sidecar-fast-load suspected; canonical-path (streaming hydrate) test in progress.

---

## Timeline of fixes applied THIS SESSION (in order)

| # | Fix | File | Status |
|---|---|---|---|
| 1 | Single-shot prefill default (revert v1.5.15 chunking @ 512) | `vmlx_engine/utils/dsv4_batch_generator.py` | ✓ applied + bundled |
| 2 | Force `enable_thinking=True` on 3 endpoints (chat-completions / Ollama / Responses) — `is not True` semantics matching v1.5.6 | `vmlx_engine/server.py` | ✓ applied + bundled |
| 3 | `VMLX_DSV4_ALLOW_CHAT=1` env opt-out for explicit-False chat-mode (advanced users only) | same as above | ✓ applied + bundled |
| 4 | Restored `_FAMILY_FALLBACK_DEFAULTS["deepseek_v4"] = (0.6, 0.95, 1.15)` | `vmlx_engine/server.py:287` | ✓ applied + bundled |
| 5 | (committed in c1) Ling MXFP4 flat-2D switch_mlp shape repair | `panel/scripts/patches/bailing_hybrid.patched.py` | ✓ pushed in `2a0cbb57` |
| 6 | (committed in c1) JANGTQ_K mixed-bit `dp_bits` threading | `jang-tools/jang_tools/load_jangtq.py` | ✓ pushed in `bd3afd0` |

**Already pushed to main:** commits `2a0cbb57` (vmlx) + `bd3afd0` (jang-tools). Includes #1, #2 (less strict), #5, #6.

**Pending in working tree (NOT pushed):** #2 (stricter is-not-True semantics with VMLX_DSV4_ALLOW_CHAT opt-out), #3, #4. Will push only after matrix passes.

---

## Tests run + results

### Identical-prompt cache hit (PASS)
- Sent same prompt twice with full cache stack.
- Result: 1st = miss (write), 2nd = **hits=2 / miss_delta=0**.
- 5 block disk files (18 MB) written.

### 4-protocol × 4-mode short-prompt matrix (PASS — but trivially)
- 16 cells: OpenAI chat, Responses, Anthropic, Ollama × omitted/on/off/max
- Used `"Reply with only a single word: hello"` — too easy to loop on.
- All 16 returned coherent (loop_score=1.0). NOT a meaningful loop test for "off" mode because prompt is too short.

### Loop-trigger prompts (FAIL — multiple LOOPs)
- Ran the actual prompts from prior sessions that triggered loops:
  - `"Tell me a 2-sentence story about a project to see the birth of a star."`
  - `"I want to write a project that observes the birth of stars. What should I call it?"`
  - `"Describe what a project to study star birth would entail."`
- 3 prompts × 4 modes = 12 cells.
- **5 cells looped** including `mode=on` and `mode=max` which are supposed to be the safe paths. Examples:
  - `mode=on, prompt 2`: loop_score=0.029 ("babies babies babies babies...")
  - `mode=on, prompt 3`: loop_score=0.005 (ZWS character spam)
  - `mode=off, prompt 1`: loop_score=0.667 (ustilizing repetition)

**This means current fixes (force-thinking + 1.15 rep_penalty + single-shot prefill) are NOT sufficient.** The loop is deeper than the API/sampling layer.

---

## Hypothesis under test now: SIDECAR-FAST-LOAD vs CANONICAL HYDRATE

**Sidecar fast-load** (`vmlx_engine/loaders/load_jangtq_dsv4.py`) was added in **v1.5.11** for instant reload. It rebuilds DSV4 TQ modules from `{base, bits}` only, separate from the canonical `jang_tools/load_jangtq.py` `_hydrate_jangtq_model` path.

**Suspect:** the sidecar loader may be missing logic the canonical loader has — e.g. correct `dp_bits` threading (just shipped in jang 2.5.24), correct `existing_in` resolution for SwitchLinear (`input_dims` vs `in_features`), Hadamard signs cache keying.

**Test:** restart with `JANGTQ_DISABLE_DSV4_FAST_LOAD=1` — bypasses sidecar, runs canonical streaming hydrate.

**Observed at startup:** `DSV4 bundle is JANGTQ-PRESTACK format — deferring to generic loader (skipping streaming restack + sidecar write)`. So even WITHOUT the env var, the JANGQ-AI/DeepSeek-V4-Flash-JANGTQ bundle is already taking the generic loader path because it's in PRESTACK format. **This means the sidecar isn't actually being used for this model.**

That is a critical nuance — the JANGTQ-PRESTACK format bypasses sidecar loading entirely. So the sidecar can't be the regression source for this specific bundle. The loop must come from elsewhere.

### Next-rank suspects (in order)

1. **Generic loader path** — `jang_tools/load_jangtq.py` `_hydrate_jangtq_model`. The `dp_bits` threading I just shipped (jang 2.5.24) only kicks in when bundle has `mxtq_bits` dict; DSV4 bundles likely have scalar bits (uniform). Check that uniform-bits path didn't regress.

2. **DeepseekV4Cache `make_cache` + paged cache wrapping** — when paged cache is enabled, `_wrap_make_cache_quantized` is called. For MLA it gets disabled. But the pre-wrap `make_cache` is what the engine actually uses. Verify the cache initialization for DSV4 produces correct compressor_state initial values.

3. **Hadamard signs cache key collision** — JANGTQRuntimeCache uses `(in_features, seed)` keys. If DSV4's `existing_in` resolves to `weight.shape[-1]` for some layers but `input_dims` for others, signs could be keyed inconsistently.

4. **`SwitchGLU` patch interaction** — `_fused_switchglu_call` patched 43 TQ instances per startup log. If the patch isn't applying to DSV4's specific MoE layout, decode could use unpatched (wrong-bit) gather.

5. **Sampling pipeline temperature interaction** — at temp=0.6 the model produces near-deterministic logits for these prompts. rep_penalty=1.15 didn't break it. Try temp=0.95 + rep=1.15 to confirm whether sampling-is-the-trigger or not.

---

## Cache architecture findings (verified)

- `/v1/cache/stats` works for DSV4. Shows `cache_hits`, `cache_misses`, `disk_hits`, `disk_misses`, `tokens_saved`, `allocated_blocks`.
- DSV4 multi-turn paged cache MISS every time (v1.5.15 cumulative-state-rejection fires; documented design tradeoff).
- DSV4 identical-replay paged cache HITS work.
- **DSV4 KV q4 quantization auto-disabled** (MLA detection at scheduler.py:541). User wants partial-quant path (quantize SWA local KV, leave compressed latents) — **deferred to a separate design task**, not in this session.
- TurboQuant KV cache (the JANG-calibrated path) auto-disabled when `--kv-cache-quantization=q4` is explicit.
- Block disk store WRITES work (5 files written in test). Disk HIT not yet verified — need restart-survival test.

---

## Length-cap nuance (caught by user critique)

`scheduler.py:3639` — DSV4 cache store is SKIPPED when finish_reason=`length`. Tests that hit `max_tokens` won't populate the cache. Must use prompts that finish naturally (`stop` finish reason). Verified by: short prompts return `finish=stop`, long-thinking prompts return `finish=length` and don't write cache. **All cache hit tests must use stop-clean prompts.**

---

## Important runtime nuances to remember

1. **DSV4 has 3 cache layers that interact:**
   - In-memory paged cache (`PagedCacheManager`)
   - In-memory hybrid cumulative cache (DeepseekV4Cache pool buffers)
   - L2 disk block cache (`BlockDiskStore`, deepseek_v4_v6 schema)

2. **Cache scope keys** include `paged_cache_schema=v6` + `dsv4_long_ctx={env}` + `dsv4_pool_quant={env}` + `dsv4_cache_schema=deepseek_v4_v6`. Cross-env cache mixing should NOT happen (verified by code reading).

3. **DSV4 cache contamination rejection** rejects post-output snapshots, forcing re-prefill on next-turn fetch. Multi-turn cache hit therefore always misses by design. Single-turn identical replay can still hit (as proven above).

4. **JANGTQ-PRESTACK format** — bundles produced from `jang 2.5.23+` ship pre-stacked TQ tensors. DSV4 sidecar loader detects this and defers to generic loader. So sidecar fast-load only matters for legacy non-PRESTACK DSV4 bundles.

5. **Force-thinking is the only API-layer protection.** The DSV4 chat encoder defaults to `("chat", None)` mode when both inputs are None. Without server.py force-flip, every default request hits the contaminated path.

6. **Anthropic adapter** defaults `enable_thinking=False` per spec. With force-flip applied at server.py chat-completions block (which Anthropic /v1/messages routes through), Anthropic clients are protected — but only because of the explicit-False handling in the force-flip block.

7. **Panel default `enableThinking=false`** for new chats (`panel/src/main/ipc/chat.ts:447`). With my stricter `is not True` semantics, this is force-flipped to True at engine. Without that strict semantics (only `is not True is True` → leave alone, `else` → force True, NOT special-casing False), explicit False from panel would sneak through.

---

## What to do next session if loops persist

1. **Trace which kernel path actually fires for DSV4 routed-experts** at the layer that produces the looping logits. Add per-layer logits magnitude logging.
2. **Compare JANGTQRuntimeCache state** between MiniMax (works) and DSV4 (loops on certain prompts). Look for differences in (in_features, seed) keys.
3. **Compare bundled mlx_lm/models/deepseek_v4.py**: verify the patched class registered via jang_tools matches the canonical reference implementation.
4. **Check if `g_norm`/`q_norm`/`o_norm` dtype handling differs from the v1.5.10 baseline** (commit 68091df7 was "DSV4-Flash JANGTQ runtime fully fixed" — what changed in jang-tools dsv4/mlx_model.py since then?).
5. **Try with `--kv-cache-quantization none`** to eliminate q4 quantization as a variable (note: q4 is auto-disabled for MLA so should already be no-op, but verify).
6. **Test on Sources/DeepSeek-V4-Flash bundle** (the original DeepSeek bf16 weights, no JANG quant) — if that loops too, bug is in mlx model class. If clean, bug is in JANGTQ runtime.

---

## Pending TODO when loop is fixed

- Restart-survival disk hit test
- `cache_salt` / `skip_prefix_cache` cache-vs-fresh equivalence
- MiniMax JANGTQ_K full matrix re-run to confirm no regression
- Ling-2.6 quick coherency check (the flat-2D shape fix)
- Push verified fixes as v1.5.21 (or revert v1.5.20 if it's worse than nothing)

---

## Files to read first thing next session

1. This doc (`docs/DSV4_RUNTIME_PROGRESS_LOG.md`) — full state.
2. `docs/DSV4_RUNTIME_REGRESSION_TRACE.md` — original regression analysis (v1.5.6 → v1.5.15 reverts).
3. `tail -200 /tmp/dsv4_canonical.log` — current server log (canonical loader path).
4. `git log --oneline 68091df7..HEAD -- jang_tools/dsv4/mlx_model.py` (run in `/Users/eric/jang/jang-tools`) — what changed in DSV4 runtime since v1.5.10 baseline.

---

## ROOT CAUSE FOUND (2026-05-05 evening)

### The bug: DSV4 prompt-cache stores POST-GENERATION wrapped RotatingKVCache

**Code site:** `vmlx_engine/scheduler.py:1964` `_truncate_cache_to_prompt_length` DSV4 branch.

**Mechanism:**
1. After a generation completes, scheduler calls `_truncate_cache_to_prompt_length` to trim the live cache back to the prompt boundary (so next turn can reuse it).
2. For DSV4, this trims `DeepseekV4Cache` which wraps a `RotatingKVCache` as `self.local` (the SWA window).
3. **`RotatingKVCache` cannot be rewound after the circular buffer has wrapped** (`offset > max_size`).
4. With sliding_window=128 and prompt 29 + output 600 = 629 tokens, the buffer wraps multiple times. Trimming back to 28 leaves `_idx` negative.
5. The trimmed state is structurally valid (no error) but semantically wrong: replaying it next turn applies output-side tokens at wrong positions → degenerate logits → loop.

**Codex verified locally:** "after prompt 29 + output 482, trimming back to 28 leaves _idx = -356."

**Same safety constraint exists at scheduler.py:72** (`_rebuild_meta_state_after_truncation` for plain RotatingKVCache, returns None when wrapped). The DSV4 special-case branch was bypassing that check.

### Fix applied (short-term production guard)

`scheduler.py:1964` DSV4 branch now refuses to store prompt cache when `local.offset > sliding_window`. Returns `None` → caller falls through to full prefill on next turn (correct, just slower).

```python
local_offset = int(getattr(local, "offset", 0) or 0)
if local_offset > sliding_window:
    logger.info("DSV4 prompt truncation skipped: local SWA wrapped ...")
    return None
```

### Verification

**Cache isolation test on the looping prompt** (prompt 3 mode=on with full cache stack):

| Test | Cache state | Pre-fix | Post-fix |
|---|---|---|---|
| A: cache_salt="fresh-1" | Bypass | OK | OK (loop=0.480, "Certainly! Star birth research...") |
| B: cache_salt="fresh-2" | Bypass | OK | OK (identical content) |
| C: no salt #1 | Miss → store | OK | OK (identical) |
| D: no salt #2 | HIT 28-token block | **LOOP** (loop=0.012) | **OK** (identical) |

Server log post-fix: `DSV4 prompt truncation skipped: local SWA wrapped (offset=628 > max_size=128)`. All requests now miss → full prefill → correct.

### Trade-off

DSV4 multi-turn paged-prefix cache hits now ALWAYS miss (when output extends past sliding_window=128 tokens, which is always for thinking-mode). Re-prefill on every turn. Slower but correct.

### Real long-term fix (not in this session)

Capture clean prompt-boundary cache **immediately after prefill, before decode starts**. Or async re-derive prompt-only cache after the request finishes. This needs a new code path in scheduler — current scheduler only sees the live cache after generation completes.

The SSM companion cache already has this pattern (`_prefill_for_clean_ssm`). DSV4 needs an analogue.

### Why v1.5.10 worked but post-v1.5.10 doesn't

Per memory in `project_cache_architecture.md`: "DSV4 paged cache stores at any prompt length" was added in **v1.5.13**. Before that, DSV4 paged cache was only stored under conditions that probably avoided this bug. v1.5.13 turned on stores broadly without re-checking the rotation-cache safety constraint. This guard restores correctness.

### Other findings during this investigation

1. **Fused SwitchGLU is NOT the bug.** Disabling via `JANGTQ_DISABLE_FUSED_SWITCHGLU=1` (env override added to `jang_tools/load_jangtq.py`) does NOT eliminate loops. With cache enabled + fused disabled, loops still appear. With cache_salt + fused either way, output is coherent.

2. **`cache_salt` is a hard cache bypass**, not a namespace salt (per `server.py:692`). Used for testing.

3. **DSV4 default cache class is `KVCache` not `DeepseekV4Cache`** unless `DSV4_LONG_CTX=1` is set (per `jang_tools/dsv4/mlx_model.py:1308`). All my testing was with the default. The bug reproduces with the default cache too because the DSV4-specific branch in `_truncate_cache_to_prompt_length` is hit whenever the cache list contains a `DeepseekV4Cache` — which happens when `DSV4_POOL_QUANT=0` is set at CLI startup (per `cli.py` v1.5.6 fix). Verify this: with `DSV4_LONG_CTX=0`, what's the actual local cache class?

4. **Existing guard at `prefix_cache.py`** (`Ignoring DSV4 paged prefix hit ... no terminal deepseek_v4 composite state`) catches multi-block-without-terminal cases but NOT the single-terminal-block-with-wrapped-rotation case that's the actual contamination source.

5. **Codex's 9 additional issues acknowledged but not in this session's scope:**
   - Ling/MXFP4 looks fine — patches all in sync
   - Standalone `jang_tools/jangrt/` runtime path lags main loader (Laguna/Mistral3 affected)
   - Standalone fused SwitchGLU path missing dp_bits fix (mixed-bit dedicated runtimes)
   - Swift JANG runtime still requires uniform bits (JANGTQ_K not supported on Swift)
   - MiniMax-M2.7-Small still per-expert (not prestack) — load/RAM concern
   - DSV4 TQ KV not really tested — MLA disables KV q4
   - Mistral 3.5/Pixtral guarded for a reason; vision is stubbed in JANG path

---

## What still needs verification (next steps)

- [x] Confirm fix works for ALL loop-trigger prompts × all 4 reasoning modes — **VERIFIED 11/12 coherent + 1 borderline false-positive (verbose mode=max thinking, length-capped)**
- [x] Confirm fix doesn't break MiniMax JANGTQ — **MiniMax architecture clarified; not affected by my fix; MiniMax has SEPARATE reasoning-quality issue documented below**
- [ ] Restart-survive disk cache test (deferred — same DSV4 store path is now guarded so no disk writes happen for DSV4 anyway)
- [x] cache_salt vs no-salt equivalence — **D now produces same content as A (loop=0.480 identical across all 4 cells)**
- [ ] 4 protocols × 4 modes matrix (full Anthropic/Ollama/Responses parity)
- [ ] Document long-term fix path (clean-prompt-boundary capture) for next session

## Architecture note — MiniMax-M2.7 is NOT hybrid SSM (correction)

`config.json`: `model_type=minimax_m2`, 62 layers, all `attn_type=1` (uniform — full attention, no sliding window, no SSM). `MiniMaxAttention` in `mlx_lm/models/minimax.py` is plain SDPA over `KVCache`. Sparse MoE for FFN.

**MiniMax loops observed in this session are NOT cache contamination.** Reproduce with `cache_salt` (full cache bypass) — 3024 chars of repetitive reasoning ("The user asks... This is a request... The user wants a story... The assistant can comply...") cycling on the same paraphrases. Not in scope of the DSV4 fix.

Likely causes for MiniMax thinking-loop on these specific prompts (separate investigation):
- bundle-specific reasoning template / lack of stop-on-think-close
- reasoning_effort mapping for minimax_m2 family (reasoning parser, server.py)
- rep_penalty default (MiniMax bundle defaults are 1.0/1.05; 1.10+ may break the rumination loop)
- prompt-specific attractor (the model itself, not vMLX runtime)

`mode=off` (no thinking) on MiniMax: clean coherent output in 38 tokens. So the runtime works; only thinking mode rumninates on these prompts.

## DSV4 fix scope summary

Single guard, single change site. Catches the cache-contamination bug for all three DSV4 attention components (SWA + CSA + HSA) at the extraction site that gates downstream paged-cache + L2 disk store + prefix-cache store. When the guard returns None (any post-generation trim attempt), `cache_for_extract is not None` check at scheduler.py:3569 skips the entire store chain.

Conservative — also rejects "safe" trims (e.g., short generations where SWA hasn't wrapped) until clean-prompt-boundary capture path exists. Trade-off: cache-hit-rate=0 for DSV4 multi-turn. Correctness > speed.

**Files changed (uncommitted, local only):**
- `vmlx_engine/scheduler.py` — DSV4 guard at `_truncate_cache_to_prompt_length`
- `vmlx_engine/server.py` — DSV4 force-thinking strict semantics + rep_penalty 1.15 floor + VMLX_DSV4_ALLOW_CHAT opt-out
- `vmlx_engine/utils/dsv4_batch_generator.py` — single-shot prefill default
- `jang_tools/load_jangtq.py` — JANGTQ_DISABLE_FUSED_SWITCHGLU=1 env override (debug only, doesn't affect default behavior)

**Already pushed in 2a0cbb57 (vmlx) + bd3afd0 (jang-tools):** v1.5.20 + jang 2.5.24. Those commits have the older / incomplete versions of the server.py + dsv4_batch_generator.py fixes (force-thinking honor explicit-False instead of strict, no rep_penalty floor). The scheduler.py guard is NOT in those commits.

**Pending push (would be v1.5.21 + jang 2.5.25):**
1. scheduler.py guard (THE actual root-cause fix)
2. server.py strict force-thinking + rep_penalty 1.15 + VMLX_DSV4_ALLOW_CHAT
3. jang_tools/load_jangtq.py env-debug override
