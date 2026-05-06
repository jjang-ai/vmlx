# DSV4 + MiniMax + Ling Fix Nuances (2026-05-05)

**Audience:** future-me (or anyone debugging this 6 months from now). Each section is one fix; the goal is to capture *what changed*, *why it works*, *what it breaks*, and *what would resurface the same bug*.

**Status framing — important:** "Coherent" in this doc means single-turn + multi-turn output is correct and reproducible. It does NOT mean "full cache stack production-ready" — DSV4 specifically runs with prefix cache effectively disabled by the truncation guard (re-prefill on every multi-turn request). MiniMax/Ling have functional cache hits but Ling has a known reconstruction-thread bug that fails-safe to full prefill. Production cache-stack readiness requires the long-term clean-prompt-boundary capture work (out of session scope).

---

## FIX 9 — DSV4 partial-MLA KV cache quantization (TQ-default for DSV4)

**File:** `vmlx_engine/scheduler.py` `_quantize_cache_for_storage` DSV4 branch + `_wrap_make_cache_quantized` MLA-disable carve-out for DSV4.
**One-liner:** DSV4 KV q4/q8 is now ENABLED by default (matching every other model). Composite-aware quantizer quantizes only the SWA local RotatingKVCache; compressor/indexer pool buffers (already-compressed latents) stay native.

### What was broken
DSV4 kv quant was auto-disabled for ALL MLA models because naive whole-cache quantization would double-quantize the kv_lora compressed latents (compressor + indexer pool buffers). Codex flagged: "DSV4 SWA+CSA/HSA needs a composite serializer/reconstructor or an explicit 'unsupported' status for TQ KV quant."

### Fix
Composite-aware path in `_quantize_cache_for_storage`:
- Detect `DeepseekV4Cache` by class name
- Quantize ONLY `layer_cache.local` (the inner RotatingKVCache holding plain SWA KV)
- Build `QuantizedKVCache` from local.keys/values via `mx.quantize(..., group_size=group_size, bits=bits)`
- Replace `layer_cache.local = qkv_local` so downstream serialization sees `(quantized_swa, native_compressor, native_indexer)`
- Compressor/indexer state passes through untouched

`_wrap_make_cache_quantized` now treats DSV4 (`_uses_dsv4_cache=True`) as NOT subject to the MLA auto-disable. The auto-disable still fires for other MLA models (DeepSeek V3, Mistral 4) without the composite path.

### What this gives users
- DSV4 prefix cache stores at q4/q8 — 2x-4x memory reduction on the SWA component (the bulk of cache memory)
- Compressed latents (CSA/HSA) stay fp16 — no double-loss
- Symmetric with `kv_cache_quantization=q4` user expectation: works for every family including DSV4

### Trade-off
- Pool buffers stay full precision, so total memory savings are partial vs full quantization
- Reconstruction must dequantize SWA local; `_dequantize_cache_for_use` handles this via the standard QuantizedKVCache path
- Tests not yet added — pin behavior in v1.5.22

### What would resurface
1. Extending the auto-disable to DSV4 family without restoring the composite path
2. Replacing DeepseekV4Cache with a different cache class name that doesn't match `"DeepseekV4Cache" in type(layer_cache).__name__`
3. A future MLA cache that ALSO contains plain KV — needs same composite branch added

---

## FIX 10 — Responses API cache_salt + skip_prefix_cache fields

**File:** `vmlx_engine/api/models.py` `ResponsesRequest`
**One-liner:** Add `cache_salt` + `skip_prefix_cache` fields to the Responses model so cache-bypass works on `/v1/responses`, not just `/v1/chat/completions`.

### What was broken
`ResponsesRequest` had `model_config = {"extra": "ignore"}` which silently dropped any `cache_salt` or `skip_prefix_cache` field a client passed. Cache bypass tests run via Responses API were not actually bypassing cache — they hit cache normally and looked like they worked when they didn't (Codex caught this).

### Fix
Added explicit fields with same semantics as ChatCompletionRequest:
```python
cache_salt: str | None = None
skip_prefix_cache: bool | None = None
```

### Verification next session
Run cache isolation test against `/v1/responses` (not just `/v1/chat/completions`) to confirm both fields propagate to `_compute_bypass_prefix_cache`.

---

## FIX 11 — `thinking_mode="high"` no longer aliases to "max"

**File:** `vmlx_engine/api/models.py` `_normalize_reasoning_alias` (both ChatCompletion and Responses request validators)
**One-liner:** "high" maps to `reasoning_effort="medium"` (verified-stable) instead of "max" (experimental, produces "Plan and Plan ... ( ( ( (" attractor on DSV4 long-form prompts).

### What was broken
`thinking_mode="high"` silently aliased to `reasoning_effort="max"`. DSV4 max-effort routes through an experimental code path that's loop-prone on long prompts. Users requesting "high" got "max" without consenting.

### Fix
"high" now joins the standard thinking tier alongside "medium". Users who want max must pass `reasoning_effort="max"` explicitly.

---

## FIX 13 — DSV4 reasoning_effort passthrough (medium/high → "high" for encoder)

**File:** `vmlx_engine/server.py` 3 DSV4 endpoints (chat-completions, Ollama, Responses)
**One-liner:** "low"/"medium"/"high" effort levels now reach the DSV4 encoder as "high" (the encoder's only non-max effort tier). Previously stripped to None — caused panel's "Reasoning" middle button to silently land on plain-thinking with no budget/close discipline.

### What was broken
Panel sends `thinking_mode="reasoning"` for the middle button → `reasoning_effort="medium"`. Server.py's DSV4 block had `if _cur_effort == "max": ... else: pop("reasoning_effort")` — stripping "medium" entirely. Encoder received None, returned `("thinking", None)` — plain thinking with no system-prompt nudge. This is exactly the loop-prone path users were hitting on long prompts.

### Fix
```python
if _cur_effort == "max":
    _ct_kwargs["reasoning_effort"] = "max"
elif _cur_effort in ("low", "medium", "high"):
    _ct_kwargs["reasoning_effort"] = "high"
else:
    _ct_kwargs.pop("reasoning_effort", None)
```

Applied symmetrically at all 3 DSV4 endpoints (chat-completions ~4179, Ollama ~4927, Responses ~6630).

### What this fixes
Panel "Reasoning" button now actually nudges DSV4 with the "high" effort hint instead of falling into default-thinking. Combined with rep_penalty 1.15 floor + hard n-gram dedup + N-1 snapshot, reasoning-mode loop susceptibility is significantly reduced.

### What would resurface
Reverting the elif branch. Or DSV4 encoder changing its accepted values away from {None, "high", "max"}.

---

## FIX 14 — DSV4 hard n-gram repetition dedup (sampling band-aid)

**File:** `vmlx_engine/utils/dsv4_batch_generator.py` `_hard_repetition_block` + `_sample`
**One-liner:** Hard-block tokens forming repeated 1-grams (3+ times in 24-token window), 2-grams, or 3-grams. Uses GENERATED tokens only (not prompt) to avoid prompt-poisoning the dedup state.

### Why this is band-aid not real fix
DSV4 long-context attention drift produces overly-peaked logits that rep_penalty 1.15× reduction can't escape. The model's logit for the attractor token (e.g., "Stanford") is so dominant that even multiplicative penalty after 50 occurrences leaves it as argmax. Verified live with VC-fund Project Plan prompt → "Stanford Stanford Stanford..." 100+ repetitions. After single-token-block, model thrashed on near-repeats: "hora horora hora era ara".

n-gram dedup catches:
1. Single-token: any token at >=3x in last 24 → -inf
2. 2-gram: prev1==a AND (a,b) seen 2+ times → block b
3. 3-gram: prev2==a, prev1==b AND (a,b,x) seen 2+ times → block x

Combined effect: even alternating-token loops (a,b,a,b,a,b) and 3-token cycles get broken.

### Critical: prompt-poison protection
The _sample method now takes BOTH `recent_tokens` (full prompt+gen, used by rep_penalty) AND `generated_tokens` (gen-only, used by hard dedup). Without this split, a prompt that legitimately repeats a word 3+ times would block that word for the entire generation.

Codex caught this — earlier version used recent_tokens for both. Bug fixed before any user impact.

### What this is NOT
- NOT a fix for the underlying DSV4 long-context attention drift
- NOT useful for non-DSV4 models (only wired into DSV4BatchGenerator)
- NOT a guarantee that all loop patterns get caught (e.g., 10-token cycles bypass)

### What would resurface
- Reverting to `recent_tokens` for both processors → prompt poisoning
- Lowering thresholds too far → blocks legitimate repeated punctuation
- Disabling via `VMLX_DSV4_HARD_REP_BLOCK=0` → loops return on long-form prompts

---

## FIX 15 — DSV4 cache snapshot bf16 round-trip safety

**File:** `vmlx_engine/utils/dsv4_batch_generator.py` `_snapshot_dsv4_cache._arr_copy`
**One-liner:** Cast bfloat16/float8 mlx arrays to fp32 BEFORE numpy roundtrip; reject the whole snapshot if any leaf copy fails.

### What was broken
`np.array(mx.array(..., dtype=mx.bfloat16))` raises:
```
ValueError: Item size 2 for PEP 3118 buffer format string B does not 
match the dtype B item size 1
```

The original `_arr_copy` caught the exception and returned None — so any DSV4 cache leaf that's bf16 would silently produce a None in the snapshot. That snapshot would replay corrupted on cache hit. Codex caught this.

### Fix
Detect bf16/float8 dtypes by string suffix, cast through fp32 staging:
```python
if str(src_dtype).endswith("bfloat16") or str(src_dtype).endswith("float8"):
    a_fp32 = a.astype(mx.float32)
    np_arr = np.array(a_fp32, copy=True)
    return mx.array(np_arr, dtype=mx.float32).astype(src_dtype)
```

Plus a hard-fail at snapshot end: if ANY leaf reported an error, reject the entire snapshot (logs warning with first 3 errors). Caller's `if r.prompt_snapshot is not None` check already handles this — caller falls through to the conservative no-store path.

### What would resurface
- Removing the dtype check → bf16 snapshots silently corrupt
- Lowering `copy_errors` to debug-only without rejecting → silent corruption
- A new dtype (fp4? int4 KV?) that np.array can't decode → also needs fp32 staging

---

## FIX 12 — Symmetric SWA-wrap guard at scheduler.py:4029 (gen_prompt_len strip)

**File:** `vmlx_engine/scheduler.py` reconstructed-DSV4Cache trim during gen_prompt_len stripping
**One-liner:** Symmetric to Fix 1's guard — refuse `cache.trim()` when reconstructing DSV4Cache for gen_prompt_len strip if SWA wrapped (current_len > sliding_window).

### What was broken
Codex caught a SECOND DSV4 trim path I missed in Fix 1. When the scheduler strips a `gen_prompt_len` prefix from cached state (used for thinking-models that prepend a `<think>` system prompt before reasoning), it reconstructs `DeepseekV4Cache` from extracted state dict and calls `cache.trim(to_trim)`. Same RotatingKVCache rewind constraint applies — if the original prefill exceeded `sliding_window`, this trim corrupts state.

### Fix
Mirror the Fix 1 guard: if `current_len > sliding_window`, refuse the trim and skip this cache (caller falls through to fresh prefill). For shorter prompts (no wrap), trim is safe and proceeds.

---

## FIX 1B — DSV4 clean prompt-boundary snapshot (REAL FIX, replaces the guard's role)

**File:** `vmlx_engine/utils/dsv4_batch_generator.py` `_snapshot_dsv4_cache` + prefill capture point; `vmlx_engine/scheduler.py` snapshot consumption at extraction site
**One-liner:** Capture deep-copy of DSV4 cache state IMMEDIATELY after prefill (before decode contaminates), use that for prefix cache + L2 disk store. The guard from Fix 1 stays as fallback for paths that don't capture a snapshot.

### Why the guard wasn't enough

Fix 1 protected correctness by refusing all post-generation cache stores. That made DSV4 prefix cache effectively unusable — every multi-turn request did a fresh full prefill (cache_hits=0 always). Operationally correct, performance-trash.

### The real fix

DSV4BatchGenerator runs prefill via `model(prompt_ids, cache=r.cache)`. Immediately after that returns, the cache state is at the prompt boundary: SWA RotatingKVCache hasn't wrapped, compressor/indexer pools contain only prompt-derived rows. **This is the moment to snapshot.**

`_snapshot_dsv4_cache(cache_list)`:
- For each `DeepseekV4Cache` layer, deep-copy `state` tuple via numpy roundtrip (`mx.array → np.array(copy=True) → mx.array`) so subsequent in-place mutations during decode don't leak into the snapshot.
- Preserve `meta_state` (sliding_window, compress_ratio, offset markers).
- Build a NEW `DeepseekV4Cache` per layer with the copied state.
- Returns a parallel list to the live cache.

Capture point: directly after `_prefill_last_logits()` returns and before `self._sample()` runs. Stored on `_Request.prompt_snapshot` and propagated to `_Response.prompt_cache_snapshot`.

Scheduler consumption: in the cache-extraction path, prefer `response.prompt_cache_snapshot` over the live `response.prompt_cache`. When snapshot exists, skip `_truncate_cache_to_prompt_length` entirely — the snapshot IS the prompt-boundary state, no rewind needed.

### Live verification (post-fix)

Cache isolation test (4-cell matrix on the previously-looping prompt):

| Test | Result |
|---|---|
| A: cache_salt="fresh-1" (bypass) | ✓ coherent loop_score=0.754, "Project Study of Star Birth..." |
| B: cache_salt="fresh-2" (bypass) | ✓ identical |
| C: no salt (fresh) → MISS, snapshot stored | ✓ identical |
| D: no salt → HIT (uses snapshot) | ✓ **identical** (was looping pre-snapshot) |

Cache stats post-test: `cache_hits=2 misses=2 tokens_saved=28 blocks=3`. **86 block-disk files** written.

Restart-survival: stop server, start with same `--block-disk-cache-dir`. `BlockDiskStore initialized: entries=3` (loaded from disk). Replay request → `paged cache hit, 28 tokens in 1 blocks` + `disk_hits=1, tokens_saved=28`. Coherent output.

### Trade-off

Memory: each prefill captures O(43 layers × prompt_len × hidden_dim) extra bytes. For DSV4-Flash @ prompt_len=512, that's ~50MB of snapshot per request. Released once the request finishes generation (snapshot is reference-counted). Acceptable.

CPU: numpy roundtrip per array. Measured ~50ms for a 30-token prompt's snapshot. Negligible vs prefill (seconds).

### What would resurface the bug

1. Removing the snapshot capture in DSV4BatchGenerator.
2. A new generator that doesn't propagate `_prompt_snapshot`.
3. Bypassing the snapshot consumption in scheduler (e.g., new code path that calls `_truncate_cache_to_prompt_length` directly without checking `response.prompt_cache_snapshot`).
4. Fix 1's guard still fires for paths without a snapshot — keep that as defensive backstop.

---

## FIX 1 — DSV4 SWA+CSA+HSA cache contamination guard (defensive backstop)

**File:** `vmlx_engine/scheduler.py` line ~1964 (`_truncate_cache_to_prompt_length` DSV4 branch)
**One-liner:** Refuse to store DSV4 prefix cache derived from post-generation live cache.

### What was broken

`DeepseekV4Cache` is a composite of three attention components:
1. **SWA** (`self.local`) — `RotatingKVCache` with `sliding_window=128`. Keys/values live in a circular buffer.
2. **CSA** — `compressor.pooled` rows. Each row summarizes `compress_ratio` consecutive raw positions.
3. **HSA** — `indexer.pooled` rows. Same pattern as CSA but for sparse attention routing.

After a generation completes, the live cache contains prompt KV + output KV. The scheduler tries to "rewind" it to prompt-only state by calling `_truncate_cache_to_prompt_length(raw_cache, prompt_len)` so subsequent same-prefix requests can hit prefix cache.

For DSV4, this rewind is fundamentally broken:

- **SWA wrap.** Once `local.offset > local.max_size` (true for any output longer than 128 tokens — i.e., any thinking-mode response), the circular buffer has wrapped multiple times. Trimming back leaves `_idx` negative. Replaying that state next turn applies output tokens at wrong positions → degenerate logits → loop.
- **CSA pool drift.** `pooled` rows are written cumulatively. After generation, pool contains both prompt-side rows AND output-side rows. `trim(n)` drops `max(1, n // ratio)` trailing rows — but: (a) the boundary rarely aligns with the prompt/output split exactly; (b) even an aligned trim leaves rows whose `key/value` were computed from a window that may have included output tokens.
- **HSA pool drift.** Same cumulative behavior as CSA.

Codex verified locally: "after prompt 29 + output 482, trimming back to 28 leaves `_idx = -356`."

### Live-test repro before fix

`/tmp/dsv4_cache_isolation.py` with full cache stack:
- A: `cache_salt="fresh-1"` (cache bypass) → coherent
- B: `cache_salt="fresh-2"` → coherent (identical content)
- C: no salt #1 (cache miss → fresh prefill, populates) → coherent
- **D: no salt #2 → CACHE HIT → loops** (loop_score=0.012, 600 tokens of garbage)

### The fix

```python
to_trim = max(0, current_len - target_len)
if to_trim > 0 and not _trust_trim:
    logger.info("DSV4 prompt cache store SKIPPED ... cannot be safely rewound ...")
    return None
```

Returning `None` from `_truncate_cache_to_prompt_length` causes the upstream caller (`scheduler.py:3569`) to skip the entire extraction + storage chain (paged store, block disk store, prefix cache store all gated on `cache_for_extract is not None`).

Override: `VMLX_DSV4_TRUST_TRIMMED_CACHE=1` keeps the v1.5.13 store-always behavior. Documented as not recommended.

### Trade-off

**DSV4 multi-turn always re-prefills. Cache-hit-rate=0 for DSV4.** Slow but correct. Real fix is "capture clean prompt-boundary cache before decode starts" — needs a new code path that hooks the cache state at prefill exit (analogous to SSM `_prefill_for_clean_ssm`). Documented as next-session work.

### What would resurface this bug

1. Removing the guard "because cache hit rate is too low."
2. Setting `VMLX_DSV4_TRUST_TRIMMED_CACHE=1` in production.
3. Adding a parallel store path that bypasses `_truncate_cache_to_prompt_length` (e.g., a streaming-store hook). Any new DSV4 cache store path must also gate on `to_trim == 0`.
4. Loosening the SWA `sliding_window` config so output ≤ window — would make wraps less likely but pool drift still exists.

### Verification anchor

Live-test with `/tmp/dsv4_cache_isolation.py` — D must produce same content as A. Server log must show `DSV4 prompt cache store SKIPPED: current_len=X target_len=Y to_trim=Z` events.

---

## FIX 2 — DSV4 force-thinking on 3 endpoints (strict `is not True` semantics)

**File:** `vmlx_engine/server.py` lines ~4079, ~4806, ~6505 (chat-completions / Ollama / Responses)
**One-liner:** `if enable_thinking is not True: chat_req.enable_thinking = True` — force True unless explicit True; honor explicit False only when env override is set.

### What was broken

DSV4-Flash bundle's chat-mode (`enable_thinking=False`) emits training-data-contaminated output: polite-assistant attractor loops ("Hello! How are you? How can I help?"), hallucinated boilerplate, mixed-language annotation leakage. v1.5.6 (commit `00a78db4`) verified empirically and added force-flip on 3 endpoints. v1.5.15 reverted with an unverified theory that scheduler-side cache fix obviated the API-layer protection.

The DSV4 chat encoder in `jang_tools/dsv4/encoding_adapter.py` `_resolve_mode_and_effort` defaults to `("chat", None)` when both `enable_thinking` and `reasoning_effort` are None. So **without server-side force-flip, every default request hits the contaminated chat path.**

### Live-test before fix

Default request `{"messages": [{"role":"user","content":"..."}]}` with no thinking field → encoder maps to chat mode → "(the project (the project ..." loop / "birth, birth, birth..." loop.

### The fix

```python
_allow_dsv4_chat = os.environ.get("VMLX_DSV4_ALLOW_CHAT", "0") in ("1","true","yes")
if chat_req.enable_thinking is False and _allow_dsv4_chat:
    # Power-user override: serve broken chat-mode for benchmark studies
    _msg_kwargs["enable_thinking"] = False
    _ct_kwargs["enable_thinking"] = False
else:
    if chat_req.enable_thinking is not True:
        logger.info("DSV4: forcing enable_thinking=True ...")
        chat_req.enable_thinking = True
    _msg_kwargs["enable_thinking"] = True
    _ct_kwargs["enable_thinking"] = True
```

Three sites, identical logic. Anthropic /v1/messages flows through chat-completions internally so it inherits this protection.

### Why "is not True" instead of "is False"

`chat_req.enable_thinking` can be:
- `True` → user explicitly requested thinking. Leave it.
- `False` → user explicitly requested chat. Honored ONLY when `VMLX_DSV4_ALLOW_CHAT=1`.
- `None` → user didn't pass anything. **Forced to True** (this is the new behavior — v1.5.20 honored None as default which fell into encoder's chat-mode default).

The panel client at `panel/src/main/ipc/chat.ts:447` defaults `enableThinking=false` for new chats. The Anthropic adapter at `vmlx_engine/api/anthropic_adapter.py:179` defaults `enable_thinking=False` per Anthropic spec. Both cases now force-flip safely. Without the strict semantics, both clients send explicit False → broken chat path.

### What would resurface this bug

1. Reverting to `if enable_thinking is True: ... elif enable_thinking is False: ...` — leaves None → encoder chat default.
2. Removing the force-flip "because the cache fix handles it" — repeats v1.5.15's mistake. Cache fix is unrelated to the bundle's chat-mode contamination.
3. Adding a fourth API endpoint (e.g., a future Vertex AI adapter) without copying the same `_is_dsv4` block.
4. Re-quantizing DSV4 with a fixed chat-mode template — would invalidate the rationale but the force-flip is still safe (forces thinking which is always more capable).

---

## FIX 3 — DSV4 rep_penalty 1.15 family floor

**File:** `vmlx_engine/server.py:287` `_FAMILY_FALLBACK_DEFAULTS`
**One-liner:** `"deepseek_v4": (0.6, 0.95, 1.15)` — empirically validated anti-loop floor.

### What was broken

DSV4 bundle's calibrated defaults are `repetition_penalty_thinking=1.0` and `repetition_penalty_chat=1.05`. Both produce degenerate token loops on long thinking chains ("birth, birth, birth..." after the model enters self-reinforcing attractor). v1.5.8 commit `44c571a6` verified `rep_penalty=1.15` "short-circuits the loop in 300-400 tokens." v1.5.15 wiped the floor on theory that cache fix removed the need.

### Live-test before fix

`/tmp/dsv4_loop_trigger.py` with default rep (1.0/1.05) at temp=0.6: loops on prompt-3 mode-on (loop_score=0.005, "ZWS character spam"), prompt-2 mode-on (loop_score=0.029, "babies babies babies").

### The fix

```python
_FAMILY_FALLBACK_DEFAULTS = {
    "deepseek_v4": (0.6, 0.95, 1.15),
}
```

`_resolve_repetition_penalty` precedence: request → bundle-calibrated → CLI default → **family fallback (THIS)** → None. Bundle still wins when present.

### Trade-off

~3pp MMLU on standard 200-Q evaluation (per v1.5.8 commit message). Users running benchmarks override per-request with `repetition_penalty: 1.05`.

### What would resurface this bug

1. Removing the family fallback "because the cache fix handles it" — repeats the v1.5.15 mistake. Even with the cache guard, single-turn loops still occur on certain prompts.
2. Re-quantizing DSV4 with a different bundle-default that's between 1.05-1.15 — bundle wins over family fallback, so floor is bypassed.

### Note on the three independent loop sources

DSV4 has three independent loop triggers, any one of which is sufficient:
1. **Chat-mode contamination** — defended by Fix 2 (force-thinking on 4 endpoints)
2. **Cumulative pool-buffer leakage** — defended by Fix 1 (cache guard)
3. **Self-reinforcing token attractor at low temp** — defended by Fix 3 (rep_penalty floor)

All three fixes are required for production-quality DSV4. Removing any single one will cause specific repro patterns to leak.

---

## FIX 4 — DSV4 single-shot prefill default

**File:** `vmlx_engine/utils/dsv4_batch_generator.py` lines ~117-145, ~167-180
**One-liner:** Default DSV4 prefill to single-shot; chunking is opt-in via `DSV4_PREFILL_STEP_SIZE>0`.

### What was broken

v1.5.6 (`00a78db4`) verified empirically that chunking the DSV4 prefill corrupts the compressor + indexer pool state mid-decode (`broadcast_shapes (1,N) (1,64,1,128)` mismatch). v1.5.15 silently re-introduced chunking at default size 512 with a comment claiming "Current DeepseekV4Cache accumulates pool state correctly across calls" — unverified, contradicted by v1.5.6's empirical 14/14 probe matrix. The jang-tools DSV4 runtime code is **byte-identical** between jang 2.5.18 (v1.5.10 baseline) and 2.5.23 (current), so the chunking corruption v1.5.6 documented is still latent in the same code path.

### The fix

```python
_dsv4_step_env = os.environ.get("DSV4_PREFILL_STEP_SIZE")
_dsv4_step = int(_dsv4_step_env) if _dsv4_step_env else 0
if _dsv4_step <= 0:
    self.prefill_step_size = 1 << 30  # sentinel — single-shot
else:
    self.prefill_step_size = _dsv4_step
```

Single-shot is achieved by setting step size larger than any real prompt; the chunked loop runs exactly once.

### Why default single-shot is safe even on long prompts

Post-warmup the model has all kernels JIT-compiled, so single-shot prefill stays under the Metal command-buffer watchdog even on 6K+ prompts. Per v1.5.6: "Post-warmup the model has all kernels JIT-compiled so single-shot prefill stays under the Metal command-buffer watchdog even on long prompts."

### What would resurface this bug

1. Setting `DSV4_PREFILL_STEP_SIZE=512` (or any positive value) — opts into the broken chunking path.
2. A future change that splits prefill into multiple `model(chunk, cache=cache)` calls without first validating the DSV4 cache is unaffected.
3. Removing the post-warmup JIT compilation — would re-expose the watchdog timeout that motivated chunking originally.

### Override

`DSV4_PREFILL_STEP_SIZE=N>0` for users who hit watchdog-kill on extreme prompts. Better: investigate why the watchdog fires and raise the timeout instead.

---

## FIX 5 — JANGTQ_DISABLE_FUSED_SWITCHGLU env override (debug only)

**File:** `jang_tools/load_jangtq.py` ~line 1098 (generic fused patch site)
**One-liner:** `JANGTQ_DISABLE_FUSED_SWITCHGLU=1` skips the fused gate+up SwitchGLU patch.

### Why

Diagnostic. Lets us isolate whether a model loop is in the fused kernel pipeline (Hadamard rotate → fused gate+up SwiGLU → rotate → gather down) or upstream/downstream. Used in this session to confirm DSV4 loops were NOT in the fused kernel — disabling fusion left the loops intact, ruling out the kernel as root cause.

### Default behavior unchanged

Without the env, the patch fires as before. With the env, mlx_lm's stock `SwitchGLU.__call__` per-projection path runs.

### What would resurface a problem

Setting this env in production — slower, possibly different numerics. Not recommended outside debugging.

---

## FIX 6 — JANGTQ_K mixed-bit `dp_bits` threading (jang 2.5.24)

**File:** `jang_tools/load_jangtq.py` `_get_compiled_decode` and `_fused_switchglu_call`
**One-liner:** `dp_bits=None` parameter (defaults to `bits` for legacy uniform); cache key extended `(in_f, out_f, bits, dp_bits, K, limit_milli)`.

### What was broken

JANGTQ_K (per-projection mixed bits) ships gate=2, up=2, down=4 (e.g., MiniMax-M2.7-JANGTQ_K). The fused gate+up kernel requires `gate.bits == up.bits` (architectural constraint). The gather_down kernel uses a SEPARATE bit width. Pre-fix, both kernels used `gp.bits` — the down_proj's 4-bit packed tensors got unpacked as 2-bit → wrong indices into the (correct) 4-bit codebook → garbage decode.

### The fix

```python
def _get_compiled_decode(in_f, out_f, bits, K, swiglu_limit=0.0, dp_bits=None):
    if dp_bits is None: dp_bits = bits  # legacy uniform-bits = byte-identical
    key = (in_f, out_f, bits, dp_bits, K, limit_milli)  # K and uniform don't share kernels
    ...
    gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, K)  # uses dp_bits
```

Caller passes `dp_bits=dp.bits` (the down-projection's actual bit width).

### Live-test

MiniMax-M2.7-JANGTQ_K coherent single-turn Q&A + multi-turn recall ("photosynthesis" returns "Photosynthesis is the process by which plants, algae, and certain bacteria...").

### Backwards compat

JANGTQ2/3/4 callers don't pass `dp_bits` (defaults to None → `bits`). Cache key extension is harmless because only K bundles enter the K branch.

### What would resurface

1. Reverting to `(in_f, out_f, bits, K, limit_milli)` cache key — would cause K and uniform layers in the same model to collide.
2. Removing `dp_bits` from `make_gather_tq_decode_per_row` — would unpack down at the wrong bit width.
3. Adding a future per-projection differential like `gate.bits != up.bits` — fused kernel constraint requires gate==up, so this would need a different fix (separate gate and up kernels).

---

## FIX 7 — Ling-2.6 / Bailing-V2.5 `bailing_hybrid` model class + flat-2D shape repair

**File:** `panel/scripts/patches/bailing_hybrid.patched.py` (build-time controlled fork of upstream mlx_lm)
**One-liner:** Vendored model class for `model_type=bailing_hybrid` (mlx_lm 0.31.3 ships none) + flat-2D switch_mlp shape repair for MXFP4 bundles.

### What was broken

User reported `ValueError: Expected shape (256, 1024, 512) but received shape (256, 524288) for parameter model.layers.1.mlp.switch_mlp.gate_proj.weight` when loading Ling-2.6-flash-MXFP4-CRACK. The MXFP4 converter produced 2D flat shape `(n_experts, out × in_per_row)` while mlx_lm's strict shape check expects 3D `(n_experts, out, in_per_row)`.

### The fix

Inside `Model.sanitize`, walk `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` keys, detect 2D `(n_exp, flat)` and reshape to 3D `(n_exp, out_dim, in_per_row)` using `moe_intermediate_size` (gate/up) or `hidden_size` (down) when divisible.

### Live-test

`/tmp/dsv4_loop_trigger.py` on Ling: 12/12 cells coherent across all 4 modes. Cache reconstruction sometimes fails-safe to full prefill (logged but doesn't corrupt output).

### What would resurface

1. New mlx_lm release that adds an upstream `bailing_hybrid` class with different field names — our patch would silently mis-load.
2. MXFP4 converter changing shape convention back to 3D — repair would be no-op (defensive only).
3. Bailing-V3 with different layer naming — pattern won't match.

### Build-time controlled fork rationale

`panel/scripts/patches/README.md` documents why we vendor patches at build time rather than monkey-patch at runtime: signed-bundle integrity, deterministic artifacts, no import-order dependencies.

---

## FIX 8 — MiniMax-M2 family rep_penalty 1.15 floor

**File:** `vmlx_engine/server.py:287` `_FAMILY_FALLBACK_DEFAULTS["minimax_m2"]`
**One-liner:** `"minimax_m2": (0.6, 0.95, 1.15)` — same value as DSV4, validated empirically.

### What was broken

MiniMax JANGTQ uniform 2-bit bundle ships `repetition_penalty=1.0` in `generation_config.json` (per startup log "merged eos_token_id from generation_config.json"). At rep=1.0, thinking-mode falls into a rumination attractor on certain prompts: produces 600+ tokens of "The user asks... This is a request... The user wants..." paraphrasing without ever closing `<think>`. Reproduces with `cache_salt` — model-level rumination, not cache contamination. mode=off (no thinking) works fine.

### Live-test rep_penalty sweep

prompt="Tell me a 2-sentence story about a project to see the birth of a star.", temp=0.6, mode=on:
- rep=1.00 → loop_score=0.250 LOOP
- rep=1.05 → loop_score=0.370 LOOP
- rep=1.10 → loop_score=0.397 LOOP (borderline)
- rep=1.15 → loop_score=0.400 OK (sweet spot lower bound)
- rep=1.20 → loop_score=0.442 OK
- rep=1.30 → loop_score=0.346 LOOP (penalty fragments output)

### Live-test before/after floor

3-prompt × 4-mode matrix:
- Before (rep default 1.0): MiniMax JANGTQ uniform = ~3/12 OK; MiniMax JANGTQ_K = 10/12 OK
- After (rep floor 1.15): MiniMax JANGTQ uniform = **11/12 OK**; MiniMax JANGTQ_K = **11/12 OK**

The 1 remaining cell on each variant is `mode=omitted` hitting length-cap with verbose-thinking = false positive on the heuristic.

### What would resurface this bug

1. Removing the `minimax_m2` family fallback "because the model bundle should fix it."
2. New MiniMax bundle that ships rep_penalty in `jang_config` — bundle wins over family fallback. If bundle ships 1.0, this floor is bypassed.
3. Re-quantizing MiniMax to add a stronger think-stop guard in template — would invalidate need for the floor but the floor is still safe.

---

## Ling-2.6 cache reconstruction limitation (NOT fixed; documented)

**Symptom:** Ling JANGTQ paged cache hits succeed but reconstruction throws `"There is no Stream(gpu, 1) in current thread"` on every hit. Scheduler safely falls back to full prefill. **No correctness regression** — Ling produces 12/12 coherent on loop matrix. Just lost cache speedup (every hit triggers re-prefill).

**Root cause:** MLX streams are thread-local. The cache reconstruction code creates MLX arrays on a thread that didn't initialize a Metal stream. Generic mlx issue affecting hybrid models (bailing_hybrid 4 MLA + 28 GLA layout). Same class as the v1.5.5 DSV4 stream-thread error, fix would be analogous to scheduler's `_step_executor` dispatch.

**Out of session scope.** Filed for separate fix.

---

## What's NOT fixed (acknowledged, not in scope)

- **DSV4 long-term fix:** capture clean prompt-boundary cache before decode (eliminates Fix 1's cache-hit-rate=0 trade-off).
- **DSV4 partial-MLA TQ KV quant** (quantize SWA local KV, leave compressed latents fp16). Currently scheduler.py:541 disables KV q4 entirely for MLA models.
- **Ling cache reconstruction thread-stream bug** (documented above).
- **Codex's other 7 issues:** standalone `jang_tools/jangrt/` runtime path lags main loader; standalone fused SwitchGLU missing dp_bits fix; Swift JANGTQ_K not yet correct; MiniMax-Small still per-expert (not prestack); Mistral 3.5/Pixtral guarded.

---

## Cross-component invariants to maintain

1. **DSV4 family-name detection** in `_model_family_for_defaults` must return `"deepseek_v4"` when `config.json` `model_type=="deepseek_v4"`. Without this, Fixes 2, 3 don't fire.
2. **DSV4 cache class detection** in `_model_uses_dsv4_cache` must find `"DeepseekV4Cache"` in `model.make_cache()` output. Without this, Fix 1's guard never fires.
3. **`_truncate_cache_to_prompt_length` is the single chokepoint** for DSV4 cache extraction. Adding alternative paths bypasses Fix 1.
4. **Force-thinking force-flip is at all 4 API surfaces** (chat-completions, Ollama, Responses, Anthropic-via-chat). Adding a fifth (Vertex, Bedrock) needs same `_is_dsv4` guard.

---

## Files changed in this session

```
vmlx_engine/scheduler.py                     # Fix 1
vmlx_engine/server.py                        # Fixes 2, 3
vmlx_engine/utils/dsv4_batch_generator.py    # Fix 4
jang_tools/load_jangtq.py                    # Fix 5 (debug env), Fix 6 (jang 2.5.24)
panel/scripts/patches/bailing_hybrid.patched.py  # Fix 7
```

## Live Matrix Test — All Models in ~/models/JANGQ/ (2026-05-05 v1.5.21 build)

| Model | arch | layers | instruct | reasoning | max | recall (multi-turn) |
|---|---|---|---|---|---|---|
| **DeepSeek-V4-Flash-JANGTQ** | deepseek_v4 | 43 | ✅ | ✅ | ✅ | ✅ teal |
| **MiniMax-M2.7-JANGTQ** | minimax_m2 | 62 | ✅ | ✅ | n/a | ✅ teal |
| **MiniMax-M2.7-JANGTQ_K** | minimax_m2 (mixed-bit) | 62 | ✅ | ✅ | n/a | ✅ teal |
| **MiniMax-M2.7-Small-JANGTQ** | minimax_m2 | 62 | ✅ | ✅ | n/a | ✅ teal |
| **Ling-2.6-flash-JANGTQ** | bailing_hybrid (4 MLA + 28 GLA) | 32 | ✅ | ✅ | n/a | ✅ teal |
| **Laguna-XS.2-JANGTQ** | laguna | 40 | ✅ | ✅ | n/a | ✅ teal |
| **Kimi-K2.6-Small-JANGTQ** | kimi_k25 | ? | ❌ pre-existing load bug | | | |

All models test prompts: `"Reply with just one word: hello."` for single-turn, `"Remember this: my favorite color is teal."` → `"What is my favorite color?"` for multi-turn recall. All passes use `cache_salt` to avoid cache poisoning.

### Kimi failure — pre-existing, NOT a regression in current fixes

```
File "mlx_vlm/tokenizer_utils.py", line 208, in __init__
    self.tokenmap = [None] * len(tokenizer.vocab)
AttributeError: TikTokenTokenizer has no attribute vocab
```

mlx_vlm's `TokenizerWrapper.__init__` assumes the wrapped tokenizer exposes `.vocab` but Kimi K2.6's TikTokenTokenizer doesn't. Same code path as Kimi K2.5 worked before — likely an mlx_vlm version drift issue. Out of session scope.

---

## Files NOT changed in this session (scope clarity)

- `jang_tools/dsv4/mlx_model.py` — DSV4 runtime is byte-identical to v1.5.10 baseline (jang 2.5.18). No changes needed.
- `vmlx_engine/loaders/load_jangtq_dsv4.py` — sidecar fast-load is bypassed for PRESTACK bundles (current standard); not in critical path.
- `vmlx_engine/prefix_cache.py` — DSV4-block storage logic correct as-is; the bug was upstream at extraction.
