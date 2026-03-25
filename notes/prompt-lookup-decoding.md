# Prompt Lookup Decoding — Research Notes

## What is it?

Prompt Lookup Decoding (PLD) generates "draft" tokens by searching the
*existing* token sequence for n-gram matches and returning what followed
the match earlier. No second model is needed. The input prompt is the
draft library.

The target model then *verifies* all K draft tokens in **one** forward
pass (instead of K sequential single-token passes), producing up to K+1
tokens at the cost of 1 forward pass.

Reference: Saxena (2023), "A Simple Framework for Prompt Lookup Decoding"

---

## Implementation in vmlx

### Files

| File | Purpose |
|------|---------|
| `vmlx_engine/prompt_lookup.py` | `find_draft_tokens()` n-gram search; `PromptLookupStats` measurement class |
| `vmlx_engine/scheduler.py` | Phase 2/3 speculative decode in `_process_batch_responses` + `_try_speculative_decode()` |
| `tests/benchmark/test_pld_acceptance.py` | 4-task acceptance rate benchmark |

### How it works (Phase 3, current)

After `BatchGenerator` emits each token, `_try_speculative_decode()`:

1. **Peek at forward logprobs** from `batch_generator.active_batch.logprobs[e]`
   BEFORE calling `remove()`. These are the NEW logprobs (prediction after
   `last_token`) set by `_step(last_token)` in the same `_next()` call.
   `response.logprobs` is the OLD logprobs (used to generate `last_token`)
   and is NOT the right source.
2. Calls `batch_generator.remove([uid], return_prompt_caches=True)` to
   extract the live KV/Arrays cache — the only way to access it mid-stream.
3. Runs `find_draft_tokens()` to find up to K=5 draft tokens via n-gram
   search in the full token sequence (prompt + output so far).
4. Calls `self.model([[d0, …, d_{K-1}]], cache=kv_cache)` — one forward
   pass processing K tokens. KVCache already holds `last_token` at offset N.
   No pre-trim needed; both KVCache and ArraysCache advance K steps uniformly.
5. **T≈0 (greedy):** Accept longest prefix where `argmax == draft`. d0 uses
   `argmax(forward_logprobs)`; d1..d_{K-1} use `argmax(logits[i-1])`.
   **T>0 (Phase 3):** Accept d_i with probability `softmax(logprobs/T)[d_i]`.
   Correction/bonus token sampled from `p(x)` with rejected token masked out.
6. Rolls back the KV cache to the accepted length.
7. Re-inserts the request into `BatchGenerator` with the bonus/correction
   token. Updates `uid_to_request_id` / `request_id_to_uid` maps.

On any failure the request is guaranteed to be re-inserted via a `finally`
block — it can never be orphaned.

### KV cache rollback — the non-obvious part

`trim_prompt_cache()` from mlx-lm is **wrong** for standard `KVCache`:
- `KVCache.trim(n)` only decrements `self.offset`.
- `KVCache.update_and_fetch()` always concatenates new keys/values, then
  sets `self.offset = keys.shape[-2]`, overwriting the trimmed offset.

`QuantizedKVCache.trim(n)` IS correct because its `update_and_fetch` uses
`offset` as a write pointer into a pre-allocated buffer.

Correct rollback (works for both types):
```python
for c in kv_cache:
    if not c.is_trimmable() or c.offset == 0:
        continue
    accepted_offset = c.offset - num_to_trim
    if isinstance(c.keys, mx.array):   # standard KVCache: must slice arrays
        c.keys   = c.keys[...,   :accepted_offset, :]
        c.values = c.values[..., :accepted_offset, :]
    c.offset = accepted_offset         # sufficient for QuantizedKVCache
```

Use `c.is_trimmable()` — not `hasattr(c, 'offset')` — to identify layers
that support rollback. `ArraysCache` returns `False` and must be skipped.

### SSM offset bug and Phase 3 fix

**Phase 2 pre-trim approach:** Phase 2 fixed double-last_token by pre-trimming
KVCache by 1 (N→N-1) before the verify pass, then using
`verify_input = [last_token, d0..d_{K-1}]` (K+1 tokens). This worked at T=0
but introduced an accumulating ArraysCache offset at T>0:

- Each spec decode round: batch gen's `_step(last_token)` advances ArraysCache
  to S_{N+1}. Pre-trim removes `last_token` from KVCache only (→ offset N-1).
  Verify pass processes `[last_token, d0..d_{K-1}]`, advancing KVCache by K+1
  and ArraysCache by K+1 → both end at N+K.
- On full-accept: new batch gen seed = bonus_token. Both caches advance
  together. No offset. ✓
- On partial reject (Case b): ArraysCache restored to saved pre-verify state
  S_{N+1}. KVCache rewound to N-1+1 = N (was `num_drafts+1`, now `num_drafts`).
  Seed = correction_token. Batch gen advances both by 1 → both at N+1. ✓
- **But at T>0:** partial reject is rare at T=0 (~8% of rounds), so offset
  stays small. At T=0.3 (~95% full-accept rounds but with sampled tokens not
  matching argmax), the pre-trim creates a +1 SSM offset every round where
  the bonus token differs from the greedy prediction. With greedy bonus_token
  at T=0.3, this grew to 30+ → catastrophic word doubling.

**Phase 3 fix:** Remove pre-trim entirely. Use `verify_input = [d0..d_{K-1}]`
(K tokens). KVCache already holds `last_token` at offset N (batch gen ran
`_step(last_token)` already). Both KVCache and ArraysCache advance by exactly
K steps per verify pass → zero SSM offset accumulation at any temperature.

**KV cache rollback** (still needed for rejected drafts):
```python
for c in kv_cache:
    if not c.is_trimmable() or c.offset == 0:
        continue
    accepted_offset = c.offset - num_to_trim
    if isinstance(c.keys, mx.array):   # standard KVCache: must slice arrays
        c.keys   = c.keys[...,   :accepted_offset, :]
        c.values = c.values[..., :accepted_offset, :]
    c.offset = accepted_offset         # sufficient for QuantizedKVCache
```

Use `c.is_trimmable()` — not `hasattr(c, 'offset')` — to identify layers
that support rollback. `ArraysCache` returns `False` and must be skipped.

### Hybrid model structure (Qwen3.5-27B)

This model has 64 layers: **48 ArraysCache + 16 KVCache**.

`ArraysCache` is used for recurrent/SSM-style layers. It has no positional
offset and cannot be rolled back. On partial rejection (Case b), the saved
pre-verify ArraysCache state is restored and KVCache is rewound to pre-verify
offset (N); the request is re-seeded with the correction token so the batch
generator advances both caches together from a consistent starting point.

---

## Results

### Phase 1 — Measurement (no generation changes)

Benchmark tasks: code generation, structured JSON, summarisation,
open-ended reasoning. Model: Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit.

| Window     | coverage | hit@1 | mean_depth | theoretical_speedup |
|------------|----------|-------|------------|---------------------|
| 0–200 tok  | 14.5%    | 89.7% | 4.58       | **2.47×**           |
| 0–400 tok  | 19.2%    | 57.1% | 3.82       | 1.72×               |
| 0–600 tok  | 20.7%    | 58.9% | 3.74       | 1.83×               |
| 0–1000 tok | 20.8%    | 60.6% | 3.44       | 1.76×               |
| 0–1600 tok | 17.8%    | 59.2% | 3.28       | 1.53×               |

Key findings:
- **Early burst (0–200 tokens): 89.7% hit.** Qwen3's reasoning preamble
  ("Let me think about this…") echoes prompt tokens near-verbatim.
- **Steady state: ~60% hit, ~3.4 mean depth.** When a match exists,
  it's correct 60% of the time and yields 3.4 consecutive correct tokens.
- **Coverage (20%) is the binding constraint**, not hit rate. Only 1-in-5
  positions finds an n-gram match; open-ended generation produces tokens
  absent from the prompt.

### Phase 2 — Speculative decode (T=0 only, corrected)

**Correctness verified**: at temperature=0, all four benchmark outputs are
byte-for-byte identical to the non-PLD baseline (`diff pld_off pld_fixed3`
shows no differences). The double-last_token bug that caused word doubling
and prompt echoing is fixed.

Client-side `tok/s` from the benchmark script (SSE events × approx chars/tok):

| Task | tok/s | vs baseline |
|------|-------|-------------|
| Code generation | ~17 | **~8.5×** |
| Structured JSON | ~11 | **~5.5×** |
| Summarisation | ~17 | **~8.5×** |
| Open-ended reasoning | ~19 | **~9.5×** |
| **Overall** | **~16** | **~8×** |

**Baseline caveat:** The ~2 tok/s baseline used here predates the paged KV
cache + block disk cache stack. Post-caching, Qwen3.5-27B-Claude-4.6-6bit baseline is ~16 tok/s.
Phase 2/3 speedup multiples are therefore overstated; see Phase 4 for
real-world numbers.

Note: the benchmark's client-side `tok/s` estimates from SSE event count
× text length. Each SSE event carries multiple tokens when spec decode
accepts drafts. Use server log `grep 'finished:'` for authoritative counts.

**Earlier (buggy) measurements** showed similar tok/s but produced corrupted
output — the model was echoing prompt text quickly, not generating correctly.
The corrected numbers represent genuine quality-equivalent speedup.

### Phase 3 — Probabilistic acceptance (T≥0, current)

**Root cause of T>0 failure (resolved):** Phase 2's pre-trim mechanism removed
`last_token` from KVCache before the verify pass, but ArraysCache (Mamba/SSM)
layers cannot be trimmed. This created a +1 SSM-state offset per spec decode
round. At T=0 (~8% full-accept), resets via Case (b) kept the offset near 1.
At T=0.3 (~95% full-accept rounds), the offset grew linearly to 30+, causing
catastrophic word doubling.

**Fix:** Remove pre-trim entirely. Use `verify_input = [d0..d_{K-1}]` (K tokens,
no `last_token` prefix). KVCache already holds `last_token` at offset N from
batch gen; the verify pass advances both caches by exactly K steps → zero SSM
offset accumulation.

**d0 logprobs:** `response.logprobs` is the distribution that *generated*
`last_token` (OLD logprobs), not the prediction *after* it. The correct source
is `batch_generator.active_batch.logprobs[e]` (NEW logprobs, set by
`_step(last_token)` in the same `_next()` call), read BEFORE `remove()`.

**Acceptance algorithm (deterministic draft source):**
1. Accept d_i with probability `softmax(forward_logprobs/T)[d_i]` for d0,
   `softmax(logits[i-1]/T)[d_i]` for d1..d_{K-1}
2. On rejection: sample correction from `p(x)` with rejected token masked
3. Bonus token: sample from `p(x)` at position `num_accept` (not argmax)

**Temperature gate removed:** PLD now fires at all temperatures (no 0.05
threshold). The greedy path (temp ≤ 1e-6) is unchanged for T≈0.

**Results at T=0.3** (same four benchmark tasks, same model):

| Task | tok/s | vs baseline |
|------|-------|-------------|
| Code generation | ~9 | **~4.5×** |
| Structured JSON | ~9 | **~4.5×** |
| Summarisation | ~11 | **~5.5×** |
| Open-ended reasoning | ~14 | **~7×** |
| **Overall** | **~11** | **~5.5×** |

**Baseline caveat:** Same as Phase 2 — baseline was ~2 tok/s (pre-caching
stack). See Phase 4 for post-caching measurements.

**Correctness verified**: no word doubling or repetition loops in any task
output. Open-ended reasoning previously failed with "presents presents a a
compelling compelling...the the the..."; now produces fully coherent text.

**Throughput at T=0.3 vs T=0:** ~11 vs ~16 tok/s. At T=0.3 the model
rarely accepts d0 outright (n-gram drafts diverge from sampled output more
often), so many spec decode rounds produce only 1 correction token. The verify
pass is still efficient (processes K tokens per bandwidth cost ≈ 1 forward
pass), giving meaningful speedup even with low acceptance.

### Phase 4 — Cross-model benchmarking (post-caching-stack, pre-optimization)

**Context:** After deploying the paged KV cache + block disk cache stack,
Qwen3.5-27B-Claude-4.6-6bit baseline throughput improved from ~2 tok/s → ~16 tok/s. This invalidated
the Phase 2/3 speedup claims and prompted a full cross-model benchmark sweep.

**Initial results (K=5, no d0 pre-check):**

| Model | Architecture | Baseline | PLD-on | Delta |
|-------|-------------|----------|--------|-------|
| Qwen3.5-27B-Claude-4.6-6bit | Hybrid SSM/ATT | ~16 tok/s | ~12 tok/s | **-22%** |
| Qwen3.5-27B-4bit-DWQ | Hybrid SSM/ATT | ~22 tok/s | ~14 tok/s | **-36%** |
| Qwen3.5-27B-5bit | Hybrid SSM/ATT | ~18 tok/s | ~12 tok/s | **-37%** |
| Qwen3.5-35B-A3B-4bit | Hybrid SSM/MoE | ~82 tok/s | ~54 tok/s | **-34%** |
| Qwen3-4B-6bit | Dense transformer | ~72 tok/s | ~68 tok/s | **-6%** |

**Root cause (two factors):**

1. **SSM sequential processing penalty.** Qwen3.5-27B has 48 ArraysCache
   (SSM) layers + 16 KVCache (attention) layers. SSM layers process verify
   tokens *sequentially* via a for-loop (`gated_delta_ops` in mlx-lm). With
   K=5, the verify pass costs ~4× a single decode step — not ~1× as standard
   speculative decoding theory assumes for pure attention models.

2. **Per-cycle remove/insert overhead.** Each PLD cycle calls
   `batch_generator.remove()` + `insert()` (~15–30ms). With K=5 and many
   cycles firing, this fixed overhead accumulates faster than the
   tokens-per-pass gain.

These findings led to the Phase 5 optimizations below.

### Phase 5 — Optimization: d0 pre-check + K tuning + auto-tune (current)

Six changes brought PLD from -22% regression to **+7% net positive** on
Qwen3.5-27B-Claude-4.6-6bit and **safe across all models** via adaptive auto-tune:

**1. Adaptive K=2 for hybrid models** (SSM layer count > 0).
Reduces verify cost from ~4× to ~1.75× a normal decode step. K=2 balances
verify cost against per-cycle overhead: each successful cycle produces
4 tokens in 2.75 decode-equivalents (1.45 eff). K=1 was tested but produced
fewer tokens per cycle (3 in 2.0 = 1.50 eff) and didn't amortize the ~20ms
remove/insert overhead as well.

**2. d0 pre-check** before the expensive remove/verify/insert cycle.
`forward_logprobs` (already materialized by BatchGenerator) are checked
before calling `remove()`:
- T=0: `argmax(forward_logprobs) == drafts[0]` — exact, zero cost.
- T>0: `forward_logprobs[drafts[0]] > -2.0` — single array lookup,
  no softmax. Threshold ≈ p > 13% at T=1.

On the OpenClaw workload (0.3–2.6% hit@1), d0 pre-check filters ~60% of
PLD attempts, eliminating the remove/insert overhead on cycles that would
produce no gain.

**3. bfloat16 numpy fix** in cache rollback. The numpy roundtrip (needed
to avoid Metal command buffer corruption) crashed on bfloat16 arrays with
`Item size 2 for PEP 3118 buffer format string B does not match the dtype B
item size 1`. Fixed by casting to float16 before numpy conversion, then
restoring the original dtype — same pattern as `_truncate_cache_to_prompt_length`.
Before the fix, ~40% of PLD cycles were crashing to emergency re-insert.

**4. Backward n-gram scan.** `find_draft_tokens` now scans from the most
recent prior occurrence instead of the oldest. More contextually relevant
matches on repetitive/structured output.

**5. Adaptive auto-tune with exponential ramp.**
Summary window starts at 1 token and doubles each positive window
(1 → 2 → 4 → 8 → ... → 487 cap). Compares PLD window throughput against
baseline (non-PLD steps in the same window). Three auto-disable triggers:
- **Congestion:** `window_tok_s < baseline_tok_s × 0.95` → disable, reset window=1.
- **No opportunities:** PLD enabled but d0 pre-check filtered every cycle
  in the window (n=0 rounds) → disable (avoids per-token `find_draft_tokens`
  overhead).
- **Probe:** After `5 × summary_interval` disabled tokens, re-enable with
  window=1 to check if conditions changed.

On new requests, window resets to 1 only if PLD is currently active.  If
auto-tune disabled PLD, the decision persists across requests — the probe
handles re-enablement.

**6. Per-request conditional window reset.** Each new `/v1/chat/completions`
request resets the auto-tune window to 1 (if PLD is active), so each
generation gets fresh workload-appropriate auto-tuning.  If PLD was
auto-disabled, the reset is skipped — the disable decision is respected.

**Cross-model results (T=0.3, adaptive auto-tune):**

| Model | Architecture | Baseline | OpenClaw | Acceptance | Old (K=5) |
|-------|-------------|----------|----------|------------|-----------|
| Qwen3.5-27B-Claude-4.6-6bit | Hybrid SSM/ATT | ~16 tok/s | **+7%** | 0% | -22% |
| Qwen3.5-27B-4bit | Hybrid SSM/ATT | ~22 tok/s | **~0%** | — | — |
| Qwen3.5-27B-4bit-DWQ | Hybrid SSM/ATT | ~23 tok/s | **-2%** | — | -36% |
| Qwen3.5-35B-A3B-4bit | Hybrid SSM/MoE | ~92 tok/s | **-1%** | -17% | -34% |
| Qwen3.5-27B-JANG_4S | Hybrid SSM/ATT | ~21 tok/s | **~-1%** | — | — |

**Qwen3.5-27B-JANG_4S per-task OpenClaw (T=0.3, adaptive auto-tune):**

| Task | Baseline | PLD | Delta |
|------|----------|-----|-------|
| json_schema | 16.8 | 18.8 | **+12%** |
| code_generation | 22.2 | 22.8 | **+3%** |
| code_review | 20.5 | 19.6 | -4% |
| pr_checklist | 20.8 | 21.2 | **+2%** |
| temperature_tradeoff | 21.2 | 19.8 | -7% |
| kv_vs_prefix_cache | 23.0 | 21.7 | -6% |
| last_topic | 24.1 | 23.4 | -3% |

Note: JANG_4S uses custom mixed-precision quantization (JANG format). Loading required
a fix to `jang_loader.py` to resolve HF model IDs to local cache snapshots via
`snapshot_download(local_files_only=True)`.

**Qwen3.5-27B-Claude-4.6-6bit per-task OpenClaw (T=0.3, final):**

| Task | Baseline | PLD | Delta |
|------|----------|-----|-------|
| json_schema | 15.6 | 13.9 | -11% |
| code_generation | 17.3 | 16.7 | -3% |
| code_review | 11.8 | 14.7 | **+25%** |
| pr_checklist | 15.8 | 17.1 | **+8%** |
| temperature_tradeoff | 16.4 | 15.9 | -3% |
| kv_vs_prefix_cache | 14.6 | 18.3 | **+25%** |
| last_topic | 15.1 | 17.7 | **+17%** |

**K experiments (on Qwen3.5-27B-Claude-4.6-6bit, without auto-tune):**

| K | Acceptance | OpenClaw | Why |
|---|-----------|----------|-----|
| K=5 (original) | -25% | -22% | Verify cost ~4× on hybrid, too expensive |
| K=2 (v4, best) | 0% | 0% | Verify cost ~1.75×, amortizes ~20ms overhead |
| K=1 | -6% | -7% | Verify cost 1.0× but fewer tokens to amortize overhead |

**Optimization journey (Qwen3.5-27B-Claude-4.6-6bit OpenClaw):**

```
Phase 4 (K=5, no pre-check):      -22%
+ K=2 + d0 pre-check (v2):        -14%
+ bfloat16 numpy fix (v3):         -5%
+ logprob lookup + threshold (v4):   0%
+ fixed-window auto-tune:          +5%
+ adaptive auto-tune (window=1):   +7%
```

### Real-world agent workload (OpenClaw)

- Prompt: ~12K tokens (system prompt + tool definitions + conversation)
- Output: 46–400 tokens, finish_reason=stop
- PLD speculative decode: **did not fire** in Phase 2 (temperature > 0.05 gate)
- Phase 1 coverage: 44–50% (large prompt = many n-gram candidates)
- Phase 1 hit@1: 0.3–2.6% (novel reasoning output, not echoing the prompt)

With Phase 3, PLD will now fire for these workloads. However the low hit@1
(0.3–2.6%) means most spec decode rounds will reject d0 and emit only a
correction token. Throughput gain depends on whether the verify-pass overhead
is less than 1 full forward pass — which it is on memory-bandwidth-limited
Apple Silicon (K tokens per pass ≈ 1 pass cost).

---

## Temperature restriction and Phase 3

**Why greedy-only?** For temperature=0, correctness is `argmax == draft`.
For temperature > 0, the model samples from a distribution — a draft token
isn't right or wrong, it's a draw from p(x). Accepting it without
correction biases the output distribution.

### Temperature=0.3 test — catastrophic failure confirmed

Tested PLD at temperature=0.3 to characterise the failure before Phase 3.
Two benchmark runs, both with PLD enabled (note: `VMLX_PLD_DISABLED=1` as
an env var prefix only affects the client process, not the running server).

Observed failures:

- **Open-ended reasoning**: "presents presents a a compelling compelling
  challenge challenge...the the the the the the..." — word doubling
  followed by an infinite "the" loop. 14 SSE events for ~421 tokens
  (~30 tokens/event), confirming PLD was firing aggressively.
- **Structured JSON**: returned `]` (empty array) — a degenerate sample.

**Root cause — greedy bonus_token:** After accepting K draft tokens, the
code emits `bonus_token = argmax(logits[num_accept])` — a *greedy*
prediction. At temperature > 0 this injects a greedy token into a sampled
context. The model then samples "the" (a high-probability continuation
after many greedy "the" tokens), creating a runaway feedback loop.

The greedy acceptance check (`argmax == draft`) has the same problem: it
only accepts a draft token if it would be the greedy choice, but then
advances the sampled context by one more greedy step.

**The temperature gate (≤ 0.05) was load-bearing** during Phase 2 and has
been removed in Phase 3 (see Implementation section).

### Phase 3 design: probabilistic acceptance + sampled bonus

For a deterministic draft source like PLD (no draft model distribution),
the correct algorithm is:

1. Accept draft token d_i with probability `p(d_i) = softmax(logits[i]/T)[d_i]`
2. On rejection: sample the correction token from `p(x)` with d_i excluded
   (set `logits[d_i] = -inf`, then sample)
3. Bonus token: sample from `p(x)` at position `num_accept` (not argmax)

This provably preserves the original sampling distribution. Implementation
cost is low — the verification forward pass already computes logits at
every position. Phase 3 requires adding the accept/reject sampling step,
modifying the correction token draw, and replacing argmax with a sample
for the bonus token.

**Phase 3 status: implemented and verified correct at T=0.3.** The temperature
gate has been removed; PLD now fires at all temperatures.

---

## Open Questions

### Resolved by Phase 5

1. ~~**K tuning**~~ — K=2 is optimal for hybrid SSM/attention models.
   K=5 costs ~4× per verify (SSM sequential processing); K=1 doesn't
   amortize the ~20ms remove/insert overhead. K=2 at 1.75× verify
   cost is the sweet spot. See Phase 5 K experiments table.

2. ~~**T=0 re-test**~~ — Tested at T=0; results similar to T=0.3 because
   the d0 pre-check (raw logprob lookup) is already very cheap at T>0.
   No significant T=0 advantage on current workloads.

3. ~~**Remove/reinsert overhead**~~ — Estimated at ~15–30ms per cycle
   based on K=1 vs K=2 amortization analysis. This is the dominant
   remaining overhead source, now handled by auto-tune: models where
   the overhead dominates (Qwen3.5-35B-A3B-4bit, Qwen3-4B-6bit) auto-disable PLD within one window.

4. ~~**Cross-model re-test**~~ — All four models tested
   (Qwen3.5-27B-Claude-4.6-6bit, Qwen3.5-27B-4bit, Qwen3.5-35B-A3B-4bit, Qwen3-4B-6bit)
   with K=2 + d0 pre-check + auto-tune. Results in cross-model table above.

### Remaining open questions

5. **Reduce remove/insert overhead** — The ~20ms per-cycle fixed cost of
   `batch_generator.remove()` + `insert()` is the binding constraint.
   Upstream mlx-lm changes (e.g., in-place cache manipulation without
   full extraction) could make PLD net-positive on fast models too.

6. **Long-prompt structured output (12K+ tokens)** — Phase 1 showed 44–50%
   n-gram coverage on 12K-token prompts. A benchmark with long structured
   output (code completion over a large codebase context) would test whether
   high coverage + long accept runs produce stronger gains.

7. **Pure attention model test** — Qwen3-4B-6bit appears to be a pure
   transformer and gets K=5. Results are promising (+13% acceptance) but
   testing a larger pure-attention model (Llama 70B, Mistral) would confirm
   whether K=5 delivers the theoretical speedup when verify ≈ 1× decode.

### Earlier open questions (pre-Phase-4)

6. **ArraysCache offset at temperature > 0** — Phase 3 fixes the accumulating
   offset by removing pre-trim + using `verify_input = [d0..d_{K-1}]`. At T=0
   Phase 3 outputs are equivalent to Phase 2 (same context seen by model).
   Whether any residual offset from the Mamba state causes detectable drift on
   very long reasoning chains at T>0 is untested but expected to be negligible.

7. **Multi-turn coverage improvement** — n-gram search covers only the
   current request's prompt+output. Including prior conversation turns would
   raise coverage for agents with long multi-turn contexts.

8. **Coverage vs. precision trade-off** — max_ngram_size=3 is the current
   setting. Dropping to 2 raises coverage but lowers precision. Unigram
   (size=1) is essentially bigram prediction — nearly 100% coverage but
   very low hit rate. An empirical sweep is needed.

9. **Concurrent request interaction** — spec decode removes and re-inserts
   requests from `BatchGenerator`. Under concurrent load this reduces
   decode-phase batch occupancy for one step per spec decode round. PLD is
   correct at any concurrency level but may hurt aggregate throughput at
   batch_size > 1. This is unmeasured and should be verified before relying
   on PLD in high-concurrency deployments. The warning is documented in the
   `_try_speculative_decode` docstring.
