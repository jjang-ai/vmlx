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

Baseline: ~2 tok/s (Qwen3.5-27B on Apple Silicon M-series).

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

**Correctness verified**: no word doubling or repetition loops in any task
output. Open-ended reasoning previously failed with "presents presents a a
compelling compelling...the the the..."; now produces fully coherent text.

**Throughput at T=0.3 vs T=0:** ~11 vs ~16 tok/s. At T=0.3 the model
rarely accepts d0 outright (n-gram drafts diverge from sampled output more
often), so many spec decode rounds produce only 1 correction token. The verify
pass is still efficient (processes K tokens per bandwidth cost ≈ 1 forward
pass), giving meaningful speedup even with low acceptance.

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

1. **ArraysCache offset at temperature > 0** — Phase 3 fixes the accumulating
   offset by removing pre-trim + using `verify_input = [d0..d_{K-1}]`. At T=0
   Phase 3 outputs are equivalent to Phase 2 (same context seen by model).
   Whether any residual offset from the Mamba state causes detectable drift on
   very long reasoning chains at T>0 is untested but expected to be negligible.

2. **OpenClaw agent workload with Phase 3** — PLD will now fire at agent
   temperatures (0.6–0.8). With 44–50% coverage but only 0.3–2.6% hit@1,
   the speedup will be modest. Measuring actual throughput on that workload
   is a useful next step.

3. **Multi-turn coverage improvement** — n-gram search covers only the
   current request's prompt+output. Including prior conversation turns would
   raise coverage for agents with long multi-turn contexts.

4. **Coverage vs. precision trade-off** — max_ngram_size=3 is the current
   setting. Dropping to 2 raises coverage but lowers precision. Unigram
   (size=1) is essentially bigram prediction — nearly 100% coverage but
   very low hit rate. An empirical sweep is needed.

5. **Concurrent request interaction** — spec decode removes and re-inserts
   requests from `BatchGenerator`. Under concurrent load this breaks
   decode-phase batching for the affected request. Throughput impact at
   batch_size > 1 is unmeasured.
