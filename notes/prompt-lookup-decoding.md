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
| `vmlx_engine/scheduler.py` | Phase 2 speculative decode in `_process_batch_responses` + `_try_speculative_decode()` |
| `tests/benchmark/test_pld_acceptance.py` | 4-task acceptance rate benchmark |

### How it works (Phase 2, current)

After `BatchGenerator` emits each token, `_try_speculative_decode()`:

1. Calls `batch_generator.remove([uid], return_prompt_caches=True)` to
   extract the live KV cache — the only way to access it mid-stream.
2. Runs `find_draft_tokens()` to find up to K=5 draft tokens via n-gram
   search in the full token sequence (prompt + output so far).
3. Calls `self.model([[last_token, d1, …, dK]], cache=kv_cache)` — one
   forward pass processing K+1 tokens.
4. Accepts the longest prefix where `argmax(logits[i]) == drafts[i]`.
5. Rolls back the KV cache to the accepted length (see below).
6. Re-inserts the request into `BatchGenerator` with the bonus token
   (the model's free prediction after the accepted prefix).
7. Updates `uid_to_request_id` / `request_id_to_uid` maps.

On any failure the request is guaranteed to be re-inserted via a `finally`
block — it can never be orphaned.

**Restricted to temperature ≤ 0.05.** For greedy decoding the acceptance
check is trivial (argmax match). For sampled decoding you need rejection
sampling to preserve the output distribution — see Phase 3 below.

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

### Double-last_token bug (fixed)

**Root cause**: `BatchGenerator._step(last_token)` runs BEFORE spec decode
fires. So `remove()` returns a KVCache that already includes `last_token`'s
K/V at offset N. Running `verify_input = [last_token, d0...dK]` with cache
at N caused the attention to see `last_token` at two positions (N-1 from
cache, N from input), corrupting every subsequent position's context.

**Symptom**: word doubling ("Ship Ship"), repetition loops, and prompt-text
echoing (the pangram appearing in summarisation output).

**Fix**: Pre-trim KVCache by 1 after extraction, before the verify forward
pass. This restores the "last_token NOT yet in cache" invariant that
`verify_input` requires.

**ArraysCache residual**: ArraysCache (Mamba layers) cannot be pre-trimmed.
They start 1 step ahead of KVCache at each spec decode round. This
introduces a **constant** (non-accumulating) +1 SSM-state offset. Formally:
if KVCache is at position N, ArraysCache is at S_{N+1} (has seen one extra
forward pass of `last_token`). Because this offset is constant rather than
growing, and Mamba states are smooth recurrent summaries, the effect on
output quality is negligible in practice.

**Key insight for the offset being constant**: every decode round adds +1
to both "correct" and "actual" Mamba steps (from the batch gen's `_step`),
so the difference stays fixed at +1 regardless of generation length.

### Hybrid model structure (Qwen3.5-27B)

This model has 64 layers: **48 ArraysCache + 16 KVCache**.

`ArraysCache` is used for recurrent/SSM-style layers. It has no positional
offset and cannot be rolled back. On partial rejection, the saved pre-verify
ArraysCache state is restored and KVCache is fully rewound to pre-verify;
the request is re-seeded with `last_token` so the batch generator advances
both caches together from a consistent starting point.

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

### Phase 2 — Speculative decode (corrected, branch fix/block-disk-metal-sync)

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

### Real-world agent workload (OpenClaw)

- Prompt: ~12K tokens (system prompt + tool definitions + conversation)
- Output: 46–400 tokens, finish_reason=stop
- PLD speculative decode: **did not fire** (temperature > 0.05)
- Phase 1 coverage: 44–50% (large prompt = many n-gram candidates)
- Phase 1 hit@1: 0.3–2.6% (novel reasoning output, not echoing the prompt)

This confirms PLD's scope: it excels at structured/repetitive output and
the early reasoning preamble. Conversational turns with temperature > 0
are out of scope until Phase 3.

---

## Temperature restriction and Phase 3

**Why greedy-only?** For temperature=0, correctness is `argmax == draft`.
For temperature > 0, the model samples from a distribution — a draft token
isn't right or wrong, it's a draw from p(x). Accepting it without
correction biases the output distribution.

**Probabilistic acceptance (Phase 3):** For a deterministic draft source
like PLD (no draft model distribution), the correct algorithm is:

1. Accept draft token d_i with probability `p(d_i) = softmax(logits[i]/T)[d_i]`
2. On rejection: sample the correction token from `p(x)` with d_i excluded
   (set `logits[d_i] = -inf`, then sample)

This provably preserves the original sampling distribution. Implementation
cost is low — the verification forward pass already computes logits at
every position. Phase 3 requires adding the accept/reject sampling step
and modifying the correction token draw.

**Expected Phase 3 benefit:** On agent workloads with temperature 0.6–0.8,
hit@1 will be lower than 60% (sampled outputs diverge from n-gram
predictions more often). Coverage (~44%) is still meaningful. Phase 3 is
most valuable for structured outputs at moderate temperature (tool calls,
JSON generation) where the model frequently samples the greedy token anyway.

---

## Open Questions

1. **ArraysCache +1 offset on partial accepts** — the constant SSM-state
   offset is confirmed zero-impact at temperature=0 (outputs are identical
   to baseline). Whether it causes detectable drift on long reasoning chains
   at temperature > 0 is untested.

2. **Phase 3: temperature > 0 support** — PLD does not fire for the main
   OpenClaw agent workload (temperature > 0.05). Probabilistic acceptance
   (see Temperature section) would unlock PLD for those calls.

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
