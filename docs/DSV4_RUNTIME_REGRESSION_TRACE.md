# DSV4-Flash Runtime Regression — Deep Trace (2026-05-05)

**Status:** Investigation in progress. NO code changes pending push.
**Symptom:** DSV4-Flash-JANGTQ at default temp=0.6 + rep=1.05 produces deterministic looping output ("Project: birth, birth, birth..." / "(the project (the project").
**User directive:** Stop chasing temp/rep_penalty bandaids — find runtime root cause.

---

## Timeline of DSV4 fixes vs regressions

| Commit | What | Status |
|--------|------|--------|
| 768df3a3 v1.5.5 | Custom DSV4BatchGenerator (single-batch, stream-thread bypass) | ✓ baseline |
| **00a78db4 v1.5.6** | **3 fixes:** (1) DSV4_POOL_QUANT=0 at CLI startup so make_cache returns DeepseekV4Cache (not PoolQuantizedV4Cache), (2) **force `enable_thinking=True` unless explicit True** (chat-mode bundle is training-data-contaminated), (3) **single-shot prefill** (chunking corrupts compressor+indexer pool state) | ✓ "14/14 probe matrix" |
| 44c571a6 v1.5.8 follow | DSV4 rep_penalty fallback bumped to 1.15 (the "polite-assistant attractor" loop) | ✓ |
| 68091df7 v1.5.10 | "DSV4-Flash JANGTQ runtime fully fixed" | ✓ verified known-good |
| ffe60039 v1.5.11 | DSV4 instant-load via pre-stacked sidecar (`loaders/load_jangtq_dsv4.py`) | added new code path |
| b3a26911 v1.5.12 | re-enable paged + block-disk L2 for DSV4 | added new code path |
| e8c4112f v1.5.13 | DSV4 paged cache stores at any prompt length | added new code path |
| **3ae0b234 v1.5.15** | **REVERTED v1.5.6 fixes (2) and (3):** force-thinking removed, single-shot prefill replaced with chunked at 512. Rep_penalty fallback wiped. New theory: scheduler.py:768 cache contamination fix obviates the v1.5.6 fixes. | ⚠ regression suspect |

---

## Confirmed regressions in v1.5.15

### R1. Chunked prefill re-introduced (REGRESSION)

**File:** `vmlx_engine/utils/dsv4_batch_generator.py:127-194`

```python
_dsv4_step = int(os.environ.get("DSV4_PREFILL_STEP_SIZE", "512"))
self.prefill_step_size = max(1, min(prefill_step_size, _dsv4_step))
...
def _prefill_last_logits(self, token_ids, cache):
    """Chunked DSV4 prefill returning logits for the last prompt token.
    The earlier custom generator used one-shot prefill because an old
    DSV4 cache bug made chunking suspicious. Current DeepseekV4Cache
    accumulates compressor/indexer pool state correctly across calls."""
```

**Problem:** the comment claims the old chunking-suspicion is gone — provides no evidence. v1.5.6 commit message contradicts this directly:

> Chunking corrupted the DSV4 compressor + indexer pool state -> broadcast_shapes (1,N) (1,64,1,128) mid-decode. Post-warmup the model has all kernels JIT-compiled so single-shot prefill stays under the Metal command-buffer watchdog even on long prompts.

**jang-tools `DeepseekV4Cache` at HEAD (2.5.23) vs v1.5.10 baseline (2.5.18):** UNCHANGED. The runtime that v1.5.6 demonstrated to corrupt under chunking is the same runtime present today.

**Hypothesis:** chunked prefill at 512 corrupts CSA/HSA pool buffers on prefill. The corrupted pool state then dominates the decode logits → peaked logits → self-reinforcing token attractor. Sampling temperature has minimal effect because the wrong-attractor logits are already saturated.

### R2. Force-thinking removed (REGRESSION)

**Files:** `vmlx_engine/server.py` — 4 endpoints (chat-completions, Ollama, Anthropic, Responses).

**v1.5.6:** `if chat_req.enable_thinking is not True: chat_req.enable_thinking = True`
**v1.5.15:** `if chat_req.enable_thinking is True: ...; elif chat_req.enable_thinking is False: ...`  (auto/None falls through to engine default).

**v1.5.6 commit explicitly states:**
> DSV4 chat-mode (enable_thinking=False) emits training-data-contaminated output (hallucinated AI-assistant boilerplate, mixed-language annotation leakage); thinking-mode is the verified-clean path for this bundle.

**v1.5.15 commit claim:** "the chat-loop bug that earlier work attributed to 'chat-mode contamination' is [actually cumulative pool-buffer contamination, fixed in scheduler]."  Two contradicting theories — only the second is implemented.

**Observation:** at temp=0.6+rep=1.05 on a SINGLE-TURN request (no multi-turn cumulative pool state), the loop still fires. That contradicts the v1.5.15 cumulative-state theory and supports v1.5.6's chat-mode contamination theory.

### R3. DSV4 rep_penalty fallback wiped (PARTIAL REGRESSION)

**File:** `vmlx_engine/server.py:287` — `_FAMILY_FALLBACK_DEFAULTS = {}` (empty).

**v1.5.8:** `"deepseek_v4": (0.6, 0.95, 1.15)` — verified to short-circuit the loop in 300-400 tokens.
**v1.5.15:** wiped with comment that the cache fix handles it.

The user is right that rep_penalty=1.15 is a bandaid. The real root cause is R1 + R2. But removing 1.15 in v1.5.15 without verifying R1 + R2 weren't ALSO regressed left the engine with neither the root fix nor the bandaid.

---

## What v1.5.15 ADDED that is correct

`scheduler.py` `_uses_dsv4_cache` detection + removal of `non_kv.discard("DeepseekV4Cache")` to route DSV4 through the hybrid cumulative-state path. This is correct for the multi-turn pool-buffer contamination case. trim(n) drops ≥1 pool row at jang-tools `dsv4/mlx_model.py:527` is also sound. Paged cache stores `deepseek_v4_pending` for non-terminal blocks + full state on terminal block. Disk cache schema versioning prevents cross-config poisoning.

This is real correctness work for **multi-turn**. But it does not address **single-turn chunked-prefill corruption** (R1) or **chat-mode contamination** (R2).

---

## Hypothesis ranking after trace

1. **R1 (chunked prefill)** — highest confidence. Direct contradiction with v1.5.6 evidence. Trivial to test by setting `DSV4_PREFILL_STEP_SIZE=999999`.
2. **R2 (chat-mode contamination)** — high confidence. Single-turn loop contradicts v1.5.15's "cumulative state" theory.
3. **R3 (rep_penalty floor wiped)** — bandaid not root cause but its removal removed the safety net.

---

## Proposed isolation test (NOT YET RUN)

Run current vmlx-engine HEAD with:

```
DSV4_PREFILL_STEP_SIZE=999999    # single-shot prefill (revert R1 via env)
```

…and a request that explicitly sets `enable_thinking=True` (works around R2 without code change). If output is coherent, R1+R2 are confirmed and the fix is to revert those two specific v1.5.15 reversions while keeping the v1.5.15 cache-schema work.

If the loop persists with both, dig deeper into Hadamard cache keying / matmul magnitude (user's hypothesis — currently lower-ranked).

---

## What is NOT regressed

- jang-tools DSV4 runtime (`dsv4/mlx_model.py`, `dsv4/encoding_adapter.py`, attention/cache classes) — unchanged 2.5.18 → 2.5.23.
- TurboQuant kernel + Hadamard butterfly — unchanged.
- JANGTQRuntimeCache signs/codebook keying — keys are `(in_features, seed)` and `(in_features, bits)`, current `getattr(existing, "in_features", None)` falls back to `weight.shape[-1]` which gives the right value for SwitchLinear (verified by Ling+MiniMax JANGTQ_K coherent live tests this session).

## Files referenced

- `vmlx_engine/utils/dsv4_batch_generator.py:127-194` (R1)
- `vmlx_engine/server.py:287, 4079-4081, 4804-4806, 6491-6493` (R2, R3)
- `vmlx_engine/scheduler.py:794-848, 1652` (correct v1.5.15 work — _uses_dsv4_cache + rep_context_size)
- `jang_tools/dsv4/mlx_model.py:515-545` (correct trim drop-1 logic)
- `00a78db4` v1.5.6 commit message (root authoritative findings)
