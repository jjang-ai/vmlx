# SSM Async Re-Derive Audit — 2026-05-06

**Spec:** `docs/superpowers/specs/2026-05-06-production-audit-design.md` §8.5, §17.5.
**Tests:** `tests/test_ssm_companion_cache.py` (39 PASS), `tests/test_dsv4_paged_cache.py::test_hybrid_ssm_rederive_uses_n_minus_one_cache_key` (PASS), `tests/test_vl_video_regression.py` SSM re-derive guards (multiple PASS).

## Spec claim vs. reality

The spec §8.5 stated: "`_prefill_for_clean_ssm` exists but is unused for `gpl > 0`."

This is **outdated**. The async re-derive infrastructure is fully wired in `vmlx_engine/scheduler.py` as of the current `main` and supports both `gpl > 0` (thinking) and `gpl = 0` (non-thinking) hybrid SSM models.

## How it works

Three sites in `scheduler.py`:

1. **Store path** (line ~4126): when generation finishes for a hybrid SSM model with prefix cache enabled, the post-generation SSM state is contaminated by gen-prompt + output tokens (gpl-contamination). Instead of either dropping the entry or storing it as `is_complete=False`, the scheduler queues a deferred clean re-prefill in `self._ssm_rederive_queue`. The queue is capped at `SSM_REDERIVE_QUEUE_CAP = 8` entries; the minimum prompt token count is `SSM_REDERIVE_MIN_TOKENS` (small prompts skip enqueue since they're unlikely to be re-requested). The same deferred path runs for `gpl == 0` (non-thinking) too.

2. **Idle handler** (line ~5478): in each scheduler step, if there are no active/waiting/unprocessed requests and `self._ssm_rederive_queue` is non-empty, pop ONE task. Run `self._prefill_for_prompt_only_cache(tokens)` to compute clean SSM state from a fresh forward pass on prompt-only tokens. Extract SSM layers from the clean cache and store via `self._ssm_state_cache.store(tokens, prompt_len, ssm_layers)`. The default `is_complete` flag is `True` for clean entries.

3. **Fetch path** (line ~3210): on a hybrid paged hit, fetch the SSM companion entry. If `is_complete=False` (contaminated state from a legacy store path), reject the hit and full-prefill. If a checkpoint at a shorter length exists with `is_complete=True`, use vmlx#91 RESUME to trim KV blocks to the checkpoint and re-run only the tail.

## RAM safety

A 2026-04-30 release-gate audit caught a real RAM leak in the prior path that
queued re-derives without bounds. Three guards landed:

- Skip enqueue if `companion_len < SSM_REDERIVE_MIN_TOKENS`.
- Skip enqueue if the LRU is already saturated.
- Cap the queue at 8 (was 20 — 60% memory reduction worst-case).

These are pinned by `tests/test_vl_video_regression.py` source-grep guards.

## Test coverage

| Test | Asserts |
|---|---|
| `test_ssm_companion_cache.py` (39 cases) | Store/fetch correctness, byte budget, key alignment N vs N-1, cross-instance disk round-trip, empty edge cases. |
| `test_dsv4_paged_cache.py::test_hybrid_ssm_rederive_uses_n_minus_one_cache_key` | Re-derive uses N-1 cache key (matches store side). |
| `test_vl_video_regression.py::test_v1384_ssm_rederive_skips_oom_prompts_not_chunks` | Re-derive does not split prompts into chunks (legacy chunking caused broadcast bugs); skips OOM-risk prompts instead. |
| `test_vl_video_regression.py::test_scheduler_has_ssm_rederive_queue_path` | Source-grep guard confirms `_ssm_rederive_queue`, `SSM_REDERIVE_QUEUE_CAP`, and `pop(0)` all present. |

Plus several other source-grep guards confirming wiring at the store, queue, and idle-handler sites.

## Live verification

Live multi-turn thinking-mode runs are out of scope for this fast audit but
recorded in the per-arch matrix follow-up (§7). The expected behavior for
turn-2 of a thinking-mode multi-turn conversation:

- Turn 1: hybrid prefill, post-gen SSM contaminated, queue async re-derive.
- Idle period: idle handler runs `_prefill_for_prompt_only_cache(prompt_tokens[:-1])`, stores clean SSM with `is_complete=True`.
- Turn 2 (same prompt prefix): fetch hits paged KV blocks AND the clean SSM companion → cache hit; only the new turn-2 user message gets prefilled.

If the idle period is too short to drain the queue (active high-throughput workload), the entry remains queued; turn 2 gets KV-only hit + full SSM prefill (acceptable graceful degradation).

## Decision

§8.5 is **complete**. The spec entry is updated to reflect the implementation state. No code change required.

## Items for next audit cycle

- Live multi-turn verification on Ling-2.6, Nemotron-Omni, Gemma-4 (the at-risk archs from §8.3 that are also hybrid). Deferred to §7 per-arch matrix.
- Live RAM ceiling verification under a 30-prompt burst to confirm the iter-126/Nemotron RAM-leak class does not recur. Deferred to §7.
