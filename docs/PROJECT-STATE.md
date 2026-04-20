# Project State

**Last Updated:** 2026-04-20

## Current Phase
- Phase: iter-96 — Feature Development (Logprobs + Benchmark Compatibility)
- Focus: OpenAI-compatible logprobs for chat/completions endpoint
- Progress: Logprobs implemented and committed. PR #99 open. Benchmark compatibility plan documented.

## Last Session
- Date: 2026-04-20
- What Done: Implemented per-token logprobs pipeline, updated e2e tests, created benchmark harness compatibility plan, submitted PR #99
- See: `docs/Handoffs/handoff-012.md`
- Commits: `6a2066d`

## Next Steps
1. Forward `logprobs`/`top_logprobs` on `/v1/completions` handler — `Sources/vMLXServer/Routes/OpenAIRoutes.swift` line ~320
2. Add `/tokenizer_info`, `/tokenize`, `/detokenize` routes — `Sources/vMLXServer/Routes/OpenAIRoutes.swift`
3. Implement `echo:true` + prompt logprob capture during prefill — `Sources/vMLXLMCommon/Evaluate.swift`
4. Legacy completions logprobs format — `Sources/vMLXServer/Routes/OpenAIRoutes.swift` + `SSEEncoder.swift`

## Blockers
- None. PR #99 awaiting review.

## Decisions Made This Session
- Post-penalty logprobs: Computed after repetition/presence/frequency penalties to match actual sampling distribution
- Chat completions format: Uses OpenAI `content` array format (not legacy flat-array)
- Prompt logprobs deferred: Requires modifying prefill pipeline — documented as Phase 3 in benchmark plan

## Files To Know
- `Sources/vMLXLMCommon/Evaluate.swift` - LogprobsCollector, TokenLogprob, Generation.logprob case
- `Sources/vMLXEngine/Stream.swift` - pendingLogprobs accumulator and flush logic
- `Sources/vMLXServer/Routes/OpenAIRoutes.swift` - Logprobs accumulation/serialization in responses
- `docs/LM-EVAL-HARNESS-COMPATIBILITY.md` - 5-phase benchmark plan
- `docs/LOGPROBS-IMPLEMENTATION.md` - Architecture reference
