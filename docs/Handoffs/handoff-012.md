# Handoff 012 - OpenAI Logprobs + Benchmark Harness Plan

**Date:** 2026-04-20
**Session Type:** Development + Documentation

## Summary
Implemented OpenAI-compatible per-token log probability collection and emission across the full vMLX generation pipeline (streaming + non-streaming). Created a 5-phase compatibility plan for lm-evaluation-harness `local-completions` backend. Submitted PR #99.

## Changes Made
- LogprobsCollector (LogitProcessor) captures post-penalty log-softmax per token
- TokenLogprob/TopTokenLogprob public types in Evaluate.swift
- Generation.logprob case emitted from generateLoopTask
- Stream.swift batches .logprob events and flushes before content yields
- ChatRequest no longer rejects logprobs in validate()
- StreamChunk gains logprobs field
- SSEEncoder encodes logprobs in SSE frames (content array format)
- OpenAIRoutes accumulates logprobs in non-streaming responses
- vMLXServer depends on vMLXLMCommon for TokenLogprob access
- E2E tests: strict logprobs validation, streaming logprobs test
- Documentation: LOGPROBS-IMPLEMENTATION.md + LM-EVAL-HARNESS-COMPATIBILITY.md

## Files Changed
- `Package.swift` - vMLXServer gains vMLXLMCommon dependency
- `Sources/vMLXLMCommon/Evaluate.swift` - LogprobsCollector, TokenLogprob, TopTokenLogprob, Generation.logprob case, GenerateParameters.logprobs/topLogprobs, TokenIterator logprob capture
- `Sources/vMLXEngine/ChatRequest.swift` - Removed logprobs rejection, added StreamChunk.logprobs field
- `Sources/vMLXEngine/Stream.swift` - pendingLogprobs accumulator, flushLogprobs(), shouldCollectLogprobs, logprobs param wiring
- `Sources/vMLXServer/Routes/OpenAIRoutes.swift` - allLogprobs accumulation and serialization in chat/completions/responses
- `Sources/vMLXServer/SSEEncoder.swift` - Logprobs in SSE frames
- `tests/e2e/harness.sh` - Strict logprobs tests, streaming logprobs test
- `docs/LOGPROBS-IMPLEMENTATION.md` - Architecture reference
- `docs/LM-EVAL-HARNESS-COMPATIBILITY.md` - 5-phase benchmark plan

## Commits
- `6a2066d` - iter-96 §123: OpenAI-compatible logprobs for chat/completions + benchmark plan

## PR
- https://github.com/jjang-ai/vmlx/pull/99 (feat/logprobs-openai-compat → dev)

## Decisions Made
- Logprobs are computed post-penalty (repetition/presence/frequency) to reflect actual sampling distribution
- OpenAI chat completions format (`content` array of objects) used, not legacy flat-array format
- Prompt logprobs (echo:true) deferred to future work — requires modifying prefill pipeline
- lm-eval compatibility broken into 5 phases documented in LM-EVAL-HARNESS-COMPATIBILITY.md

## Next Steps
1. Phase 2: Forward `logprobs`/`top_logprobs` params on `/v1/completions` handler (~10 lines)
2. Phase 1: Add `/tokenizer_info`, `/tokenize`, `/detokenize` routes for remote tokenizer support
3. Phase 3: Implement `echo:true` + prompt logprob capture during prefill (large change)
4. Phase 4: Legacy completions logprobs format (flat arrays: tokens, token_logprobs, top_logprobs, text_offset)
5. Phase 5: End-to-end `lm_eval --model local-completions` integration testing

## Blockers
- None. PR #99 awaiting review.
- Push access to `jjang-ai/vmlx` uses fork workflow (`SandorDobi/vmlx` fork).

## Notes
- `generate_until` tasks in lm-eval already work via `/v1/completions`
- `loglikelihood` tasks require echo + prompt logprobs (Phase 3+4)
- The harness sends `logprobs: 1` (integer) not `logprobs: true` on the completions endpoint — worth verifying serde handles both
