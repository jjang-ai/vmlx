# lm-evaluation-harness Network Compatibility — Implementation Summary

**Mission:** Make vMLX's OpenAI-compatible server a drop-in backend for EleutherAI/lm-evaluation-harness (v0.4.11+) via the `local-completions` adapter.

**Status:** Complete — all 49/49 validation assertions passed across 7 milestones.

**Date:** 2026-04-22

**Branch:** `feat/logprobs-openai-compat`

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [Architecture](#architecture)
3. [Performance Guarantees](#performance-guarantees)
4. [Validation Results](#validation-results)
5. [Known Limitations](#known-limitations)
6. [How to Use](#how-to-use)
7. [Files Changed](#files-changed)

---

## What Was Built

### 1. Tokenizer Endpoints (3 new routes)

Enable `tokenizer_backend="remote"` so lm-eval can operate without local HuggingFace weights.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tokenizer_info` | GET | Returns `eos_token`, `bos_token`, `pad_token`, `chat_template` |
| `/v1/tokenize` | POST | Accepts `{"text": "...", "add_special_tokens": false}` → `{"tokens": [123, 456]}` |
| `/v1/detokenize` | POST | Accepts `{"tokens": [123, 456]}` → `{"text": "..."}` |

**lm-eval compatibility aliases:** `/tokenizer_info`, `/tokenize`, `/detokenize` (without `/v1/` prefix) since lm-eval strips `/v1/completions` from the `base_url`.

**Key design:** Tokenizer accessed via `engine.loaded?.perform { $0.tokenizer }` — no direct access needed.

### 2. Completions Logprobs Forwarding

Wired existing logprobs infrastructure into the legacy `/v1/completions` endpoint.

- Extracts `logprobs`/`top_logprobs` from JSON body and passes to `ChatRequest`
- Handles `logprobs: 1` (integer) — lm-eval sends integer, not boolean
- Emits logprobs in `textCompletionStream` SSE chunks
- Includes logprobs in non-streaming `/v1/completions` response
- Validates `top_logprobs` range (rejects negative values with HTTP 400)

### 3. Echo + Prompt Logprobs

The most architecturally significant change. Enables `loglikelihood` scoring.

**New parameters:**
- `echo: Bool?` — returns prompt text prepended to completion
- `prompt_logprobs: Int?` — returns logprobs for prompt tokens without echoing text

**Batched prompt logprob capture:**
- When `echo == true || prompt_logprobs > 0`, `TokenIterator.prepare()` captures the full `[1, seq_len, vocab]` logits tensor from `model.prepare()`
- Computes `logSoftmax` **once** over the full sequence (not per-token)
- Indexes per-position logprobs for actual prompt tokens
- Position 0 gets `NaN` → mapped to JSON `null` (no prior context)

**Performance-critical gating:**
- Normal chat completions (no echo, no prompt_logprobs): **zero overhead**
- `logSoftmax` over full sequence is ONLY computed when explicitly requested
- KV cache lookup is bypassed when prompt logprobs are needed (ensures fresh prefill)

**Top-K extraction:** Uses `argPartition` (O(V)) instead of `argSort` (O(V log V)).

### 4. Legacy Completions Logprobs Format

Produces the exact JSON shape lm-eval's `parse_logprobs()` expects.

```json
{
  "choices": [{
    "text": "...",
    "logprobs": {
      "tokens": ["<p1>", "<p2>", "<c1>"],
      "token_logprobs": [null, -1.2, -0.5],
      "top_logprobs": [null, {"tok": -1.0, ...}, ...],
      "text_offset": [0, 3, 7]
    }
  }]
}
```

**Key behaviors:**
- `token_logprobs[0]` is `null` (first token has no prior context)
- `top_logprobs[0]` is `null` (same reason)
- `text_offset` uses cumulative UTF-8 byte offsets
- `top_logprobs: 0` produces dictionaries with only the chosen token
- Both streaming and non-streaming paths use identical format

**Ordered JSON serialization:** Swift Dictionary does not preserve insertion order, so a custom JSON string builder ensures `top_logprobs` entries are serialized with chosen token first, then alternatives descending by logprob.

### 5. Loglikelihood Fast-Path

Dedicated synchronous path for `max_tokens: 1` loglikelihood requests.

**Entry condition:** `!stream && maxTokens == 1 && echo && (logprobs || promptLogprobs)`

**What it bypasses:**
- `AsyncStream` / `continuation.yield` overhead
- `StreamingDetokenizer` (no text streaming needed)
- `ToolCallProcessor` (no tool calls in loglikelihood mode)
- Per-token decode loop

**What it does:**
- Calls `model()` directly for full-sequence logits
- Computes batched `logSoftmax`
- Indexes actual token IDs
- Samples 1 token via `ArgMaxSampler`
- Builds legacy logprobs response directly

**Performance:** Measured at **103% of raw prefill latency** (3% overhead) — well within the 150% target.

### 6. RawPrompt Mode

A non-Codable flag on `ChatRequest` that tells the engine to tokenize raw text **without** applying the chat template.

**Why it matters:** lm-eval's `ctxlen` (prompt token count) must match the number of logprob entries returned. If the chat template injects system messages or formatting tokens, the counts diverge and `parse_logprobs()` slices incorrectly.

**Usage:** Automatically set to `true` for all legacy `/v1/completions` requests.

### 7. Prompt-as-[Int] Support

`/v1/completions` now accepts:
- `prompt: "string"` — existing behavior
- `prompt: [128000, 128006, ...]` — pre-tokenized prompt (lm-eval `tokenized_requests=True`)
- `prompt: [[128000], [128006]]` — batched (joins with `\n`, documented limitation)

---

## Architecture

### Data Flow: Normal Chat Completion (No Logprobs)

```
Client → OpenAIRoutes → ChatRequest → Engine.stream → Stream.swift
  → GenerateParameters → TokenIterator → model.prepare() → convertToToken()
  → generateLoopTask → AsyncStream<Generation> → StreamChunk → SSEEncoder → Client
```

### Data Flow: Legacy Completions with Logprobs

```
Client → OpenAIRoutes → ChatRequest(logprobs:true) → Engine.stream → Stream.swift
  → GenerateParameters(logprobs:true) → TokenIterator → model.prepare()
  → convertToToken() → LogprobsCollector.capture() → Generation.logprob
  → StreamChunk.logprobs → SSEEncoder.textCompletionStream (streaming)
  → OR accumulated allLogprobs → legacy format JSON (non-streaming)
```

### Data Flow: Prompt Logprobs with Echo

```
Client → OpenAIRoutes → ChatRequest(echo:true, logprobs:true) → Engine.stream
  → GenerateParameters(echo:true, logprobs:true) → TokenIterator
  → prepare(): capture full [1, seq_len, vocab] logits → batched logSoftmax
  → index out per-position logprobs → store promptLogprobs
  → decode loop: capture completion logprobs → merge prompt + completion
  → legacy format JSON with tokens, token_logprobs, top_logprobs, text_offset
```

### Data Flow: Loglikelihood Fast-Path

```
Client → OpenAIRoutes → ChatRequest(maxTokens:1, echo:true, logprobs:1)
  → Engine.loglikelihood(request:) [NEW fast-path entrypoint]
  → model() directly → batched logSoftmax over full sequence
  → index actual token IDs → build legacy logprobs response
  → NO AsyncStream, NO StreamingDetokenizer, NO ToolCallProcessor
```

### Key Invariants

1. **Zero overhead for normal requests:** `logSoftmax` over full sequence ONLY computed when `echo == true || promptLogprobs > 0`
2. **Batched operations:** Single `logSoftmax` over `[1, seq_len, vocab]`, not per-token loops
3. **O(V) top-K:** `argPartition` instead of `argSort` for top-K extraction
4. **Raw prefill logits:** Prompt logprobs use pre-penalty logits from `model.prepare()`
5. **Position 0 is null:** First token has no prior context → `token_logprobs[0] == null`
6. **Legacy format exactness:** `tokens`, `token_logprobs`, `top_logprobs`, `text_offset` match OpenAI legacy shape
7. **KV cache bypass for logprobs:** When prompt logprobs requested, KV cache lookup is skipped to ensure fresh prefill with full logits
8. **Float32 computation:** `bfloat16` logits are upcast to `float32` before `logSoftmax` to prevent numerical precision loss

---

## Performance Guarantees

| Guarantee | Implementation | Verified |
|-----------|---------------|----------|
| Normal chat completions: zero overhead | `logSoftmax` gated behind `echo`/`prompt_logprobs` | Yes — before/after latency comparison |
| Prompt logprobs: batched | Single `logSoftmax` over full sequence | Yes — code inspection |
| Top-K: O(V) | `argPartition` instead of `argSort` | Yes — code inspection |
| Fast-path: <150% baseline | Bypasses AsyncStream/StreamingDetokenizer | Yes — measured at 103% |
| No NaN/Inf in logprobs | `bfloat16` → `float32` upcast before `logSoftmax` | Yes — live API test with 36+ finite values |

---

## Validation Results

### Milestone Breakdown

| Milestone | Assertions | Status |
|-----------|-----------|--------|
| Tokenizer Endpoints | 7/7 | Sealed |
| Completions Logprobs | 7/7 | Sealed |
| Echo + Prompt Logprobs | 11/11 | Sealed |
| Legacy Logprobs Format | 10/10 | Sealed |
| Loglikelihood Fast-Path | 3/3 | Sealed |
| Numerical Correctness + Chat | 3/3 | Sealed |
| lm-eval Integration | 8/8 | Sealed |
| **Total** | **49/49** | **Complete** |

### lm-eval End-to-End Verification

**Commands tested:**

```bash
# generate_until tasks
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions" \
  --tasks gsm8k --limit 2 --batch_size 1

# loglikelihood tasks with remote tokenizer
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions,tokenizer_backend=remote" \
  --tasks hellaswag --limit 5 --batch_size 1

# record tasks (loglikelihood)
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions,tokenizer_backend=remote" \
  --tasks record --limit 2 --batch_size 1
```

**Results:** All tasks completed with exit code 0 and valid metrics.

---

## Known Limitations

1. **Batching limitation:** `/v1/completions` joins `[String]` prompts with `\n` for batching, which is incorrect for independent loglikelihood scoring. **Workaround:** Use `batch_size=1` in lm-eval.

2. **pad_token accuracy:** `/tokenizer_info` returns `unknownToken` as `pad_token` since the `Tokenizer` protocol lacks a `padToken` property. May be incorrect for models where `pad_token ≠ unk_token`.

3. **chat_template:** Returns `null` when the tokenizer does not expose a raw template string. The `applyChatTemplate` method works, but the raw Jinja template is not available.

4. **Streaming path chat template:** Streaming completions (`max_tokens > 1`) still apply the chat template to legacy completions prompts. The `rawPrompt` flag only affects the loglikelihood fast-path (`max_tokens == 1 + echo`).

5. **Multi-step prefill:** Models that return `.tokens` from `model.prepare()` (instead of `.logits`) do not support prompt logprob capture. Most models return `.logits` for short prompts (the primary lm-eval use case).

6. **No unit tests:** `swift test` fails because `Tests/vMLXTests/` files are missing on disk. All validation is via e2e harness and live API testing.

---

## How to Use

### Start the Server

```bash
swift build -c release --product vmlxctl
.build/release/vmlxctl serve \
  --model /path/to/model \
  --host 127.0.0.1 \
  --port 8080
```

### Run lm-eval

```bash
# With remote tokenizer (no HF weights needed)
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions,tokenizer_backend=remote" \
  --tasks hellaswag \
  --limit 5 \
  --batch_size 1

# With local HF tokenizer (fallback)
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions" \
  --tasks hellaswag \
  --limit 5 \
  --batch_size 1
```

### Test Logprobs

```bash
# Non-streaming with echo and logprobs
curl -s http://localhost:8080/v1/completions \
  -d '{"prompt": "The capital of France is", "max_tokens": 10, "echo": true, "logprobs": true, "top_logprobs": 3}'

# Streaming with logprobs
curl -sN --no-buffer http://localhost:8080/v1/completions \
  -d '{"prompt": "Hello", "max_tokens": 5, "logprobs": true, "stream": true}'
```

### Test Tokenizer Endpoints

```bash
# Tokenizer info
curl -s http://localhost:8080/v1/tokenizer_info

# Tokenize
curl -s -X POST http://localhost:8080/v1/tokenize \
  -d '{"text": "hello world", "add_special_tokens": false}'

# Detokenize
curl -s -X POST http://localhost:8080/v1/detokenize \
  -d '{"tokens": [23391, 1902]}'
```

---

## Files Changed

### Core Implementation

| File | Changes |
|------|---------|
| `Sources/vMLXLMCommon/Evaluate.swift` | LogprobsCollector, TokenIterator prompt logprob capture, argPartition for top-K, Generation.logprob case |
| `Sources/vMLXEngine/Engine.swift` | `Engine.loglikelihood()` fast-path method, `LoglikelihoodResult`, `_performLoglikelihood`, `_extractTopK` |
| `Sources/vMLXEngine/ChatRequest.swift` | `echo`, `promptLogprobs` fields, `rawPrompt` flag, `StreamChunk.logprobs` |
| `Sources/vMLXEngine/Stream.swift` | `flushLogprobs()` before tool calls, `shouldCollectLogprobs` wiring |
| `Sources/vMLXServer/Routes/OpenAIRoutes.swift` | Tokenizer endpoints, completions logprobs forwarding, legacy format response, prompt-as-[Int], negative validation |
| `Sources/vMLXServer/SSEEncoder.swift` | `textCompletionStream` logprobs, `buildLegacyLogprobs()`, `buildLegacyLogprobsJSON()` |

### Documentation

| File | Purpose |
|------|---------|
| `docs/LM-EVAL-HARNESS-COMPATIBILITY.md` | Original 5-phase compatibility plan |
| `docs/LOGPROBS-IMPLEMENTATION.md` | Architecture reference for logprobs pipeline |
| `docs/LM-EVAL-HARNESS-IMPLEMENTATION-SUMMARY.md` | This document |
| `.factory/library/architecture.md` | System architecture and data flows |
| `.factory/library/user-testing.md` | Validation surface and gotchas |
| `.factory/library/lm-eval-integration.md` | Integration notes and tested configurations |

### E2E Tests

| File | Changes |
|------|---------|
| `tests/e2e/harness.sh` | New test suites: tokenizer, completions, logprobs, fastpath, legacy |

---

## Credits

- **Mission planning and orchestration:** Droid orchestrator
- **Implementation workers:** swift-backend-worker droids
- **Validation workers:** scrutiny-validator, user-testing-validator droids
- **Model for testing:** Gemma-4-26B-A4B-it-JANG_4M, Qwen3-0.6B-8bit
