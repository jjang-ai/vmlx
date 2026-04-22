# Architecture — vMLX lm-eval Compatibility

**What belongs here:** High-level system design, component relationships, data flows, and invariants relevant to the lm-evaluation-harness compatibility mission.
**What does NOT belong here:** Implementation details, file-level code patterns, or service ports (use `.factory/services.yaml`).

---

## Components

### vMLXServer (HTTP Layer)
- **Hummingbird 2.x** router with OpenAI-compatible routes
- `OpenAIRoutes.register(on:router:engine:)` wires all endpoints
- JSON request decoding: mixed Codable (`ChatRequest`) + `JSONSerialization` (legacy completions)
- JSON response encoding: `JSONSerialization` into `[String: Any]` dictionaries
- Engine access: `Engine` actor passed into route closures

### vMLXEngine (Orchestration Layer)
- `Engine` actor: loads models, manages `ModelContainer`, streams generation
- `Engine.stream(request:id:)` → `AsyncThrowingStream<StreamChunk, Error>`
- `ModelContainer.perform(_:)` → safely access `ModelContext` (model, tokenizer, processor)
- `Stream.swift`: builds `GenerateParameters`, manages `StreamingDetokenizer`, `ToolCallProcessor`

### vMLXLMCommon (Core Layer)
- `TokenIterator`: prefill + decode loop, holds `LogprobsCollector`
- `LogprobsCollector`: `LogitProcessor` that captures per-token logprobs via `logSoftmax`
- `GenerateParameters`: sampling config + `logprobs`, `topLogprobs`, `echo`, `promptLogprobs`
- `Generation` enum: `.chunk`, `.info`, `.toolCall`, `.logprob(TokenLogprob)`

## Data Flows

### Normal Chat Completion (No Logprobs)
```
Client → OpenAIRoutes → ChatRequest → Engine.stream → Stream.swift
  → GenerateParameters → TokenIterator → model.prepare() → convertToToken()
  → generateLoopTask → AsyncStream<Generation> → StreamChunk → SSEEncoder → Client
```

### Legacy Completions with Logprobs (New)
```
Client → OpenAIRoutes → ChatRequest(logprobs:true) → Engine.stream → Stream.swift
  → GenerateParameters(logprobs:true) → TokenIterator → model.prepare()
  → convertToToken() → LogprobsCollector.capture() → Generation.logprob
  → StreamChunk.logprobs → SSEEncoder.textCompletionStream (streaming)
  → OR accumulated allLogprobs → legacy format JSON (non-streaming)
```

### Prompt Logprobs with Echo (New)
```
Client → OpenAIRoutes → ChatRequest(echo:true, logprobs:true) → Engine.stream
  → GenerateParameters(echo:true, logprobs:true) → TokenIterator
  → prepare(): capture full [1, seq_len, vocab] logits → batched logSoftmax
  → index out per-position logprobs → store promptLogprobs
  → decode loop: capture completion logprobs → merge prompt + completion
  → legacy format JSON with tokens, token_logprobs, top_logprobs, text_offset
```

### Loglikelihood Fast-Path (New)
```
Client → OpenAIRoutes → ChatRequest(maxTokens:1, echo:true, logprobs:1)
  → Engine.loglikelihood(request:) [NEW fast-path entrypoint]
  → ctx.model() directly (NOT model.prepare() — prepare() returns .tokens discarding logits)
  → batched logSoftmax over full sequence
  → index actual token IDs → build legacy logprobs response
  → NO AsyncStream, NO StreamingDetokenizer, NO ToolCallProcessor
```

**Why `model()` instead of `model.prepare()`**: `LLMModel.prepare()` returns `.tokens` (discarding intermediate logits), which makes it unsuitable when full-sequence logits are needed for logprob capture. The fast-path calls `ctx.model()` directly with a fresh KV cache to obtain the full logit tensor `[1, seq_len, vocab]`.

## Tokenizer Protocol Limitations

The `Tokenizer` protocol (vMLXLMCommon/Tokenizer.swift) has limited metadata surface:
- **Available:** `bosToken`, `eosToken`, `unknownToken`
- **Not available:** `padToken` — no property exists on the protocol. The tokenizer endpoints map `pad_token` to `unknownToken` as a pragmatic stand-in. For models where `pad_token ≠ unk_token`, this will be incorrect.
- **Not available:** Raw `chat_template` string — only `applyChatTemplate(messages:...)` is exposed (a method, not the template itself). The raw template lives in the model configuration layer (`CapabilityDetector`), not on the Tokenizer protocol. The tokenizer endpoints return `chat_template: null`.

Future work requiring these properties will need to either extend the Tokenizer protocol or access the model configuration layer directly.

## Key Invariants

1. **Zero overhead for normal requests**: `logSoftmax` over full sequence is ONLY computed when `echo == true || promptLogprobs > 0`. Normal chat completions use the existing path unchanged.
2. **Batched operations**: Prompt logprobs use single `logSoftmax` over `[1, seq_len, vocab]`, not per-token loops.
3. **O(V) top-K**: `argPartition` instead of `argSort` for top-K extraction.
4. **Raw prefill logits**: Prompt logprobs use pre-penalty logits from `model.prepare()`. Penalties are not applied to prompt positions.
5. **Position 0 is null**: First token has no prior context → `token_logprobs[0] == null`.
6. **Legacy format exactness**: `tokens`, `token_logprobs`, `top_logprobs`, `text_offset` must match OpenAI legacy shape exactly for lm-eval `parse_logprobs()`.
7. **`.tokens` fallback limitation**: When `model.prepare()` returns `.tokens` (instead of `.logits`), prompt logprobs cannot be captured — `promptLogprobsResult` remains nil with no error. This affects only model types that return `.tokens` from `prepare()`; standard LLM models return `.logits`.
8. **bfloat16 upcast before logSoftmax**: MLX's `logSoftmax` on bfloat16 tensors can produce NaN/-Inf. All logprob computation paths must upcast to float32 before calling `logSoftmax`. There are 4 upcast locations: `LogprobsCollector.capture()` (Evaluate.swift:~370), sampler logit processing (Evaluate.swift:~463), prompt logprobs in `TokenIterator` (Evaluate.swift:~1146), and `Engine.loglikelihood` (Engine.swift:~1677).
