# lm-evaluation-harness Network Compatibility Plan

> Status: **Planning** — not yet implemented.
> Date: 2026-04-20
> Depends on: logprobs infrastructure (already merged in iter-96).

## 1. Goal

Make vMLX's OpenAI-compatible server work as a drop-in backend for
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(v0.4.11+) via the `local-completions` model adapter, using **only** HTTP
requests — no local weights or HuggingFace tokenizer required.

## 2. Current State

| Feature | Status | Notes |
|---------|--------|-------|
| `/v1/completions` endpoint | **Exists** | Wraps prompt as user message, dispatches through `Engine.stream`. Returns `text_completion` format. |
| `logprobs` param on completions | **Not forwarded** | `ChatRequest` supports `logprobs`/`topLogprobs`, but the `/v1/completions` handler does not pass them from the JSON body. |
| `echo` param | **Not supported** | `ChatRequest` has no `echo` field. The completions handler ignores it. |
| Prompt logprobs | **Not captured** | `LogprobsCollector` only records logprobs for *generated* tokens. Prompt token logits are discarded during prefill. |
| Legacy completions logprobs format | **Not emitted** | The completions handler response has no `logprobs` key in `choices[]`. The SSE encoder for `textCompletionStream` does not emit logprobs. |
| `/tokenizer_info` endpoint | **Missing** | Required by `tokenizer_backend="auto"` (remote tokenizer detection). |
| `/tokenize` endpoint | **Missing** | Required by `RemoteTokenizer.encode()`. |
| `/detokenize` endpoint | **Missing** | Required by `RemoteTokenizer.decode()`. |
| `generate_until` tasks | **Works now** | Greedy/temperature generation through `/v1/completions` with `prompt` + `max_tokens` already functions. |
| `loglikelihood` tasks | **Blocked** | Requires `echo: true` + `logprobs: N` + legacy flat-array format. |

### What `generate_until` needs (already works)

```
POST /v1/completions
{
  "model": "...",
  "prompt": "The capital of France is",
  "max_tokens": 32,
  "temperature": 0,
  "stop": ["\n"]
}
```

Response: `choices[0].text` — already returned correctly.

### What `loglikelihood` needs (does NOT work yet)

```
POST /v1/completions
{
  "model": "...",
  "prompt": "<ctx+continuation tokens>",
  "temperature": 0,
  "max_tokens": 1,
  "logprobs": 1,
  "echo": true
}
```

Expected response shape:

```json
{
  "choices": [{
    "text": "...",
    "index": 0,
    "finish_reason": "length",
    "logprobs": {
      "tokens": ["<p1>", "<p2>", ..., "<c1>"],
      "token_logprobs": [null, -1.2, -0.5, ...],
      "top_logprobs": [null, {"tok": -1.0, ...}, ...],
      "text_offset": [0, 3, 7, ...]
    }
  }],
  "usage": { ... }
}
```

The harness's `parse_logprobs()` slices `token_logprobs[ctxlen:-1]` to extract
only the continuation's logprobs and sums them. The `[ctxlen:-1]` slice drops
the final `null` (EOS position) and takes only the continuation portion.

## 3. Implementation Plan

### Phase 1: Tokenizer Endpoints

Enable `tokenizer_backend="auto"` → `tokenizer_backend="remote"` in lm-eval.

#### 3.1 `GET /tokenizer_info`

Returns tokenizer metadata consumed by `RemoteTokenizer`.

```json
{
  "eos_token": "</s>",
  "bos_token": "<s>",
  "pad_token": "<pad>",
  "chat_template": "{% for message in messages %}..."
}
```

Implementation: read from the loaded `Tokenizer` instance already held by the engine.

#### 3.2 `POST /tokenize`

Tokenizes a string into token IDs.

Request: `{"prompt": "hello world", "add_special_tokens": false}`
Response: `{"tokens": [1234, 5678]}`

#### 3.3 `POST /detokenize`

Decodes token IDs back to a string.

Request: `{"tokens": [1234, 5678]}`
Response: `{"prompt": "hello world"}`

### Phase 2: Forward `logprobs`/`top_logprobs` on `/v1/completions`

**Minimal change** — the `ChatRequest` already has `logprobs`/`topLogprobs`
fields decoded from JSON, but the `/v1/completions` handler uses the
*programmatic* init and doesn't pass them.

Changes:
- `OpenAIRoutes.swift` line ~320: extract `logprobs`, `top_logprobs` from
  the JSON body and pass them through to `ChatRequest`.
- Since `ChatRequest.init(model:messages:...)` doesn't have `logprobs`/`topLogprobs`
  parameters, either:
  - (a) Add them to the programmatic init, or
  - (b) Set them after construction: `chatReq.logprobs = obj["logprobs"] as? Bool`

This enables **completion logprobs** (generated tokens only) — useful for
debugging but insufficient for lm-eval's `loglikelihood` scoring.

### Phase 3: `echo` Support + Prompt Logprobs

This is the most architecturally significant change.

#### 3.4 `echo` Parameter

- Add `echo: Bool?` to `ChatRequest`.
- Forward from `/v1/completions` JSON body.
- Propagate to `GenerateParameters`.

#### 3.5 Prompt Logprob Capture During Prefill

**The core challenge**: during prefill, the model processes all prompt tokens
in a single batched forward pass. The logits at each position represent the
model's prediction for the *next* token. To get each prompt token's logprob,
we need the logit at position `i-1` evaluated over the token at position `i`.

Current flow in `TokenIterator.prepare()`:

```
model.prepare(input) → case .logits(result):
  convertToToken(logits: result.logits)  // only uses logits[-1]
```

The `result.logits` tensor is `[1, seq_len, vocab]` — it contains logits for
*every* position, but only the last one is used.

**Proposed approach:**

Add a `PromptLogprobsCollector` that, when `echo: true`:

1. In `prepare()`, after `model.prepare()` returns `.logits(result)`:
   - Take `result.logits[0]` → shape `[seq_len, vocab]`
   - For each position `i` in `1..<seq_len`:
     - Compute `logSoftmax(logits[i])` → full log-prob distribution
     - Look up `logProbs[promptTokens[i]]` → the logprob of the actual next token
     - Record `(token: decode(promptTokens[i]), logprob: val, topLogprobs: topN)`
   - Position 0 gets `null` (no prior context to score against)
2. Store the collected `promptLogprobs: [TokenLogprob?]` on the iterator.
3. Expose via `TokenIteratorProtocol.promptLogprobs`.

**Performance consideration**: This requires a full `logSoftmax` over the
vocabulary for each prompt position. For a 128K vocabulary and 2048 prompt
tokens, that's 2048 softmax operations. This adds ~10-20% latency to the
prefill phase. Can be optimized later with:
- Batched logSoftmax (already a single tensor operation)
- Only computing for positions > ctxLen when only continuation logprobs
  are needed (though lm-eval needs all positions since it slices with
  `[ctxlen:-1]`)

#### 3.6 `GenerateParameters` Extension

Add:
```swift
public var echo: Bool = false
```

When `echo == true && logprobs`, the `TokenIterator` creates a
`PromptLogprobsCollector` alongside the existing `LogprobsCollector`.

### Phase 4: Legacy Completions Logprobs Response Format

#### 3.7 Non-Streaming Response

In the `/v1/completions` handler, after generation completes:

```swift
if !allLogprobs.isEmpty || promptLogprobs != nil {
    let logprobsDict = buildLegacyLogprobs(
        promptLogprobs: promptLogprobs,
        completionLogprobs: allLogprobs,
        tokenizer: tokenizer
    )
    choice["logprobs"] = logprobsDict
}
```

Where `buildLegacyLogprobs` produces:

```json
{
  "tokens": ["<p1>", "<p2>", ..., "<c1>"],
  "token_logprobs": [null, -1.2, -0.5, ...],
  "top_logprobs": [null, {"tok": -1.0, ...}, ...],
  "text_offset": [0, 3, 7, ...]
}
```

Note: `token_logprobs[0]` is `null` (first token has no context).
`token_logprobs[-1]` may be `null` (EOS). The harness slices
`[ctxlen:-1]` to get continuation logprobs only.

#### 3.8 Streaming Response

Add logprobs to `textCompletionStream` SSE chunks:

```json
{
  "choices": [{
    "text": "hello",
    "logprobs": {
      "tokens": ["hello"],
      "token_logprobs": [-0.5],
      "top_logprobs": [{"hello": -0.5, "world": -2.1}],
      "text_offset": [0]
    }
  }]
}
```

For `echo: true`, the first chunk should contain all prompt tokens' logprobs.

### Phase 5: Integration Testing

#### 3.9 E2E Test: `generate_until`

Already works. Verify with:

```bash
lm_eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions" \
  --tasks hellaswag \
  --limit 5
```

Uses `tokenizer_backend=huggingface` (falls back when remote endpoints
are absent). Requires a HuggingFace tokenizer name matching the model.

#### 3.10 E2E Test: `loglikelihood`

After all phases complete:

```bash
lm_eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions,tokenizer_backend=remote" \
  --tasks hellaswag \
  --limit 5
```

Uses remote tokenizer (Phase 1) + echo logprobs (Phases 2-4).

## 4. Dependency Graph

```
Phase 1 (Tokenizer Endpoints)
  └─→ Enables: tokenizer_backend="remote" (no local HF weights needed)

Phase 2 (Forward logprobs on /v1/completions)
  └─→ Enables: completion-only logprobs (debugging, partial support)

Phase 3 (echo + Prompt Logprobs)
  ├─→ Depends on: Phase 2
  └─→ Enables: loglikelihood scoring

Phase 4 (Legacy Format)
  ├─→ Depends on: Phase 3
  └─→ Enables: parse_logprobs() compatibility

Phase 5 (Integration Testing)
  ├─→ Depends on: Phase 1 + Phase 4
  └─→ Validates: full lm-eval compatibility
```

## 5. Estimated Effort

| Phase | Scope | Effort |
|-------|-------|--------|
| Phase 1: Tokenizer Endpoints | 3 new routes, simple wrapper around existing `Tokenizer` | Small |
| Phase 2: Forward logprobs | ~10 lines in completions handler | Trivial |
| Phase 3: echo + Prompt Logprobs | New collector, modify `prepare()`, propagate through pipeline | Large |
| Phase 4: Legacy Format | Response serialization in completions handler + SSE encoder | Medium |
| Phase 5: Integration Testing | e2e harness invocations | Small |

## 6. Risks and Open Questions

- **Prompt logprob accuracy**: The logits from `model.prepare()` must be the
  *pre-penalty* logits at each position. If any logit processing (e.g.
  repetition penalty) is applied during prefill, the prompt logprobs may
  differ from what the harness expects. Need to verify that `prepare()` returns
  raw model outputs.

- **Memory**: Storing prompt logprobs for long contexts (e.g. 32K tokens)
  requires ~32K * (1 + topN) * (sizeof(Float) + tokenString) bytes. This is
  manageable but should be profiled.

- **`max_tokens: 1` behavior**: lm-eval sends `max_tokens: 1` for loglikelihood.
  The current `/v1/completions` handler maps this to `ChatRequest.maxTokens = 1`.
  This means only **one** generated token. The harness expects the response
  to contain prompt + 1 generated token's logprobs, and slices with
  `[ctxlen:-1]`. The `-1` drops the final position. We need to verify that the
  single generated token's logprob appears at position `ctxlen` in the array,
  and that position `ctxlen + 1` (the token after the one generated, which
  should be `null`) is handled correctly.

- **Batching**: lm-eval may batch multiple requests in a single API call.
  The current `/v1/completions` handler joins `[String]` prompts with "\n".
  This is incorrect for loglikelihood scoring where each prompt is independent.
  Proper multi-choice batching would need `N` choices in the response.
  For now, running with `batch_size=1` in lm-eval avoids this issue.

- **`tokenized_requests`**: When `tokenizer_backend="remote"`, the harness
  sends raw strings (not token IDs) to the API. vMLX's completions handler
  already handles string prompts, so this should work. When `tokenizer_backend`
  is `"huggingface"`, the harness tokenizes locally and sends token ID lists.
  The current handler only handles `String` and `[String]` prompts — it would
  need to also accept `[Int]` (token IDs) for that mode.
