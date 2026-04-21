# User Testing

**What belongs here:** Testing surface findings, required tools, resource costs, and gotchas for validators.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Validation Surfaces

### 1. HTTP API (curl-based)
- **Tool:** `curl`
- **Setup:** Start vMLX server with a loaded model
- **Test pattern:** POST/GET requests, JSON response validation
- **Cost:** Light — each request is a single HTTP round-trip

### 2. SSE Streaming (curl + parser)
- **Tool:** `curl` + `python -c` or `awk` for SSE parsing
- **Setup:** Same as HTTP API
- **Test pattern:** Stream capture, per-chunk validation
- **Cost:** Light — streaming adds minimal overhead

### 3. lm-eval CLI Integration
- **Tool:** `~/.local/pipx/venvs/lm-eval/bin/lm-eval`
- **Setup:** vMLX server running + lm-eval with `local-completions` backend
- **Test pattern:** Full task evaluation, results JSON validation
- **Cost:** Heavy — model inference for each example; `hellaswag --limit 5` takes ~30-60s depending on model size

## Resource Cost Classification

| Surface | Per-Validator RAM | Per-Validator CPU | Max Concurrent |
|---------|-------------------|-------------------|----------------|
| HTTP API | ~50 MB (curl) | Negligible | 5 |
| SSE Streaming | ~50 MB (curl) | Negligible | 5 |
| lm-eval CLI | 2-4 GB (model + inference) | 2-4 cores (Metal) | 1 |

**Rationale:** The vMLX server loads the model into GPU/ANE memory. Only ONE lm-eval validator can run at a time because the server is a singleton process. Multiple HTTP validators can run concurrently against the same server instance.

## Gotchas

- **Port 8765 is occupied** by a Python process. Use port 8080 for mission validation.
- **Model required:** The server MUST have a model loaded before testing. Start server with `--model <path>`.
- **lm-eval venv:** Use `~/.local/pipx/venvs/lm-eval/bin/lm-eval` (not system `lm-eval`).
- **openai package:** Already injected into lm-eval venv.
- **Batching limitation:** `/v1/completions` joins `[String]` prompts with `\n`. Use `batch_size=1` in lm-eval for correct loglikelihood scoring.
- **No unit tests on disk:** `swift test` currently fails (no test files). E2E harness is the primary validation tool.
- **Build time:** `swift build --product vmlxctl` takes ~2-5 minutes on first build.
- **Tokenizer endpoint paths:** All tokenizer endpoints are under `/v1/` prefix: `/v1/tokenizer_info`, `/v1/tokenize`, `/v1/detokenize` (not `/tokenizer_info` etc.).

## Flow Validator Guidance: HTTP API

**Surface:** HTTP API (curl-based)
**Base URL:** `http://127.0.0.1:8080`
**Isolation:** All validators share the same server instance and model. HTTP requests are stateless and idempotent for tokenizer endpoints, so multiple validators can run concurrently without interference.
**Shared state:** The server's loaded model/tokenizer is shared and read-only for tokenizer tests. No mutations occur.
**Constraints:**
- Do NOT restart the server or change the loaded model.
- Do NOT test "no model loaded" scenarios (that requires a dedicated server instance).
- Rate-limit to avoid overwhelming the server (add small delays between requests if needed).
- All responses are JSON; validate with `python3 -m json.tool` or `jq`.
- The `/v1/tokenizer_info` response may not include `chat_template` (can be `null`).

## Flow Validator Guidance: Echo + Prompt Logprobs (HTTP API + SSE)

**Surface:** HTTP API (curl-based) + SSE Streaming for `/v1/completions`
**Base URL:** `http://127.0.0.1:8080`
**Model:** `Gemma-4-26B-A4B-it-JANG_4M`
**Isolation:** All validators share the same server instance. Completions requests are stateless — each request is independent. Multiple validators can run concurrently without interference since the server handles requests sequentially but statelessly.
**Shared state:** The server's loaded model/tokenizer is read-only. No persistent state mutations between requests.
**Constraints:**
- Do NOT restart the server or change the loaded model.
- Rate-limit to avoid overwhelming the server (add `sleep 0.5` between requests if needed).
- All responses are JSON; validate with `python3 -m json.tool` or `jq`.
- SSE streams should be captured with `curl -N --no-buffer` and parsed with Python.
- Gemma models use chat templates that wrap the prompt — the echoed text will include template tokens (e.g., `<bos>`, `<|turn>`, etc.) before the actual prompt text.
- `token_logprobs[0]` should be `null` (no prior context for first token).
- For `max_tokens: 0` tests, some models may not support zero generation — if the server returns an error, document it.
- When testing `echo: true` without `logprobs`, the prompt text MUST appear at the start of `choices[0].text`.
- When testing `prompt_logprobs: N` without `echo`, the text MUST NOT include the prompt but logprobs MUST still be present.
