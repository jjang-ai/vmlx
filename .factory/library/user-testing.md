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
