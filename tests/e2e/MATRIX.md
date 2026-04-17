# vMLX E2E Test Matrix

Empirical tests run by `tests/e2e/harness.sh`. Each row = (model, suite,
setting variant). JSON-lines results live in `/tmp/vmlx-e2e/*.jsonl`.

## Cases (implemented in harness.sh)

| id | case | what it proves |
|----|------|----------------|
| S1 | `models_list`          | `/v1/models` populated + dedupe clean |
| S2 | `basic_chat`           | Non-streaming OpenAI chat returns content |
| S3 | `sse_stream`           | Streaming chunks arrive, TTFT + tok/s measurable |
| S4 | `metrics_endpoint`     | `/metrics` Prometheus exposition 200 |
| S5 | `ollama_tags`          | `/api/tags` Ollama compat 200 |
| S6 | `max_tokens`           | `max_tokens=4` → ≤4 tokens + `finish_reason=length` |
| F1 | `multiturn_prefix_cache` | 2nd turn TTFT ≤ 1st (prefix hit) |
| F2 | `stop_sequences`       | Custom `stop` strips output + `finish_reason=stop` |
| F3 | `json_mode`            | `response_format=json_object` returns parseable JSON |
| F4 | `ollama_chat`          | `/api/chat` returns content |
| F5 | `anthropic_messages`   | `/v1/messages` returns `content[0].text` |
| F6 | `concurrent`           | 3 parallel requests all succeed (no Metal crash) |

## Models on disk (Apr 17 audit)

Size + shard count from `tests/e2e/audit-disk.sh` (resolves HF symlinks):

### HF cache (fully downloaded)
| model | size | family | notes |
|-------|------|--------|-------|
| google/gemma-4-26B-A4B-it            | 48 GB | gemma4 | text, bf16 |
| mlx-community/Llama-3.2-1B-Instruct-4bit | 0.6 GB | llama | tiny, fast smoke |
| mlx-community/Qwen3-0.6B-8bit        | 0.6 GB | qwen3 | tiny, fast smoke (baseline) |
| mlx-community/Qwen3-Embedding-0.6B-8bit | 0.6 GB | qwen3 | embedding-only |
| mlx-community/whisper-tiny-mlx(-4bit) | 0.1 GB | whisper | audio |
| mlx-community/gemma-4-e2b-it-4bit    | 3.3 GB | gemma4 | text |
| mlx-community/gemma-4-e4b-it-4bit    | 4.9 GB | gemma4 | text |
| Qwen/Qwen3.6-35B-A3B                 | 67 GB | qwen3_6 | MoE, large |
| sentence-transformers/all-MiniLM-L6-v2 | 0.1 GB | bert | embedding |

### User dir `~/.mlxstudio/models/MLXModels/`
JANGQ-AI / OsaurusAI / dealignai family variants — see full list in
`tests/e2e/audit-disk.sh` output. Notable:

| model | size | family | cache | JANG |
|-------|------|--------|-------|------|
| Nemotron-Cascade-2-30B-A3B-JANG_2L-CRACK | 10.3 GB | nemotron_h | hybrid | 2L |
| Nemotron-Cascade-2-30B-A3B-JANG_4M       | 17.0 GB | nemotron_h | hybrid | 4M |
| Nemotron-3-Super-120B-A12B-JANG_2L-CRACK | 43.3 GB | nemotron_h | hybrid | 2L |
| Gemma-4-31B-it-JANG_4M                   | 21.1 GB | gemma4     | kv     | 4M |
| Mistral-Small-4-119B-JANG_2L             | 37.4 GB | mistral4   | mla    | 2L |
| Mistral-Small-4-119B-JANG_4M-CRACK       | 64.3 GB | mistral4   | mla    | 4M |
| MiniMax-M2.7-JANG_2L                     | 62.6 GB | minimax_m2 | kv     | 2L |
| MiniMax-M2.7-JANG_3L                     | 88.6 GB | minimax_m2 | kv     | 3L |
| MiniMax-M2.7-JANGTQ-CRACK (dealignai)    | 56.5 GB | minimax_m2 | kv     | native TQ |
| Qwen3.6-35B-A3B-JANGTQ2/4 (OsaurusAI)    | 10.8/18.3 GB | qwen3_6 | hybrid | native TQ |
| Qwen3.5-VL-* (dealignai)                 | 3-18 GB   | qwen3_5_vl | hybrid | JANG 4S/4K |
| GPT-OSS-120B-MLX-CRACK                   | 60.8 GB | gpt_oss    | kv    | — |

## Run plan (one sweep = one Ralph iteration)

### Tier 1: fast smoke (runs every iteration, < 2 min total)
- [x] Qwen3-0.6B-8bit — smoke + full
- [ ] Llama-3.2-1B-Instruct-4bit — smoke + full
- [ ] gemma-4-e2b-it-4bit — smoke + full

### Tier 2: family representatives (picked one per arch, < 30 min each)
- [ ] Nemotron-Cascade-2-30B-A3B-JANG_2L-CRACK — hybrid SSM path
- [ ] Gemma-4-31B-it-JANG_4M — Gemma4 MoE path
- [ ] MiniMax-M2.7-JANG_2L — MLA-less minimax
- [ ] MiniMax-M2.7-JANGTQ-CRACK — JANGTQ native fast path
- [ ] Mistral-Small-4-119B-JANG_2L — MLA path
- [ ] Qwen3.6-35B-A3B (Qwen/…) — BF16 baseline
- [ ] Qwen3.6-35B-A3B-JANGTQ2 — JANGTQ native on qwen3_6

### Tier 3: VL + audio
- [ ] Qwen3.5-VL-4B-JANG_4S-CRACK — multi-image chat
- [ ] Qwen3.5-VL-9B-JANG_4S-CRACK — multi-turn VL cache test
- [ ] whisper-tiny-mlx — /v1/audio/transcriptions
- [ ] all-MiniLM-L6-v2 — /v1/embeddings

### Tier 4: edge cases
- [ ] Model with missing shards → expect `EngineError.loadFailed` (not hang)
- [ ] Concurrent 5-request burst — verify lock FIFO, no crash
- [ ] Prefix-cache hit measurement (t1 prefill vs t2 cached)
- [ ] JSON schema mode (not just json_object)
- [ ] Tools: bash tool round-trip + tool_choice=required
- [ ] Stop button mid-generation → quick cancel
- [ ] Large context (8k+ tokens) + prefix eviction
- [ ] Switching models without server restart

## Issue log (this iteration)

### ✅ Fixed
- **Concurrent request segfault** (Metal `_status < MTLCommandBufferStatusCommitted`):
  added `GenerationLock` actor, wrapped `performStreamingGeneration` with
  FIFO acquire/release. See `Sources/vMLXEngine/GenerationLock.swift`.

### 🔎 Open / observed
- **Qwen3-0.6B JANG-stamp log line shows `modality=vision`** in detection
  log from scanning OTHER library entries — NOT the loaded model. Just
  chatty log; verify real capabilities on the LOADED model are correct.
- **SSE `tps=30.7` on Qwen3-0.6B-8bit** — matches reference Swift MLX on
  this model class; fine for 0.6B 8-bit baseline.
- **`json_mode` on Qwen3-0.6B-8bit** returned an API error wrapped in a
  JSON dict — the harness matched against "is a dict" so the test passed
  but the model's actual JSON adherence wasn't exercised. Tighten the
  harness to check for the expected `color: blue` field.
- **`stop_sequences`** flagged `ok=false` because of the shell-grep
  fencepost, but `finish_reason=stop` did fire. Test-harness nit.
