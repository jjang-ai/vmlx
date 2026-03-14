# vMLX Feature Parity + Superiority Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add all missing features from oMLX (multi-model, menu bar, Anthropic API, memory enforcement, per-model settings, reranker, OCR, SSE keep-alive, Homebrew) while verifying all existing features remain working.

**Architecture:** Hybrid multi-model (Electron manages multiple single-model server processes), Electron tray with detach-on-close, Anthropic adapter converting to existing Chat Completions pipeline, closed source with cask-only Homebrew.

**Key Decisions:**
- Multi-model: Electron spawns separate `vmlx-engine serve` processes per model (crash isolation, existing server code untouched)
- Menu bar: Part of Electron app, window close keeps app alive in tray (toggle in settings)
- Admin dashboard: Skip (Electron app IS the dashboard)
- Anthropic API: Claude Code focused (streaming, tools, thinking blocks)
- Verification: Existing 2300+ tests + new integration test suite
- Homebrew: Cask only (closed source, points to DMG on releases repo)

---

## Architecture Overview

```
Layer 4: Distribution
  └── Homebrew cask (brew install --cask vmlx)

Layer 3: Electron App (panel/)
  ├── Menu bar tray (Tray API, process manager)
  ├── Per-model settings UI (settings panel extension)
  └── Multi-model session switching

Layer 2: Python Server (vmlx_engine/)
  ├── AnthropicAdapter (/v1/messages → internal chat completions)
  ├── RerankerEngine (/v1/rerank, CausalLM + encoder scoring)
  ├── OCR model configs (auto-detect, optimized prompts)
  └── SSE keep-alive (heartbeat during long prefills)

Layer 1: Verification
  ├── Run existing 2300+ tests
  └── New integration test suite (HTTP → server → response)
```

Build order: Layer 1 → 2 → 3 → 4 (each depends on the one below).

---

## 1. Multi-Model Serving (Hybrid Process Architecture)

Electron app becomes a process manager. Each model gets its own `vmlx-engine serve` process on a unique port.

```
Electron App (process manager)
├── Model A: vmlx-engine serve --model Qwen3-30B --port 8000
├── Model B: vmlx-engine serve --model Nemotron-H --port 8001
├── Model C: vmlx-engine serve --model bge-m3 --port 8002
└── ProcessManager
    ├── spawn(model, port, config) → ChildProcess
    ├── kill(model)
    ├── health_check() → poll /health on each port
    ├── memory_total() → sum GPU memory from all processes
    └── auto_evict() → kill LRU idle process when over budget
```

**Integration points:**
- `sessions.ts` already tracks local sessions with port — extend to per-model ports
- `chat.ts` builds request URLs from session config — port comes from session, zero changes
- Python server code stays completely untouched — each process is single-model
- Process isolation: crashing VLM doesn't kill text LLM

**New file:** `panel/src/main/process-manager.ts` (~300 lines)

---

## 2. Menu Bar Tray

Electron Tray API. Window close keeps app alive in system tray.

```
Tray icon states:
  ● Green  — at least one model serving
  ● Yellow — starting up / loading model
  ● Gray   — no models loaded

Context menu:
  ├── "vMLX — 2 models loaded"
  ├── ─────────────
  ├── Qwen3-30B (port 8000) ▸
  │   ├── ✓ Running — 4.2 GB
  │   ├── Temp: 0.7 | Top-P: 0.9
  │   ├── Restart
  │   └── Stop
  ├── Nemotron-H (port 8001) ▸
  │   ├── ✓ Running — 8.1 GB
  │   └── Stop
  ├── ─────────────
  ├── Load Model... (opens model picker)
  ├── Memory: 12.3 / 96 GB
  ├── ─────────────
  ├── Copy API URL
  ├── Open vMLX Window
  └── Quit vMLX
```

**Integration:**
- `main/index.ts` — `app.on('window-all-closed')` checks tray setting before quitting
- Tray reads model list from process manager
- Settings in SQLite: `tray_enabled`, `tray_close_to_tray`

**New files:**
- `panel/src/main/tray.ts` (~200 lines)
- `panel/src/renderer/src/components/settings/TraySettings.tsx` (~60 lines)

---

## 3. Anthropic Messages API

Thin adapter: Anthropic wire format → existing Chat Completions pipeline → Anthropic SSE events.

```
POST /v1/messages
  → AnthropicAdapter.to_chat_completion(request)
  → existing stream_chat_completion()
  → AnthropicAdapter.from_chat_completion(chunks)
  → SSE: message_start, content_block_delta, message_stop
```

**Request mapping:**
- `messages[].content` (content blocks) → `messages[].content` (string/parts)
- `system` (top-level) → `messages[0].role = "system"`
- `tools` (name/input_schema) → `tools` (function format)
- `tool_use` block → `tool_calls` on assistant
- `tool_result` block → `role: "tool"` message
- `thinking.type = "enabled"` → `enable_thinking = true`

**Streaming SSE:** message_start, content_block_start/delta/stop, message_delta, message_stop

**New file:** `vmlx_engine/api/anthropic_adapter.py` (~400 lines)
**Modified:** `server.py` — add `/v1/messages` route

---

## 4. Process Memory Enforcement

Electron-side monitor polling all server processes.

```typescript
class MemoryEnforcer {
  pollIntervalMs = 5000
  maxMemoryGB: number  // default: system RAM - 8GB

  poll():
    for each process → fetch /health → read gpu_memory
    if total > max:
      1. Kill expired TTL (idle > configured, not pinned)
      2. Kill LRU (oldest lastUsed, not pinned)
      3. If all pinned → tray notification
}
```

**Integration:**
- `/health` already returns GPU memory — add `last_request_time` (3 lines)
- Tray shows live memory gauge
- Settings: max memory slider, per-model pin toggle, default TTL

**New file:** `panel/src/main/memory-enforcer.ts` (~150 lines)

---

## 5. Per-Model Settings

SQLite table storing per-model configuration. Passed as CLI flags when spawning.

```sql
CREATE TABLE model_settings (
  model_path TEXT PRIMARY KEY,
  alias TEXT,
  temperature REAL,
  top_p REAL,
  max_tokens INTEGER,
  ttl_minutes INTEGER,
  pinned BOOLEAN DEFAULT FALSE,
  port INTEGER,
  cache_quant TEXT,  -- q4/q8/none
  disk_cache_enabled BOOLEAN DEFAULT FALSE,
  reasoning_mode TEXT DEFAULT 'auto'  -- auto/on/off
);
```

Spawning maps settings to CLI flags — Python server already accepts all of them.

**New files:**
- `panel/src/main/db/model-settings.ts` (~80 lines)
- `panel/src/renderer/src/components/settings/ModelSettings.tsx` (~150 lines)

---

## 6. Reranker Engine

New `/v1/rerank` endpoint with two scoring backends:

- **Encoder** (ModernBERT, XLM-RoBERTa): mlx-embeddings sequence classification
- **CausalLM** (Qwen3-Reranker): yes/no logit scoring via generate

**New file:** `vmlx_engine/reranker.py` (~200 lines)
**Modified:** `server.py` — add `/v1/rerank` route (~40 lines)

---

## 7. OCR Model Support

3-4 new `_register()` calls in `model_configs.py`:
- Match: `deepseek-ocr`, `dots-ocr`, `glm-ocr`, `florence2`
- Set `is_mllm = True`, default OCR system prompt, stop tokens
- Existing VLM pipeline handles everything else

~30 lines total.

---

## 8. SSE Keep-alive

In streaming paths, emit SSE comment if no token for 15 seconds:

```python
if time.time() - last_emit > 15.0:
    yield ": keep-alive\n\n"
    last_emit = time.time()
```

~10 lines in both Chat Completions and Responses API streaming generators.

---

## 9. Integration Test Suite

```
tests/integration/test_server_endpoints.py
├── test_health, test_models
├── test_chat_completion (stream + non-stream)
├── test_chat_tools (call → result → response)
├── test_chat_reasoning (thinking blocks)
├── test_responses_api, test_responses_tools
├── test_anthropic_messages, test_anthropic_tools, test_anthropic_thinking
├── test_embeddings, test_rerank
├── test_cancel, test_cache_stats, test_cache_warm
├── test_vlm_image, test_sse_keepalive
└── test_concurrent (4 parallel)
```

Run: `python tests/integration/test_server_endpoints.py --model <small-model> --ci`

**New files:** `tests/integration/test_server_endpoints.py` (~500 lines), `conftest.py`

---

## 10. Homebrew Cask

Public repo `vmlxllm/homebrew-vmlx` with cask pointing to DMG:

```ruby
cask "vmlx" do
  version "1.2.2"
  sha256 "<sha256>"
  url "https://github.com/vmlxllm/vmlx-releases/releases/download/v#{version}/vMLX-#{version}-arm64.dmg"
  app "vMLX.app"
end
```

Install: `brew tap vmlxllm/vmlx && brew install --cask vmlx`

---

## Files Summary

### New Files (Python engine)
- `vmlx_engine/api/anthropic_adapter.py` (~400 lines)
- `vmlx_engine/reranker.py` (~200 lines)
- `tests/integration/test_server_endpoints.py` (~500 lines)
- `tests/integration/conftest.py` (~50 lines)

### New Files (Electron panel)
- `panel/src/main/process-manager.ts` (~300 lines)
- `panel/src/main/tray.ts` (~200 lines)
- `panel/src/main/memory-enforcer.ts` (~150 lines)
- `panel/src/main/db/model-settings.ts` (~80 lines)
- `panel/src/renderer/src/components/settings/TraySettings.tsx` (~60 lines)
- `panel/src/renderer/src/components/settings/ModelSettings.tsx` (~150 lines)

### Modified Files
- `vmlx_engine/server.py` — /v1/messages route, /v1/rerank route, SSE keep-alive, last_request_time in /health
- `vmlx_engine/model_configs.py` — OCR model registrations
- `panel/src/main/index.ts` — tray integration, window-close behavior
- `panel/src/main/ipc/sessions.ts` — multi-model port mapping
- `panel/src/main/db.ts` — model_settings table migration
- `panel/src/renderer/src/components/settings/` — TraySettings, ModelSettings panels

### New Repo
- `vmlxllm/homebrew-vmlx` — Homebrew cask formula

### Estimated Total
- ~2,100 lines new code
- ~100 lines modified in existing files
- 10 new files, 6 modified files, 1 new repo
