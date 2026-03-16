<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://vmlx.net/logos/png/wordmark-dark-600x150.png">
    <source media="(prefers-color-scheme: light)" srcset="https://vmlx.net/logos/png/wordmark-light-600x150.png">
    <img alt="vMLX" src="https://vmlx.net/logos/png/wordmark-transparent-600x150.png" width="400">
  </picture>
</p>

<p align="center">
  <strong>The most complete MLX inference engine for Mac</strong>
</p>

<p align="center">
  Voice, vision, Mamba, 5-layer caching, 20+ agentic tools.<br>
  Run any MLX model locally with full prefix caching, paged KV, q4/q8 quantization, and continuous batching.
</p>

<p align="center">
  <a href="https://vmlx.net">Website</a> &bull;
  <a href="https://huggingface.co/JANGQ-AI">Models</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a>
</p>

---

## Features

- **Multi-Model Management** — Run multiple models on different ports simultaneously, each with independent configuration
- **Full Chat Interface** — Streaming responses, markdown rendering, code highlighting, reasoning boxes, tool call widgets
- **Reasoning Models** — Thinking/reasoning extraction with auto-scroll, collapsible reasoning boxes (Qwen3, DeepSeek-R1, GLM-4.7, GPT-OSS, Phi-4, Gemma 3)
- **Tool Calling (MCP)** — Integrated tool execution with inline result widgets, auto-continue for multi-step tool chains
- **Paged KV Cache** — Memory-efficient caching with prefix sharing and block-level disk persistence (L2 cache)
- **KV Cache Quantization** — q4/q8 compression of cached prompts for reduced RAM usage
- **Prefix Cache** — Memory-aware or entry-count based, with TTL expiration and warm-up support
- **Disk Cache** — Persistent prompt caches that survive server restarts (both legacy and block-based L2)
- **Continuous Batching** — High throughput for multiple concurrent users
- **Benchmarks** — Built-in benchmark suite measuring TTFT, TPS, prompt processing speed, with history tracking
- **Cache Inspector** — Real-time prefix cache stats, entry browser, warm-up, and cache management
- **Chat Export/Import** — Export chats as JSON or Markdown, import from ChatGPT/Claude/vMLX formats
- **Remote Sessions** — Connect to any OpenAI-compatible endpoint (not just local vMLX Engine)
- **Per-Chat Overrides** — Temperature, top_p, max_tokens, system prompt, stop sequences per conversation
- **Bundled Python** — Optional self-contained Python distribution for zero-dependency deployment
- **Auto-Detection** — Model architecture, parser, and cache type detected from config.json
- **Vision/Multimodal** — Image and video input for VLM models (Qwen-VL, Gemma 3, LLaVA, etc.)
- **Performance Hints** — User-friendly explanations alongside every server setting

## Quick Start

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Build + package for production
npm run build
npx electron-builder --mac --dir

# Install to Applications
cp -R release/mac-arm64/vMLX.app /Applications/
```

---

## How It Works

### 1. First Launch — Setup

On first launch, vMLX checks for a vMLX Engine installation. If not found, it offers **one-click install** via `uv` (preferred) or `pip3` (Python 3.10+ required). Optionally bundles a self-contained Python 3.12 distribution with all dependencies.

### 2. Dashboard — See All Sessions

The dashboard shows all your vMLX Engine sessions as cards. Each session represents one model loaded on one port. Sessions can be running, stopped, loading, or in an error state.

If vMLX Engine is already running (started from terminal), click **Detect Processes** — the app scans for running `vmlx-engine serve` processes, health-checks each one, and automatically creates session records.

### 3. Create a Session — Pick Model + Configure

Click **New Session** to launch the two-step wizard:

1. **Select Model** — Scans configured directories for MLX-format models. Auto-detects architecture, parser, and cache type from config.json.
2. **Configure Server** — Every vMLX Engine parameter is exposed with detailed tooltips and plain-language performance hints:
   - **Server**: host, port (auto-assigned), API key, rate limit, timeout
   - **Concurrent Processing**: max sequences, prefill/completion batch sizes, continuous batching
   - **Prefix Cache**: enable/disable, memory-aware vs entry-count, memory limits, TTL
   - **Paged KV Cache**: block size, max blocks, block disk cache (L2)
   - **KV Cache Quantization**: q4/q8 with configurable group size
   - **Disk Cache**: legacy and block-based persistent caching
   - **Performance**: stream interval, max tokens
   - **Tools (MCP)**: MCP config, auto tool choice, tool call parser, reasoning parser
   - **Additional**: raw CLI arguments

Click **Launch** — the app spawns `vmlx-engine serve` and shows live server logs. When the health endpoint responds OK, you're taken into the session.

### 4. Inside a Session — Chat + Panels

Each session shows:
- **Header**: model name, `host:port`, PID, health status, TPS counter, Stop button
- **Chat**: full conversational interface with streaming, markdown, code highlighting, reasoning boxes, tool call widgets
- **Chat History**: persisted per model path — unload a model today, reload it tomorrow, your chats are still there
- **Chat Settings** (gear icon): per-chat inference parameters — temperature, top_p, max_tokens, system prompt, stop sequences, reasoning toggle
- **Server Settings** (gear icon): inline server configuration editor with all parameters
- **Cache Panel**: real-time prefix cache stats, entry browser, warm-up, clear controls
- **Benchmark Panel**: run built-in benchmark suite, view history, compare runs

Multiple sessions can run simultaneously on different ports.

### 5. About — vMLX Engine Management

Access via the **About** button in the title bar. Check for vMLX Engine updates, install/upgrade with streaming terminal output, and view release notes. Auto-updates the bundled engine when source version changes.

---

## Architecture

```
App.tsx (view routing)
├── SetupScreen         → First-run vMLX Engine installer gate
├── SessionDashboard    → Grid of session cards (home screen)
├── CreateSession       → Two-step wizard (model picker → config → launch)
├── SessionView         → Header + ChatInterface + Settings drawers (per-session)
│   ├── ChatSettings    → Per-chat inference params drawer
│   ├── ServerSettings  → Inline server config drawer
│   ├── CachePanel      → Real-time cache stats + management
│   └── BenchmarkPanel  → Performance benchmarking suite
├── SessionSettings     → Full-page vMLX Engine server config editor
└── About               → UpdateManager + app info
```

### Three-Layer Electron Architecture

```
┌─────────────────────────────────────────────────┐
│  Renderer (React + TypeScript + Tailwind)        │
│  SetupScreen / SessionDashboard / CreateSession  │
│  SessionView / ChatInterface / UpdateManager     │
│  CachePanel / BenchmarkPanel / ReasoningBox      │
└────────────────────┬────────────────────────────┘
                     │  IPC (contextBridge)
┌────────────────────┴────────────────────────────┐
│  Preload (preload/index.ts)                      │
│  window.api.sessions / chat / models / vllm      │
│  window.api.cache / benchmark / export           │
└────────────────────┬────────────────────────────┘
                     │  ipcMain.handle
┌────────────────────┴────────────────────────────┐
│  Main Process (Node.js)                          │
│  SessionManager  → spawn/kill vMLX Engine processes │
│  DatabaseManager → SQLite WAL (chats, sessions)  │
│  VllmManager     → install/update/detect vMLX Engine│
│  IPC Handlers    → sessions, chat, models, vllm  │
│                  → cache, benchmark, export       │
└─────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/main/sessions.ts` | `SessionManager` — multi-instance lifecycle, process detection, health monitoring, buildArgs |
| `src/main/database.ts` | SQLite WAL schema + CRUD for sessions, chats, messages, folders, overrides, benchmarks, settings |
| `src/main/vllm-manager.ts` | vMLX Engine detection, install (uv/pip streaming), update, version checking |
| `src/main/ipc/sessions.ts` | IPC handlers: list, get, create, start, stop, delete, detect, update |
| `src/main/ipc/chat.ts` | Chat handlers: sendMessage (dual API SSE streaming), tool calls, abort, TPS tracking |
| `src/main/ipc/models.ts` | Model scanning, directory management, config.json detection |
| `src/main/ipc/benchmark.ts` | Benchmark suite: TTFT, TPS, prompt processing speed, history |
| `src/main/ipc/cache.ts` | Cache management: stats, entries, warm, clear |
| `src/main/ipc/export.ts` | Chat export (JSON/Markdown) and import (ChatGPT/Claude/vMLX) |
| `src/main/ipc/vllm.ts` | vLLM install/update IPC handlers with streaming log output |
| `src/main/index.ts` | App lifecycle: startup adoption, global monitor, graceful quit |
| `src/preload/index.ts` | IPC bridge exposing `window.api` to renderer |
| `src/renderer/src/App.tsx` | View routing: setup / dashboard / create / session / settings / about |
| `src/renderer/src/components/sessions/SessionConfigForm.tsx` | Full server config form with tooltips + performance hints |
| `src/renderer/src/components/sessions/CachePanel.tsx` | Real-time cache stats, entry browser, warm/clear controls |
| `src/renderer/src/components/sessions/BenchmarkPanel.tsx` | Benchmark runner and history viewer |
| `src/renderer/src/components/chat/ChatInterface.tsx` | Main chat UI: streaming, tool widgets, reasoning boxes |
| `src/renderer/src/components/chat/ReasoningBox.tsx` | Collapsible thinking/reasoning content display |
| `src/renderer/src/components/chat/ToolCallWidget.tsx` | Inline tool call + result display |

### Database Schema

```sql
-- Sessions: one per model path
sessions (id, model_path UNIQUE, model_name, host, port, pid, status, config JSON, timestamps)

-- Chats: tied to model_path for per-model history
chats (id, title, folder_id, model_id, model_path, timestamps)

-- Messages: with reasoning content support
messages (id, chat_id, role, content, reasoning_content, tool_calls, metrics, timestamps)

-- Benchmarks: performance history per model
benchmarks (id, session_id, model_path, model_name, results_json, created_at)

-- chat_overrides, folders, settings: supporting tables
```

### Dual API Support

vMLX supports both OpenAI APIs:
- **Chat Completions API** (`/v1/chat/completions`) — Standard chat with streaming
- **Responses API** (`/v1/responses`) — Extended format with reasoning events

Both APIs support: streaming, tool calls, reasoning extraction, usage tracking, and abort.

---

## Requirements

- macOS 26+ (Tahoe) — Apple Silicon required (MLX Metal shaders require Metal 4.0)
- Node.js 18+
- vMLX Engine installed (auto-installed on first launch, or manually via `uv tool install vmlx-engine`)
- MLX-format models (configurable scan directories, defaults: `~/.lmstudio/models/`, `~/.cache/huggingface/hub/`)

---

## Supported Models

vMLX auto-detects model architecture from `config.json` and selects the appropriate parser configuration. Fine-tunes inherit the base model's parser (e.g., a Llama fine-tune named "Nemotron-Orchestrator" gets `llama` parsers).

### Text Models — Tool & Reasoning Parsers

| Model Family | Tool Parser | Reasoning Parser | Notes |
|-------------|-------------|-----------------|-------|
| Qwen3 / Qwen3-Coder / QwQ | `qwen` | `qwen3` | 0.6B–235B, MoE variants |
| Qwen3-Next (hybrid Mamba) | `qwen` | `qwen3` | Mamba + Transformer hybrid |
| Qwen2.5 / Qwen2.5-Coder / Qwen2 | `qwen` | — | |
| Llama 4 Scout / Maverick | `llama` | — | MoE architecture |
| Llama 3.3 / 3.2 / 3.1 / 3 | `llama` | — | Yi uses same parser |
| Mistral Large/Small/Nemo/7B | `mistral` | — | |
| Mixtral 8x7B / 8x22B | `mistral` | — | MoE |
| Codestral / Devstral | `mistral` | — | Coding-focused |
| Gemma 3 (text-only) | `hermes` | `deepseek_r1` | Supports batching/paged cache |
| Phi-4 Mini / Medium | `hermes` | — | |
| Phi-4 Reasoning / Reasoning Plus | `hermes` | `deepseek_r1` | |
| DeepSeek-R1 (native 671B) | `deepseek` | `deepseek_r1` | R1-Distill uses base arch parser |
| DeepSeek-V3 / V2.5 / V2 | `deepseek` | — | |
| GLM-4.7 / GLM-Z1 | `glm47` | `deepseek_r1` | `<think>` tags |
| GLM-4.7 Flash / GPT-OSS | `glm47` | `openai_gptoss` | Harmony `<\|channel\|>` protocol |
| GLM-4 | `glm47` | — | Tools only, no reasoning |
| MiniMax M1 / M2 / M2.5 | `minimax` | `qwen3` | MoE |
| Granite 3.x / Granite-Code | `granite` | — | IBM |
| Nemotron (native arch) | `nemotron` | — | Hybrid Mamba+Transformer |
| Kimi-K2 / Moonshot | `kimi` | — | |
| xLAM | `xlam` | — | Salesforce |
| StepFun Step-3.5 | `step3p5` | `qwen3` | |
| Functionary v2/v3/v4r | `functionary` | — | MeetKai |
| Hermes fine-tunes | `hermes` | — | NousResearch |

### Vision / Multimodal Models

These use SimpleEngine (no batching, paged cache, or KV quant):

| Model Family | Tool Parser | Notes |
|-------------|-------------|-------|
| Qwen3-VL / Qwen2.5-VL / Qwen2-VL | `qwen` | Vision-Language |
| Gemma 3 (multimodal) | `hermes` | Has `deepseek_r1` reasoning |
| Pixtral 12B / Large | `mistral` | Mistral vision |
| Phi-4 Multimodal | — | Vision only |
| Phi-3 Vision | — | |
| DeepSeek-VL / VL2 | — | |
| LLaVA | — | |
| InternVL | — | |
| MiniCPM-V | — | |
| Molmo | — | |
| PaliGemma / MedGemma | — | Medical / research |
| SmolVLM | — | HuggingFace |
| Yi-VL | — | |

### Other Architectures

| Model Family | Cache Type | Notes |
|-------------|-----------|-------|
| Jamba | hybrid | Mamba+Transformer |
| Falcon Mamba | mamba | Pure SSM |
| RWKV 5/6 | mamba | Recurrent |
| Command-R / R+ | kv | Cohere |
| EXAONE | kv | LG AI Research |
| OLMo / OLMo2 | kv | AI2 |
| InternLM 3 / InternLM | kv | |
| Gemma 2 | kv | |
| StarCoder 2 | kv | Code only |
| StableLM | kv | |
| Baichuan | kv | |

---

## Development

```bash
npm run dev          # Development mode (hot reload)
npm run build        # Build for production
npm run typecheck    # TypeScript validation
npm run lint         # ESLint
```

### Build & Install Script

```bash
# Full build pipeline with pre-flight checks
./scripts/build-and-install.sh
```

The build script runs TypeScript checks, Python syntax validation, registry sync verification, and API parity checks before building.

### Project Structure

```
src/
├── main/                           # Electron main process
│   ├── index.ts                    # App lifecycle, startup
│   ├── sessions.ts                 # SessionManager (multi-instance)
│   ├── database.ts                 # SQLite WAL schema + queries
│   ├── vllm-manager.ts            # vMLX Engine install/update/detect
│   └── ipc/
│       ├── sessions.ts             # Session IPC handlers
│       ├── chat.ts                 # Chat IPC + dual-API SSE streaming + abort
│       ├── models.ts               # Model scanning + directories
│       ├── benchmark.ts            # Benchmark runner + history
│       ├── cache.ts                # Cache stats/entries/warm/clear
│       ├── export.ts               # Chat export/import
│       └── vllm.ts                 # vLLM install/update handlers
├── renderer/                       # React UI
│   └── src/
│       ├── App.tsx                 # View routing
│       ├── components/
│       │   ├── setup/              # SetupScreen (first-run installer)
│       │   ├── sessions/           # Dashboard, Card, Create, View, Settings,
│       │   │                       # ConfigForm, CachePanel, BenchmarkPanel
│       │   ├── chat/               # ChatInterface, ChatList, ChatSettings,
│       │   │                       # Messages, ReasoningBox, ToolCallWidget
│       │   └── update/             # UpdateManager
│       └── index.css               # Tailwind + custom classes
└── preload/
    ├── index.ts                    # IPC bridge (contextBridge)
    └── index.d.ts                  # TypeScript declarations
```

---

## Distribution

```bash
npm run build                        # Build app
npx electron-builder --mac --dir     # Package as .app
```

Output: `release/mac-arm64/vMLX.app`

Deploy: `cp -R release/mac-arm64/vMLX.app /Applications/`

---

## Troubleshooting

### vMLX Engine not found
On first launch, the app offers one-click install. If you prefer manual install:
```bash
uv tool install vmlx-engine          # Recommended (fastest)
pip3 install vmlx-engine             # Alternative (needs Python 3.10+)
```

### Models not detected
Add custom model directories via the directory manager in the Create Session wizard. Default scan locations:
- `~/.lmstudio/models/`
- `~/.cache/huggingface/hub/`

### Session won't start
- Check the loading screen logs for errors
- Verify the model path exists and contains valid MLX weights
- Ensure the port isn't already in use (the app auto-assigns ports)
- Check system memory — large models need significant RAM

### Chat history missing
- Chat history is tied to exact `modelPath`. If you moved the model, chats won't appear.
- Database location: `~/Library/Application Support/vmlx-engine/chats.db`

### Process detected but not adopted
- Click "Detect Processes" on the dashboard
- The process must respond to `/health` endpoint
- Only processes running `vmlx-engine serve` are detected

### Vision models using SimpleEngine
- VLM/multimodal models automatically use SimpleEngine (no batching, paged cache, or KV quant)
- This is a hardware limitation of multimodal processing on Apple Silicon
- Text-only models get full BatchedEngine features

---

## Credits

- **[vMLX Engine](https://github.com/jjang-ai/vmlx)** — Apple Silicon inference engine
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's ML framework
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** — LLM inference
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** — Vision-language models
- **[Electron](https://www.electronjs.org/)** — Desktop app framework

---

## License

Apache 2.0 — Copyright 2026 JANGQ AI. See [LICENSE](LICENSE).
