<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://vmlx.net/logos/png/wordmark-dark-600x150.png">
    <source media="(prefers-color-scheme: light)" srcset="https://vmlx.net/logos/png/wordmark-light-600x150.png">
    <img alt="vMLX" src="https://vmlx.net/logos/png/wordmark-transparent-600x150.png" width="400">
  </picture>
</p>

<p align="center">
  <strong>Native macOS AI inference — local models, remote endpoints, zero config</strong>
</p>

<p align="center">
  <a href="https://vmlx.net">Website</a> · <a href="panel/CHANGELOG.md">Panel Changelog</a> · <a href="CHANGELOG.md">Engine Changelog</a> · <a href="docs/">Documentation</a>
</p>

---

## What is vMLX?

vMLX is a native macOS application for running AI models on Apple Silicon. It bundles a custom inference engine with a full-featured desktop interface — manage sessions, chat with models, download from HuggingFace, connect to remote APIs, and use agentic tool-calling workflows.

- **Local inference** with GPU acceleration via MLX
- **Remote endpoints** — connect to any OpenAI-compatible API
- **HuggingFace downloader** — search, download, and serve models in-app
- **Built-in tools** — file I/O, shell, search, image reading, ask_user interrupt
- **MCP integration** — Model Context Protocol tool servers (local sessions)

---

## Key Features

### Inference Engine (v0.2.9)

| Feature | Description |
|---------|-------------|
| **Paged KV Cache** | Memory-efficient caching with prefix sharing and block-level reuse |
| **KV Cache Quantization** | Q4/Q8 quantized cache storage (2–4× memory savings) |
| **Prefix Cache** | Token-level prefix matching for fast prompt reuse across requests |
| **Continuous Batching** | Concurrent request handling with slot management |
| **VLM Caching** | Full KV cache pipeline for vision-language models (Qwen-VL, Gemma 3, etc.) |
| **Mamba Hybrid Support** | Auto-detects mixed KVCache + MambaCache models (Qwen3.5-VL, Qwen3-Coder-Next, Nemotron) |
| **Streaming Detokenizer** | Per-request UTF-8 buffering — emoji, CJK, Arabic render correctly |
| **Request Cancellation** | Stop inference mid-stream via API or connection close |
| **OpenAI-Compatible API** | Chat Completions + Responses API with full streaming support |
| **Speculative Decoding** | Draft model acceleration (20-90% speedup, zero quality loss) |

### Desktop App (Panel v0.3.10)

| Feature | Description |
|---------|-------------|
| **Multi-session** | Run multiple models simultaneously on different ports |
| **Remote endpoints** | Connect to OpenAI, Groq, local vLLM, or any compatible API |
| **HuggingFace browser** | Search, download, and install MLX models with progress tracking |
| **Agentic tools** | File I/O, shell, search, image reading with auto-continue loops (up to 10 iterations) |
| **Per-chat settings** | Temperature, Top P/K, Min P, Repeat Penalty, Stop Sequences, Max Tokens |
| **Reasoning display** | Collapsible thinking sections for Qwen3, DeepSeek-R1, GLM-4.7 |
| **Tool parsers** | hermes, pythonic, llama3, mistral, minimax, qwen3, nemotron, step3p5, and more |
| **Auto-detection** | Reads model config JSON for automatic parser and cache type selection |
| **Persistent history** | SQLite-backed chat history with metrics, tool calls, and reasoning content |
| **Live metrics** | TTFT, tokens/sec, prompt processing speed, prefix cache hits |

---

## Quick Start

### Desktop App (recommended)

```bash
# Clone and build
git clone https://github.com/vmlxllm/vmlx.git
cd vmlx/panel

# Install dependencies
npm install

# Development mode
npm run dev

# Build and install to /Applications
bash scripts/build-and-install.sh
```

### Engine Only (CLI)

```bash
# Install
uv tool install git+https://github.com/vmlxllm/vmlx.git
# or
pip install git+https://github.com/vmlxllm/vmlx.git

# Start server
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# With continuous batching
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# With API key
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --api-key your-key

# With speculative decoding (20-90% faster)
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 \
  --speculative-model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --num-draft-tokens 3
```

### Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat Completions API (streaming) |
| `POST /v1/responses` | Responses API (streaming) |
| `GET /v1/models` | List loaded models |
| `GET /health` | Server health + model info |
| `POST /v1/mcp/execute` | Execute MCP tool |
| `GET /v1/cache/stats` | Prefix cache statistics |
| `POST /v1/cache/warm` | Pre-warm cache with prompt |
| `DELETE /v1/cache` | Clear prefix cache |
| `POST /v1/chat/completions/{id}/cancel` | Cancel inference (save GPU) |
| `POST /v1/embeddings` | Text embeddings (mlx-embeddings) |

---

## Reasoning Models

Extract thinking process from reasoning-capable models:

```bash
vmlx-engine serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

| Parser | Models | Format |
|--------|--------|--------|
| `qwen3` | Qwen3, QwQ, MiniMax M2/M2.5, StepFun | `<think>` / `</think>` tags |
| `deepseek_r1` | DeepSeek-R1, Gemma 3, Phi-4 Reasoning | Lenient `<think>` (handles missing open tag) |
| `openai_gptoss` | GLM-4.7, GLM-4.7 Flash, GLM-Z1, GPT-OSS | Harmony `<\|channel\|>analysis/final` protocol |

---

## Tool Calling

Built-in agentic tools available in the desktop app:

| Category | Tools |
|----------|-------|
| **File** | read_file, write_file, edit_file, patch_file, batch_edit, copy, move, delete, create_directory, list_directory, read_image |
| **Search** | search_files, find_files, file_info, get_diagnostics, get_tree, diff_files |
| **Shell** | run_command, spawn_process, get_process_output |
| **Web** | fetchUrl, brave_search |
| **Utility** | ask_user (interactive interrupt) |

Plus MCP tool server passthrough for local sessions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    vMLX Desktop App                      │
│              (Electron + React + TypeScript)              │
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│   Local vmlx-engine     │  │   Remote Endpoints   │
│   (spawned process)  │  │ (OpenAI, Groq, etc.) │
└──────────────────────┘  └──────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│                    vMLX Engine Engine                        │
│         (FastAPI + MLX inference + caching)               │
└─────────────────────────────────────────────────────────┘
              │
    ┌─────────┼──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐┌────────────┐
│ mlx-lm ││mlx-vlm ││mlx-aud ││mlx-embed   │
│ (LLMs) ││(Vision)││(Audio) ││(Embeddings)│
└────────┘└────────┘└────────┘└────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│                     Apple MLX                            │
│             (Metal GPU + Unified Memory)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop app | Electron 28 + React 18 + TypeScript |
| Styling | Tailwind CSS |
| Database | SQLite (WAL mode, better-sqlite3) |
| Inference engine | vMLX Engine v0.2.8 (Python, FastAPI) |
| ML framework | Apple MLX (Metal GPU acceleration) |
| Build | electron-vite + electron-builder |
| Tests | Vitest (panel: 80 tests), pytest (engine: 1295+ tests) |
| Python | Bundled relocatable Python 3.12 |

---

## Recent Changes

### Engine v0.2.9 (2026-03-05)
- **Speculative decoding**: `--speculative-model` + `--num-draft-tokens` CLI flags for 20-90% throughput boost
- **RotatingKVCache confirmed**: Sliding window attention support verified across all cache modules
- 21 new tests for speculative decoding

### Engine v0.2.8 (2026-03-03)
- **Multi-turn VLM fix**: `model_dump(exclude_none=True)` prevents Jinja2 template from double-counting image tokens in multi-turn conversations
- **Hybrid cache fix**: `_fix_hybrid_cache()` returns fresh full cache on mismatch instead of corrupt short cache
- **SimpleEngine MLLM kwargs**: `reasoning_effort` and `chat_template_kwargs` now properly forwarded in VLM paths
- **1295+ tests**: Comprehensive model config registry tests (89) + MLLM serialization tests (64)

### Panel v0.3.10 (2026-03-02)
- **Bug fix**: `abortByEndpoint()` now correctly aborts remote session chat requests
- **HF model sizes**: Search results show model file sizes from safetensors metadata
- **Test suite**: 80 tests across 3 test files (vitest)

See [Panel Changelog](panel/CHANGELOG.md) and [Engine Changelog](CHANGELOG.md) for full history.

---

## Current Version

**Engine v0.2.9** / **Panel v0.3.10** — macOS Apple Silicon (M1, M2, M3, M4)

## Links

- **Website**: [vmlx.net](https://vmlx.net)
- **Contact**: admin@vmlx.net

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
