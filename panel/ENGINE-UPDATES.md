# vMLX Engine Maintenance Guide

How to maintain the vmlx-engine engine and push updates to users.

## Architecture

```
vMLX.app/Contents/Resources/
├── bundled-python/
│   └── python/
│       ├── bin/python3              ← Relocatable Python 3.12 interpreter
│       └── lib/python3.12/
│           └── site-packages/
│               ├── vmlx_engine/        ← Engine (pip installed from source)
│               ├── mlx/
│               ├── transformers/
│               └── ...              ← All dependencies
├── vmlx-engine-source/                 ← Engine source (auto-copied from repo)
│   ├── pyproject.toml
│   └── vmlx_engine/
└── app/                             ← Electron app
```

### How it works

1. **On startup**, the app compares installed engine version (in bundled-python)
   against source version (in vmlx-engine-source/pyproject.toml)
2. If versions differ, it auto-runs `pip install --force-reinstall --no-deps`
   from the bundled source (takes ~5s, 30s timeout)
3. **Session launch** uses `python3 -m vmlx_engine.cli serve ...` instead of the
   `vmlx-engine` binary, avoiding shebang path issues in relocatable Python

### Fallback behavior

If bundled Python is not present (dev mode, or older builds):
- Falls back to system-installed `vmlx-engine` binary (uv, pip, homebrew)
- SetupScreen appears if no installation found
- All existing behavior preserved

## Update Scenarios

### 1. Updating vmlx-engine source code (frequent)

Engine-only changes: model configs, parsers, server logic, API handlers.

1. Edit files in `vmlx_engine/`
2. **Bump version** in `pyproject.toml`
3. Rebuild: `cd panel && npm run build && npx electron-builder --mac`
4. The source is auto-copied to `Resources/vmlx-engine-source/` by electron-builder
5. On user's machine: app detects version mismatch on startup, auto-reinstalls (~5s)

### 2. Updating Python dependencies (rare)

When MLX, transformers, or other deps need version bumps.

1. Edit dependency versions in `pyproject.toml` AND `scripts/bundle-python.sh`
2. Re-run: `bash scripts/bundle-python.sh` (~10-15 min)
3. Rebuild entire app: `npm run build && npx electron-builder --mac`
4. Users need to download a new DMG

### 3. Adding new model configs / parsers

Pure Python changes — only step 1 above needed.

- `vmlx_engine/model_configs.py` — model family detection
- `vmlx_engine/reasoning/*.py` — reasoning parsers
- `vmlx_engine/tools/*.py` — tool call parsers

## Key files that MUST stay in sync

| Python | TypeScript | What |
|--------|-----------|------|
| `vmlx_engine/model_configs.py` | `panel/src/main/model-config-registry.ts` | Model family detection, parser/cache defaults |
| `vmlx_engine/api/models.py` | `panel/src/main/ipc/chat.ts` | Request fields (stream_options, enable_thinking) |
| `vmlx_engine/server.py` | `panel/src/main/ipc/chat.ts` | Both Chat Completions + Responses API paths |

## Parser & Cache Architecture

### Reasoning Parsers (server-side)

| Parser | Models | Format |
|--------|--------|--------|
| `qwen3` | Qwen3, Qwen3-Coder, QwQ-32B, StepFun, MiniMax-M2.5 | `<think>...</think>content` (strict) |
| `deepseek_r1` | DeepSeek-R1, R1-Distill, R1-0528 | `<think>...</think>content` (lenient) |
| `openai_gptoss` | GLM-4.7, GLM-4.7 Flash, GLM-Z1, GPT-OSS | `<\|channel\|>analysis...final` (Harmony) |

### Cache Types

| Type | Models | `usePagedCache` | Notes |
|------|--------|-----------------|-------|
| `kv` | Most models (Qwen, Llama, Mistral, DeepSeek, etc.) | `false` | Standard KV cache, positional, sliceable |
| `mamba` | Falcon-Mamba, Mamba, RWKV, Qwen3-Next | `true` | Cumulative state, auto-uses paged cache |
| `hybrid` | Nemotron, Jamba | `true` | Mixed KV+Mamba layers, auto-uses paged cache |
| `rotating_kv` | Sliding-window models | `false` | Positional, handled same as KV |

Scheduler auto-detects hybrid models and switches to paged cache even if user didn't enable it.

### KV Cache Quantization

Quantizes stored KV cache entries in the prefix cache using `QuantizedKVCache` (q4/q8).
Reduces prefix cache memory by 2-4x, enabling more cached prefixes and better hit rates.

**Architecture: Storage-boundary quantization**
- During generation: full-precision `KVCache`/`BatchKVCache` (no quality loss)
- Storage to prefix cache: `KVCache` → `QuantizedKVCache` (2-4x memory savings)
- Retrieval from prefix cache: `QuantizedKVCache` → `KVCache` (dequantize for BatchGenerator)

Key design decision: `BatchGenerator` (mlx-lm) doesn't natively support `QuantizedKVCache`.
Rather than monkey-patching `model.make_cache()` (which crashes `to_batch_cache()` and
`_merge_caches()`), we quantize/dequantize at the prefix cache boundary.

**CLI:** `--kv-cache-quantization none|q4|q8`, `--kv-cache-group-size 64`
**UI:** "KV Cache Quantization" section in SessionConfigForm
**Config:** `kvCacheQuantization`, `kvCacheGroupSize` in SessionConfig

**Quantization flows:**
- `_quantize_cache_for_storage(cache)` — converts KVCache layers → QuantizedKVCache via `mx.quantize()`
- `_dequantize_cache_for_use(cache)` — converts QuantizedKVCache layers → KVCache via `mx.dequantize()`
- 3 storage points: paged (pre-extraction), memory-aware (post-truncation), legacy (post-truncation)
- 3 retrieval points: paged (post-reconstruct), memory-aware (post-fetch), legacy (post-fetch)
- Safety fallback: `mamba_cache.py` handles QuantizedKVCache in `to_batch_cache()`/`_merge_caches()`

**Works with:** all 3 cache backends, continuous batching, paged cache, hybrid models, tool calls, reasoning

### Cache Reuse API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/cache/stats` | GET | Cache statistics (entries, memory, hit/miss, quantization info) |
| `/v1/cache/entries` | GET | List cached prefixes with token counts and memory |
| `/v1/cache/warm` | POST | Pre-warm cache with system prompts: `{"prompts": ["..."]}` |
| `/v1/cache` | DELETE | Clear cache: `?type=prefix\|multimodal\|all` |

## What is NOT affected by bundled Python

These subsystems were verified safe during implementation:

- **Chat API / Responses API** — HTTP-based, agnostic to Python type
- **Chat templates** — model-specific, read from config.json
- **Tool calling** — MCP and built-in tools work via HTTP
- **Prefix cache / paged cache** — args built before spawn, identical for both paths
- **Tool/reasoning parser resolution** — auto/explicit/disabled tri-state unchanged
- **Model directory scanning** — pure filesystem operations
- **Session health monitoring** — polls `/health` endpoint regardless of spawn method

## Build Workflow

```bash
# One-time: Bundle Python (or when deps change)
cd panel && bash scripts/bundle-python.sh

# Regular builds (engine source auto-copied)
npm run build && npx electron-builder --mac

# Full build + install to /Applications
bash scripts/build-and-install.sh
```

## CRITICAL: Proprietary Model Restrictions

**MiniMax Prism Pro** model weights are **proprietary**. NEVER upload, distribute,
bundle, or include MiniMax Prism Pro weights in any public release, DMG, repository,
or distribution channel. This applies to all forms of the model (original, quantized,
MLX-converted). All other open-weight models (Qwen, DeepSeek, Llama, Gemma, etc.)
are fine to reference and distribute normally.

## Testing checklist

```bash
# 1. Verify bundled Python
bundled-python/python/bin/python3 -c "import vmlx_engine; print(vmlx_engine.__version__)"
bundled-python/python/bin/python3 -m vmlx_engine.cli --help

# 2. Build and package
npm run build && npx electron-builder --mac

# 3. Install and test
# - App launches without SetupScreen
# - Create session, start model, chat works
# - Tool calls work (if MCP configured)
# - Prefix cache / paged cache settings apply
# - KV cache quantization (q8): same prompt cached at reduced memory
# - Cache API: curl http://localhost:8000/v1/cache/stats
# - Process adoption works after restart

# 4. Engine update test
# - Bump version in pyproject.toml
# - Rebuild → app auto-updates engine on startup
```
