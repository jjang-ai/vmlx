# Build, Test & Deploy

Complete guide for building the vMLX desktop app, running the test suite, and deploying to `/Applications`.

## Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Node.js** ≥ 18 and **npm** ≥ 9
- **Python** 3.12+ with the vllm-mlx venv (`.venv/`)

## Building the Desktop App

### Development Mode

```bash
cd panel
npm install     # first time only
npm run dev     # launches Electron with hot reload
```

### Production Build

```bash
cd panel
npm run build   # compile TypeScript + bundle renderer
npm run dist    # package as macOS .app via electron-builder
```

The built app appears at `panel/release/mac-arm64/vMLX.app`.

### Install to /Applications

```bash
# Kill any running instance, copy, strip quarantine, launch
killall vMLX 2>/dev/null || true
rm -rf /Applications/vMLX.app
cp -R panel/release/mac-arm64/vMLX.app /Applications/
xattr -cr /Applications/vMLX.app
open /Applications/vMLX.app
```

> **Note on Apple Sandbox**: The app currently runs without App Sandbox entitlements. For Mac App Store distribution, sandbox entitlements (file access, network, subprocess spawning) will need to be configured in `build/entitlements.mas.plist`.

---

## Running the Test Suite

### Engine Tests (Python)

The engine test suite lives in `tests/` and uses **pytest** with the project's `.venv`:

```bash
# Run ALL engine tests
.venv/bin/python -m pytest tests/ -v

# Run specific test suites
.venv/bin/python -m pytest tests/test_reasoning_tool_interaction.py -v   # 61 tests
.venv/bin/python -m pytest tests/test_tool_fallback_injection.py -v      # 4 tests
.venv/bin/python -m pytest tests/test_tool_format.py -v                  # 54 tests

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=vllm_mlx
```

#### Key Test Files

| File | Tests | What it covers |
|------|-------|----------------|
| `test_reasoning_tool_interaction.py` | 61 | Reasoning parser + tool parser cross-interaction, think tag handling, content deduplication, streaming edge cases |
| `test_tool_fallback_injection.py` | 4 | Template tool injection fallback (Qwen thinking-off, generic models) |
| `test_tool_format.py` | 54+ | Tool format conversion, tool_choice filtering, response_format strict, model config flags |
| `test_mllm_scheduler_stability.py` | — | MLLM batching concurrency, ghost request detection, queue bounds |
| `test_hybrid_batching.py` | — | Mamba/SSM hybrid cache routing, VL model paged cache |
| `test_paged_cache.py` | — | Block allocation, LRU eviction, hash dedup, COW, quantization |

### Panel Tests (TypeScript)

The panel test suite uses **vitest**:

```bash
cd panel
npx vitest run          # run all 80+ tests
npx vitest run --watch  # watch mode for development
```

---

## Pre-Build Checklist

Before every production build, run this checklist:

```bash
# 1. Run engine tests (should be 1000+ passed)
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5

# 2. Run panel tests (should be 80+ passed)
cd panel && npx vitest run 2>&1 | tail -5

# 3. Build and package
npm run build && npm run dist

# 4. Install and launch
killall vMLX 2>/dev/null || true
rm -rf /Applications/vMLX.app
cp -R release/mac-arm64/vMLX.app /Applications/
xattr -cr /Applications/vMLX.app
open /Applications/vMLX.app
```

---

## Feature Cohesion Matrix

All features must work together. This matrix shows the interactions:

| Feature | Works With | Key Test Coverage |
|---------|-----------|-------------------|
| **Continuous Batching** | Prefix Cache, Paged Cache, KV Quant, VL Models | `test_mllm_scheduler_stability.py` |
| **Paged Cache** | Prefix Cache (required), Continuous Batching, Mamba Hybrid | `test_paged_cache.py`, `test_hybrid_batching.py` |
| **KV Quantization** | Paged Cache, VL Models | `test_paged_cache.py` |
| **Mamba/SSM Hybrid** | Batching (auto-fallback), Legacy Cache | `test_hybrid_batching.py` |
| **Tool Calling** | All tool parsers, Reasoning parsers, VL Models | `test_tool_format.py`, `test_reasoning_tool_interaction.py` |
| **Tool Fallback Injection** | All models (Qwen, Llama, etc.), thinking ON/OFF | `test_tool_fallback_injection.py` |
| **Reasoning Parsers** | Tool parsers, Streaming, Content dedup | `test_reasoning_tool_interaction.py` |
| **VL (Vision-Language)** | MLLM Scheduler, Paged Cache, Vision Cache | `test_mllm_scheduler_stability.py` |

### Dependency Chain

```
Continuous Batching ─→ Prefix Cache ─→ Paged Cache
                                     ─→ KV Quantization  
                                     ─→ Disk Cache (exclusive with Paged)

Mamba Hybrid Models ─→ Auto-fallback to Legacy Cache
                     ─→ Batching still works (non-paged mode)

Tool Calling ─→ Chat Template (tools kwarg)
             ─→ Fallback Injection (if template drops tools)
             ─→ Tool Parser (qwen/llama/mistral/hermes/deepseek/etc.)
             ─→ Reasoning Parser (strips <think> before tool parsing)
```

---

## Tool Fallback Injection

When a model's chat template silently drops tool definitions (e.g., Qwen 3.5 with `enable_thinking=False`), the engine automatically:

1. Detects the missing tools by checking if the first tool name appears in the rendered prompt
2. Injects a standard XML `<tool_call>` instruction set into the system message
3. Re-applies the chat template with modified messages (tools removed from kwargs)

This is model-agnostic — works for any model family, not just Qwen. The fallback lives in `vllm_mlx/api/tool_calling.py::check_and_inject_fallback_tools()` and is called from both `SimpleEngine` and `BatchedEngine`.

---

## Versioning

- **Engine**: `pyproject.toml` → `version = "0.2.7"`
- **Panel**: `panel/package.json` → `"version": "0.3.10"`
- **Changelogs**: `CHANGELOG.md` (engine), `panel/CHANGELOG.md` (panel)

Always update CHANGELOG entries before building a release.
