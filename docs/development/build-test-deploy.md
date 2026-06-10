# Build, Test & Deploy

Complete guide for building the vMLX desktop app, running the test suite, and deploying to `/Applications`.

## Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Node.js** ≥ 18 and **npm** ≥ 9
- **Python** 3.12+ with the vmlx-engine venv (`.venv/`)

## Building the Desktop App

### Makefile (recommended)

From the **repo root**, run **`make`** or **`make help`** for all targets. Make builds prerequisites automatically (e.g. `make dmg` runs `deps` and checks the bundle first).

```bash
make                 # or: make help
make doctor          # see what's missing
make first-dmg       # once: deps + bundle + DMG + SHA256SUMS
make sync            # refresh vmlx_engine + jang in bundled-python/
make app             # staged panel/release/mac-arm64/vMLX.app (no DMG)
make install         # app if needed + /Applications/vMLX.app
make engine-app      # sync + app after engine/jang edits
make engine-and-install  # sync + app + install (common local loop)
make release         # DMG + SHA256SUMS (distribution)
make engine-dmg      # sync + DMG + SHA256SUMS (ship after engine/jang edits)
make verify-hashes   # optional QA on release/SHA256SUMS
make clean           # remove build outputs (dist/, release/)
make clean-all       # also remove .venv, bundled-python, node_modules
make uninstall       # remove /Applications/vMLX.app
make freshen         # git fetch + checkout latest vmlx tag (or main)
make freshen-jang    # git fetch + checkout jang branch with step37/
make freshen-all     # both repos
make stash           # before freshen — saves tracked edits
make stash-apply     # after freshen — restore (keeps stash entry)
make stash-pop       # restore and remove stash entry
```

For **Python engine / pytest only** (no UI):

```bash
make engine-install              # .venv + pip install -e ".[dev,jang,image]"
make engine-install-local-jang   # same, plus editable local jang-tools
make test-engine
```

`uv tool install vmlx` installs the **CLI only** — it does not build the Electron UI. Use `make first-dmg` to test the desktop app.

**jang-tools** is resolved automatically (first match wins):

1. `VMLX_JANG_TOOLS_SOURCE` if set  
2. `../jang/jang-tools` beside the vmlx repo (typical: clone [jangq](https://github.com/jjang-ai/jangq) as a sibling)  
3. `../jangq/jang-tools`  
4. `~/jang/jang-tools` (legacy)

For vMLX 1.5.56+, that checkout must include `jang_tools/step37/` — run `make freshen-jang` after clone.

### Development Mode

```bash
cd panel
make deps            # or: npm install --ignore-scripts
make dev             # launches Electron with hot reload
```

### Production Build

**Local testing** (staged `.app`, no DMG):

```bash
make first-dmg       # once: deps + bundle (includes first DMG)
make sync            # after vmlx_engine / jang edits
make app             # or: make engine-app  (sync + app)
make install         # copies/signs to /Applications/vMLX.app
# one-shot after engine edits:
make engine-and-install
```

**Distribution** (reuses staged `.app` when current):

```bash
make install         # build/sign staged .app → /Applications
make release         # DMG from that staged .app, then remove release/mac-arm64/
make engine-dmg      # check-sync + release after engine/jang edits
```

When sources change, stale DMGs and `SHA256SUMS` are removed automatically. After `make release`, the staged `.app` is deleted (DMG is authoritative); a later `make install` with no source changes installs from the current DMG without rebuilding.

### npm scripts vs Makefile

| Goal | Use | Avoid |
|------|-----|--------|
| Local install to /Applications | `make install` or `make engine-and-install` | expecting `make install` to build a DMG |
| Local staged `.app` only | `make app` or `make engine-app` | `npm run dist` (release gates) |
| Local installable DMG | `make release` | `npm run dist` (runs release manifest gates) |
| Panel TypeScript only | `npm run build` | expecting a DMG |
| Full Python rebundle | `make bundle` | `npm run build:full` unless you mean it |
| Official release ship | `npm run dist` or `scripts/build-release-dmgs.sh` | — |

`npm run build` no longer runs `bundle-python.sh` (that was causing accidental 30-minute rebundles).

### Resume after failure

| Step failed | Re-run |
|-------------|--------|
| `make bundle` | `make bundle` |
| `make app` | `make app` |
| `make release` | `make release` |
| Engine/jang edit (local) | `make engine-app` or `make engine-and-install` |
| Engine/jang edit (ship) | `make engine-dmg` |
| Full reset | `make first-dmg` |

Staged app: `panel/release/mac-arm64/vMLX.app`. Ship artifacts: `panel/release/vMLX-*-arm64.dmg` and `panel/release/SHA256SUMS`.

### Install to /Applications

Local `make release` builds are ad-hoc signed; **bundled Python dylibs must be signed before launch**. Use `make install` (do not `cp` the `.app` directly). `make install` auto-syncs engine/jang changes and rebuilds the staged `.app` when sources are newer than the last build.

```bash
make install
open /Applications/vMLX.app
```

Custom destination: `APP_DEST="$HOME/Applications/vMLX.app" make install`

Manual copy without signing will fail at runtime with unsigned `libpython3.12.dylib`. If you must copy by hand, run `panel/scripts/install-from-release.sh` afterward (it signs all bundled Mach-O files, then seals the app).

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
.venv/bin/python -m pytest tests/ --cov=vmlx_engine
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

This is model-agnostic — works for any model family, not just Qwen. The fallback lives in `vmlx_engine/api/tool_calling.py::check_and_inject_fallback_tools()` and is called from both `SimpleEngine` and `BatchedEngine`.

---

## Versioning

- **Engine**: `pyproject.toml` → `version = "0.2.7"`
- **Panel**: `panel/package.json` → `"version": "0.3.10"`
- **Changelogs**: `CHANGELOG.md` (engine), `panel/CHANGELOG.md` (panel)

Always update CHANGELOG entries before building a release.
