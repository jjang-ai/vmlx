# Changelog

## v0.3.9 — 2026-02-15 — Paged Cache Default, Streaming Smoothness, GLM-4.7 Verified

### Changes
- **Paged KV cache ON by default**: `usePagedCache` now defaults to `true` for all models (reduces memory fragmentation, better for long contexts).
- **Smoother streaming**: IPC throttle reduced from 80ms (~12 fps) to 32ms (~30 fps) for visually smoother token rendering.
- **GLM-4.7 parser label fix**: Reasoning parser dropdown correctly shows GLM-4.7 under "GPT-OSS / Harmony" instead of under "Qwen3".

### Verified
- GLM-4.7 Flash auto-detection: `glm4_moe_lite` model_type → `glm47-flash` family → `glm47` tool parser + `openai_gptoss` reasoning parser. Confirmed correct.
- TTFT measurement: Accurate (fetchStartTime → firstTokenTime). Prefix cache hits visible as near-zero TTFT.
- API key enforcement: End-to-end (UI → `--api-key` → `Authorization: Bearer` header → timing-safe comparison on server).
- System prompt: Properly injected as first message (completions) or `instructions` (responses API).
- All 15 chat settings verified to take effect in API requests.
- Both Chat Completions and Responses API have full parity (streaming, tools, reasoning, usage).
- OpenCode/Cline compatibility: Standard OpenAI-compatible endpoints with Bearer token auth.

---

## v0.3.8 — 2026-02-15 — Tool Loop Hardening, Abort Safety, Toast Notifications

### Bug Fixes
- **Abort during tool execution**: Added abort check between each tool in the execution loop. Previously, clicking Stop during multi-tool execution waited for ALL tools to finish (could be 60s+ for shell commands).
- **Tool argument parse crash**: Malformed JSON in tool call arguments no longer crashes the entire tool loop — returns error for that tool and continues.
- **Auto-continue token threshold**: Fixed using cumulative token count across all iterations. Now tracks per-iteration tokens so the 100-token threshold works correctly for each follow-up.
- **SSE parser state leak**: Follow-up requests now reset `currentEventType` before streaming, preventing misparsed first chunks from stale event types.
- **Client-side tool call buffering**: Reduced false positives — markers must appear at start of a line, not mid-sentence when model explains tool syntax.
- **Tool call ID collisions**: Replaced `Date.now() + random(6)` with UUID-based IDs (collision-free).
- **Chat settings reset**: Reset now preserves non-inference settings (working directory, system prompt, tool toggles, wire API) instead of wiping everything.
- **Dynamic require**: Replaced `require('../model-config-registry')` with proper static import.

### Improvements
- **Smart enable_thinking default**: `enable_thinking` now defaults to `true` only for models with a reasoning parser (Qwen3, DeepSeek-R1, GLM-4, etc.). Non-reasoning models no longer send unnecessary `enable_thinking=true`.
- **Tool result truncation**: All tool results are now truncated to 50KB to prevent context overflow on large file reads or command outputs. Truncation message indicates original size.
- **Start button in SessionView**: Stopped/errored sessions can now be restarted from within the session view header.
- **Full session lifecycle events**: SessionView now listens to starting/ready/stopped/error events for real-time status updates.
- **Toast notification system**: Replaced all 9 browser `alert()` calls with themed in-app toast notifications. Supports error/warning/info/success types with auto-dismiss (10s errors, 6s others). Matches app dark theme with backdrop blur and color-coded styling.
- **MIT LICENSE added**: For GitHub distribution readiness.

### Cleanup
- Removed 14 dead dependencies (62 packages): zustand, lucide-react, 8x @radix-ui/*, class-variance-authority, clsx, tailwind-merge
- Removed dead preload functions: `removeStreamListener()`, `removeListeners()` (dangerous removeAllListeners pattern, zero callers)
- Removed dead `ipcMain.on('ping')` debug handler
- Removed `require('electron').shell` → proper static `import { shell }`
- Deleted 4 stale test-*.js files and theme-preview.html
- Build script now kills running instances before deploying

---

## v0.3.7 — 2026-02-15 — Per-Category Tool Toggles, Code Block Styling, StepFun Fix

### Bug Fixes
- **StepFun parser**: Fixed invalid `step3p5` tool-call-parser (not a valid vllm-mlx value). StepFun models now correctly use `qwen` parser since they're Qwen3-architecture based.
- **Dead prose CSS**: Installed `@tailwindcss/typography` plugin — `prose`/`prose-invert` classes on assistant messages now actually apply styling. Previously the classes had no effect.
- **Code copy buttons**: Fixed non-functional copy buttons — DOMPurify correctly strips `onclick` handlers; switched to React event delegation.
- **Dynamic require**: Replaced `require('electron').shell` with proper static import of `shell` from electron.

### New Features
- **Per-category tool toggles**: Built-in tools can now be individually toggled by category (File I/O, Search, Shell, Web Search, URL Fetch) in Chat Settings under the Agentic section.
- **Code block copy buttons**: Each code block now shows a language label and a hover-revealed "Copy" button for one-click copying.
- **Code block styling**: Proper borders, backgrounds, and inline code styling via Typography plugin + custom CSS overrides.
- **Start button in SessionView**: Stopped/errored sessions can now be restarted directly from the session interior header without navigating back to the dashboard.
- **MIT LICENSE**: Added LICENSE file for GitHub distribution.

### Cleanup
- **Removed 14 dead dependencies**: `zustand`, `lucide-react`, 8x `@radix-ui/*`, `class-variance-authority`, `clsx`, `tailwind-merge` — none were imported anywhere in src/. Removed 62 packages total.
- **Removed dead preload code**: `removeStreamListener()` and `removeListeners()` used dangerous `removeAllListeners` pattern and had no callers — all components use individual unsubscribe functions.
- **Removed dead ping handler**: `ipcMain.on('ping')` debug handler removed from main/index.ts.
- **Deleted stale test files**: 4 root-level `test-*.js` files removed.
- **Deleted theme-preview.html**: Dev artifact removed.

### Changes
- Centralized `filterTools()` function replaces 4 inline filter blocks in chat.ts
- 5 tool category sets: `FILE_TOOLS`, `SEARCH_TOOLS`, `SHELL_TOOLS`, `WEB_SEARCH_TOOLS`, `FETCH_TOOLS`
- 4 new `chat_overrides` columns: `fetch_url_enabled`, `file_tools_enabled`, `search_tools_enabled`, `shell_enabled`
- Chat override inheritance includes all 4 new tool category fields
- Custom `marked.Renderer` for code blocks with copy button and language label
- `@tailwindcss/typography` added to tailwind.config.js plugins
- Full prose variable override set in index.css for dark theme consistency
- SessionView now listens for `starting`, `ready`, `stopped`, and `error` events for real-time status updates
- Build script updated to kill running instances before deploying

---

## v0.3.6 — 2026-02-15 — Streaming Fixes, Model Detection, Web Search Toggle

### Bug Fixes
- **Streaming content continuity**: Pre-tool text no longer disappears when follow-up stream starts after tool execution. Content from all tool iterations is accumulated and displayed continuously.
- **Abrupt response endings**: Default max_tokens increased from 2048 to 4096 to prevent mid-response cutoffs during agentic tool-use conversations.
- **Content flush before tool execution**: Renderer receives a content snapshot before blocking tool execution, eliminating the "frozen" appearance during tool runs.
- **GLM-4.7 model detection**: Added `glm4_moe` and `glm4_moe_lite` model_type entries; GLM-Z1 pattern now matches; reasoning parser (`openai_gptoss`) added for GLM-4.7 family.
- **Nemotron-Orchestrator-8B detection**: Config.json `model_type` override correctly identifies Qwen3-based fine-tunes regardless of model name (prevents misclassification as Nemotron hybrid architecture).

### New Features
- **Separate Web Search toggle**: Web Search & URL Fetch can be independently enabled/disabled per chat under Agentic settings, without affecting other built-in coding tools.

### Changes
- `webSearchEnabled` column added to `chat_overrides` (defaults to enabled)
- Tool filtering uses `WEB_TOOL_NAMES` set to exclude `web_search`/`fetch_url` when disabled
- `allGeneratedContent` now included in `chat:stream` emissions for continuous display
- Final saved message combines content from all tool iterations

---

## v0.3.5 — 2026-02-15 — User-Configurable API Keys

### Breaking Changes
- **Removed hardcoded Brave Search API key** — users must provide their own key via About > API Keys

### New Features
- **API Keys section in About page**: Brave Search API key input with show/hide toggle, persistent SQLite storage
- **Settings IPC bridge**: `settings:get/set/delete` handlers for app-level key-value settings
- **Preload API**: `window.api.settings` namespace for renderer access

### Changes
- `executor.ts` reads Brave key from `db.getSetting('braveApiKey')` instead of file-based config
- Removed `tools-config.json` dependency — no more plaintext API keys on disk
- `BRAVE_API_KEY` environment variable still works as fallback

---

## v0.3.4 — 2026-02-06 — In-App Installer, Code Review Fixes

### New Features

#### One-Click vLLM-MLX Installer
- **First-run setup gate**: App shows SetupScreen on launch if vLLM-MLX is not installed, blocking access until installation succeeds
- **Auto-detect install methods**: Checks for `uv` (preferred) then `pip3` with Python >=3.10 validation
- **Streaming terminal output**: Real-time install/upgrade logs shown in a terminal-style viewer
- **One-click install in About page**: UpdateManager rewritten with Tailwind, supports streaming install and upgrade
- **Cancel support**: Users can abort in-progress installs
- **Smart detection**: Resolves symlinks to correctly identify uv-installed binaries at `~/.local/bin`

#### New Files
| File | Purpose |
|------|---------|
| `src/renderer/src/components/setup/SetupScreen.tsx` | First-run blocker with auto-detect + streaming install |

#### Rewritten Files
| File | Changes |
|------|---------|
| `src/main/vllm-manager.ts` | Added uv support, `detectAvailableInstallers()`, `installVllmStreaming()`, `cancelInstall()`, symlink resolution |
| `src/main/ipc/vllm.ts` | New IPC handlers for streaming install, detect-installers, cancel-install; getter pattern for window |
| `src/renderer/src/components/update/UpdateManager.tsx` | Rewritten with Tailwind (removed CSS dependency), streaming install/update support |

#### Modified Files
| File | Changes |
|------|---------|
| `src/main/index.ts` | Passes `() => mainWindow` getter to vllm handlers |
| `src/preload/index.ts` | Added `detectInstallers`, `installStreaming`, `cancelInstall`, `onInstallLog`, `onInstallComplete` |
| `src/preload/index.d.ts` | Updated types for all new vllm + chat APIs |
| `src/renderer/src/App.tsx` | Added `setup` view type, SetupScreen gates app access |
| `src/renderer/src/components/sessions/CreateSession.tsx` | Updated error message to reference About page installer |

#### Deleted Files
| File | Reason |
|------|--------|
| `src/renderer/src/components/update/UpdateManager.css` | Replaced by Tailwind classes |

### Code Review Fixes (from v0.3.3)

#### Process Lifecycle Safety
- **SIGTERM-first shutdown**: `stopAll()` sends SIGTERM, waits 3s, then SIGKILL (was immediate SIGKILL)
- **Kill timeout**: `killChildProcess` has 15s hard timeout that resolves even if process hangs
- **Quit timeout**: App quit has 8s `Promise.race` to prevent hanging on `stopAll()`
- **Loading exclusion**: `resolveServerEndpoint` only matches sessions with `status === 'running'` (not `loading`)

#### SSE Streaming Safety
- **AbortController**: Chat SSE fetch now uses `AbortController` for proper cancellation
- **Per-chat concurrency guard**: `activeRequests` Map prevents double-send per chat
- **New `chat:abort` handler**: Cancels active generation via IPC

#### Database & Performance
- **WAL mode**: SQLite now uses `journal_mode = WAL` for concurrent read performance
- **Module-level constants**: `TEMPLATE_STOP_TOKENS` and `TEMPLATE_TOKEN_REGEX` moved out of message handler
- **ppSpeed guard**: Guarded against Infinity with `> 0.001` threshold

#### Window Reference Safety
- **Getter pattern**: All IPC handlers (`sessions`, `chat`, `vllm`) use `() => BrowserWindow | null` getter to survive macOS window recreation
- **Chat stream cleanup**: `onStream`/`onComplete` return unsubscribe functions (matching session event pattern)

#### UI Safety
- **Launch button guard**: Disabled during launch in CreateSession (`disabled={launching || !selectedModel}`)
- **Unmount guards**: SetupScreen and UpdateManager use `mountedRef` to prevent setState on unmounted components

---

## v0.3.3 — 2026-02-05 — Stability, Settings Panel & Cleanup

### Bug Fixes

#### Health Monitor No Longer Marks Sessions Down on Single Failure
- Added `failCounts` map with 3-strike threshold
- Counter resets on any successful check

### New Features

#### Server Settings Side Panel
- "Server" button in SessionView opens an inline right-side drawer instead of navigating to a separate page
- Drawer includes the full SessionConfigForm with Save, Save & Restart, and Reset buttons
- Only one settings panel (Chat or Server) can be open at a time

### Cleanup

#### Removed Ghost Chat Override Settings
- Removed `topK` and `repeatPenalty` from `ChatOverrides` interface and DB methods
- These were never accepted by vLLM-MLX's OpenAI-compatible endpoint

#### Removed Dead Preload APIs & Handler Files
- Removed `server.*`, `models.list/load/unload/download`, `update.*`, `inference.*`, `config.*` preload APIs
- Deleted dead IPC handlers: `ipc/server.ts`, `ipc/config.ts`, `ipc/inference.ts`
- Deleted dead UI components: `ModelSelector.tsx`, `ServerConfig.tsx`

#### Verified All CLI Flags
- Confirmed ALL 23 flags in `buildArgs()` are valid against `vllm-mlx serve --help`

---

## v0.3.2 — 2026-02-05 — Chat Accuracy, Metrics & Organization

### Bug Fixes
- Removed non-functional `top_k` and `repeat_penalty` chat settings (not accepted by vLLM-MLX API)
- Partial responses now saved on streaming error with `[Generation interrupted]` marker

### New Features
- **Prompt processing speed (pp/s)**: Uses `stream_options.include_usage` for prompt token metrics
- **Chat search**: Search titles and message content
- **Chat rename**: Inline rename via pencil icon
- **Chat folders**: Create folders, move chats between folders

---

## v0.3.1 — 2026-02-05 — Settings, Chat Controls & Stability Fixes

### New Features
- **Session Settings page**: Full-page config editor for all vLLM-MLX parameters
- **Chat Settings drawer**: Per-chat inference controls (temperature, top_p, max_tokens, system prompt, stop sequences)
- **Shared SessionConfigForm**: Extracted reusable config form component
- **Session Card configure button**: Gear icon navigates to SessionSettings

### Bug Fixes
- **Sessions no longer die when navigating away**: Preload `on*` methods return targeted unsubscribe functions
- **macOS traffic light overlap**: Added 72px left padding
- **Settings renamed to About**: Avoids confusion with session/chat settings

---

## v0.3.0 — 2026-02-05 — Session-Centric Multi-Instance Manager

### Breaking Changes
Complete redesign from tab-based single-server to session-centric multi-instance manager.

### New Features
- **Session Dashboard**: Grid of session cards with real-time status
- **Two-step Creation Wizard**: Model picker + full server configuration
- **Session View**: Per-session chat with header, streaming, and settings
- **Multi-instance**: Run multiple vLLM-MLX servers simultaneously
- **Process adoption**: Detect and adopt running `vllm-mlx serve` processes on startup
- **Chat history per model path**: Persistent across load/unload cycles
- **Configurable model directories**: Add/remove scan directories via UI

### Architecture
- `SessionManager` replaces `ServerManager` — manages N processes via Map
- `sessions` table in SQLite, `model_path` column on `chats`
- Global health monitor (5s interval) with per-session health checks
- Graceful stop: SIGTERM with SIGKILL fallback

---

## v0.2.0 — 2026-02-04 — Chat & Server Fixes

- SSE streaming with proper line buffering
- Chat template stop sequences (ChatML, Llama 3, Phi-3, Mistral, Gemma)
- Template token regex stripping from streamed deltas
- Multi-server chat routing
- Auto-detect running vLLM processes
- Window destruction crash fix

---

## v0.1.0 — 2026-02-04 — Initial Release

- Basic Electron + React + TypeScript app
- vLLM-MLX process management
- Chat interface with SQLite persistence
- Model detection

---

**Current Version:** v0.3.9
**Status:** Production-ready on macOS Apple Silicon
