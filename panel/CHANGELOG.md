# Changelog

## v1.2.1 — 2026-03-09 — Tool Calling Fix, MCP Safety

### Fixed
- **Critical: Auto-tool-choice default blocked all tool calling**: `DEFAULT_CONFIG.enableAutoToolChoice` was `false`, which blocked auto-detection via `??` operator (`false ?? true` returns `false`). Changed to `undefined` (auto-detect from model config). MCP tools and built-in tools now work out of the box.
- **MCP tool result truncation**: MCP tool results had no size limit (unlike built-in tools' 50KB cap), risking context overflow with large tool outputs. Now applies same truncation.
- **buildCommandPreview diverged from buildArgs**: SessionSettings command preview used `if (effectiveAutoTool)` but actual buildArgs uses `if (effectiveAutoTool || config.enableAutoToolChoice === undefined)`. Preview now matches actual behavior.
- **Old config migration**: Stored sessions with `enableAutoToolChoice: false` (the broken default) auto-migrate to `undefined` when loaded from database.

### Tests
- 1595 engine tests, 542 panel tests (2137 total)
- 6 new enableAutoToolChoice regression tests in settings-flow
- 6 new MCP tool result truncation tests in comprehensive-audit

## v1.2.0 — 2026-03-09 — Download Progress Fix, Deep Stability Audit

### Fixed
- **Critical: HuggingFace download stuck at 0%**: tqdm progress bars use `\r` carriage returns that arrive as multi-segment chunks. Parser now splits on `\r`/`\n`, takes the highest percent from each chunk, and strips ANSI escape codes. Downloads show real-time progress from the first byte.
- **NaN model age / Unknown author in HF browser**: HuggingFace list API omits `lastModified` and `author` fields. Added `mapHFModel()` helper that uses `createdAt` as date fallback and extracts author from `modelId.split('/')[0]`. Both `searchHF` and `getRecommendedModels` use it.
- **macOS 15 launch blocked (GitHub #10)**: `minimumSystemVersion` was set to `26.0.0` (macOS Tahoe), blocking all users on Sonoma (14) and Sequoia (15). Fixed to `14.0.0`.
- **Download progress bar visibility**: Progress bar now appears at 0% (`p.percent != null` instead of `p.percent > 0`).
- **Download marker cleanup on error**: Failed downloads now clean up the `.downloading` marker file, preventing models from appearing permanently stuck.
- **Responses API error events**: SSE parser now handles `response.error` and `response.failed` events, surfacing server errors to the user instead of silently dropping them.
- **Stale lock timeout cap**: Lock timeout escalation now capped at 10 minutes (`Math.min(existing.timeoutMs + 30_000, 10 * 60 * 1000)`).
- **Sparse tool calls array**: Out-of-order tool call indices no longer crash — placeholder entries are initialized for gaps.
- **4xx error detail parsing**: HTTP error responses with JSON bodies now extract and display the `detail` message instead of generic status text.
- **Database migration atomicity**: All schema migrations now wrapped in transactions — partial migration failures can't corrupt the database.
- **Auto-add `--enable-auto-tool-choice`**: When a tool call parser is configured, `--enable-auto-tool-choice` is now automatically added to engine args if not explicitly disabled.

### Engine (vmlx-engine 0.2.18)
- Fix CancelledError SSE hang, paged cache block leak on abort, VLM disk cache key mismatch
- Fix KV dequantize None crash (3 callers), reasoning trailing window false positives
- Add DeepSeek Unicode tool markers, reduce fallback threshold (10→3), raise strip_partial min (3→5)
- Fix tool fallback all-tools check, Mistral JSON validation
- Stop token leak on abort, ghost request 30s time-based reaping

### Tests
- 1595 engine tests, 530 panel tests (2125 total)
- 14 new regression tests covering all audit fixes

## v1.1.4 — 2026-03-07 — Tool Choice Fix, First-Launch UX & Input Validation

### Fixed
- **Critical: tool_choice="none" content swallowing**: Streaming Chat Completions silently swallowed content when tool markers were detected with `tool_choice="none"`. Content buffering now correctly disabled.
- **suppress_reasoning leaks**: Responses API `reasoning.done` event and Chat Completions reasoning-only fallback no longer leak reasoning when suppressed.
- **Non-streaming tool_choice="none"**: Both API paths now skip tool parsing when tools are suppressed.
- **PagedCacheManager input validation**: Rejects `block_size=0` and `max_blocks<2` with clear errors instead of crashing later.
- **First-launch empty state**: Auto-creates initial chat for new users. Remote sessions skip `detectConfig` (no filesystem path).
- **About page version**: Now reads dynamically from Electron `app.getVersion()` instead of hardcoded.

### Engine (vmlx-engine 0.2.12)
- Fix streaming `tool_call_active` gating, suppress_reasoning guards, PagedCacheManager validation
- Add hybrid detection logging, memory cache fallback warning
- 14 new invariant tests

## v1.1.3 — 2026-03-07 — Stop Button, Cache OOM & Reasoning Fix

### Fixed
- **Stop button stays red forever**: Pressing Stop during generation (especially mid-tool-call) left the button stuck in "Stop" state. Now immediately clears UI state after signaling abort.
- **Setup screen on first launch after update**: Engine auto-update raced with the setup check, causing the install screen to flash on first launch after updating. Engine update now completes before the window loads.
- **ask_user tool abort handling**: Aborting during an `ask_user` tool call no longer hangs the IPC handler for up to 5 minutes.
- **Hybrid VLM paged cache OOM crash**: Hybrid models (Qwen3.5-VL) with paged cache would crash with `kIOGPUCommandBufferCallbackErrorOutOfMemory` after a few requests. Block ref_counts were never decremented when cache hits couldn't be used (missing SSM state), causing blocks to accumulate until Metal GPU memory exhausted.
- **Reasoning toggle for always-thinking models**: When reasoning is toggled off for models like MiniMax M2.5, thinking text is now fully hidden instead of being shown as regular content. Users see a brief pause then only the final answer.
- **macOS Gatekeeper blocking app launch**: DMG is now properly notarized with Apple, so users can install and launch without security warnings.

## v1.1.2 — 2026-03-06 — Reasoning Parser Fix

### Fixed
- **Reasoning parser for always-thinking models**: MiniMax M2.5 and similar models that always inject `<think>` regardless of `enable_thinking` setting now correctly classify reasoning vs content when user disables reasoning
- **Parser dropdown UI**: Labels now include model names directly (e.g., "Qwen3 — Qwen / QwQ / MiniMax / StepFun"), help panel auto-opens when manually selecting a parser, more comprehensive model compatibility lists

### Engine (vmlx-engine 0.2.10)
- Fix `effective_think_in_template` override bug in both Chat Completions and Responses API paths

## v1.0.0 — 2026-03-05 — Production Release

### Production Readiness
- **App identity**: Bundle ID `net.vmlx.app`, copyright JANGQ AI, proprietary license
- **Code signing**: Developer ID Application: JANGQ AI (55KGF2S5AY), hardened runtime
- **Branded DMG**: Custom Midnight Steel background with vMLX logo and drag-to-install arrow
- **CSP header**: Content Security Policy on renderer window (blocks XSS, restricts connect/font/img sources)
- **Version 1.0.0**: Updated across package.json, Info.plist, and all metadata

### New Features
- **Date/Time tool**: Models can query current date, time, and timezone for time-aware responses
- **Embeddings tab**: Generate and compare embeddings with cosine similarity
- **Performance tab**: Real-time token generation speed monitoring
- **Cache management tab**: View cache stats, entries, warm up, and clear caches
- **420 tests** across 6 test suites (vitest)

### Engine (vmlx-engine 0.2.8)
- **KV cache quantization**: Storage-boundary q4/q8 compression (full precision during generation)
- **Prefix cache visibility**: `cached_tokens` reported in OpenAI-compatible API response
- **Mamba/SSM support**: BatchMambaCache with batch filtering, merging, and KV quantization safety
- **Vision-language + caching**: Only MLX engine where VL models work with full 5-layer caching stack
- **50+ auto-detected architectures**, 14 tool call parsers, 4 reasoning parsers

---

## v0.3.10 — 2026-03-02 — Remote API Audit, HF Downloader, Test Suite

### Bug Fixes
- **Remote abort fix**: `abortByEndpoint()` now correctly matches remote sessions — previously, stopping a remote session failed to abort in-flight chat requests because the `endpoint` field was never set on active request entries.

### Improvements
- **HF model sizes**: HuggingFace search results now display model file sizes (extracted from `safetensors` metadata with fallback parameter estimation).
- **HF model link**: Each model card in the Download tab now has a ↗ button to open the HuggingFace model page in the browser.
- **Download status fallback**: When tqdm progress can't be parsed (e.g., during "Fetching N files" phase), the download status bar now shows the raw status text instead of a blank pulsing dot.

### New
- **Test suite**: Added vitest test infrastructure with 80 tests across 3 test files:
  - `remote-session.test.ts` — URL resolution, auth headers, modelPath format, abort tracking, stale lock recovery, health check paths
  - `request-builder.test.ts` — Completions/Responses API parameter forwarding, remote gating, tool format, filterTools
  - `download-manager.test.ts` — tqdm parsing, size formatting, number formatting, timeAgo, model size extraction, download queue logic

### Verified (Remote API Audit)
- All sampling parameters (temperature, top_p, top_k, min_p, repeat_penalty, stop, max_tokens, reasoning_effort) correctly forwarded for remote sessions
- `chat_template_kwargs` correctly excluded for remote sessions
- MCP tools correctly blocked with clear error message
- Tool format uses correct wire format per API (Completions: wrapped, Responses: flat)
- Health check uses `/v1/models` for remote, `/health` for local
- Dampened fail counting (every 3rd failure) for remote health monitor
- 15-second recently-healthy optimization skips redundant health checks
- Stale lock recovery works with session-specific timeout + 30s buffer
- Abort handler sends fire-and-forget cancel request (silently fails for providers without cancel endpoints)
- Remote sessions correctly reset to 'stopped' on app restart

---

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
- **StepFun parser: Fixed invalid `step3p5` tool-call-parser (not a valid vmlx-engine value). StepFun models now correctly use `qwen` parser since they're Qwen3-architecture based.
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

#### One-Click vMLX Engine Installer
- **First-run setup gate**: App shows SetupScreen on launch if vMLX Engine is not installed, blocking access until installation succeeds
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
- These were never accepted by vMLX Engine's OpenAI-compatible endpoint

#### Removed Dead Preload APIs & Handler Files
- Removed `server.*`, `models.list/load/unload/download`, `update.*`, `inference.*`, `config.*` preload APIs
- Deleted dead IPC handlers: `ipc/server.ts`, `ipc/config.ts`, `ipc/inference.ts`
- Deleted dead UI components: `ModelSelector.tsx`, `ServerConfig.tsx`

#### Verified All CLI Flags
- Confirmed ALL 23 flags in `buildArgs()` are valid against `vmlx-engine serve --help`

---

## v0.3.2 — 2026-02-05 — Chat Accuracy, Metrics & Organization

### Bug Fixes
- Removed non-functional `top_k` and `repeat_penalty` chat settings (not accepted by vMLX Engine API)
- Partial responses now saved on streaming error with `[Generation interrupted]` marker

### New Features
- **Prompt processing speed (pp/s)**: Uses `stream_options.include_usage` for prompt token metrics
- **Chat search**: Search titles and message content
- **Chat rename**: Inline rename via pencil icon
- **Chat folders**: Create folders, move chats between folders

---

## v0.3.1 — 2026-02-05 — Settings, Chat Controls & Stability Fixes

### New Features
- **Session Settings page**: Full-page config editor for all vMLX Engine parameters
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
- **Multi-instance**: Run multiple vMLX Engine servers simultaneously
- **Process adoption**: Detect and adopt running `vmlx-engine serve` processes on startup
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
- vMLX Engine process management
- Chat interface with SQLite persistence
- Model detection

---

**Current Version:** v1.2.0
**Status:** Production release — macOS Apple Silicon
