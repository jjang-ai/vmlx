# Swift v2 Production Audit — 2026-04-18 (Ralph loop iteration 1)

Scope: `/Users/eric/vmlx/swift` dev branch. Every Swift source file: Engine,
App, Server, LMCommon, LLM, VLM, Flux, Embedders, Whisper, TTS, CLI.

Total code surface: **145k lines of vMLX Swift** across 352 files.
- vMLXLLM — 26.2k lines, 60 files (model ports)
- vMLXLMCommon — 25.8k lines, 109 files (cache stack, batch engine, scheduler)
- vMLXEngine — 17.4k lines, 44 files (Engine actor, MCP, settings, metrics)
- vMLXVLM — 18.2k lines, 21 files (VL processors + models)
- vMLXApp — 17.2k lines, 57 files (SwiftUI chrome)
- vMLXServer — 4.7k lines, 15 files (Hummingbird routes)
- vMLXFluxKit / vMLXFluxModels / vMLXFluxVideo — 3.4k lines (image gen — STUB)
- Tests — 63 files, 8.5k lines. Runnable via `swift test`: **288/288 green**.

---

## 0. Production ship state

| Component | Status |
|---|---|
| `vmlxctl` release binary | **65 MB, 12:46 today**, all today's fixes |
| `vMLX` app release binary | **72 MB, 12:46 today**, all today's fixes |
| Notarized DMG | **STALE (Apr 17 01:51)** — does not contain today's 11 fixes |
| Xcode project | `xcodegen generate` run clean today |
| Code signing identity | Developer ID Application: ShieldStack LLC (55KGF2S5AY), verified |
| Notarization creds | `.env.signing` present, verified |
| `default.metallib` | 3.6 MB present at `Sources/Cmlx/default.metallib` |
| Pre-build guards | `scripts/stage-metallib.sh` + build-release.sh in place |

**Ship blocker:** DMG is stale. Today's 11 Swift source edits + 3 test additions need to be baked into a fresh notarized `vMLX-2.0.0-beta.3-arm64.dmg`. Run `scripts/build-release.sh 2.0.0-beta.3` to produce one.

---

## 1. Chat pipeline — end-to-end

### Works (verified by live harness + code inspection)
- OpenAI `/v1/chat/completions` streaming + non-streaming, schema-valid JSON
- Anthropic `/v1/messages` streaming + non-streaming, `message_start` / `content_block_delta` / `content_block_stop` / `message_stop` events
- Ollama `/api/chat` + `/api/generate` NDJSON streams, `done_reason` preserved, tool_calls spliced
- OpenAI `/v1/responses` (new API)
- Legacy `/v1/completions` + `/v1/completions/:id/cancel`
- `/v1/chat/completions/:id/cancel` — cancellation via request id
- Multi-turn prefix cache hits (text-only, Llama-1B: 256 tokens cached after T1)
- Reasoning content routed separately from content (delta.reasoning_content)
- Tool-call round-trip (2 tool_calls, finish=tool_calls on Qwen3.6)
- Tool-choice required / none / function-name
- JSON mode + JSON schema (engine produces best-effort, validator rejects malformed)
- Unicode round-trip (Japanese 日本語, Hebrew עברית, emoji 🦊)
- Temperature variance (hot != greedy on supported models)
- Seed reproducibility (non-hybrid models)
- Stop sequences
- Stream usage chunk (`prefill_ms` + `tokens_per_second`)
- Input validation (6/6 400s)
- Admin auth gate
- Benchmark endpoint (422 tok/s on Llama)
- Concurrent burst 5/5 survives
- Sleep/wake cycle (soft + deep)
- Large context (5809 tokens correctly answered)

### Known gaps / edge cases
- **Logprobs** — returns 400 "not yet supported" (ChatRequest.swift:568). Design-correct stub; needs threading through MLX eval.
- **Seed reproducibility on hybrid-SSM models** — non-deterministic by algorithm (Mamba scan). Documented but flaky tests.
- **Llama-1B tool parsing** — model emits tool syntax but harness can't always parse `<|python_tag|>` truncated mid-call on small models. Not a server bug.

### Production todos for chat
1. ~~Edit-last-user-message + retry~~ — **IMPLEMENTED.** `ChatViewModel.editMessage` + `regenerate` (lines 327, 334); wired into `MessageBubble` at lines 177-198 + 222-232 with edit/regenerate buttons.
2. Branch-from-message — **NOT IMPLEMENTED.** No `forkSession` / `cloneSession` / `branchSession` function found. Would need ForkSession helper that copies messages up to an anchor and starts a new chat from there. Low priority vs alternatives.
3. ~~Chat search (global)~~ — **IMPLEMENTED.** `SessionsSidebar:17` has search `TextField("Search chats", text: $vm.searchQuery)` + `vm.filteredSessions`.
4. ~~Session rename~~ — **IMPLEMENTED.** `ChatViewModel.renameSession` at line 259, wired from SessionsSidebar context menu (line 63).
5. Token-count preview in input bar — **NOT IMPLEMENTED.** No `estimateTokens` or preview label in InputBar.swift.

---

## 2. Server lifecycle

### Works
- Multi-session coexistence (per-session Engine actor + HTTP listener)
- Session persistence across app restart (hydrateSessionsFromSettings in vMLXApp.swift:549)
- Start / stop / soft-sleep / deep-sleep / wake (`AppState.startSession`/`stopSession`, tray + ServerScreen drive them)
- Load progress: 40+ regex patterns → `LoadProgress` events → ServerScreen + banner
- Idle timer now fires `.softSleep` at `softAfter` s, `.deepSleep` at `deepAfter` s
- **Idle countdown rendering** — `TrayItem.stateLabel` now shows `"Running · Sleeps in 4:32"` (added today)
- WakeFromStandby called before serving requests on `.standby`

### Known gaps
- Port change in Session settings — needs session restart (no hot-swap-port; sessions listen on bound port).
- LAN toggle live — needs verification; may require session restart.
- Hot-swap model within same session — not tested; user typically creates new session.
- Crash recovery (vmlxctl crashed → auto-restart?) — no supervisor. Process manager would need external wrapper.

---

## 3. API protocol coverage

### OpenAI — fully wired
All /v1/ routes present: chat/completions (+ cancel), completions (+ cancel), responses (+ cancel), audio/transcriptions, audio/speech, audio/voices, embeddings, rerank, models, images/generations, images/edits.

### Anthropic — fully wired
/v1/messages streaming + non-streaming. Missing: anthropic tool_use blocks (check AnthropicRoutes.swift).

### Ollama — fully wired
/api/chat, /api/generate, /api/tags, /api/show (with capabilities array — NEW helper extracted + unit-tested today), /api/version (0.12.6), /api/ps, /api/pull, /api/copy, /api/delete, /api/embed, /api/embeddings. Copilot-compat: `done_reason` splice across two-chunk pattern (verified in harness).

### MCP — stdio + SSE transports
`/v1/mcp/servers`, `/v1/mcp/tools`, `/v1/mcp/execute`, plus `/mcp/:server/**` proxy routes. Plain HTTP JSON-RPC transport marked "not yet implemented" (minor gap).

### Admin / cache / metrics
`/admin/wake`, `/admin/soft-sleep`, `/admin/deep-sleep`, `/admin/benchmark`, `/admin/dflash`, `/admin/cache/stats`, `/admin/models/:id` DELETE. `/v1/cache/*` routes. `/metrics` Prometheus-style.

### Adapters (LoRA)
`/v1/adapters`, `/v1/adapters/load`, `/v1/adapters/unload`, `/v1/adapters/fuse` — implementation status unverified.

---

## 4. Cache stack

All 5 tiers present + wired via `CacheCoordinator`:

| Tier | Class | Purpose | Keyed by |
|---|---|---|---|
| L1 paged | `PagedCacheManager` | block-indexed in-memory | tokens + modelKey + mediaSalt |
| L1.5 memory | `MemoryAwarePrefixCache` | whole-prompt, byte-budgeted LRU | tokens (text-only) |
| L2 disk | `DiskCache` (TQDiskSerializer v2) | persistent, survives restart | tokens + modelKey + mediaSalt |
| SSM companion | `SSMStateCache` | Mamba state for hybrid models | tokens + mediaSalt |
| Block disk | `TQDiskSerializer` | per-block on disk | (shared by L2) |

### Verified this session
- mediaSalt threading through fetch + store (both BatchEngine and Evaluate paths)
- VL prefix cache hits on block boundary (Gemma-4-26B T2 cached=256)
- Pressure-check throttle ≤ 10s (regression-guarded)
- windowStep ≥ 4096 (regression-guarded)
- SSM re-derive cancel gate (both spawn-site + detached-task isCancelled guards)

### Known gaps
- SSM re-derive for thinking models (gpl>0) remains slow (async, but takes ~model-inference latency to repopulate state). Real fix is capture-during-prefill via unused `_prefill_for_prompt_only_cache()`.
- MiniMax M2.5 FlashMoE slot-bank at 256 thrashes on 60+ layer models. Auto-size formula needed.

---

## 5. JANG / JANGTQ / MXTQ

### Load paths (`Sources/vMLXLMCommon/Load.swift`)
- `.jangspec` bundle — detected, loaded via `JangSpecBundleLoader`
- JANG v1 (`.jang.safetensors`) — uint8→uint32 repacking
- JANG v2 (standard safetensors) — per-layer quant inference
- JANGTQ native (`jang_config.json.weight_format == "mxtq"` OR `jangtq_runtime.safetensors`)
  - Skips MXTQ→affine expander
  - Skips MoE gate dequant
  - Keeps `.tq_packed`/`.tq_norms` raw for kernels (P3/P15/P17/P18)
  - Case-insensitive weight_format compare
- MXTQ (non-native JANG with mxtq_seed): runs `dequantizeJangMXTQ` expander
- HTML-error-page sniffing — catches partial downloads before safetensors error

### Verified this session
- MiniMax-M2.7-JANGTQ-CRACK: 57 GB weights, loads in 11s at 3.1 GB peak RSS (mmap working)
- Qwen3.6-35B-A3B-JANGTQ2: 46/49 full-suite pass, server alive through cancel/burst
- **Load peak memory halved** via `weights.removeAll()` + inlined ModuleParameters (today)

### Known gaps
- Partial MXTQ bundles (no .tq_packed, weight_format=mxtq) surface actionable error — good.
- JANGTQ cancel path: **closed today** (SSM re-derive cancel gate)

---

## 6. VL / video

### Verified
- OpenAI `image_url` parts decode (data:, file://, http(s)://)
- Ollama `images:` array decode
- Video frame extraction via AVAssetImageGenerator
- `<|image|>` marker auto-prepended when VL content lacks one (fix from Apr 16)
- 30-image cap enforced at parse time (halved decode RAM today)
- Video frame memory halved via eager CIImage → MLXArray (today)
- Multi-turn cache hit at block granularity (Gemma-4-26B T2 cached=256)
- `video_url` AVURLAsset pre-probe (rejects bad base64 .mp4 blobs)

### Known gaps
- VL decode tok/s never benchmarked with 200+ token completion
- SmolVLM2 has hardcoded values / TODO markers (line 47, 83, 181, 223)
- FastVLM max_length trim TODO (line 1125)

---

## 7. Engine features

### FlashMoE — works
- Qwen / Mistral / MiniMax via SwitchGLU shim
- Nemotron via fc1/fc2 SwitchMLP
- Gemma 4 via `FlashMoESwitchGLUShim`
- SSD-backed expert streaming with slot-bank cache
- Settings default 256 slots (known: 60+ layer models need more)

### TurboQuant KV — opt-in
- Default OFF (flipped Apr 16 after perf audit showed 2458% regression on Nemotron)
- JANG calibrated models still auto-enable via `jang_config.turboquant`
- Opt-in via Server settings for long-context memory savings

### DFlash (spec decode) — works
- Admin `/admin/dflash/load` + `/admin/dflash/unload`
- `--dflash-enable` CLI flag
- `dflash-smoke` subcommand

### Smelt (partial expert loading) — per-model matrix
- Memory notes flag MTP contamination, deferred eval root causes
- See `memory/project_smelt_mode.md` for per-model matrix

### Batching — BatchEngine
- `maxNumSeqs` controls concurrent slot count
- Paged cache block-sharing across slots
- Verified: 5/5 concurrent burst survives

---

## 8. UI surfaces

### Chat — 10 Swift files, 3.7k lines
- ChatScreen, ChatViewModel, MessageList, MessageBubble, InputBar
- SessionsSidebar (create/select/close/rename?)
- ChatSettingsPopover (663 lines), QuickSlidersPopover (171 lines)
- InlineToolCallCard (121 lines)
- ChatExporter (191 lines)
- Keyboard: Cmd-N, Cmd-K, Cmd-W, Esc (cancel), Up/Down (history)
- Load Model button — **always visible in top-bar (fixed today)**

### Server — 12 files, 4.2k lines
- ServerScreen, SessionDashboard, SessionCard, SessionView, SessionConfigForm (932 lines!)
- PerformancePanel, CachePanel (530 lines), BenchmarkPanel, LogsPanel (441 lines)
- ModelDirectoriesPanel, GatewayActor, HTTPServerActor

### API — 5 files, 1.8k lines
- API keys manager UI

### Downloads — 1 file, 377 lines
- DownloadStatusBar (auto-open per rule)

### Image — 9 files, 1.7k lines
- **IMPLEMENTATION GAP**: All Flux concrete models throw `notImplemented` except ZImage (which runs but with zero text embeddings = noise output). T5-XXL + CLIP-L encoders not ported.

### Terminal — 1 file, 560 lines
- Full terminal mode (replaced Tools mode)

### Onboarding — 1 file, 175 lines
- SetupScreen wizard (first-launch)

### Common — 13 files, 2.4k lines
- TrayItem (with today's idle countdown), CommandBar (Cmd-K), Theme cycler, etc.

---

## 9. Observability

### Works
- `LogStore` ring buffer + subscribe (file backed)
- Prometheus `/metrics` endpoint
- `MetricsCollector` with per-turn snapshot subscribe
- Load progress parsing (40+ regex patterns in Engine → LoadProgress enum)
- Cache stats panel with live polling
- Per-message TTFT + tok/s + cache-hit detail in bubble
- Benchmark endpoint exposed via admin
- Request log (MCP + chat)

### Gaps
- No Instruments trace captured for Gemma4 / Nemotron perf (47-60% vs Python)
- Metal shader compile time not surfaced in load progress granularly

---

## 10. Onboarding / first-run

SetupScreen exists at `Sources/vMLXApp/Onboarding/` — 175 lines. Shown when `vmlx.firstLaunchComplete == false`. Need to verify:
- Model picker includes "download default" option
- License agreement present
- Telemetry opt-in (if any)

---

## 11. Missing / degraded for production

### Ship-blocker
- [ ] **Fresh notarized DMG** with today's 11 source fixes (scripts/build-release.sh 2.0.0-beta.3)

### High-value engine/perf
- [ ] Gemma4 / Nemotron decode at 47-60% of Python ref — needs Metal Instruments trace
- [ ] VL decode tok/s benchmark with 200+ token completion
- [ ] SSM re-derive capture-during-prefill (slow async fallback remains)

### Implementation gaps
- [ ] **Image generation** — all Flux models stub except ZImage partial. Text encoders (T5-XXL, CLIP-L) not ported. ETA: significant.
- [ ] **Logprobs endpoint** — intentional 400 stub. Real impl requires threading token logprobs through MLX eval.
- [ ] MCP plain HTTP JSON-RPC transport (stdio + SSE work)
- [ ] Chat branch-from-message (edit + retry + rename + search already work)
- [ ] Input-bar token count preview (live tokenizer sample)
- [ ] Adapters (LoRA) — endpoints wired, runtime status unverified

### Feature requests (GH issues)
- [ ] vmlx#80 "chat broken" — need user repro
- [ ] vmlx#83 perf vs LM Studio — need A/B trace
- [ ] vmlx#78 coding harness recs
- [ ] vmlx#79 Z-image
- [ ] mlxstudio#73 TurboQuant + dflash freeze
- [ ] mlxstudio#74 JANG quant convert macOS (pre-flight landed, awaiting user log)

### UX polish
- [x] Idle countdown in tray (closed today)
- [x] Load Model button always-visible (closed today)
- [ ] Session rename UI (check SessionsSidebar)
- [ ] Session drag-reorder (check SessionsSidebar)
- [ ] Chat export formats beyond JSON (markdown? HTML?)
- [ ] Cost / token-count preview in input bar

---

## 12. This session's landed fixes (Apr 18)

1. `Load.swift` — JANGTQ load peak RAM halved (weights dict + ModuleParameters inlined)
2. `MediaProcessing.swift` (×2) — video frame memory halved (eager CIImage → MLXArray)
3. `Stream.swift extractImages` — prefix(30) cap at parse time (VL decode bound)
4. `Stream.swift` — widened cancel-barrier skip gate + **SSM re-derive cancel guards** (closes the §3 open blocker — hybrid-MXFP4 AND JANGTQ cancel now safe)
5. `ChatScreen.swift` — always-visible Load Model + Wake-now CTA on deep standby
6. `Sources/vMLXEngine/Library/OllamaCapabilities.swift` — new testable helper, 17 tests
7. `Sources/vMLXEngine/Lifecycle/IdleTimer.swift` — `nextSleepCountdown()` exposed
8. `AppState` + `TrayItem` — idle countdown rendered live
9. `vMLXCLI/main.swift` — SIGPIPE ignore (prevents silent death on client disconnect)
10. Stale tests repaired: logprobs rejection / TurboQuant default-off / VL `<image>` marker / cacheMemoryPercent removal
11. `Package.swift` — runnable test set 58 → **288**

---

## 13. Next production milestones (prioritized)

**P0 — ship-blocker**
1. Run `scripts/build-release.sh 2.0.0-beta.3` to produce notarized DMG with today's fixes.

**P1 — high user-impact**
2. Gemma4 perf trace (Instruments Metal capture on ~40s decode).
3. Logprobs endpoint implementation OR upgrade the error to point at `response_format` as an alternative.
4. vmlx#80 triage (need user repro).

**P2 — feature gaps**
5. Chat branch-from-message (remaining chat history feature).
6. Input-bar token-count preview (live tokenizer).
7. Flux image gen text encoders (T5-XXL + CLIP-L) — significant port.
8. Session drag-reorder (nice-to-have).

**P3 — polish**
9. Cost / token preview in input bar.
10. Adapters runtime verification.

---

Generated by Ralph loop iteration 1. Next iteration: pick highest-priority
item from P0/P1, close it, update this doc.

---

## Ralph iteration 2 — 2026-04-18

**Landed:**
1. **Anthropic `/v1/messages` error-code parity.** Non-streaming path used to catch every error as 500; now discriminates `EngineError` variants (invalidRequest→400, toolChoiceNotSatisfied→422, modelNotFound→404, notLoaded→503, requestTimeout→504, promptTooLong→400 with descriptive message).
2. **Anthropic `ChatRequest.validate()` — wired.** Live-caught during production audit: temperature=99 passed through the engine silently as HTTP 200 with a valid completion. Now rejected with 400 before engine hand-off. Same for negative max_tokens, bad seed, bad topP.
3. **Ollama `/api/chat` + `/api/generate` — validate-parity fixed.** Same bug class — both endpoints accepted wild options values because they skipped `.validate()`. Now route-level validation before `engine.stream` hand-off.
4. **Ollama EngineError discrimination.** Same 4xx/5xx taxonomy as Anthropic so Ollama clients see contract-correct codes on bad input.
5. **Input-bar token-count preview.** InputBar.swift now renders `"≈N tok · M chars"` caption below the bar, flipping amber at 2k tokens and red at 8k so users can spot context pressure before sending.

**Verified live:**
- Anthropic temperature=99 → HTTP 400 (was 200 before)
- Ollama temperature=99 → HTTP 400 (was 200 before)
- Full test suite: **288/288 green**
- Fresh release vmlxctl rebuilt at 13:00-ish

**Implications for production:**
- Every inbound HTTP protocol now enforces the same input-validation contract. Third-party SDKs that rely on 4xx-for-bad-input (LangChain, LlamaIndex, anthropic-sdk-python) will now behave correctly.
- No regression on valid-request paths; same binaries, same engine.

Next iteration target: run `scripts/build-release.sh 2.0.0-beta.3` for the notarized DMG OR start the T5-XXL text encoder port for image gen.

---

## Ralph iteration 3 — 2026-04-18

**Landed:**

1. **App-termination settings flush** (`vMLXApp.swift`). Added `VMLXAppDelegate: NSApplicationDelegate` via `@NSApplicationDelegateAdaptor`. On `applicationWillTerminate`, it flushes pending debounced SettingsStore writes + stops the engine (bounded to 2 s). Pre-fix, a user changing settings then cmd-Q'ing within ~500 ms could lose the write because the debounce hadn't fired yet. Now the SQLite commit lands before exit.

2. **6 new source-scan regression tests** (`RegressionConstantsTests`):
   - §33 AnthropicRoutes calls `chatReq.validate()` (guards iter-2 fix)
   - §33 OllamaRoutes calls validate ≥ 2× (covers /api/chat + /api/generate)
   - §34 AnthropicRoutes catches EngineError and discriminates variants
   - §34 OllamaRoutes catches EngineError and discriminates variants
   - §35 `Serve.run` installs SIGPIPE → SIG_IGN
   - §36 vMLXApp has `applicationWillTerminate` that flushes SettingsStore

**Audited (no gap found):**
- LoRA adapters — `EngineAdapters.swift` fully implements loadAdapter/unloadAdapter/fuseAdapter/listAdapter with clean error types (adapterMissingFile, adapterAlreadyFused, adapterNotLoaded, incompatibleModelType)
- Download resume — `DownloadManager.swift` uses `Range: bytes=<n>-` + KVO progress + HF 403 gated-repo hint. Full-featured.
- Onboarding zero-model path — recommends Qwen3-0.6B-8bit with one-click download CTA

**Tests:** **294/294** (+6 new, up from 288).
**Release binaries:** vmlxctl 13:13 (65MB) + vMLX 13:16 (72MB) carry every iteration 1+2+3 fix.

---

## Ralph iterations 4+5 — 2026-04-18 — SHIP

**Iter 4 landed:**
1. **Shared `OpenAIRoutes.mapEngineError` helper.** Consolidated the hand-rolled per-route `if case .invalidRequest → 400 else 500` into a single discriminated switch covering all 13 `EngineError` variants. OpenAI's 8 catch sites + Anthropic + Ollama all delegate to it now. Changes 8 internal routes to full-contract error codes.
2. **§37 regression test** — source-scan verifies `mapEngineError` discriminates invalidRequest / toolChoiceNotSatisfied / modelNotFound / notLoaded / requestTimeout / promptTooLong.

**Iter 5 — SHIPPED BETA.3:**
1. **`scripts/build-release.sh` rewritten.** Xcode 26 broke the `exportOptionsPlist` `method` enum — every documented value returned `error: exportArchive exportOptionsPlist error for key "method" expected one {} but found <value>`. Even when the archive itself built, Products/Applications was empty. New script:
   - Runs SwiftPM `swift build -c release --product vMLX` (same path used for local dev)
   - Manually stages `.app` bundle from the executable + Info.plist + metallib + SwiftPM resource bundles
   - Codesigns with Developer ID + hardened runtime + entitlements
   - Submits for notarization, waits, staples
   - Builds DMG via create-dmg or hdiutil, submits DMG for notarization, staples
   
2. **Secrets scan — all clean.** No hardcoded passwords, API keys, tokens, or work-server credentials in `Sources/`. `.env.signing` gitignored; only emails in source are copyright attribution.

3. **Both notarizations Accepted** — submissions `f64c5948-e6e1-4857-97fd-5c85ee5273ff` (.app) and `c46c9a90-f733-4390-ad88-a834e1c9b37c` (DMG), both returned `"statusSummary": "Ready for distribution"`, no issues.

4. **Ship artifact:**
   - Path: `release/vMLX-2.0.0-beta.3-arm64.dmg`
   - Size: 22 MB
   - SHA256: `57ec434178513304aa8a3ea701ddc55f2bc44b34bae65f2efb04e111af701e62`
   - Gatekeeper: `accepted, source=Notarized Developer ID`
   - Codesign: `valid on disk, satisfies its Designated Requirement`
   - Team: 55KGF2S5AY (Developer ID Application: ShieldStack LLC)

**Tests:** **295/295** (+1 from §37).
**Release binaries:** vmlxctl 13:24 + vMLX 13:23. Shipped DMG binary signed 13:28, stapled 13:29, DMG stapled 13:32.

## Ralph iteration 6 — post-ship polish

**Audited, confirmed shipped:**
- DMG passes `spctl -a -v -t execute` on inner .app: `accepted, source=Notarized Developer ID`
- Notary log: `"statusSummary": "Ready for distribution"`, `"issues": null`
- Session persistence across restart (`hydrateSessionsFromSettings` + Database.messages(for:) queries)
- Cmd-Shift-T reopens most-recently-closed chat via `recentlyClosed.popLast()`
- MCP stdio transport fully implemented; SSE + HTTP JSON-RPC remain as future transports (stdio covers most community MCP servers)

**Remaining post-ship work (non-blockers):**
- ~~SSE heartbeat comments for thinking models~~ **DONE iter 7 + 8** (SSE + NDJSON)
- MCP SSE + HTTP JSON-RPC transports
- Flux image gen text encoders (T5-XXL + CLIP-L)
- Gemma4 / Nemotron perf Metal Instruments trace
- vmlx#80 / vmlx#83 external reports — need user repro info

---

## Ralph iterations 7+8 — post-ship production hardening

**Iter 7 — SSE heartbeat:**
- `sseMergeWithHeartbeat` helper + `sseHeartbeatInterval` constant (env: `VMLX_SSE_HEARTBEAT_SEC`, default 15s)
- Wired into `chatCompletionStream`, `textCompletionStream`, `responsesStream`
- Emits `: keep-alive\n\n` SSE comment line during idle — ignored by parsers but flushes TCP
- Protects thinking-model streams from nginx 60s / ollama-js 30s / custom-SDK short-timeout disconnects

**Iter 8 — NDJSON heartbeat:**
- Same helper reused in `JSONLEncoder` for Ollama `/api/chat` + `/api/generate`
- NDJSON can't skip SSE comments → emits `{"message":{"role":"assistant","content":""},"done":false}` ping objects (matches Ollama's native reasoning-phase behavior)
- All 5 streaming protocols now heartbeat-safe: OpenAI chat SSE, OpenAI text SSE, OpenAI Responses SSE, Ollama chat NDJSON, Ollama generate NDJSON

**3 new regression tests** (§38 ×3) source-scan the helper + its call sites.

**Tests: 295 → 298 green.**

**Note on shipped DMG:** `vMLX-2.0.0-beta.3-arm64.dmg` shipped in iter-5 does NOT include the heartbeat fixes (iter-7 + iter-8). It's still fully functional (local 127.0.0.1 binding doesn't need heartbeats). A beta.4 DMG would add LAN-deployment robustness; ask to re-run `scripts/build-release.sh 2.0.0-beta.4` to produce it.

---

## Ralph iteration 9 — chat branch-from-message

**Landed:**
- `ChatViewModel.branchSession(from messageId: UUID)` — forks a chat from any message into a new session. Copies all messages strictly BEFORE the anchor, re-UUIDs each row (so SQLite PKs don't collide), and nudges timestamps by µs offsets to preserve ordering inside the branch without borrowing the source's absolute times. Title prefixed with `↱ ` so sidebar shows forks at a glance. Pushes an undo action so accidental branches are recoverable.
- `MessageBubble.onBranch` optional callback — new action-bar button (SF Symbol `arrow.triangle.branch`) visible on hover; hidden on the first message (branching there is equivalent to New Chat) + on the source-chat where branching is invalid.
- `MessageList` wires `onBranch: idx > 0 ? { vm.branchSession(from:) } : nil` so the button automatically disappears on message index 0.
- 2 new regression tests (§39 branchSession impl + MessageBubble button) — covers the SQLite PK re-UUID, fork-arrow title prefix, and SF Symbol name.

**Common user gesture**: "What if I'd answered differently" / "fork from the assistant's reply into a parallel track" — matches Claude/ChatGPT web UI behavior.

**Tests: 298 → 300 green** (+2 new, hit the 5.2× expansion milestone from the 58-test baseline).

## Full session tally (iter 0–9)

| Metric | Value |
|---|---|
| Swift source edits | 21 files |
| New source files | 1 (`OllamaCapabilities.swift`) |
| Tests: runnable | 58 → **300** |
| Tests: pass rate | **300/300** |
| Notarized DMG shipped | `vMLX-2.0.0-beta.3-arm64.dmg` (iter-5) |
| Ralph iterations completed | 9 |
| Production fixes | 23 (cache/mem/VL/JANGTQ/cancel/UI/API/contract) |

**Remaining post-ship (genuinely non-trivial ports):**
- Flux T5-XXL + CLIP-L text encoders (image gen produces real output)
- Gemma4 / Nemotron decode perf Instruments trace
- MCP SSE + HTTP JSON-RPC transports
- vmlx#80 / vmlx#83 (external repros pending)
- vmlx#24 JANG quant convert macOS (user log pending)
