---
active: true
iteration: 34
session_id: 
max_iterations: 0
completion_promise: null
started_at: "2026-04-20T00:02:24Z"
---

# vMLX Swift v2 Production Readiness — Self-Refining Audit

## Original user directive (iter-1)

Go out of ur way to search and think further of how UI and user usage
would be on: loading models, unloading, auto sleep, unload, wake load,
JIT, hybrid SSM, cache hitting, stats, performance, how in-app chat
works, connections to all API sides, when starting up model the server
settings offered, how it applies, making sure it saves along with chat
settings, making sure no duplicate server setting or chat or any other
buttons duplicate or missing or empty or hardcoded.

Think really deeply and be systematic in questioning. Make a list and
go down no matter how small and tiny the feature and consideration is.
Check from most important and root level stuff first and going outwards
to cover all tiny things documented. Consider edge cases + issues that
users have posted on GitHub for vmlx (python) and mlxstudio — so no
further similar errors occur. Tasks not mutually exclusive — plan and
do it all comprehensively.

## How I drive progress (self-refinement protocol)

Every iteration I:
1. Pick ONE bucket from the audit checklist below that's still `[ ]`.
2. Trace root → leaf: engine code → settings → UI → live behavior.
3. Fix or clearly label any silent-drop, duplicate, missing, or
   hardcoded control I find. Add a §N regression guard in
   `Tests/vMLXTests/RegressionConstantsTests.swift`.
4. Live-test when feasible (vmlxctl + curl or real model load).
5. Check tick box below, log the fix in
   `~/.claude/projects/-Users-eric-vmlx/memory/project_swift_iter33_iter34_app_audit.md`.
6. Append a "findings" bullet under the current bucket in THIS file.
7. Before closing the turn, update this file's `## Next focus`
   section so the next iteration lands on sharper instructions.

Rules: NEVER skip tests. NEVER ship a silent no-op — either wire it
or label it clearly. NEVER add "coming soon" without a comment in the
Swift source explaining WHY it's not wired + a §N regression guard.

## Audit checklist (root → outward)

### Lifecycle & state
- [x] Tray Start / Stop / Restart button wiring (§84 — tray Stop now
      tears down HTTP listener via stopSession).
- [x] `/health` reports real engine state (§65).
- [x] Soft vs deep sleep icon distinction in tray (§69).
- [x] SSM re-derive counter + breadcrumbs (§74, §75, §67).
- [x] JIT wake on every engine-hitting OpenAI route (§89, iter-60):
      embeddings, images/generations, images/edits, audio/transcriptions,
      audio/speech, rerank. Previously only chat routes woke.
- [x] Idle timer countdown: gate on `.running` (§90, iter-61). After
      manual softSleep, IdleTimer's `softFired` stayed false (only the
      auto-fire path flips it), so `nextSleepCountdown()` kept
      returning a remaining time while engine was in `.standby(.soft)`.
      Tray showed moon icon + "Sleeps in 3:45" simultaneously. Now
      gates in the AppState poll loop on `engine.state == .running`.
- [x] Gateway route wake coverage (§91, iter-62). Gateway
      `/v1/embeddings` had its own inline handler that skipped the
      wake call — fixed. Chat + completions route through
      `handleOpenAIChat` which already waked. No other non-chat
      gateway routes exist (v1 gateway intentionally omits Ollama/
      Anthropic per design).
- [x] Stream cancel + engine state audit (iter-62): `cancelStream()`
      only cancels Task handles + drains `streamTasksByID` map —
      never touches `EngineState`. Engine stays `.running` through
      cancel, next request works. No bug to fix.

### Settings save coherence
- [x] reasoningEffort / stopSequences / systemPrompt / maxToolIterations
      / hideToolStatus / workingDirectory wired (§59–§66).
- [x] enableSSMCompanion gate (§82), defaultReasoningParser override
      (§87), chatTemplate override actually wires (§88, iter-60).
- [x] rateLimit (§79), TLS (§80), CORS (§78) flow from settings to
      per-session server + gateway.
- [x] pagedCacheBlockSize exposed as session override (§76).
- [x] Chat/Session save-path debounce parity (§101, iter-72).
      Both setChat + setSession route through the same 500ms
      debounce, same cancel-on-reset pattern, same per-id task
      map, same flushPending drain on quit. Field-name non-overlap
      documented (iter-61). But found a real bug: `flushPending`
      directly iterated `sessionSaveTasks` + `chatSaveTasks` while
      their per-id flush functions called `removeValue(forKey:
      id)` — dictionary mutation during iteration. Risks crash on
      quit with multiple chats/sessions; in release may silently
      skip/repeat entries. Fixed: snapshot `Array(keys)` before
      iterating, look up tasks via subscript.
- [x] Chat draft stash audit (§99, iter-70). Drafts keyed by CHAT
      session UUID, not server session UUID — Engine.stop() never
      touches ChatViewModel state, so server-lifecycle can't clobber
      drafts. Real gap found elsewhere: `deleteSession` +
      `clearAllSessions` didn't drop `drafts[id]`, causing orphan
      entries (harmless cross-chat, but a slow memory leak and
      undo-restore of active chat landed on a blank composer
      despite a resurrected dict entry). Fixed: both paths now
      snapshot-then-drop, and undo closures restore both the dict
      AND live compose state on the resurrected active chat.

### Duplicate / missing / empty / hardcoded
- [x] `/Users/eric` hardcoded paths swept (§77).
- [x] Unwired Built-in Tools collapsed to Shell + Coming Soon (§70).
- [x] wireApi section removed from chat popover (§71).
- [x] Distributed / Smelt / Block disk cache / FlashMoE prefetch
      labeled coming-soon (§81, §83, §84, §86).
- [x] Duplicate-button scan (§98, iter-69). Audited every view with
      ≥3 Buttons. All label duplicates are intentional: (a) Cancel
      buttons in confirmation dialogs (destructive-action pair),
      (b) Copy buttons for distinct targets (endpoint URL vs API
      key), (c) SessionCard context-menu Stop/Wake mirroring inline
      actions for right-click discoverability, (d) SessionView
      state-branch switch rendering Cancel/Stop/Reconnect per state
      so every state has an escape hatch. §98 pins the state-branch
      pattern so a future "DRY up duplicates" cleanup can't silently
      collapse the switch and remove state-specific affordances.
- [x] Placeholder fallback for empty global defaults (§92, iter-62).
      SmeltMode / dflashDrafterPath / dflashTapLayers / distributedHost
      all defaulted to `""` globally; the TextField placeholder
      rendered as blank, looking like the input was disabled. Each
      now guards `.isEmpty ? hint : value` with an explanatory hint.

### Cache / hybrid / SSM stats
- [x] CachePanel architecture section surfaces rotating/mamba/TQ (§57).
- [x] TrayItem header pills for hybrid/SWA/TQ (§58).
- [x] reDerives counter rendered in CachePanel (§67).
- [x] CACHE-MATRIX.md doc corrected (memory default ON, paged sub-block
      limitation) (iter-47).
- [~] SSM companion `hits` counter live-verify — DEFERRED: requires
      loading a hybrid SSM model (Nemotron-H / Qwen3.5-A3B) with
      block_size ≤ prompt_len. Blocked by RAM constraint + test
      harness environment lacks MLX Metal library (all
      MLXArray-touching tests fatal-trap at MLX init). Needs a
      Metal-enabled dev machine with spare RAM.
- [~] L2 disk cache restart round-trip (iter-74 §103) — DEFERRED:
      synthetic XCTest written + reverted because the Swift test
      harness here can't init MLX Metal. Even an XCTSkipIf guard
      doesn't fire because MLX fatal-traps before the test body
      runs. The TQDiskSerializer round-trip IS covered by
      `@Test`-based TQDiskSerializerTests (testRoundTripKVCacheSimple
      / Mamba / Quantized / Hybrid / Rotating) which exercise the
      same code path but also require Metal — they're silently not
      running in this harness. Needs a Metal-available env to run.

### Model loading (UX flows)
- [x] ModelLibrary dedup by SHA-path + shard fingerprint (§67 tests).
- [x] Empty / stub-snapshot rejection (§67).
- [x] `/api/ps` uses displayName not HF snapshot hash (§64).
- [x] `vmlxctl ls` SIGSEGV fix — printf %s on Swift String (§62).
- [x] Failure modes: corrupted shard, missing tokenizer.json,
      unrecognized model_type (§95, iter-66). `ModelFactoryError` +
      `JangLoaderError` had crafted `LocalizedError.errorDescription`
      messages but NO `CustomStringConvertible` conformance.
      Engine's `fail("\(error)")` produced enum-case gore
      (`loadFailed("Shard X …")`) instead of the intended
      (`JANG load failed: Shard X …`). Fixed: both types now conform
      to `CustomStringConvertible` forwarding to `errorDescription`.
      Load.swift already has defensive guards for: HTML error-page
      shards, empty-weights bundles, JANGTQ misconfig. TokenizerLoader
      retries with patched class name on `unsupportedTokenizer`. The
      messages are now truthful end-to-end; runtime test asserts
      `"\(ModelFactoryError.unsupportedModelType(...))"` equals the
      crafted string.
- [x] "Load Model" progress banner audit (§97, iter-68). Server-tab
      `SessionLoadBar` already renders phase + NN% text + LoadingBar +
      label — no regression. TrayItem renders NN% or label. Chat-side
      `EngineStateBanner` showed only bar + label, no text %. Fixed:
      chat banner title is now `"Loading model… NN%"` when
      `loadProgress.fraction` is known, falls back to plain
      `"Loading model…"`. All three surfaces now surface the number
      explicitly.

### Chat streaming + API-parity
- [x] OpenAI / Anthropic / Ollama streaming event shapes verified
      end-to-end (iter-41).
- [x] Hybrid model SSM re-derive live-verified (iter-45: stateCount=46
      on Nemotron).
- [x] Ollama `/api/generate` timing envelope (§93, iter-63).
      total_duration / load_duration / prompt_eval_duration /
      eval_duration were missing — added mapping from ms to nanos.
      Also wired JIT wake into `ollamaEmbedHandler`.
- [x] Multi-turn VL image cache audit (§100, iter-71). Verified
      mediaSalt (FIX-G-O) propagates through every VL-capable cache
      call site: Evaluate.swift fetch+store, BatchEngine.swift
      fetch+store, Stream.swift pre-fetch probe. All three consistent.
      Also cleaned up an inert+misleading `isMLLM: isMLLM` arg in
      the pre-fetch probe — it only affected SSM companion
      boundary, but the probe's returned `matched` is from the paged
      tier (isMLLM-agnostic), so the flag did nothing. Python
      parity for the N-1 boundary doesn't apply to Swift because
      Swift has no separate MLLM batch generator split path —
      Evaluate.swift + BatchEngine.swift both use the `isMLLM:
      false` default consistently for store AND fetch.

## Known gaps (ship with caveat)

1. Nemotron-H `model.prepare` quirk: iter-45 root fix helps with chat-
   length prompts but longer prompts that cross the prefill step
   boundary have a second MLX gap where `MambaCache.state` partially
   populates. Separate investigation — don't regress the re-derive
   fix while chasing it.
2. `pagedCacheBlockSize` default=64 means sub-64-token prompts bypass
   paged entirely. Docs updated, UI override exposed (§76), but
   consider lowering default to 32 after a benchmark shows the
   shorter-block overhead is minimal.
3. `enableAutoToolChoice` is a vLLM-compat flag with no runtime gate
   in Swift. Kept for CLI compat; audit comment in SettingsTypes.
4. **iter-64 perf caveat**: single-model perf reads on Llama-1B
   (bench-direct 461 tok/s) don't generalize to mixed-model or
   concurrent-session scenarios. TQ default-on has a real compress-
   dequant cost — re-benchmark Gemma4 MoE, Nemotron-H, Qwen3.5-A3B
   side-by-side under a no-other-agents RAM budget before publishing
   tok/s numbers.
5. TQ default-on reinstated in iter-64. Iter-16 measured 25% regression
   on Nemotron with TQ on. Guards (MLA skip, KVCacheSimple-only
   compression) should protect hybrid families, but verify on isolated
   benchmark. Env `VMLX_DISABLE_TURBO_QUANT=1` gives instant A/B.

## Next focus (set at end of iter-61)

### Resolved this iter
- §90 Idle countdown phantom-label fix.
- Settings-store debounce coherence: confirmed `VMLXAppDelegate.applicationWillTerminate`
  calls `engine.settings.flushPending()` with a 2s timeout — no lost
  writes on quit.
- ChatSettings ↔ SessionSettings field overlap: no duplicate field
  names (Chat uses bare `temperature`, Session uses `defaultTemperature`).
  Resolver applies `r?.x ?? c?.x ?? s?.defaultX ?? g.defaultX`
  correctly. ChatSettings-exclusive fields (reasoningEffort,
  stopSequences, tools, toolChoice, maxToolIterations, hideToolStatus,
  workingDirectory) flow via the in-app ChatViewModel path which
  reads `chatOverrides` directly. HTTP path expects clients to send
  them in the request body. Documented as design choice.

### Resolved this iter
- Ollama streaming NDJSON encoders now emit the same timing envelope
  as the non-stream handler. Extracted `JSONLEncoder.applyOllamaTimings`
  helper; all three call sites (non-stream /api/generate, streaming
  /api/chat, streaming /api/generate) use it. **Live-verified** on
  Llama-1B: /api/chat stream total=45ms/prefill=18ms/decode=26ms,
  /api/generate stream total=55ms/prefill=4ms/decode=50ms — both
  done:true chunks carry all four nanosecond fields.

### Still open
1. ~~**Model loading failure modes.**~~ **DONE iter-66 (§95).**
   `ModelFactoryError` + `JangLoaderError` had crafted
   `LocalizedError.errorDescription` messages but were missing
   `CustomStringConvertible`, so `"\(error)"` produced enum-case
   gore. Added conformance forwarding to errorDescription. Existing
   defensive guards in Load.swift (HTML-error-page shards,
   empty-weights, JANGTQ misconfig) and TransformersTokenizerLoader
   (unsupportedTokenizer retry + Jinja render fallback) already
   handled the actual failure modes; the bug was only in the surface
   layer where messages got munged. Runtime test pins both error
   types' interpolation.
2. **SSM companion hits live-verify with block_size ≤ prompt_len.**
   Iter-46 identified that paged-cache block alignment gates SSM
   hits. Drop block_size to 32 via session override, run a 2-turn
   Nemotron thinking chat, confirm hits tick upward.
3. **L2 disk cache round-trip after process restart.** Load JANG
   model → issue prompt → kill server → restart with same model →
   issue same prompt → expect cached_tokens > 0 from disk.
4. ~~**`/v1/responses` end-to-end test.**~~ **DONE iter-67 (§96).**
   Code audit found two gaps: (a) no `chatReq.validate()` pre-flight
   — bad max_output_tokens / temperature leaked past the 400 wall,
   (b) both non-stream and streaming `response.completed` usage
   envelopes emitted bare `{input_tokens, output_tokens,
   total_tokens}` without the timing quartet. Extracted shared
   `OpenAIRoutes.responsesUsageEnvelope(_:)` used at both call
   sites; added validate() call. §96 regression guard pins helper
   shape, both call sites delegating, and validate() count ≥ 2.
5. ~~**Anthropic `/v1/messages` timing parity.**~~ **DONE iter-65 (§94).**
   Added `AnthropicRoutes.usageEnvelope(_: includeInputTokens:)` shared
   helper. Both non-stream response and streaming `message_delta` now
   emit `input_tokens`, `output_tokens`, `tokens_per_second`, `ttft_ms`,
   `prefill_ms`, `total_ms`. Anthropic spec only requires the first
   two; extras are tolerated by anthropic-sdk-python/ts which ignore
   unknown keys. §94 regression guard verifies helper + both call
   sites delegate (no inline regressions).

## Scoreboard
- 353/353 source-scan tests green, 113/113 regression guards + 15 matrix
  rows (§57–§117)
- iter-89 work:
  1. WhisperAudio.decodeData temp-file leak (§117) — **real resource
     leak**. `decodeData(_:fileExtension:)` registered its cleanup
     `defer { try? FileManager.default.removeItem(at: tmp) }` AFTER
     the `try data.write(to: tmp)` call. If the write throws (disk
     full, permission race, sandbox change, partial-write that
     succeeds-then-fails mid-way) the defer was never installed, so
     the partial `/tmp/vmlx-whisper-<UUID>.<ext>` file leaked. Under
     sustained audio-transcription traffic the temp dir would
     accumulate stale files indefinitely. Fix: move the defer
     BEFORE the write — standard Swift idiom for exception-safe
     cleanup of pre-created resources. §117 regression guard scans
     the decodeData body and asserts `defer` offset < write offset.
- iter-88 work:
  1. cancelStream snapshot-then-drain (§116) — **real concurrency
     bug**. `Engine.cancelStream()` used `for (id, task) in
     streamTasksByID { … streamTasksByID.removeValue(forKey: id) }`
     — classic mutation-during-iteration. The comment block
     literally promised "snapshot keys first so the subsequent
     removeValue doesn't mutate during iteration" but the code
     skipped the snapshot. Swift debug builds trap on "Dictionary
     was mutated while being enumerated"; release rode COW-on-
     mutate buffer swap to avoid visible crash but iterated in
     undefined state. Same pattern fixed in iter-72 (§101) for
     SettingsStore.flushPending — the fix here mirrors that one:
     snapshot Array(dict), cancel each task, then removeAll() in
     one shot. §116 regression guard pins the snapshot call + the
     removeAll form + asserts the per-iteration removeValue is
     absent.
- iter-87 work:
  1. Auth timing-oracle fix (§115) — **SECURITY FIX**. Both the user
     API-key check (BearerAuthMiddleware) and the admin-token check
     (AdminAuthMiddleware) compared the header vs the expected token
     with Swift's stdlib `==` operator on `String`. That implementation
     is variable-time: it early-exits on the first mismatching byte.
     On a vMLX server bound to 0.0.0.0 (the optional LAN toggle), any
     LAN peer can send successive auth attempts varying one byte at a
     time and recover the token byte-by-byte by measuring response
     latency — classic timing-oracle attack. Admin token gates the
     destructive surface (soft/deep sleep, cache flush, adapter fuse,
     on-disk model delete) so recovering it is worst-case a full
     engine compromise. Fix: added
     `BearerAuthMiddleware.constantTimeEquals(_:_:)` — a UTF-8 byte
     XOR-diff across the longer of the two inputs, iteration count
     pinned to max-length so length mismatch doesn't leak via loop
     count either. AdminAuth reuses the same helper. §115 regression
     guard pins both middlewares' call sites + asserts the dangerous
     `==` forms are absent.
- iter-86 work:
  1. GatewayActor model-resolve tolerant-match bug (§114) — **real
     production bug**. The tolerant match previously accepted
     `k.hasSuffix("/" + model) || k.hasSuffix(model)`. The second
     clause allowed ANY suffix to match, so a client request of
     `model: "4bit"` collided with `mlx-community/gemma-4-e2b-it-4bit`
     (suffix matches). Under parallel load with multiple engines
     registered, `resolve()` returned whichever one was visited first
     in the dictionary iteration order — non-deterministic dispatch
     to the wrong engine. Fix: drop the unbounded `hasSuffix(model)`
     fallback, keep only the `"/" + model` boundary form, so bare-
     repo name `gemma-4-e2b-it-4bit` still matches `mlx-community/
     gemma-4-e2b-it-4bit` but a substring like `4bit` or `-4bit`
     never does. Verified locally — §114 regression guard pins the
     boundary form and asserts the dangerous fallback is absent.
- iter-85 work:
  1. Unload memory reclamation audit (§113) — **REAL USER-VISIBLE FIX**.
     `Engine.stop()` and `Engine.deepSleep()` dropped their model
     references to nil (`loaded = nil` + `cacheCoordinator = nil`)
     but never told MLX to flush its pooled Metal buffer cache. When
     the Swift references drop to zero ARC, MLX retains the buffers
     internally in its allocator until the next allocation pressure
     forces a flush — so a user who deep-sleeps a 30GB Nemotron
     expecting their RAM back watched Activity Monitor continue to
     show the process holding all 30GB for minutes or until the next
     model load event. Cross-checked against
     `Sources/MLX/Memory.swift:356` — `Memory.clearCache()` is the
     public API for "cause all cached buffers to be deallocated" and
     is thread-safe via the shared evalLock. Safe to call after
     nil-ing because it only drops buffers that aren't currently
     referenced, and both paths have already cancelled in-flight
     streams (stop) or cleared the CacheCoordinator (deepSleep)
     before reaching the call. Added the call to both paths with a
     doc comment explaining why the RSS drop behavior matters; §113
     regression guard pins both call sites so a future refactor
     can't silently drop either one.
- iter-84 work (pushed to origin/dev as 71c5b1f):
  1. Swift 6 actor-isolation cleanup (§112). `Engine` is an actor,
     so every stored property (`settings: SettingsStore`) was
     actor-isolated by default — UI reads of `app.engine.settings`
     emitted "actor-isolated property 'settings' cannot be accessed
     from outside of the actor" warnings (slated to become errors
     in Swift 6 language mode). SettingsStore is itself an actor,
     so the reference doesn't need engine-level serialization —
     marked `nonisolated public let`. Also dropped several
     redundant `await`s in `Task { }` closures inside actor methods
     (Task inherits the enclosing actor's isolation in Swift 6, so
     sync-flush helpers don't need an actor hop). And fully qualified
     `ChatRequest.ToolChoice.none` at AnthropicRoutes.swift:634 to
     silence the Optional-enum ambiguity warning. Release build now
     down to 1 warning (upstream MLX/IO.swift void-return).
- iter-83 work (pushed as d219861):
  1. UpdateService auto-updater audit (§111) — **DEFENSIVE FIX**.
     `UpdateInfo.htmlURL` came from the GitHub API's `html_url`
     field and was passed to `NSWorkspace.shared.open(url)` on
     the Download-button click without any scheme/host
     validation. `NSWorkspace.open` is scheme-agnostic — a
     compromised or MITM'd API response could set `html_url` to
     `javascript:`, `file:///...`, `data:`, or a typosquatted
     phishing URL (`github.com.evil.com`) and the call would
     dispatch it verbatim. Added
     `UpdateAvailableService.validatedReleaseURL(_:)` that
     requires `https://` scheme + host in allowlist
     (`github.com` / `api.github.com`, case-insensitive).
     Invalid URLs fall back to a hardcoded safe URL. Added 2
     behavioral tests (accepts 4 legitimate URLs / rejects 12
     dangerous patterns incl. typosquat, non-https, JS/file/
     data/custom schemes) in UpdateServiceTests — but that
     file can't be wired into Package.swift due to the same
     atomics-transitivity build issue that blocks
     MultipartFormParserTests (vMLXApp transitively links
     vMLXServer → atomics). Source-scan guard in
     RegressionConstantsTests pins the wiring and the four
     validator requirements (https, host allowlist, two hostnames).
- iter-82 work:
  1. DownloadManager audit (§110) — **DEFENSIVE FIX**. The
     `sib.rfilename` from the HuggingFace API response flowed
     into `destDir.appendingPathComponent(rfilename)` with only
     an extension allow-list. A compromised HF API, MITM, or
     malicious mirror could return `rfilename:
     "../../../.ssh/authorized_keys"` — the extension check
     passed (still ends in plausible suffixes if wrapped), and
     POSIX path resolution would collapse the `..` segments to
     write OUTSIDE the cache dir. Added
     `DownloadManager.isSafeFilename(_:)` that rejects: absolute
     POSIX paths, UNC-style Windows paths, any `..` segment
     (split on both `/` and `\`), empty/whitespace strings, and
     null-byte injection. `fetchSiblings` filter calls it BEFORE
     the extension check so traversals never reach the download
     path. Added 2 behavioral tests in DownloadManagerTests (12
     rejection patterns + 8 legitimate filenames) + source-scan
     guard in RegressionConstantsTests.
- iter-81 work:
  1. BashTool shell-exec safety (§109) — two real safety fixes.
     **(a) Unbounded output**: `readDataToEndOfFile()` on stdout +
     stderr blocked until the process closed the pipe. A command
     producing gigabytes (`cat /dev/urandom`, `yes`, `dd if=/dev/zero`)
     would allocate unbounded memory in the engine actor. A
     malicious model (prompt-injection via web content) could OOM
     vMLX via a single bash tool call. Fix: `readCapped(_:cap:)`
     chunked reader enforces 1 MiB/stream, drops excess, appends
     a truncation marker so the caller can detect the bound.
     **(b) SIGTERM-only kill**: `process.terminate()` sends SIGTERM
     but `trap '' TERM; sleep 99999` ignores it, and the pipe
     reader below blocks until pipe close — which never happens
     while the process is alive. BashTool.run would lock up
     forever on timeout. Fix: SIGKILL escalation via
     `Darwin.kill(pid, SIGKILL)` after 2-second grace period.
     §109 regression guard pins both the capped readers and the
     SIGKILL fallback.
- iter-80 work:
  1. Audio transcription multipart audit (§108). The route
     extracts a file extension from the multipart `filename`
     (or the JSON `file_extension` field) and passes it to
     `WhisperAudio.decodeData`, which joins it into a temp
     file path via `URL.appendingPathComponent`. Forward
     slashes in the string would produce non-well-formed
     paths (`/tmp/vmlx-whisper-UUID.bar/qux`). Not actively
     exploitable today — `Data.write(to:)` fails if the
     intermediate directory doesn't exist — but a race
     against another process that creates the expected
     subdirectory could misdirect writes. Added
     `OpenAIRoutes.sanitizeFileExt(_:)` helper that enforces
     ≤8 ASCII alphanumeric chars, lowercased, with nil return
     on anything else. Both call sites (multipart + JSON body)
     now route through the sanitizer; rejected inputs fall
     back to the default "wav". §108 source-scan guard pins
     both call sites + helper visibility. Behavioral test of
     the sanitizer lives in MultipartFormParserTests (not
     wired into Package.swift due to atomics-transitivity
     build issue with vMLXServer tests, but kept for future
     harness enablement).
- iter-79 work:
  1. RequestLogger sensitive-data audit (§107). Confirmed the
     middleware only reads method/path/status/elapsed — no
     auth headers, no request/response bodies, no query
     strings. Swept all `engine.logs.append` / `engine.log`
     call sites — the two `\(error)` interpolations
     (Stream.swift:107 + SSMReDerive:113) carry engine-internal
     errors, not user request data. LogStore is bounded at
     5000-line ring buffer so log-pump adversaries can't
     balloon memory. Added §107 regression guard: scans
     RequestLogger.swift for forbidden reader patterns
     (`request.headers[.authorization]`, `request.body`,
     `response.body`, `request.uri.query`) and asserts they're
     absent; also asserts the positive fields
     (`request.method`, `request.uri.path`) are present.
- iter-78 work:
  1. MCP routes audit trail (§106). HTTP access logs captured
     "POST /v1/mcp/execute 200" but not WHICH tool was invoked
     — the namespaced name lives in the request body,
     invisible to access-level logging. `/mcp/:server/**` had
     the method in the URL path (visible in HTTP access logs)
     but no engine-side log on error. Added
     `engine.log(.info/.warn/.error, "mcp", ...)` on entry +
     success/fail for both routes. Arguments/params deliberately
     NOT logged — they may carry user prompts, file contents,
     credentials. Broader audit: MCP routes correctly validate
     serverName against config.servers (can't inject new
     servers via HTTP); name/serverName/method all empty-
     checked; bearer auth covers MCP routes (admin auth does
     not — MCP tools are user-opt-in via mcp.json so the user's
     API key = user's tools policy is acceptable).
- iter-77 work:
  1. RateLimitMiddleware Swift 6 concurrency fix (§105). The
     middleware's `handle(_:context:next:)` is an async method
     but it called `NSLock.lock()` / `unlock()` on a stored
     state struct's lock directly. Swift 6 marks those APIs
     unavailable from async contexts because a lock acquisition
     that's interrupted by an actor hop before the paired
     unlock violates the sync-mutex contract. SourceKit had
     been warning about this for several iterations (the
     repeated "'lock' is unavailable from asynchronous
     contexts" flag). Fixed by migrating to
     `OSAllocatedUnfairLock<[String: [Date]]>` with `.withLock`
     scoped access — the closure body runs synchronously and
     the lock is released before any suspension point can
     happen. Functional behavior unchanged (same sliding-window
     logic, 60s window, 429 response). Added `import os` +
     one regression guard pinning the withLock pattern.
- iter-76 work:
  1. Bearer auth audit (§104). Three improvements:
     (a) `/health` (GET-only, exact path) exempt from bearer
         gate. External monitoring probes (uptime, K8s liveness,
         tray pulse) no longer need the API key to verify
         liveness. Only returns `{state, model_name}`, no
         sensitive material.
     (b) Accepts `x-api-key: <key>` as equivalent to
         `Authorization: Bearer <key>`. Anthropic SDK's default
         header convention now works against our /v1/messages
         endpoint. Previously an Anthropic-SDK client got 401
         even with a valid key.
     (c) CORSMiddleware allowHeaders on both Server +
         GatewayServer now include `x-api-key` + `x-admin-token`
         so browser-hosted clients can send them (preflight
         OPTIONS would otherwise reject custom headers). 3
         regression guards.
- iter-75 work:
  1. Admin auth gate audit (§103) — **SECURITY FIX**. The previous
     middleware only gated `/admin/*` and `/v1/cache/*`. Four
     destructive routes outside those prefixes bypassed the gate:
     - POST /v1/adapters/load (arbitrary-path LoRA load)
     - POST /v1/adapters/unload
     - POST /v1/adapters/fuse (**permanent** weight fusion)
     - DELETE /api/delete (**permanent** on-disk model wipe)
     A LAN peer with just the user API key could fuse a rogue
     adapter into base weights or delete models from disk. Fixed
     by extending `AdminAuthMiddleware.handle` with explicit
     match clauses. The safe read-path `GET /v1/adapters` stays
     open. Also corrected a stale comment at
     AdminRoutes.swift:176-177 that falsely claimed
     `/v1/cache/clear` bypasses admin auth. 2 regression guards
     pin both the route list and the corrected comment.
- iter-74 work:
  1. Attempted synthetic L2 disk cache restart round-trip test.
     Discovery: the current test harness fatal-traps at MLX Metal
     init ("Failed to load the default metallib"), which means any
     test touching MLXArray (including existing `@Test`-based
     TQDiskSerializerTests) cannot run here — those existing tests
     are silently dead in this environment. My XCTest guard
     couldn't soft-skip because MLX crashes before the test body
     runs. Reverted the test file; documented both remaining
     open items as DEFERRED blocked-on-env (need Metal-available
     dev machine + isolated RAM to actually exercise). The disk
     serializer code path IS correct (code audit confirmed
     mediaSalt + isMLLM + modelKey isolation in iter-71) — we just
     can't run the round-trip end-to-end under this harness.
- iter-73 work:
  1. Confirmed user concern ("TextToolTokenLoopHandler only wires
     ToolCallProcessor, no ReasoningParser → `<think>...</think>`
     leaks as raw .chunk") is NOT a bug in our Swift path: Stream.swift
     receives `.chunk(text)` and pipes through
     `reasoningParser.extractReasoningStreaming` at line 1204 BEFORE
     emitting to user. Reasoning is properly split downstream.
  2. 15-row matrix test suite `ToolCallReasoningMatrixTests` exercises
     real multi-factor combinations: (Qwen3 / Mistral / DeepSeek-R1 /
     no-parser reasoning) × (JSON / XML-function / Mistral inline /
     none tool format) × (thinking on / off) × (single / multi tool
     call) × (multi-chunk streaming cadence) × (reasoning
     rerouting under §15).
  3. **Real bug found + fixed in ToolCallProcessor**: when a single
     chunk contained `"text<tool_call>...</tool_call>"`, the
     collectingToolCall branch silently dropped the `leadingToken`
     prefix that was split off during `.normal → .potentialToolCall`
     transition. Added `lead + trailing` concat to preserve. Real in
     non-streaming replay + BPE batched-token deliveries.
- iter-72 work:
  1. SettingsStore debounce parity + snapshot-safe flush (§101).
     Confirmed Chat/Session save paths share the same 500ms
     debounce + cancel-on-reset + per-id Task tracking +
     flushPending-on-quit. Real bug found: flushPending
     iterated `sessionSaveTasks` / `chatSaveTasks` directly while
     their flush callbacks called `removeValue(forKey: id)` —
     dictionary mutation during iteration. Swift crashes in
     debug with "Dictionary was mutated while being enumerated";
     release may silently skip/repeat. Fixed: snapshot
     `Array(dict.keys)` before iterating, subscript-lookup per
     iteration. Real risk for 10-chat users quitting the app.
- iter-71 work:
  1. Multi-turn VL image cache audit (§100). Verified mediaSalt
     flows through every VL-capable cache site (Evaluate,
     BatchEngine, Stream pre-fetch) — fetch/store parity is
     complete. Cleaned up an inert `isMLLM: isMLLM` flag in
     Stream.swift's pre-fetch probe that only affected SSM
     companion boundary (not exposed to the probe's returned
     `matched` count) and misleadingly implied Swift has a
     separate MLLM batch generator split path. Added 2 guards:
     mediaSalt-present at all three sites + Stream pre-fetch
     doesn't re-add isMLLM.
- iter-70 work:
  1. Chat draft stash audit (§99). Chat drafts are keyed by chat
     session UUID; server-session lifecycle (Engine.stop / wake)
     cannot reach the stash. Real gap: `deleteSession` +
     `clearAllSessions` dropped the chat from SQLite but left
     `drafts[id]` as an orphan — slow memory leak for users who
     repeatedly delete drafts-in-progress chats, and undo-restore
     landed on a blank composer despite the dict entry surviving.
     Fix: snapshot-then-drop on both paths, undo closures re-seat
     the draft dict AND restore live `inputText`/`pendingImages`/
     `pendingVideos` when the resurrected chat becomes active.
- iter-69 work:
  1. Duplicate-button scan (§98). Zero accidental duplications
     found across all Chat/Server/API/Image/Terminal views. All
     label repeats are intentional patterns: confirmation-dialog
     Cancel pairs, distinct-target Copy buttons, context-menu +
     inline action mirrors (SessionCard Stop/Wake), state-branch
     switch affordances (SessionView Cancel/Stop/Reconnect per
     engine state). Added 2 guards pinning the state-branch and
     context-menu-mirror patterns so a future refactor can't
     silently collapse them.
- iter-68 work:
  1. Load-Model progress banner parity (§97). Chat-side
     `EngineStateBanner` was the only one of three load-progress
     surfaces (Server SessionLoadBar, TrayItem, Chat banner) that
     rendered visual bar + label but NO text percent. Fixed:
     chat banner title is now `"Loading model… NN%"` when
     `loadProgress.fraction` is known, plain `"Loading model…"`
     as fallback. Audit confirmed Server + Tray already emit NN%.
- iter-67 work:
  1. OpenAI Responses API parity (§96). Extracted
     `OpenAIRoutes.responsesUsageEnvelope(_:)` shared helper;
     non-stream `/v1/responses` + streaming `response.completed`
     event both delegate. Timings (`tokens_per_second`, `ttft_ms`,
     `prefill_ms`, `total_ms`) now land on every Responses surface,
     matching chat/completions (§64), Ollama (§93), Anthropic (§94).
     Added missing `chatReq.validate()` pre-flight — was the last
     chat-shape route skipping it, silently returning 200 + partial
     stream on bad temperature / max_output_tokens.
- iter-66 work:
  1. Model-load failure-mode audit (§95). `ModelFactoryError` +
     `JangLoaderError` now conform to `CustomStringConvertible` so
     `"\(error)"` interpolation in Engine's fail path + UI banner
     emits the crafted `errorDescription` string rather than raw
     enum-case gore. Existing Load.swift guards for HTML error-page
     shards + empty-weight bundles + JANGTQ misconfig now surface to
     users as actionable messages; TokenizerLoader's class-name
     retry + Jinja-rendering fallback already handled the other
     failure modes. Two runtime tests (not source-scan) pin the
     interpolation so a future accidental LocalizedError-only revert
     fails the test suite.
- iter-65 work:
  1. Anthropic `/v1/messages` timing parity (§94). Shared helper
     `AnthropicRoutes.usageEnvelope(_:includeInputTokens:)` replaces
     two inline `{input_tokens, output_tokens}` dicts. Non-stream
     response + streaming `message_delta` both carry
     `tokens_per_second / ttft_ms / prefill_ms / total_ms` when
     engine reports them — matches OpenAI (§64) and Ollama (§93).
     Streaming final event now also emits real `input_tokens`
     instead of the stub `0` from `message_start`.

- iter-64 work:
  1. `JSONLEncoder.applyOllamaTimings` shared helper unified non-
     stream + streaming timing envelopes. Live-verified with Llama-1B
     (streaming /api/chat done chunk carries total=45ms,
     prefill=18ms, decode=26ms).
  2. TurboQuant KV cache default flipped back to **ON** per user
     directive. `TurboQuantDefaultTests` updated. MLA + hybrid-SSM
     guards in `maybeQuantizeKVCache` unchanged so those families
     still bypass compression.
  3. OpenAI non-stream `/v1/chat/completions` usage envelope now
     includes `tokens_per_second`, `ttft_ms`, `prefill_ms`, `total_ms`
     for parity with the SSE streaming path.
  4. Per-token metrics actor-hop cut 8× via batching — accumulate
     elapsed ms + count, flush every 8 tokens or at end-of-stream.
     Tray 1 Hz TPS display stays accurate (~12 batches/s at 100
     tok/s) while reducing scheduler contention.
  5. **Perf caveat** (user reminder): do NOT assume current tok/s
     generalizes — other agents are holding RAM, measurements must
     be redone isolated; concurrent-session loads will differ.
