---
active: true
iteration: 53
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
- 369/369 source-scan tests green, 129/129 regression guards + 15 matrix
  rows (§57–§133)
- iter-107 work:
  1. Rerank cosine scores silent zero (§133) — **same root-cause
     class as §132**, different code path. Engine.rerank reuses
     embeddings() to batch query+doc vectors, then cosine-similarity
     pairs. The inner vec() helper did
     `row["embedding"] as? [Double]`. Engine stores [Float]; cast
     returned nil; vec() returned []; cosineSimilarity's empty-
     vector guard returned 0; every score was 0; `scored.sort` on
     identical scores is a stable no-op, so users got documents
     back in INPUT order. Usually looked "reasonable" in short-list
     dev tests, masking the bug in production. Fix: cast to [Float]
     (the real type), map to Double for the math. §133 pins the
     [Float] cast + the Double upcast + absence of the old buggy
     line. Build clean, guard green.
- iter-106 work:
  1. Embeddings API audit (§132) — three real bugs. (a) Ollama
     `/api/embeddings` + `/api/embed` cast `$0["embedding"] as?
     [Double]` on the engine dict. Engine stores vectors as [Float];
     Swift array casts don't bridge across Float/Double, so compactMap
     ALWAYS returned []. Every Ollama embed call silently shipped
     empty vectors — LangChain/llama-index/Copilot/Open-WebUI affected.
     Fixed: cast to [Float]. (b) Engine.embeddings had no empty-input
     guard; `inputs = []` crashed at `MLX.stacked([])`. Now throws
     400 like OpenAI. (c) OpenAI Python SDK ≥1.47 defaults
     `encoding_format="base64"` and decodes into numpy float32 on
     receive; we were silently ignoring the field, SDK clients
     crashed on parse. Now honored, default stays "float" for
     curl/JSON. Build clean, §132 guard green.
- iter-105 work:
  1. DiskCache corrupt-entry DB row orphan (§131) — **real budget
     accounting bug**. `DiskCache.fetch()` catches deserialize
     errors (corrupt safetensors files from partial write / disk
     full / format version drift), logs to stderr, and removes
     the file on disk. But previously it did NOT drop the SQLite
     row. `evictIfNeeded` calculates total-cache-size with
     `SELECT COALESCE(SUM(file_size), 0) FROM cache_entries` —
     so an orphan row claiming 50MB for a missing file inflated
     the computed total and caused premature eviction of
     healthy entries. The orphan row was functionally dead
     (fetch short-circuits on `fileExists` before deserialize)
     but budget-inflating until true-LRU eventually touched it
     (could be many turns for a cold entry). Fix: new private
     `deleteRow(hash:)` helper under the shared SQLite lock.
     fetch's corrupt-file branch calls it in the same
     `lock.withLock { misses += 1; deleteRow(hash: hash) }`
     scope. §131 regression guard pins the helper + the fetch
     call-site + the parameterized DELETE.
- iter-104 work (no-code audit):
  1. **SQL safety sweep** — no user-data SQL interpolation. All
     `sqlite3_exec` call sites pass schema / PRAGMA strings only;
     every CRUD path uses `sqlite3_prepare_v2` + `sqlite3_bind_*`
     parameterized queries. Two table-name-interpolating private
     helpers in SettingsDB.swift (`deleteRow` + read helper) are
     called with hardcoded table names only — no user reach.
     Checked: Database.swift, ImageHistoryStore.swift,
     BenchmarkPanel.swift, ModelLibraryDB.swift, SettingsDB.swift,
     DiskCache.swift. No fix needed.
  2. **Env-var parsing sweep** — all `VMLX_*` env reads are
     range-checked or limited to "1"-vs-other boolean semantics.
     `VMLX_SSE_HEARTBEAT_SEC` has `v >= 0` check (0 disables,
     positive enables, no upper bound but unbounded values are
     functionally-equivalent to 0). `VMLX_MINIMAX_TOPK` has both
     lower (>= 1) and upper (<= numExpertsPerTok) bounds.
     Boolean-style (`VMLX_DISABLE_CACHE_COORD`, `VMLX_DISABLE_VL_RACE_BARRIER`,
     `VMLX_DISABLE_PER_TOKEN_METRICS`, `VMLX_DISABLE_SSM_RE_DERIVE_ASYNC`,
     `VMLX_VL_DEBUG`, `VMLX_COMPILED_DECODE`, `VMLX_DISABLE_TURBO_QUANT`,
     `VMLX_CAPS_LOG`, `VMLX_DISABLE_COMPILE_DECODE`, `VMLX_OLLAMA_DEBUG`)
     all use `== "1"` which is safe — any other value including
     empty, "true", "yes" is treated as off. No fix needed.
- iter-103 work:
  1. FluxBackend output PNG leak (§130) — **real resource leak,
     extends §126 sweep**. FluxBackend cleans up input files
     (src-*.png + mask-*.png) per-call, but the OUTPUT PNG is
     the caller's return value (handed to ImageScreen /
     ChatViewModel for display), so nobody deletes it. In
     practice users rarely save every generated image → every
     image generation leaks 2-10 MB into
     `/var/folders/.../T/vmlx-flux-out/`. Dozens of images per
     day × weeks of usage = many GB of orphaned PNGs. The §126
     startup sweep only matched top-level `vmlx-*` prefixes;
     files inside the `vmlx-flux-out` directory don't carry
     any prefix (they're named `out-<hash>.png`), so they
     never got caught. Fix: extended
     `sweepAgedVMLXTempArtifacts` with a second pass that
     walks inside `vmlx-flux-out/` and removes files older
     than the same 1h cutoff, without removing the directory
     itself (FluxBackend recreates it per-call). §130
     regression guard asserts the "vmlx-flux-out" reference,
     the iter-103 marker, and the mtime-cutoff inner check.
- iter-102 work:
  1. enableAutoToolChoice audit-note accuracy (§129) — **docs-
     vs-behavior drift**. Known Gap #3 has said "kept for CLI
     compat — audit comment in SettingsTypes". The comment
     existed (iter-50) and claimed the flag is retained "so
     `--enable-auto-tool-choice` on the command line doesn't
     error out". Audit: grep'd Sources/vMLXCLI for that flag
     and found zero hits. `vmlxctl` uses ArgumentParser with
     strict flag definitions, so passing `--enable-auto-tool-choice`
     to vmlxctl actually DOES error with "unexpected argument".
     The CLI-compat claim was wrong. The flag IS still useful
     for JSON config migration (Python users' dumped
     GlobalSettings carry the key, we decode it without
     complaining) but the CLI angle is a myth. Fix: rewrote the
     audit note to match reality — accurate description of
     what the flag does (nothing) and why we keep it (JSON
     schema forward-compat for migration). §129 regression
     guard pins the corrected phrasing and asserts the stale
     CLI claim is gone.
- iter-101 work:
  1. ChatRequest unsupported-field doc drift (§128) — **docs-vs-
     behavior mismatch audit**. The header block at the top of
     `ChatRequest` claimed `n` / `logprobs` / `top_logprobs` /
     `frequency_penalty` / `presence_penalty` / `logit_bias`
     were all "rejected with 400 until real support lands". But
     `validate()` had been relaxed (audit 2026-04-15 after
     LangChain + OpenAI SDK v1 defaults broke ~100% of
     vanilla-config callers) to accept frequency_penalty +
     presence_penalty + top_logprobs + logit_bias silently
     (range-checked only; engine ignores them). An auditor
     reading the struct would think those four still 400'd. No
     user-visible bug today but future contributors wanting to
     understand the contract got misleading signals. Fix: rewrite
     the doc header to reflect actual semantics — `n != 1` and
     `logprobs == true` still hard-reject, the other four
     range-checked + accepted silently. Per-field doc comments
     also updated. §128 regression guard asserts the new header
     phrasing ("partially-supported", "accepted silently") AND
     that the dead "blanket 400" claim pattern is absent.
- iter-100 work:
  1. Mid-load cancel race (§127) — **real UX bug**. `stop()`
     cancelled `currentLoadTask` and wiped `loaded` /
     `loadedModelPath` / `lastLoadOptions`, but the mlx-swift
     load path (safetensors mmap, tokenizer init, Metal warmup)
     doesn't honor Swift task cancellation — it runs to
     completion. The load body then committed
     `setLoaded(container)` + transitioned to `.running` a few
     seconds later, overriding stop()'s wipe. User clicked
     Stop, saw the state go `.stopped`, then watched the model
     spring back to `.running` with a fresh engine they just
     tried to abort. Real-reproducible on any load taking
     >5 seconds (every realistic model). Fix: checkpoint
     `Task.isCancelled` right before the final
     `setLoaded(container)`. If cancelled, flush MLX pooled
     buffers (§113), yield `.failed("cancelled")`, return.
     Engine stays in whatever state stop() put it in, memory
     is reclaimed. §127 regression guard asserts the
     checkpoint lives BEFORE setLoaded (ordering matters) AND
     that the cancel bail-out flushes Memory + yields cancelled.
- iter-99 work:
  1. Engine startup sweep for orphan vmlx-* temp artifacts (§126)
     — **crash recovery + per-call leak containment**.
     `ChatRequest.VideoURL.loadVideoLocalURL` writes decoded video
     bytes to `vmlx-video-<UUID>.<ext>` when an HTTP API client
     posts a `data:video/...` or `https://` URL. `Stream.extractVideos`
     uses the local URL during prefill and does NOT clean up on the
     SUCCESS path (only on the "no video track detected" reject).
     Every HTTP VL video request with inline payload left an orphan
     in `/var/folders/.../T/vmlx-video-*`. The §117 WhisperAudio
     defer and §125 tokenizer-shadow defer both cover the happy
     path, but a crash mid-operation still leaves orphans. Fix:
     at `Engine.init` fire a detached task that sweeps `vmlx-video-*`
     + `vmlx-whisper-*` + `vmlx-tokenizer-shadow-*` files older
     than 1 hour. Crash-recovers prior runs AND makes per-call
     leaks self-bounded. 1h TTL is lenient enough that in-flight
     work never gets swept (even a 30-min video inference
     completes well inside the window). §126 regression guard
     pins the init call, helper definition, all three file
     prefixes, and the 3600-second TTL.
- iter-98 work:
  1. Tokenizer shadow-dir temp leak (§125) — **real resource leak**.
     `TransformersTokenizerLoader.makeShadowWithPatchedTokenizerClass`
     creates a UUID'd dir under the system temp with symlinks +
     a patched `tokenizer_config.json`, returns the URL, and the
     retry path at `loadTokenizer` passes it to
     `AutoTokenizer.from(modelFolder:)` exactly once — but nothing
     cleans it up. Every JANG model with a custom tokenizer class
     name (TokenizersBackend, Qwen3Tokenizer, etc.) hit this path
     and left an orphaned `/var/folders/.../T/vmlx-tokenizer-shadow-<UUID>/`
     until the process exited or macOS periodic cleanup ran (days
     to weeks later). Heavy-model-swap sessions accumulated
     hundreds of these. Fix: defer-remove after the
     `AutoTokenizer.from` call. Same pattern as §117 (WhisperAudio).
     §125 regression guard pins the defer.
- iter-97 work:
  1. RemoteEngineClient UTF-8 corruption (§124) — **real i18n
     data-corruption bug**. All three streaming parsers
     (`streamOpenAI`, `streamOllama`, `streamAnthropic`) used
     `Unicode.Scalar(byte)` + `Character(scalar)` to accumulate
     each HTTP byte as a separate Unicode scalar in
     U+0000..U+00FF. Works for ASCII, corrupts UTF-8 multi-byte
     sequences. A non-ASCII byte `0xE2` (start of ✅) became
     scalar U+00E2 instead of being part of `[0xE2, 0x9C, 0x85]`.
     `data(using: .utf8)` then re-encoded each char as a 2-byte
     UTF-8 sequence, producing garbled bytes different from the
     input. JSON decode still succeeded (structural chars are
     ASCII) but string VALUES were mojibake. Emoji, accented
     chars, CJK — all broke on remote OpenAI/Ollama/Anthropic
     streaming responses. Fix: accumulate `[UInt8]`, detect line
     boundary on raw byte 0x0A, decode via
     `String(bytes: lineBytes, encoding: .utf8)`. UTF-8-safe and
     drops a per-byte allocation. Applied to all three streamers.
     §124 regression guard asserts the old pattern is gone + the
     new pattern appears in all three sites.
- iter-96 work:
  1. API key revoke stale-enforcement bug (§123) — **real security
     UX bug**. The revoke confirmation dialog promised "Any client
     using this key will immediately lose access", but
     `APIKeyManager.revoke(id:)` only deleted the row from SQLite +
     Keychain — it did NOT update `settings.apiKey`, which is what
     `BearerAuthMiddleware` actually enforces (middleware takes a
     single `apiKey: String?` at init, checked via constant-time
     compare in §115). Result: if the revoked key happened to be
     the one currently in `settings.apiKey`, the middleware's in-
     memory copy was never invalidated and clients kept
     authenticating with the "revoked" key indefinitely. The UI
     promise was a lie. Fix: in the revoke dialog action, capture
     the value being revoked BEFORE deletion, call a new
     `resyncBearerIfRevoked(wasValue:)` helper that checks whether
     the revoked value equals `settings.apiKey` and, if yes,
     `applySettings` with `remaining.first?.value` (most-recent
     remaining key) or nil (disable bearer auth when no keys
     remain). The guard `g.apiKey == revokedValue` ensures we
     never touch the enforcement key on revokes of NON-active
     keys. §123 regression guard pins all four checkpoints.
- iter-95 work:
  1. ChatExporter VL media data-loss (§122) — **real archival
     bug**. `exportToJSON` at `version: 1` serialized only
     role/content/reasoning/toolCallsJSON/createdAt, silently
     dropping three ChatMessage fields:
       - `imageData: [Data]` (inline VL image payloads)
       - `videoPaths: [String]` (attached video URLs)
       - `toolStatuses: [String: ToolCallStatus]` (tool-call phases)
     A user exporting a VL chat with images to JSON got a text-
     only archive. `exportToMarkdown` was worse — didn't even
     mention that media was attached. Fix: JSON schema bumped to
     `version: 2`, MessageBlock gained `imagesBase64: [String]?`
     (base64-encoded payloads) and `videoPaths: [String]?`.
     Markdown export now emits `_N image(s) attached_` and a
     video path list. base64 only goes into JSON (archival);
     Markdown stays small for paste-into-docs use. toolStatuses
     kept out-of-scope — it's lifecycle metadata keyed by
     transient tool_call ids that don't round-trip. §122
     regression guard pins the new schema version + the two new
     fields + the image-count banner in Markdown.
- iter-94 work:
  1. DownloadManager cancel status-flip (§121) — **real UX bug**.
     `cancel(_:)` synchronously set job status to `.cancelled` AND
     cancelled both the outer workTasks Task + the URLSessionDownloadTask.
     But when URLSession fired its completion handler with
     `URLError(.cancelled)` (CFNetwork -999), `run(id:)` caught it
     via the generic catch (the error is NSError, NOT Swift's
     `CancellationError`, so `catch is CancellationError` missed
     it), and flipped status from `.cancelled` to `.failed` with
     cryptic `"The operation couldn't be completed (Cocoa error
     -999)"`. User clicked Cancel, saw "cancelled" briefly, then
     "failed". Fix: in the generic catch, early-return on
     `URLError(.cancelled)`, on `Task.isCancelled`, or when
     `_jobs[id].status` is already `.cancelled` — any of those
     three means the user intentionally stopped the download.
     §121 regression guard pins all three early-return conditions.
- iter-93 work:
  1. MCP dead-subprocess teardown (§120) — **real lifecycle bug**.
     `MCPStdioClient.handleEOF` correctly fails all pending
     continuations + nils its own `process` when the child
     subprocess exits. But `MCPServerManager.clients[name]` held
     the zombie reference AND `statuses[name]` stayed at
     `.connected`. Next `executeTool`/`rawCall` on that server
     flowed through the client → stale stdin pipe → IO error.
     User saw `/v1/mcp/servers` report healthy while every tool
     call 500'd, and subsequent calls kept hitting the zombie
     instead of lazy-restarting the subprocess. Fix: wrap both
     `executeTool` and `rawCall` in a do-catch matching
     `MCPError.processFailure`. On that error, `markServerDead`
     drops the client from the registry, clears the tool cache,
     and transitions status to `.error` with a descriptive
     reason. The NEXT call sees `clients[name] == nil` and takes
     the lazy-start branch — automatic self-heal. §120 regression
     guard pins both call sites + the helper definition + the
     `.processFailure`-specific match (don't teardown on any
     generic error, only on actual subprocess death).
- iter-92 work:
  1. Terminal cwd UI↔engine drift (§119) — **real UX bug**.
     `TerminalScreen` kept its own `@State cwd` separate from
     `Engine.terminalCwd` (the authoritative bash working dir used
     by `ToolDispatcher.executeBashTool`). The raw-shell fallback
     path updated UI.cwd (§87 iter-56) but not engine.terminalCwd;
     the model-mode bash path updated engine.terminalCwd
     (ToolDispatcher:144) but not UI.cwd. Result: after a model-
     mode `cd foo`, the TerminalScreen header still showed the
     old path but the next bash invocation ran in the new dir —
     "header says I'm in ~/, but `ls` returns a different listing".
     And after a raw-shell `cd foo`, loading a model and switching
     to model-mode bash started from stale engine cwd, losing the
     UI-visible cwd change. Fix: on model-mode .done pull
     engine.terminalCwd back into UI.cwd; on raw-shell update,
     push the new value into engine via `updateTerminalCwd`.
     `updateTerminalCwd` is a no-op when the value matches, so
     bidirectional sync is idempotent. §119 regression guard
     asserts both call sites exist.
- iter-91 work (no-code audit):
  1. **Cross-protocol error-path sweep.** After the §118 SSE fix,
     verified the other three streaming encoders + all non-streaming
     error paths are already correct:
     - `JSONLEncoder.ollamaChatStream` / `ollamaGenerateStream`: on
       error emit `{error, done: true}` object then `writer.finish`
       + `return` — no success-shaped final chunk. Correct.
     - Non-stream `/v1/chat/completions` (OpenAIRoutes.swift:136-149):
       catches EngineError → `mapEngineError` (400/404/422/503/504);
       generic → 500 via `errorJSON`. Correct.
     - Non-stream `/v1/messages` (AnthropicRoutes.swift:72-84): same
       mapping through `OpenAIRoutes.mapEngineError`. Correct.
     - Ollama non-stream (OllamaRoutes.swift:302-508): all EngineError
       catch sites route through `mapEngineError`. Correct.
     No fix needed; logged as a clean audit so the next iteration
     doesn't revisit.
- iter-90 work:
  1. SSE encoder error-path truthfulness (§118) — **real wire-
     protocol bug**. All three SSE encoders
     (`chatCompletionStream`, `textCompletionStream`,
     `responsesStream`) caught an upstream throw, emitted a proper
     error event, AND THEN emitted a success-shaped final frame:
     `finish_reason: "stop"` on chat/text, `response.completed`
     with `status: "completed"` on responses. OpenAI SDK clients
     treat those success frames as "clean completion" — they log
     usage, close UI spinners, mark the response successful in
     telemetry. An error event immediately followed by a stop
     chunk gave inconsistent signals: the error gets logged, then
     the success frame papers over it, and the client ends up in
     an ambiguous state. Fix: track `hadError: Bool` across each
     do-catch. On the error path, skip the final finish-reason
     frame and the usage frame; for the Responses API additionally
     switch `response.completed` → `response.failed` with
     `status: "failed"` so SDK clients see the actual spec event.
     Anthropic's `/v1/messages` stream already handled this
     correctly (return early after the error event) — its pattern
     was the model for this fix. §118 regression guard asserts
     `hadError = true` appears in all three catch blocks, the
     `response.failed` + `"failed"` strings are present for the
     Responses path, and the `if !hadError` guard shows up at
     final-frame sites. Also updated the §N guard for iter-88's
     cancelStream fix that was matching the old mutation-during-
     iteration pattern (accidental: the §116 fix changed the code
     shape that an earlier §N regression guard pinned).
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
