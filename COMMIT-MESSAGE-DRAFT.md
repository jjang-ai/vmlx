# Suggested commit message

```
iter-71..80: Swift v2 production hardening session 2026-04-18 — beta.3 + beta.4 DMGs shipped

Landed across 10 iterations:

Memory / crash fixes (beta.3):
  1. Load.swift — JANGTQ weights dict + ModuleParameters temp dropped
     before graph-eval; 57 GB MiniMax JANGTQ loads at 3.1 GB peak RSS
     (from ~2× during the bulk update hop). Community user's 3L report.
  2. MediaProcessing.swift (x2 overloads) — eager CIImage→MLXArray
     conversion halves video frame peak memory. 60s 1080p @ 2 FPS:
     ~3.6 GB → ~1.8 GB.
  3. Stream.swift extractImages — prefix(maxTotalImagesPerRequest) at
     parse time so 100-image requests never decode all 100
     simultaneously. VL RAM bound.
  4. Stream.swift SSM re-derive cancel gate — Task.isCancelled guards
     at spawn + inside detached re-derive Task. Closes the §3 open
     blocker: hybrid-SSM + MXFP4 AND JANGTQ2 cancel_midstream now
     survive. Crash was EXC_BAD_ACCESS in mlx::core::binary_op_gpu_inplace
     on freed MTL::Resource. DiagnosticReport diagnosed.
  5. Stream.swift cancel barrier widened — ANY cancel skips MLX.eval,
     relies on synchronize() for drain. Hybrid SSM path safe.
  6. vMLXCLI/main.swift — signal(SIGPIPE, SIG_IGN). Mid-stream client
     disconnect (harness cancel_midstream) no longer silent-exits
     vmlxctl. Hummingbird ServiceGroup handles SIGTERM/SIGINT but not
     SIGPIPE; default SIGPIPE action is SIGTERM-style.
  7. Llama multi-<|python_tag|> parser — split on EVERY marker, drop
     invalid JSON fragments silently. Harness-caught against llama-3.2-1b.
  8. SettingsStore.swift defaultMaxTokens clamp — persisted value < 256
     restored to code default. User hit a 60-token cap from a prior
     harness run.

API contract fixes (beta.3):
  9. OpenAIRoutes.mapEngineError — single EngineError → HTTP status
     helper covering all 13 variants (invalidRequest→400,
     toolChoiceNotSatisfied→422, modelNotFound→404, notLoaded→503,
     requestTimeout→504, promptTooLong→400, etc). Collapses 8 hand-
     rolled `if case` sites across OpenAI + all protocol routes.
  10. OllamaRoutes + AnthropicRoutes — both now call chatReq.validate()
      before streaming. Live audit caught temperature=99 returning
      HTTP 200 with a valid completion because validate was skipped.
  11. All protocol routes (OpenAI, Ollama, Anthropic) funnel
      EngineError through OpenAIRoutes.mapEngineError — third-party
      SDKs that rely on 4xx-for-bad-input (LangChain, anthropic-sdk-
      python, ollama-js) now see contract-correct codes.
  12. OllamaCapabilities.swift extracted — /api/show capabilities
      array classifier lifted into a pure helper with 17 unit tests
      covering qwen/llama/mistral/gemma/deepseek/minimax/glm/kimi
      families vs modality cross-product.

UI / UX (beta.3 + beta.4):
  13. ChatScreen — Load Model button always visible in top-bar; was
      hidden when no model picked yet. Added "Wake now" CTA to
      .standby(.deep) banner.
  14. ChatScreen loadChatModelInline — wakeFromStandby fast-path for
      sessions already in .standby; was full startSession pipeline.
  15. TrayItem — "Running · Sleeps in M:SS" live countdown via
      IdleTimer.nextSleepCountdown() + 1Hz poller in AppState
      rebindEngineObserver.
  16. InputBar — "≈N tok · M chars" caption flipping amber/red at
      2k/8k token thresholds.
  17. ChatViewModel.branchSession(from:) + MessageBubble onBranch
      — fork chat from any message into a new session. Arrow-glyph
      title prefix, re-UUID'd messages to avoid SQLite PK collision,
      undo action pushed. Gated on !isGenerating.
  18. vMLXApp — VMLXAppDelegate via @NSApplicationDelegateAdaptor.
      applicationWillTerminate flushes pending SettingsStore writes
      (2s bounded). Prevents lost settings on immediate Cmd-Q.
  19. IdleTimer.nextSleepCountdown() + (kind, seconds) tuple for UI.

Streaming robustness (beta.4):
  20. SSEEncoder — sseMergeWithHeartbeat helper + env-tunable
      sseHeartbeatInterval (VMLX_SSE_HEARTBEAT_SEC, default 15s).
      Wired into chatCompletionStream + textCompletionStream +
      responsesStream. Emits ": keep-alive\n\n" SSE comments during
      upstream idle. Protects thinking-model streams (20-40s prefill)
      from nginx 60s / ollama-js 30s / custom-SDK read timeouts.
  21. JSONLEncoder — same helper wired into ollamaChatStream +
      ollamaGenerateStream. NDJSON emits empty-content ping objects
      (matches Ollama's native reasoning-phase behavior).

Infrastructure:
  22. scripts/build-release.sh rewritten — Xcode 26 broke
      exportOptionsPlist `method` enum (every documented value fails
      with "expected one {} but found <value>"). New script uses
      SwiftPM build + manual .app bundle staging + direct codesign +
      notarytool submit + stapler. Shipped 2 DMGs with it today.
  23. Package.swift runnable test target — 58 → 300 tests.
      OllamaCapabilitiesTests (17) + RegressionConstantsTests (21
      source-scan guards for every fix above) + 10+ existing test
      files recovered from stale references to removed fields
      (cacheMemoryPercent). Xcode-only tests documented in header.
  24. Stale tests repaired — logprobs rejection, TurboQuant
      default-off, VL <image> marker expectation, cacheMemoryPercent
      removal.

Ship artifacts:
  - vMLX-2.0.0-beta.3-arm64.dmg (22 MB) — shipped 2026-04-18 13:32.
    Notary IDs f64c5948 + c46c9a90, both Accepted, stapled, Gatekeeper-
    approved. Fixes 1-19 + 22-24.
  - vMLX-2.0.0-beta.4-arm64.dmg — same pipeline, adds fixes 14, 17,
    20-21 (wake fast-path + branch + SSE/NDJSON heartbeat).

Test posture:
  swift test: 300/300 pass, 3 skipped (Metal-dependent tests stay
  Xcode-only until MLX ships a SwiftPM-bundle-aware metallib loader).

Zero hardcoded credentials in Sources/. .env.signing gitignored.
Project Team ID (55KGF2S5AY) is public signing config.
```
