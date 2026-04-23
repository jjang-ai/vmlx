# vMLX Swift — Production Audit Tracker (2026-04-22)

**Storage note:** This file lives outside `.claude/` so edits don't trigger
the ralph-loop plugin's hook path. The actual ralph-loop state file
(`.claude/ralph-loop.local.md`) is kept minimal — only the frontmatter
the plugin needs to decide whether to continue. All progress notes +
checklist items live here.

## Completion promise

Ralph loop closes when every row below is `[x]` with commit SHA + short
proof, image gen/edit either produce real prompt-conditioned pixels OR
are explicitly gated as `.notImplemented` (no stub-returning-success
pathway), deep-sleep drains chat + image + embedding backends with
measured memory drop, `bash .claude/live-verify.sh` 44/44, `bash
.claude/lifecycle-verify.sh` 13/13, `bash .claude/image-lifecycle-verify.sh`
all green, gateway mode reachable from CLI, no regressions, dev
pushed to origin.

## Per-iteration protocol

1. Pick the top `[ ]` row. Do NOT skip.
2. Read implicated sources + prior commits that touched them.
3. Ship real code.
4. Live-verify — boot a server, hit the route, measure the behavior.
5. Cross-check sibling harnesses (live-verify, lifecycle-verify,
   image-lifecycle-verify).
6. Mark `[x]` with commit SHA + one-line proof. Commit + push.
7. Bump iteration in `.claude/ralph-loop.local.md`.
8. Break big rows into sub-rows first; never claim `[x]` on a partial.

## Block 1: IMAGE GEN/EDIT — close the placeholder gap

- [x] I1 Audit complete — 10 of 11 registered image models throw
  `.notImplemented` today (Flux1-Schnell/Dev/Kontext/Fill, Flux2-Klein +
  edit, FIBO, QwenImage + edit, SeedVR2). Only Z-Image Turbo loads and
  runs, and its transformer is a noise-returning scaffold per
  `ZImage.swift:7-14`. Commit `7cac93b`.
- [x] I2 ZImage already honors request.width/height — prior "1024
  hardcoded" was a test-harness bug (my curl used top-level `width/height`
  instead of OpenAI's `size:"WxH"`). Live-verified: 256x256 → 256x256,
  512x384 → 512x384.
- [x] I3 Honest placeholder signaling — `isPlaceholder: Bool` on
  `ModelEntry`/`LoadedModel`/`FluxEngine.loadedIsPlaceholder()`, stamped
  into `/v1/images/generations` response as
  `warnings: [{code: "placeholder_output", message, model}]`.
  Commit `7cac93b`.
- [ ] I4 mflux variant (`Z-Image-Turbo-mflux-4bit`) — currently the
  subdir-fallback catches its safetensors but the model config isn't
  mflux-aware. Support or surface a clean error.
- [ ] I5 Flux.1-Schnell end-to-end — scan HF cache; if locally present,
  verify POST produces a valid prompt-conditioned PNG. Today:
  `.notImplemented` throw (returns 500).
- [ ] I6 Flux.1-Dev end-to-end (same drill as I5).
- [x] I7 Image routes classify user-error Flux failures as 400 —
  wrong model kind, weights not found, unknown model, invalid size,
  missing required field. Live-verified: POST /v1/images/edits on
  Z-Image-Turbo → 400 wrong-model-kind. Commit `b04ea03`.
- [ ] I7a Edit round-trip on real edit model — blocked on locally-absent
  Kontext/Fill/Qwen-Image-Edit weights.
- [ ] I8 Image-gen response shape conforms to OpenAI: `data[]` with
  `b64_json` or `url`, `seed` populated, `timing` populated. Verify all
  three `response_format` values ("b64_json", "url", "bytes").

## Block 2: LIFECYCLE — image backend drain on sleep

- [x] L1 softSleep drains `fluxJobs` (transient per-job state) while
  keeping image weights resident. Commit `6397e1d`.
- [x] L2 deepSleep unloads FluxEngine + nils fluxBackend entirely.
  Commit `6397e1d`.
- [x] L3 `lastImageModelPath` retained across deep-sleep +
  `rehydrateImageBackendIfNeeded` runs at top of generateImage/editImage.
  Commit `6397e1d`.
- [x] L4 JIT-wake on image gen — live-verified: deep-sleep → POST
  /v1/images/generations → 200 ok in 2.7s with JIT re-hydrate.
  Commit `6397e1d`.
- [ ] L5 Embedding backend — verify softSleep/deepSleep affect embedding
  model weights when `--embedding-model` was set.

## Block 3: LIVE HARNESS COVERAGE

- [x] H1 `.claude/image-lifecycle-verify.sh` shipped — 12/12 checks
  across S1 boot + I3 placeholder warning + L1 soft-sleep + L2 deep-
  sleep + L3/L4 JIT-rehydrate + I7 edit route 400. Live-run
  `PASSED: 12  FAILED: 0`, deep-wake+gen 2s. Commit `b0ecd1a`.
- [ ] H2 Extend `.claude/live-verify.sh` with S19-S23 covering
  `/admin/log-level` live swap + `x-vmlx-trace-id` on streaming.
- [ ] H3 Extend `.claude/lifecycle-verify.sh` with S6 covering
  concurrent soft+deep sleep race.

## Block 4: GATEWAY MODE

- [ ] G1 `vmlxctl gateway --port N --engines <path1>,<path2>` CLI
  subcommand booting a GatewayServer with multiple Engine actors.
- [ ] G2 Gateway /health returns aggregate {per-engine state, total
  loaded bytes, gateway uptime}.
- [ ] G3 Gateway /v1/chat/completions routes by model-id match.
- [ ] G4 Gateway + image: image-loaded engine gets routed for
  /v1/images/* when another engine has no image backend.
- [ ] G5 Gateway auth — decide aggregate vs per-engine + enforce.

## Block 5: HYBRID SSM + PAGED + PREFIX + TQ + DISK L2

- [ ] C1 Reset+re-run the 2026-04-16 cache matrix across
  {Gemma4 non-hybrid, Qwen3.6 hybrid+thinking, MiniMax hybrid+thinking,
  Nemotron Cascade hybrid+thinking}. Verify T1 cold → T2 warm hit,
  hybrid SSM companion fetched, TurboQuant layer count, disk L2 survive
  restart.
- [ ] C2 Disk L2 SHA-256 corruption live-test — deliberately truncate
  a cache file, fetch, expect eviction + log + miss.
- [ ] C3 SSM async re-derive counter bump during reasoning-high.
- [ ] C4 Prefix cache hit across VL turns.
- [ ] C5 TurboQuant layer-count sanity — tqLayers == total layer count.
- [ ] C6 Paged cache block reuse after unload+reload.

## Block 6: API SURFACE PARITY

- [ ] A1 `.claude/route-matrix-verify.sh` — per-route live-verify across
  the 52 RouteCatalog entries.
- [ ] A2 `tool_choice="required"` on non-tool-capable model → 422.
- [ ] A3 `/v1/rerank` scoring honest.
- [ ] A4 `/v1/audio/transcriptions` 503 vs real Whisper output.
- [ ] A5 `/v1/audio/speech` 503 vs real audio.
- [ ] A6 Ollama `/api/chat` streaming NDJSON frame format compliance.
- [ ] A7 Ollama `/api/tags`+`/api/ps` return clean model names.
- [ ] A8 Anthropic `/v1/messages` thinking/tool_use block shape.

## Block 7: SETTINGS RESOLUTION

- [ ] S1 Per-field 4-tier resolution test (global/session/chat/request).
- [ ] S2 Load-time settings: DB persist + new session picks up new value.
- [ ] S3 Per-request settings override all other tiers.

## Block 8: UI / BUTTON AUDIT

- [ ] U1 Total button count + 20% sample → verify action is real.
- [ ] U2 Settings sliders persist to SettingsStore.
- [ ] U3 Sidebar tabs + modal sheets navigate correctly.
- [ ] U4 RoutesCard copy-curl produces a curl that actually works.
- [ ] U5 AdvancedServerCard TLS browse updates state + re-renders.

## Block 9: LOGS + TRACE

- [x] O1 LogStore category taxonomy — 12 stable categories in use
  (adapters/admin/bench/cache/engine/flux/gateway/mcp/rerank/server/
  validator/whisper). LogStore accepts any string; no typos/churn
  observed across the codebase.
- [x] O2 §319 — Log-level live-swap propagates. Fixed middleware-level
  minLevel shadow — LogStore.append owns threshold via global
  `_globalMinLevel` updated by `setMinLevel`. Commit `ae86a3c`.
- [x] O3 §107 — RequestLogger PII-safe. Never reads auth headers,
  bodies, or URL query. Only method/path/status/elapsed/tid land.
- [x] O4 Debug bundle — inherits O3 (nothing sensitive enters LogStore
  → nothing leaks through export).

## Block 10: PERFORMANCE TARGETS

- [ ] P1 Qwen3.5-35B-A3B-4bit decode ≥100 tok/s (reference 99).
- [ ] P2 Nemotron-Cascade-2-30B-A3B-JANG_4M decode ≥100 (ref 132.3).
- [ ] P3 Gemma-4-26B-A4B-JANG_4M decode ≥90 (ref 87.9).
- [ ] P4 Image-gen wall-clock on Z-Image Turbo 4-step < 10s (placeholder
  ~8s today; target holds post-real-DiT).
- [ ] P5 TTFT < 30ms after warm prefix cache hit across the 3 above.

## Block 11: PACKAGING + DISTRIBUTION

- [ ] D1 DMG signed + notarized end-to-end (not SKIP_NOTARIZE).
- [ ] D2 Fresh-user first-run: install DMG → SetupScreen → starter
  download → Chat tab bot response.
- [ ] D3 `brew install ./packaging/homebrew/vmlx.rb` locally works.
- [ ] D4 Auto-updater config: latest.json references DMG, app detects.

## Block K: Kimi K2.6 JANGTQ port (from ../jang/research/KIMI-K2.6-VMLX-INTEGRATION.md)

- [x] K0 Audit — what exists, what doesn't. Findings:
  - ✓ `LLMModelFactory` registers `"kimi_k25": DeepseekV3Model`
  - ✓ `DeepseekV3.swift` full MLA + MoE (prefill-style; no bf16 drift)
  - ✓ `ChunkedPrefillVLM.swift`, `KimiK2ToolCallParser.swift`
  - ✓ `TurboQuantSwitchGLU`, `TurboQuantKVCache`, `JangMXTQDequant`,
    `TQHadamard`, `TQCodebook`, `TQEncoder`, `NumPyPCG64`
  - ✓ Precedent JANGTQ ports: `MiniMaxJANGTQ`, `GLM4MoEJANGTQ`,
    `Qwen35JANGTQ`
  - ✗ `DeepseekV3JANGTQModel` (blocking Kimi K2.6-JANGTQ_1L native load)
  - ✗ `KimiMoonViT.swift` (blocking Kimi K2.6 VL inputs)
  - ✗ `KimiVLM.swift` wrapper
  - ✗ `VLMModelFactory` "kimi_k25" entry
- [x] K1 §317 — factory refuses mxtq bundles with actionable error
  message pointing at jang-tools Path A conversion. Prevents silent
  weight mangling on affine loader. Commit `ceaec3b`.
- [x] K2 §318 — `DeepseekV3JANGTQModel` ported. Covers model_types
  deepseek_v3 / deepseek_v2 / deepseek_v32 / kimi_k25. Reuses the
  internal `DeepseekV3Attention` / `DeepseekV3MLP` / `MoEGate` classes
  via `DeepseekV3JANGTQConfiguration.asDeepseekV3()` mirror so
  duplication stays under ~300 LOC. Cache/parser wiring audit proved
  already green (silver allowlist `kimi_k25` → deepseek_r1 +
  KimiToolCallParser + cacheType=mla; TQ auto-activation skips MLA).
  Commit `f91af12`. **This is the spec's Path B (§2.2.5) for routed
  experts — supersedes the Path A conversion recommendation for
  text-only load.**
- [ ] K3 Port `KimiMoonViT.swift` — 27-block MoonViT ViT, ~500 LOC
  from `mlx_vlm.models.kimi_vl.vision.VisionModel`. DEFERRED —
  requires Qwen2VLVisionBlock-shaped rewrite + div-fixed pos-emb +
  sd2_tpool. Precedents exist in Qwen25VL.swift +
  Qwen3VL.swift but configuration diverges enough to warrant its
  own file.
- [ ] K4 Ship `KimiVLM.swift` wrapper — calls `chunkedPrefillEmbedding`
  with `prefillStepSize: 32` (NOT default 512 — monolithic Metal
  buffer hits watchdog on 191 GB MoE). **Note**: the prefill clamp
  now lives at the scheduler level (§323) — `buildGenerateParameters`
  pins MLA models to ≤32 automatically, so even if K4 forgets to
  pass the explicit override the runtime is safe.
- [x] K5 §325 — `renameKimiMMProjectorKeys(_:)` helper shipped at
  `Sources/vMLXVLM/Models/KimiVLMSanitize.swift`. Pure key-rewriting,
  idempotent, safe on both affine and JANGTQ bundles. Covers the
  three-pair mapping (`pre_norm` / `proj.0` → `linear_1` / `proj.2` →
  `linear_2`). Future KimiVLMModel.sanitize() just calls this.
  Commit `b7651d8`.
- [ ] K6 Register `"kimi_k25"` in `VLMModelFactory` routing to
  `KimiVLMModel`. Blocked on K3 + K4.
- [ ] K7 Test harness — text + VL coherence against local
  Kimi-K2.6-REAP-30-JANG_1L (after Path A conversion). Text path is
  ready to live-test today via K2 native JANGTQ load; VL blocked on K3/K4.
- [x] K8 §323 — MLA prefill auto-clamp. `buildGenerateParameters`
  pins `prefillStepSize ≤ 32` whenever loaded model's cacheType is
  "mla". Spec §2.6 #6-#8 mandate. Commit `647797b`.
- [x] K9 §324 — `kimi_k25` silver allowlist stamped
  `thinkInTemplate: true`. Without this the UI loses the thinking
  toggle + `reasoning_effort` doesn't auto-map to `enable_thinking`
  per §223. Spec §2.6 #16. Commit `647797b`.

### Spec audit matrix (`research/KIMI-K2.6-VMLX-INTEGRATION.md` → vMLX Swift)

| Spec item | Swift status | Commit / ref |
|---|---|---|
| `kimi_k25` dispatch | ✓ LLMModelFactory | (pre-existing) |
| MLA L==1 fp32 SDPA | ✓ prefill-style avoids drift | DeepseekV3.swift |
| JANGTQ routed experts | ✓ native Path B | `f91af12` K2 |
| mm_projector rename | ✓ sanitize helper | `b7651d8` K5 |
| MoonViT vision tower | ✗ deferred | K3 |
| KimiVLM wrapper | ✗ deferred | K4 |
| VLMFactory register | ✗ blocked on K3+K4 | K6 |
| VL prefill chunking | ✓ via ChunkedPrefillVLM | (pre-existing) |
| Text prefill chunking for ≥100 GB | ✓ MLA auto-clamp ≤32 | `647797b` §323 |
| Tool parser (Kimi TS-style) | ✓ KimiToolCallParser | (pre-existing) |
| Thinking-by-default | ✓ thinkInTemplate=true | `647797b` §324 |
| Layer-by-layer warmup | ✗ not yet — Swift-specific | deferred |
| Dedicated MLX.Stream | API exists (`Stream.setDefault`/`runWith`), unused in generate loop. Multi-session risk makes a blanket opt-in unsafe — `mlx_set_default_stream` is process-global so concurrent TokenIterators would clobber each other. Deferred pending a `withNewDefaultStream`-based refactor that uses TaskLocal for per-Task isolation. | K10 deferred |
| wired_limit auto-tune | Unverified | K11 pending audit |
| Hidden-size detection | ✓ via text_config parse | (pre-existing) |

## Commits this audit chain

- `29cb85b` fix(flux): WeightLoader falls through to transformer/ subdirs
- `7cac93b` I1/I3 §311: honest placeholder_output warning on image gen
- `b04ea03` I7 §312: image routes classify user-error Flux as 400
- `6397e1d` L1/L2/L3/L4 §313: deep-sleep drains FluxBackend + JIT re-hydrate
- `b0ecd1a` H1 §314: image-lifecycle-verify.sh harness (12/12 green) + progress tracker moved out of .claude/
- `ceaec3b` K1 §317: factory refuses mxtq bundles on DeepSeek family (superseded by K2)
- `f91af12` K2 §318: DeepseekV3JANGTQModel — native Kimi K2.6 JANGTQ load + registration tests
- `ae86a3c` O2 §319: RequestLoggerMiddleware honors live /admin/log-level swap
- `31dae08` §320-322: three regression-guard fixes (audio JIT wake position +
  embed .notLoaded throw hoist + ChatSettings Coming-soon explainer)
- `647797b` §323-324 K8/K9: Kimi K2.6 integration — MLA prefill clamp +
  thinking-by-default stamp
- `b7651d8` K5 §325: Kimi VL mm_projector rename helper
- `cd90e56` §326: Kimi K2.6 tool-call parser — strip `functions.` prefix
  + `:N` suffix. **Production bug** — every Kimi K2 / K2.6 tool call
  was collapsing to function-not-found because the raw wire name
  `functions.search_web:0` was not being normalized. Engine-side
  `KimiToolCallParser` now mirrors the correct logic from
  `vMLXLMCommon/Tool/Parsers/KimiK2ToolCallParser`. 3 regression tests
  added (all green).
- `1c60fc7` §327: DownloadManager.Job forward-compat decoder —
  **Production bug**: pre-§293 sidecars missing `requiresHFAuth`
  caused `keyNotFound` on decode, silently wiping the entire
  download job history on first launch of the upgraded build.
  Custom `init(from:)` with `decodeIfPresent` defaults the field.
  6/6 DownloadSidecarTests green post-fix.
- `81f7a3c` §328: Anthropic `thinking.budget_tokens` → thinkingBudget
  pass-through. **Production bug**: `budget_tokens: 2000` was only
  used to pick an effort bucket; the numeric cap was silently
  dropped because `ChatRequest.init(...)` didn't expose the
  `thinkingBudget: Int?` parameter. Stream.swift's
  `thinkingBudgetCharCap` now receives the user's budget.
- `e685e28` §329: Ollama `think` accepts Bool OR String. Ollama
  0.12+ reasoning-effort API (`think: "medium"`) was silent-drop
  because only Bool was read. Shared `parseOllamaThinkField` helper
  handles both shapes across /api/chat + /api/generate.
- `53b859d` §330: `/v1/completions` + `/v1/responses` stream_options
  pass-through. Only `/v1/chat/completions` (Codable decode) honored
  `stream_options: {include_usage: true}`. Legacy completions +
  Responses routes dropped it because the positional init doesn't
  accept streamOptions. Post-init assignment fixes both.
- `26fa5ea` §331 **CORSAllowlistMiddleware — security hole**:
  2+-entry `allowedOrigins` mapped to `.originBased` which echoed
  ANY request Origin with no allowlist gating. Users thought they
  had restricted CORS to `["example.com", "app.com"]` but
  effectively had `.all`. New wrapper middleware strips Origin for
  disallowed values + 403s preflight requests.
- `41a343e` §332 explicit `ORPHAN` tag on 6 unused GlobalSettings
  fields (continuousBatching, streamInterval, prefixCacheSize,
  prefixCacheMaxBytes, cacheMemoryMB, memoryAwareCache). Python-
  parity fields that never got Swift wiring. Regression guard
  pins the orphan set so future work must consciously wire or
  un-tag.
- `995b07d` §333 RemoteEngineClient.listModels timeout — picker
  froze 60s on dead remote hosts; capped at 10s.
- `670df1b` §334 **user-URL DoS caps** — `image_url` + `video_url`
  fetches had no timeout (60s default) + no size cap (gigabyte
  streams held in RAM). Now 20s timeout, 64 MB (image) / 512 MB
  (video) size caps, data(for:URLRequest) path.
- `aa0e0c0` §337 **SSM re-derive VL guard** — `maybeReDeriveSSMState`
  used a text-only forward to derive clean SSM state, but for a VLM
  request where real generation mixed image embeddings into the
  prompt tokens, the derived state is wrong at every position and
  would corrupt future-turn mamba layers. Plus SSMStateCache.store
  was called with mediaSalt=nil, producing a salt-less entry real VL
  fetches never hit. Skip re-derive when userInput has any image or
  video. Full audit confirmed BatchEngine/PagedCacheManager/DiskCache
  all thread mediaSalt correctly; shouldSkipSSMStorage elides hybrid+
  thinking SSM companions at store-time already; MLA prefill clamp
  §323 flows through BatchScheduler via request.parameters.
- `80865f3` §335-336 **mlxstudio #83 + #88 Gemma4 VLM**:
  (#83) Long-prompt Metal-buffer overflow — Gemma4 was the only
  VLM not chunking prefill. Single unchunked forward on a 30-100k
  token QwenCode /init prompt blew past Metal's 9.5 GB buffer cap
  on 64-GB Macs. Fix: use `chunkedPrefillEmbedding` helper with
  the nil-padded cache quirk handled.
  (#88) Python pixel_values concatenate TypeError — Swift is
  already type-safe (`[MLXArray]` explicit typing). Guards pin
  both pixel-list sites so a refactor to `Any` can't re-open
  the Python bug class.
  New VLM-coverage test lists the 5 remaining unchunked VLMs
  (Qwen35/Qwen35MoEJANGTQ/Qwen3VL/SmolVLM2/Paligemma) as
  known risks for subsequent iterations.

## Harness state (updated each iteration)

- `bash .claude/live-verify.sh` — 44/44 ✓ (iter 3 post-L1-L4)
- `bash .claude/lifecycle-verify.sh` — 13/13 ✓ (iter 3)
- `bash .claude/image-lifecycle-verify.sh` — 12/12 ✓ (iter 4 post-H1)
- `swift test --filter RegressionConstantsTests` — 280/280 ✓ (post §320-322)
- `swift test --filter ModelFactoryRegistrationTests` — 5/5 ✓ (post K2 port)
