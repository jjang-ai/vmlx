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

- [ ] H1 Ship `.claude/image-lifecycle-verify.sh` covering I2/I3/L1-L4
  as bash harness with RAM/GPU delta measurement via `/metrics`.
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

- [ ] O1 LogStore category taxonomy — only known categories used.
- [ ] O2 Log-level live-swap propagates through RequestLoggerMiddleware.
- [ ] O3 RequestLogger no-PII guard — no prompts, completions, keys.
- [ ] O4 Debug bundle → zero hits for `sk-`/`hf_`/chat content.

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

## Commits this audit chain

- `29cb85b` fix(flux): WeightLoader falls through to transformer/ subdirs
- `7cac93b` I1/I3 §311: honest placeholder_output warning on image gen
- `b04ea03` I7 §312: image routes classify user-error Flux as 400
- `6397e1d` L1/L2/L3/L4 §313: deep-sleep drains FluxBackend + JIT re-hydrate

## Harness state (updated each iteration)

- `bash .claude/live-verify.sh` — 44/44 ✓ (iter 3 post-L1-L4)
- `bash .claude/lifecycle-verify.sh` — 13/13 ✓ (iter 3)
- `bash .claude/image-lifecycle-verify.sh` — NOT YET SHIPPED (H1 row open)
