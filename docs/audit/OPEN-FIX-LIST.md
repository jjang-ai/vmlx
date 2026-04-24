# vMLX Swift — Open Fix List

Systematic tracker for every outstanding issue the user has surfaced.
Updated per iteration. Numbered with § tags that match commits.

## Status legend

- ✅ = shipped this session
- 🚧 = in progress this session
- 📋 = queued, design decided
- ❓ = needs user direction
- ⏭ = deferred (not tractable in one turn, needs live model + time)

---

## Open fixes

### Priority 1 — correctness / release blockers

| # | Issue | Status | Section |
|---|-------|--------|---------|
| F1 | **Image generation fails** ("Image generation for this model is not yet ported. Swift FluxKit infrastructure (scheduler, VAE, DiT, loaders) is in place but individual model generate() methods are still scaffolded.") | ⏭ | vMLXFluxKit — real scaffolded, not a quick fix |
| F2 | **`[Image #N]` placeholder text** leaking into UI instead of real thumbnail | 🚧 | ChatInterface / MessageBubble image rendering |
| F3 | **Block disk cache** — storage layer + paged-cache integration | 🚧 | §355 storage layer in progress, paged-cache hook follow-on |

### Priority 2 — UX / copyability

| # | Issue | Status | Section |
|---|-------|--------|---------|
| F4 | **Text selection** — users can't copy error messages, notifications, settings field labels, message bodies. SwiftUI needs `.textSelection(.enabled)` applied broadly, and modifier key drag should allow multi-region selection (macOS standard) | 🚧 | §357 app-wide sweep |
| F5 | **i18n long tail** — ~175 raw `Text("…")` literals remain in SessionConfigForm / APIScreen / MCPPanel / ImageSettings / SetupScreen / AdvancedServerCard / HuggingFaceTokenCard / RequestLogPanel | 📋 | Opportunistic per-view, hookify rule enforces new adds |
| F6 | **VoiceOver / accessibility labels** on image-remove buttons, video-remove buttons, tool-call cards | 📋 | `.accessibilityLabel("Remove image N")` patterns exist but still in English |

### Priority 3 — shipped this session (ref)

| # | Issue | Status | Section |
|---|-------|--------|---------|
| ✓ | PR #99 logprobs wire-format | ✅ | cb01f95 |
| ✓ | §347 tokenizer routes (lm-eval Phase 1) | ✅ | 8c8324b |
| ✓ | §349 i18n foundation (en/ja/ko/zh-Hans, compile-time enforced) | ✅ | ae5d4ce → 004e168 |
| ✓ | §350 Jinja inline-if-without-else | ✅ | 7dd86b5 |
| ✓ | §352 /v1/completions legacy logprobs wire fix | ✅ | 389cda7 |
| ✓ | §354 TurboQuant-as-default cache picker + zombie purge | ✅ | 9c6974b |

### Priority 3.5 — new asks this session

| # | Issue | Status | Shipped as |
|---|-------|--------|------------|
| N1 | **CLI `vmlxctl chat` plain REPL** → real agentic | ✅ | §367 |
| N2 | **Agentic loop in CLI** | ✅ | §367 — Stream.swift ToolDispatcher + maxToolCalls cap |
| N3 | **UI Terminal tab visible tool-call surface + interleaved reasoning** | ✅ | §371 — TerminalTurn.Role += .reasoning with auto-collapse + dim styling; tool-call surface via InlineToolCallCard already in place |
| N4 | **Default chat settings from generation_config.json** | ✅ | §367 CLI + §368 engine API + UI placeholder |
| N5 | **User-side reflection of N4 in SessionConfigForm** | ✅ | §368 — sparkles-prefixed caption "Model recommends: temp=X, top_p=Y…" above inference fields; reads live from Engine.readGenerationConfig(at:) |
| N6 | **Help/info tooltips translated** | 🚧 partial | Major help strings done; remaining are `.help()` field tooltips in SessionConfigForm rows |
| N7 | **Terminal tool scope flags** — read-only / no-network / no-destructive / sandbox-cwd, prompt-level not tool-schema-level | ✅ | §369 |
| N8 | **Interleaved reasoning display + verbose mode** — stream demuxer for content/reasoning/tool-calls with ANSI color prefixes | ✅ | §370 |

### Priority 4 — live-test findings (Nemotron session)

| # | Finding | Status |
|---|---------|--------|
| T1 | Multi-turn cache hit works: T1 33/39 → T2 53/59 → T3 44/82, memory+ssm(46)+disk-backfill+gp(6) | ✅ verified |
| T2 | Hybrid SSM cache architecture active: 29 layers = 6 kvSimple + 23 mamba, hybridSSMActive=true | ✅ verified |
| T3 | Reasoning ON/OFF honored on all 3 APIs (OpenAI reasoning_effort, Anthropic thinking.budget_tokens, Ollama think) | ✅ verified |
| T4 | Only 1 model family tested live (Nemotron). Qwen3.6 JANGTQ4, Gemma4, MiniMax, GPT-OSS pending live-test | ⏭ wake-sleep matrix ready, awaiting user to start sessions |

### Priority 5 — architectural / design

| # | Issue | Status |
|---|-------|--------|
| A1 | Gateway port collision with Ollama default (8080) | 📋 suggest defaulting to 8888 if 8080 taken |
| A2 | Gateway: same model on 2 sessions → first-registered wins (ambiguous resolve) | 📋 flag to user with warning banner |
| A3 | Gateway: image/audio routes pin to default engine — no fan-out | 📋 doc'd 404 tells caller to use per-session port |
| A4 | `SpeculativeTokenIterator.collectedLogprobs` returns `[]` (latent silent drop — not used today) | ⏭ only bites if spec-decode wires up |
| A5 | VoiceOver / Accessibility — broad sweep beyond just labels | ⏭ |

---

## Acknowledged-incomplete areas

### Image generation (§F1)

**Honest status**: Flux scheduler + VAE + DiT + loaders ARE present under `Sources/vMLXFluxKit/`. What's missing is the per-model `generate()` method bodies — they exist as scaffolds that throw "not yet ported". The `docs/` tracking README details which models.

**Not safely fixable in one turn** because correctly porting each model's sampler loop requires:
- Matching the Python reference numerically (Schnell vs Dev vs Z-Image vs Klein differ in subtle ways)
- Large unit tests against Python-generated reference images
- Metal kernel timing validation

**What CAN be done now**: clean up the error message so it tells the user which models ARE ported vs which aren't, instead of a generic "scaffolded" message.

### Block disk cache (§F3 / §355)

**Plan**:
1. `BlockDiskCache.swift` — SQLite-indexed block store, safetensors payload per block, LRU eviction (THIS TURN)
2. Round-trip test (THIS TURN)
3. Wire into cache stats reporter (THIS TURN)
4. Hook into `PagedCacheManager` block-fill site (FOLLOW-ON — requires live model + paged cache to validate)

Default stays OFF until step 4 lands + lives through a multi-turn matrix run.

---

## Master commit chain this session

```
9c6974b §354 cache-settings unification
599f13c §352b release-test cached_tokens path
389cda7 §352 /v1/completions legacy logprobs
26f07af §353 wake-sleep matrix
96419f5 §351 final pre-release live-test
7dd86b5 §350 Jinja inline-if-without-else
004e168 §349 ChatSettingsPopover section labels
59a69d6 §349 MessageBubble + SessionsSidebar
8d9e616 §349 TrayItem + CORS TODO close
f9f863b §349 DownloadsWindow + menu
943807f §349 catalog + format helper
ae5d4ce §349 i18n foundation
03bcf1d §348 audit scorecard
8c8324b §347 tokenizer routes
cb01f95 PR #99 logprobs
```
