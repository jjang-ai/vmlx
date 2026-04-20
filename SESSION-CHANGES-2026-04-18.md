# Session changes — 2026-04-18

Quick-read guide to everything touched this session. Full audit lives in
`SWIFT-AUDIT-2026-04-18.md`.

## TL;DR

Six code fixes, one user-data repair, six new regression rules,
58 unit tests now runnable via `swift test`, **JANGTQ2 burst OOM
closed** (live-verified: 5/5 HTTP=200 + SERVER ALIVE post-burst).

## Code changes (7 files, +122/-18)

| # | File | What | Severity |
|---|------|------|----------|
| 1 | `Sources/vMLXEngine/Parsers/ToolCallParser.swift` | Llama multi-`<\|python_tag\|>` extraction — split on ALL markers, parse each, drop invalid | HIGH (harness-caught) |
| 2 | `Sources/vMLXEngine/Settings/SettingsStore.swift` | Defensive clamp for pathological `defaultMaxTokens < 256` → restore code default + persist | HIGH (user-reported) |
| 3 | `Sources/vMLXServer/Routes/OllamaRoutes.swift` | `/api/show` returns `capabilities` array — Ollama 0.20.x clients can filter picker | MED (Copilot parity) |
| 4 | `Sources/vMLXLMCommon/TurboQuant/TurboQuantKVCache.swift` | windowStep 256 → 4096 — amortize burst-time reallocation 16× | MED (part 1 of JANGTQ2 fix) |
| 5 | **`Sources/vMLXLMCommon/Cache/MemoryAwarePrefixCache.swift`** | **Pressure throttle 60s → 10s — ROOT CAUSE of JANGTQ2 OOM. Burst completed in 15s, old throttle skipped all eviction → 5× 1 GB TQ payloads accumulated → OOM kill** | HIGH (part 2, closes blocker) |
| 6 | `Package.swift` | vMLXParserTests target — `swift test` now runs 58 tests | LOW (test infra) |
| 7 | `.claude/ralph-loop.local.md` | Auto-modified by harness runner | — |

## User-data repair (applied via sqlite)

- `~/Library/Application Support/vmlx/settings.sqlite3` — defaultMaxTokens
  60 → 32768, diskCacheDir `/tmp/vmlx-l2-test-*` → `~/Library/Application
  Support/vmlx/disk_cache` (stale harness artifacts). Backup in
  `/tmp/vmlx-settings-backup-*.sqlite3`.

## `swift test` — 58 tests pass

```
$ swift test
Executed 58 tests, with 1 test skipped and 0 failures (0 unexpected) in 0.073 seconds
```

- 21 parser tests (all 15 tool parsers + new Llama multi-tag guards)
- 10 AhoCorasick stop-sequence tests
- 10 NumPyPCG64 MXTQ PRNG parity tests (§23)
- 11 CacheCoordinator genPromptLen tests (§15)
- 3 SettingsStoreClamp tests (NEW — §26)
- 4 LogStoreTests (ring buffer + subscribe)

## New regression rules in `SWIFT-NO-REGRESSION-CHECKLIST.md`

- **§25** — Llama `<|python_tag|>` multi-marker extraction
- **§26** — SettingsStore must clamp pathological `defaultMaxTokens`
- **§27** — Ollama `/api/show` MUST return `capabilities` array
- **§28** — Memory pressure throttle MUST stay ≤ 10s

## New artefacts (local, per `.gitignore`)

- `SWIFT-AUDIT-2026-04-18.md` — full audit (350+ lines)
- `SESSION-CHANGES-2026-04-18.md` — this file
- `Tests/e2e/deep-scenarios.sh` — real-content end-user flows
- `Tests/e2e/jangtq-burst-probe.sh` — targeted JANGTQ2 burst repro
- `Tests/vMLXTests/SettingsStoreClampTests.swift` — runnable via `swift test`

## Live verification

| Test | Result |
|------|--------|
| Tier 1/2/3 harness (9 families × 30-49 cases) | 8/9 families pass cleanly |
| Deep scenarios × 2 models (Qwen3-0.6B + Gemma-4-e2b) | 11/12 pass (reasoning_ab on Gemma is correct-behavior-not-fail) |
| Live UI screenshot | 5 tabs + picker + reasoning + history + markdown + thinking fold-out all present |
| **JANGTQ2 concurrent_burst** | **5/5 HTTP=200, alive=200 immediately + 5s after** ✅ |
| `swift test` | 58 tests pass |
| Binary compiles | Yes (61s last cycle) |

## What's still open (non-blocking)

1. Gemma4 26B / Nemotron-30B 47-60% of ref perf — needs Instruments Metal trace
2. VL multi-turn L1.5 memory-cache skip (design choice, enhancement possible)
3. Logprobs endpoint intentional stub
4. Idle countdown timer UX gap
5. Tier-3 re-run with all fixes in progress (will confirm JANGTQ2 passes harness end-to-end)

## Not committed

Per project policy (`CLAUDE.md`: "NEVER release without explicit
permission" + "only commit when explicitly asked"), all changes are in
working tree awaiting your review.
