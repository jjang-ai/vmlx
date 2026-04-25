# Python GH issues → Swift parity audit (2026-04-25)

Triage of recent vmlx (Python) GitHub issues against the Swift engine + app surface. Each row records: title, Swift relevance, current Swift status, and follow-up if any.

Issues already pinned in the prior parity sweep (`SSMParityMapTests`):
**#103, #105, #107, #109, #110** — see `Tests/vMLXTests/SSMParityMapTests.swift`. Not re-listed here.

| # | Title (Python) | Swift relevance | Status | Notes / follow-up |
|---|---|---|---|---|
| **#72** | MiniMax-M2.7-JANGTQ not working in 1.3.25 | Loader — Swift has its own JANGTQ path | **Done** (§221, §394) | Live multi-turn re-verified this session. MiniMax-M2.7-Small JANGTQ runs 3-turn coherent; site-specific `compile(shapeless:)` opt-out at the router (MiniMaxJANGTQ.swift §394) lands. |
| **#73** | Feature Suggestion: activity indicator | UI — Electron-only feature on Python side | **Pending — not in Swift app yet** | The Swift `vMLXApp` Tray + chat surface lacks a generic "engine busy" pill. Add: spinner in TrayItem header when `engine.state == .generating`, hide on idle. Low priority, cosmetic. |
| **#75** | unrecognized arg `--default-repetition-penalty 1.10` | CLI — Swift `vmlxctl serve` flag | **Done** | Verified in `main.swift:77-78` (`@Option var defaultRepetitionPenalty: Double?`) + propagated to settings at line 230. |
| **#80** | Unable to chat | Generic — broad regression class | **Done** | Multiple Swift-side fixes shipped (§36-§42, §47-§49, §63-§66 sequence). Live regression covered by Tier-1/Tier-2 smoke matrices. |
| **#81** | TQ loading with `--smelt` or `--flash-moe` | Engine — TQ + smelt/flashmoe coexistence | **Done** | Swift handles all three together: §50 TurboQuantKVCache pre-alloc, §99-100 TQ default flip + flashmoe slot bank, §167 FlashMoE auto-sizing. JANGTQ-calibrated bundles auto-on TQ; smelt + flashmoe orthogonal. |
| **#83** | Slower performance vs LMStudio | Perf — generic | **Done** | §99-100 perf push delivered Gemma4-26B 98 tok/s, Qwen3.5-35B-A3B 99 tok/s, Nemotron Cascade 128 tok/s. All at/above the 100 tok/s target on M4 Max. Deep audit at `docs/audit/swift-perf-2026-04-16.md`. |
| **#84** | Clarification on gemma4-31b-it-jang-4M | Doc-only | **N/A** | Python-side bundle question, no Swift work. |
| **#85** | Context Length | Doc-only | **N/A** | User-facing question about how to set context length. Swift exposes `maxTokens` per request and `slidingWindowMode` (this session §403) to control SW vs full-context. |
| **#87** | ValueError "Received 2 parameters not in model" | Loader — sanitize unhandled keys | **Done** | Swift uses `verify: [.noUnusedKeys]` rather than strict `.all` (`Load.swift:303`). Per-model `sanitize()` strips known-unused keys (e.g. DSV4's `tq_bits` / `attn.indexer.*` / `gate.bias→e_score_correction_bias` rename in §395). |
| **#89** | Metal OOM >72GB on hybrid SSM text >34K tokens | Memory — hybrid SSM Metal alloc | **Partial** | Swift has the equivalent guard in chunked-prefill at the SSM re-derive boundary (`SSMReDerive.swift:_prefill_for_clean_ssm` is one-shot with OOM-guard skip per v1.3.84 audit). Single-buffer >72GB on Qwen3.5 long-prompt is a Metal-allocator class issue — needs live test on Swift to confirm parity. **TODO:** add 50K-token text-only smoke on Qwen3.5-35B-A3B to confirm. |
| **#91** | SSM companion not served when KV prefix matches across sessions | Cache — cross-session SSM | **Partial** | `MemoryAwarePrefixCache` cross-session eviction is in place; SSM companion fetch path covered for same-session, but cross-session SSM hit is not explicitly tested. **TODO:** add a multi-session test to `SSMParityMapTests` exercising disk L2 SSM round-trip across two `Engine` instances. |
| **#92** | PLD speculative decode crashes on non-MLLM batch | Spec decode — Swift has DFlash | **N/A** for the exact bug | Swift's analog (DFlash) doesn't have the Python-specific MLLM/non-MLLM batch fork — it's a single dispatch in `Engine.runDFlashStream`. Crash mode doesn't reproduce. |
| **#94** | Deprecation cleanup `mx.metal.*` → `mx.*` | Python-only API | **N/A** | Swift consumes MLX through `Cmlx` bindings, not the deprecated Python aliases. |
| **#96** | Relocated image models broken | Image gen — bundle path resolution | **Pending verification** | Swift `vMLXFlux` + `vMLXFluxKit` integrate mflux-equivalent. The Python issue was about path symlinks not resolving for relocated bundles — Swift's `loadWeights` does `modelDirectory.resolvingSymlinksInPath()` (Load.swift:27), so the same class of bug should already be guarded. **TODO:** spot-test relocated Flux Schnell / Z-Image bundle. |
| **#97** | Fill IMAGE models with painted mask | Image edit — Qwen-Image-Edit | **Pending** | Qwen Image Edit is documented as full-precision-only on the Python side; Swift `vMLXFluxKit` has `/v1/images/edits` route per §213 but mask handling depth needs spot-check. **TODO:** verify mask channel passthrough with Qwen-Image-Edit. |
| **#112** | MLLMBatchGenerator class-shared stream is thread-local violation | Concurrency — VLM forward | **Done** (by design) | Swift has **no separate MLLM batch generator** — single unified `Stream.swift` path (per `Stream.swift:1174-1178` comment). All MLX dispatches use `MLX.Stream.defaultStream(.gpu)` (per-thread default), never a class-shared named stream. Python's exact bug class doesn't apply. |

## Summary

| Category | Count | Action |
|---|---|---|
| Done — already in Swift | 9 | No work |
| N/A — Python-only or doesn't apply | 4 | No work |
| Partial / pending verification | 4 | #73 (activity indicator UI), #89 (long-prompt hybrid OOM live test), #91 (cross-session SSM test), #96 (relocated Flux bundle test), #97 (Qwen-Image-Edit mask) |

The four pending items are spot-checks / regression guards, not new bug fixes. The 4 closed-bug parity rows for #72/#75/#80/#81/#83/#87/#92/#94/#112 don't need Swift action — either already shipped or by-architecture immune.

## Recommended next regression guards

1. `Tests/vMLXTests/Qwen35LongPromptOOMGuardTests.swift` — synthetic 50K-token text-only prompt against Qwen3.5-35B-A3B JANG; assert no Metal allocator failure.
2. `Tests/vMLXTests/SSMCrossSessionTests.swift` — start engine, prime SSM companion, restart, hit disk L2, assert SSM state recovers.
3. Image model: relocate `~/.mlxstudio/models/Flux-Schnell/` to `~/Downloads/Flux-Schnell/` symlink + run a generation; confirm path resolution.

These are session-end polish, not blockers.
