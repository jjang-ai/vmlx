# PROGRESS — vMLX Swift dev branch

Newest entries at top.

---

## 2026-04-15 — Dev build revived + DFlash v2 + F-G cross-family fixes

### Runtime-critical build fix

Metallib bundling was broken — every model load failed with
`Failed to load the default metallib`. Two causes:

1. `SWIFTPM_BUNDLE` define in `Package.swift` pointed at the stale
   upstream name `mlx-swift_Cmlx` (from when this was a vendored copy);
   SwiftPM produces `vmlx_Cmlx` because our package is named `vmlx`.
   Fixed — see `Package.swift:~115`.
2. The SwiftPM bundle on CLI targets is a flat dir (no
   `Contents/Resources/` hierarchy) so the `load_swiftpm_library`
   search in `device.cpp` couldn't find `default.metallib` inside it.
   Mitigation: colocate `mlx.metallib` next to the binary so the
   first-try `load_colocated_library(device, "mlx")` finds it. Helper
   at `scripts/stage-metallib.sh [debug|release]` after every
   `swift build`.

Live verified end-to-end on three models via `vmlxctl chat`:

| Model | Result |
|---|---|
| mlx-community/Llama-3.2-1B-Instruct-4bit | ✅ loads, generates |
| mlx-community/Qwen3-0.6B-8bit | ✅ loads, generates |
| mlx-community/gemma-4-e2b-it-4bit | ✅ loads, generates (Jinja fallback to built-in ChatML) |

### JANG-DFlash — end-to-end integration (v2)

Full speculative-decoding pipeline now plumbed through the stream path.
Previously just a CLI smoke test (`vmlxctl dflash-smoke`); now a
first-class Engine feature with real settings + lifecycle + API.

- **Settings** (`Sources/vMLXEngine/Settings/SettingsTypes.swift`):
  7 new fields on `GlobalSettings` + `SessionSettings` override —
  `dflash`, `dflashDrafterPath`, `dflashBlockSize` (16),
  `dflashTopK` (4), `dflashNumPaths` (60), `dflashTapLayers`
  (`"10,22,34,46,58"`), `dflashTargetHiddenDim` (3072). Merged
  through the 4-tier resolver in `SettingsStore`.
- **Engine state** (`Sources/vMLXEngine/EngineDFlash.swift`, new):
  `loadDFlashDrafter(from:config:)`, `unloadDFlashDrafter()`,
  `dflashIsReady()`, `dflashDrafterPath()`, `makeDFlashSpecDec(settings:)`.
  Drafter↔target `tapDim` shape check refuses mismatched checkpoints
  with a clear error. `autoLoadDFlashDrafterIfConfigured` restores
  the drafter from persisted settings on every `Engine.load`.
  `bindDFlashTargetIfEligible` auto-detects MiniMax / Mistral 4 /
  DeepSeek V3 targets and installs the right `JangDFlashTarget`
  adapter at load time.
- **Stream hot-path short-circuit** (`Sources/vMLXEngine/StreamDFlash.swift`,
  new): `tryDFlashGenerationPass` engages when enabled + drafter +
  target + no tools + no images. Coordinator fetch → restoreLayerData
  into pre-warmed cache → `cachedGenerate(cache:prefixMatched:)` so
  multi-turn prefix + L2 disk + SSM companion reuse lands across
  turns just like the standard path. Post-generation
  `storeAfterGeneration` persists the final cache + SSM state.
  Usage reports `cacheDetail: "dflash+{tier}(N blocks)"` so the UI
  surfaces the tier hit.
- **Lifecycle**: `Engine.stop()` + `deepSleep()` zero `_dflashDrafter`
  / `_dflashTarget` so a subsequent load of a different family
  doesn't dispatch through a dead adapter.
- **Target adapters**: `MiniMaxDFlashTarget` (MiniMax), new
  `Mistral4DFlashTarget` + `DeepseekV3DFlashTarget` with
  `callAsFunctionWithTaps` overloads on the model classes. All tap
  post-decoder-block `h` (not internal MLA K/V) to preserve drafter
  training contract.
- **Admin API** (`Sources/vMLXServer/Routes/AdminRoutes.swift`):
  `GET /admin/dflash` returns status blob; `POST /admin/dflash/load`
  with `{"path": "..."}` loads a drafter; `POST /admin/dflash/unload`
  drops it.
- **/v1/models enrichment**: currently-loaded model entry gains
  `vmlx.speculative_decoding = {kind:"dflash", drafter:"..."}` when
  a drafter is bound and compatible.
- **CLI** (`Sources/vMLXCLI/main.swift`): `--dflash`,
  `--dflash-drafter PATH`, `--dflash-block-size`, `--dflash-top-k`,
  `--dflash-num-paths`, `--dflash-tap-layers`,
  `--dflash-target-hidden-dim`. Drafter auto-loads after target if
  both flags are set.
- **UI** (`Sources/vMLXApp/Server/SessionConfigForm.swift`): DFlash
  toggle + drafter path + 4 sliders + tap-layer CSV. Smelt row
  relabelled `"Smelt mode (Python engine only)"` and
  `Stream.performOneGenerationPass` emits a one-shot warning per
  request when `smelt=true` so it's no longer a silent dead setting.

### F-G cross-family audit fixes (23 items scoped; 5 landed)

From `SWIFT-PER-FAMILY-MATRIX-2026-04-15.md`:

- **F-G1** — SSMStateCache `mediaSalt` propagation (P0 — Qwen3.5-VL
  hybrid multi-turn was silently colliding image-A state with
  image-B text prefix). `SSMStateCache.makeKey` + `store` + `fetch`
  + `fetchEntry` now accept `mediaSalt:`; `CacheCoordinator` call
  sites all updated.
- **F-G2** — BaichuanM1 `CacheList` disk serialization walker (P0 —
  BaichuanM1 wraps `(MambaCache, KVCacheSimple)` per layer and the
  old serialize ladder fell through to `.skip`, dropping all nested
  state). New `LayerKind.cachelist = 7` + compound `{i}_{sub}`
  indexing + per-sub kind tag + `deserializeCacheListLayer` +
  `restoreCacheListLayer` by sub-index.
- **F-G3** — Llama 4 dedicated model class (P0 — silver row pointed
  at generic `LlamaModel` which has no iRope or MoE). New
  `Sources/vMLXLLM/Models/Llama4.swift` (405 lines): iRope every 4th
  layer with attention temperature tuning, QK-norm on
  use_rope+use_qk_norm layers, interleaved MoE with SwitchGLU +
  shared expert + top-1 sigmoid routing, mixed
  `ChunkedKVCache`/`KVCacheSimple` per layer, gate_up_proj sanitize.
- **F-G4** — Gemma 3 tool parser `"hermes"` → `"gemma4"` (silver
  row was dropping tool calls silently).
- **F-G5** — Gemma 3 mixed `sliding_attention` + `full_attention`
  detection helper added to `CapabilityDetector` (won't
  misclassify as SSM hybrid now; runtime already safe via
  RotatingKVCache + SLIDING-1 disk round-trip).

Remaining F-G items (F-G6 through F-G23) tracked; FlashMoE family
conformance (F-G6..F-G9) was parked per user directive in favor of
DFlash focus.

### Docs

- `README.md` — build instructions now call out `stage-metallib.sh`;
  DFlash section replaces the old "deferred" note; smelt labelled
  honestly as "Python engine only".
- This `PROGRESS.md`.
