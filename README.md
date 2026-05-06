# vMLX — Swift

**The entire MLX inference stack, from Metal kernels to SwiftUI, in a
single SwiftPM package.** No external `path:` dependencies. No upstream
drift risk. We control every layer — kernel compile flags, quant
kernels, attention paths, scheduler, tokenizer, chat loop, HTTP
routes, desktop UI.

**Canonical home:** `./`
**Build:** `swift build` → `.build/release/vmlx` CLI · `xcodegen` for the macOS app
(4 unrelated Jinja-parser repros skip)
**Binaries:** `vmlxctl` (CLI), `vMLX` (SwiftUI app)

---

## Why this exists

Two forcing functions:

1. **App Store / sandbox.** Electron + bundled-Python can't ship to
   MAS. Pure Swift + SwiftUI can.
2. **Control the stack.** Every time we need a kernel tweak, a
   scheduler change, or a quant path, we'd rather not coordinate with
   upstream `mlx-swift`, `mlx-swift-examples`, or `vmlx-swift-lm`.
   Vendor everything and commit fixes directly.

As of 2026-05-02, vMLX Swift is the **main app** (Python panel
demoted to legacy, see [vmlx_swift_v2_known_issues.md](
https://github.com/jjang-ai/vmlx/blob/dev/) tracker). Vendoring is
complete: `Cmlx` + all 8 MLX Swift targets live next to our `vMLX*`
targets under one `Package.swift`.

**Current `dev` HEAD highlights** (newest at top, full log in
`PROGRESS.md`):

- **JangPress (axis E weight tier) — default ON @ pct=70.**
  Routed-MoE expert weights tier-out to `madvise(DONTNEED)` (mmap
  failsafe — kernel ignores under low pressure, reclaims up to 70%
  of routed mass under actual pressure). 6+ MoE families covered by
  13 tile-name regex patterns (Qwen3.5/3.6 MoE, Mistral 4, DSV3.x,
  Kimi K2, Laguna, MiniMax M2.x, Holo3, DSV4, Nemotron H Cascade /
  MXFP4 / Cascade-2). `--enable-jangpress` / `--jangpress-compress-
  pct` / `--jangpress-backend {mmap|mach|none}` /
  `--jangpress-force-mode {soft|force}`. `GET /v1/cache/jangpress`
  returns live state. See `Sources/vMLXLMCommon/Cache/
  JANGPRESS-PRODUCTION.md`.
- **mlx#3461 / mlx#3462 patch (iter 126).** Vendored mlx 0.31.1's
  `commandBufferWithUnretainedReferences()` + Swift structured
  concurrency MLXArray drop hits Metal "Invalid Resource" race
  under TurboQuant KV cache load. Patched `device.cpp:405` to
  retained `commandBuffer()` (~2.4% throughput cost — safe trade
  per upstream measurement on M5 Max @ qwen35-35b-a3b B=17/B=32).
- **kvCacheQuantization ↔ enableTurboQuant CLI symmetry (iter 127,
  parity with mlxstudio#138).** SettingsStore resolver derives Bool
  from canonical String, so `vmlx serve --kv-cache-quantization
  none` now actually disables TQ; `--disable-turbo-quant` also
  demotes the string. No more silent re-derive overriding the
  user's choice.
- **Engine.LoadOptions ↔ GlobalSettings alignment (iter 101).**
  Bare-init defaults match GlobalSettings tier-1 — fixes 3 prod
  call sites that were silently shipping stricter caches than
  configured.
- **MXTQ PRNG parity — RESOLVED (iter 93).** `NumPyPCG64.swift`
  ships bit-identical PCG64 port; `JangMXTQDequant` uses it for
  sign generation. MiniMax-M2.7-JANGTQ-CRACK decodes at 46.59 tok/s
  (Python ref 44.3).
- **JANGTQ-VL thread fix (iter 33).** Dedicated single-worker
  `mllm-worker_0` executor for VLM `step()` — fixes Stream(gpu, 1)
  Metal crashes on Qwen3.6-MoE-VL JANGTQ bundles. Live-verified
  Qwen3.6-35B-A3B-JANGTQ4: T2 30/31 cached (97% prefix hit).
- §341–§346, §338, §339, §340 (Jinja `not in`, MCP toggle resolver,
  thinking-leak audits, JANGTQ `mxtq_bits` flat-or-dict, JSON-
  Schema tool arg coercion, image model folder discovery, MCP
  one-click import — all already covered in PROGRESS.md).

---

## Stack layout

```
swift/
├── Package.swift                    # 21 local targets, 5 external deps
├── Package.resolved
├── project.yml                      # XcodeGen spec (optional)
├── scripts/
│   ├── build-release.sh             # xcodegen → archive → notarize → DMG
│   └── notarize-only.sh             # resubmit-only path
│
├── Sources/
│   │
│   │ ─── MLX runtime (vendored from mlx-swift @ vmlx-0.31.3) ───
│   ├── Cmlx/                        # ~23 MB — mlx + mlx-c C++ submodule
│   │                                #   checkouts with vmlx-patches-0.31.3,
│   │                                #   metallib staging, Metal kernels
│   ├── MLX/                         # core tensor API
│   ├── MLXNN/                       # nn.Module + layers
│   ├── MLXFast/                     # scaled dot product, layer norm, rope
│   ├── MLXFFT/                      # FFT ops
│   ├── MLXLinalg/                   # cholesky, qr, svd, etc
│   ├── MLXOptimizers/               # adamw, sgd, lion
│   ├── MLXRandom/                   # rng + distribution ops
│   │
│   │ ─── vMLX layer (our code) ───
│   ├── vMLXLMCommon/                # vendored mlx-swift-examples LMCommon
│   │   ├── Cache/                   # KVCache, PagedCacheManager, SSM
│   │   │                            #   companion, DiskCache, TQDiskSerializer,
│   │   │                            #   ChunkedPrefillVLM helper
│   │   ├── BatchEngine/             # continuous batching
│   │   ├── FlashMoE/                # SSD-streamed expert loader (Phase 1 + 2a)
│   │   ├── TurboQuant/              # TurboQuantKVCache
│   │   ├── Evaluate.swift           # generate loop
│   │   ├── LanguageModel.swift
│   │   ├── KVCache.swift
│   │   ├── JangMXTQDequant.swift    # MXTQ packed dequant w/ NumPyPCG64
│   │   ├── NumPyPCG64.swift         # PCG64 parity for MXTQ sign generation
│   │   ├── Load.swift
│   │   └── ChunkedPrefillVLM.swift
│   │
│   ├── vMLXLLM/                     # LLM model implementations (~50)
│   │   └── Models/                  # Qwen3/3MoE/3Next/3.5, Mistral, Mistral4,
│   │                                #   MiniMax, Nemotron-H, Gemma4Text,
│   │                                #   GLM4MoE, Jamba, FalconH1,
│   │                                #   GraniteMoeHybrid, LFM2/LFM2MoE,
│   │                                #   MiMoV2Flash, BaichuanM1, DeepSeekV3,
│   │                                #   GPT-OSS, AfMoE, BailingMoe, …
│   │
│   ├── vMLXVLM/                     # Vision-language models (~15)
│   │   └── Models/                  # Qwen25VL, Qwen2VL, Qwen3VL, Qwen35,
│   │                                #   Qwen35MoE, Gemma3, Gemma4, Mistral3,
│   │                                #   Mistral4VLM, Pixtral, Idefics3,
│   │                                #   Paligemma, FastVLM, GlmOcr, LFM2VL
│   │
│   ├── vMLXEmbedders/               # embedding models
│   │
│   ├── vMLXFluxKit/                 # DiT + T5/CLIP + VAE + flow-match scheduler
│   ├── vMLXFluxModels/              # Flux1 Schnell/Dev/Kontext/Fill, Flux2Klein,
│   │                                #   QwenImage, ZImage, SeedVR2, FIBO
│   ├── vMLXFluxVideo/               # WAN 3D video model
│   ├── vMLXFlux/                    # public facade (ImageGenRequest,
│   │                                #   FluxEngine, LatentSpace, WeightLoader,
│   │                                #   JangSupport, ModelRegistry)
│   │
│   ├── vMLXEngine/                  # our wrapper over the vendored layers
│   │   ├── Engine.swift             # load / stream / cache stats / mcp / flash moe
│   │   ├── Stream.swift             # generation loop w/ tool dispatch + §15 routing
│   │   ├── ChatRequest.swift
│   │   ├── Settings/                # 4-tier global → session → chat → request
│   │   ├── Cache/                   # CacheCoordinator (paged + disk + SSM)
│   │   ├── Parsers/                 # reasoning + tool-call parser registries
│   │   ├── Library/                 # ModelLibrary + DB + FSEvents watcher
│   │   ├── MCP/                     # real stdio JSON-RPC 2.0 client
│   │   ├── Tools/                   # BashTool + ToolDispatcher
│   │   ├── ModelCapabilities.swift  # 4-tier auto-detection
│   │   ├── CapabilityDetector.swift
│   │   ├── DownloadManager.swift    # background resumable HF downloads
│   │   ├── IdleTimer.swift
│   │   ├── MetricsCollector.swift
│   │   ├── FluxBackend.swift        # bridge to vMLXFlux
│   │   └── ImageGen.swift
│   │
│   ├── vMLXServer/                  # Hummingbird routes
│   │   ├── Server.swift
│   │   ├── Routes/                  # OpenAI / Ollama / Anthropic / Admin / MCP
│   │   ├── SSEEncoder.swift
│   │   ├── JSONLEncoder.swift
│   │   └── Auth.swift               # Bearer middleware
│   │
│   ├── vMLXApp/                     # SwiftUI app — 5 modes
│   │   ├── vMLXApp.swift
│   │   ├── Chat/                    # ChatScreen + ChatViewModel + MessageBubble
│   │   ├── Server/                  # SessionDashboard + HTTPServerActor
│   │   ├── Image/                   # ImageScreen + ImageModelPicker + Gallery
│   │   ├── Terminal/                # TerminalScreen w/ bash tool
│   │   ├── API/                     # APIScreen w/ LAN QR
│   │   ├── Settings/
│   │   ├── Downloads/
│   │   └── Common/
│   │
│   ├── vMLXTheme/                   # Linear-inspired color/typography tokens
│   │
│   └── vMLXCLI/                     # `vmlxctl serve / chat / pull / ls`
│       └── main.swift
```

**23 targets.** **5 external deps only:** `swift-numerics`,
`hummingbird`, `swift-argument-parser`, `swift-transformers`,
`Jinja`. Nothing else.

Audio targets (added 2026-04-14):

```
│   ├── vMLXWhisper/                 # MLX Whisper ASR — WhisperLoader,
│   │                                #   WhisperModel/Decoder/Tokenizer,
│   │                                #   WhisperAudio (mel + resampling).
│   │                                #   Auto-transcodes legacy .npz → .safetensors.
│   ├── vMLXTTS/                     # TTS — TTSEngine facade + PlaceholderSynth
│   │                                #   (deterministic 24 kHz WAV tone, real bytes).
│   │                                #   Kokoro neural backend scaffolded (9-step plan
│   │                                #   in KokoroBackend.swift), not live yet.
```

---

## Build

```sh
# Requires: macOS 14+, Xcode 15.4+ (Swift 5.10), xcodegen (brew install xcodegen)

git clone -b dev https://github.com/jjang-ai/vmlx.git
cd vmlx

# CLI — SwiftPM produces .build/<target-triple>/<cfg>/vmlxctl
swift build -c release
# CRITICAL: colocate the Metal kernel library so binaries can load
# models. Without this every load fails with "Failed to load the
# default metallib." The helper copies `vmlx_Cmlx.bundle/default.metallib`
# to `<build>/mlx.metallib` which is the first path mlx-swift's
# `load_colocated_library` tries.
./scripts/stage-metallib.sh release

# macOS app — xcodegen wraps the SwiftPM vMLXApp target into a signable bundle
xcodegen
open vMLX.xcodeproj
# Xcode → Run. Ad-hoc signing works for local dev; set your own DEVELOPMENT_TEAM
# in project.yml if you want to distribute the .app.

# Use the CLI
.build/release/vmlx serve --model /path/to/model
.build/release/vmlx chat  --model /path/to/model
.build/release/vmlx pull  mlx-community/Qwen3-32B-4bit
.build/release/vmlx list

# Build the SwiftUI app via XcodeGen + sign + notarize + DMG
./scripts/build-release.sh
```

**Binaries after build:**

```
.build/arm64-apple-macosx/debug/vMLX       # SwiftUI app
.build/arm64-apple-macosx/debug/vmlxctl    # CLI
```

---

## Downloading models

vMLX uses the standard HuggingFace cache layout, so anything you've
already downloaded with `huggingface-cli` or `transformers` will be
auto-detected on first launch.

**Three ways to start a download:**

1. **Image tab → model picker** — every Flux / Z-Image / Qwen Image
   model has a Download button. The progress window pops open
   automatically; nothing is ever silent.
2. **CLI:** `.build/release/vmlx pull mlx-community/Qwen3-32B-4bit`
3. **HTTP:** `POST /api/pull {"name":"<repo>"}` (Ollama-shape NDJSON
   stream — useful for scripting from another machine).

**Gated repos (Llama, Gemma, Mistral large):**

Some models require accepting a license on huggingface.co before they
can be downloaded. To use them with vMLX:

1. Visit the model page on huggingface.co and click **Request access**.
   Wait for approval.
2. Generate a token at <https://huggingface.co/settings/tokens> (Read
   scope is enough).
3. In the app, open the **API** tab → **HuggingFace access token** card,
   paste the token, click **Save & Test**. The token is stored in the
   macOS Keychain (never in plaintext on disk) and pushed into every
   download manager so subsequent gated downloads succeed.

**Speed and resume:**

Downloads stream directly to disk via `URLSessionDownloadTask` (no
byte-by-byte async iteration overhead). A 5-second sliding-window speed
metric drives the live MB/s readout — *not* a count of file shards.
Pause/resume use HTTP `Range: bytes=N-` requests, so paused downloads
pick up from the exact byte they stopped — no re-downloading.

**Where files land:**

```
~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/main/
```

The Server tab's Model Library scans this path plus any user-added
directories. Add custom dirs from Server tab → Model Directories panel.

---

## Runtime surfaces

**CLI (`vmlxctl`)**
- `serve --model PATH [--host H] [--port P] [--api-key K] [--json-progress]`
- `chat --model PATH [--system S]`
- `pull REPO`
- `ls`

**HTTP server (`vMLXServer`)**
- OpenAI: `/v1/{chat/completions, completions, responses, embeddings,
  models, rerank, images/generations, images/edits,
  audio/transcriptions, audio/speech}`
  - `/v1/responses` — full Responses API: string + structured `input`
    array (`message` / `function_call` / `function_call_output` /
    `input_text` / `input_image`), tools, tool_choice,
    `reasoning.effort` → reasoning_effort bucketing. Streaming SSE
    emits the Responses event family (`response.created`,
    `output_item.added`, `output_text.delta`,
    `reasoning_summary_text.delta`, `function_call_arguments.delta`,
    `output_item.done`, `response.completed`, `[DONE]`). Non-streaming
    emits `output[]` with `reasoning` / `message` / `function_call`
    blocks. See `Routes/OpenAIRoutes.swift` + `SSEEncoder.responsesStream`.
  - `/v1/audio/transcriptions` — Whisper multipart form (file, model,
    language, response_format, task, prompt, temperature). Formats:
    `json` (default), `text`, `verbose_json`, `srt`, `vtt`. Lazy-loads
    from `~/.cache/huggingface/hub`. Live-verified on
    `mlx-community/whisper-tiny-mlx`.
  - `/v1/audio/speech` — TTS returning real 24 kHz mono WAV. Currently
    ships `PlaceholderSynth` tone backend (advertised via
    `X-vMLX-TTS-Backend: placeholder-tone`). Kokoro neural backend
    scaffolded — not yet live.
- Ollama: `/api/{chat, generate, embeddings, embed, tags, show, ps,
  version, pull}` — `/api/chat` honors `tools` (fixed 2026-04-14).
- Anthropic: `/v1/messages` (streaming + vision blocks + `document` PDF
  / text-url / image-url, `server_tool_use`, `web_search_tool_result`).
- Admin: `/health`, `/admin/{soft-sleep, deep-sleep, wake,
  cache/stats, models/:id}` — `wake` replays `lastLoadOptions` and
  accepts `{model}` override.
- MCP: `/v1/mcp/{tools, servers, execute}`, `/mcp/:server/:method`
  (raw JSON-RPC 2.0 passthrough — body `{params:{...}}` or raw params
  dict, e.g. `resources/list`, `prompts/get`).

**SwiftUI app (`vMLX`)**
- 5 modes: Chat, Server, Image, Terminal, API
- Per-chat model picker, sessions sidebar, message bubbles with
  streaming + reasoning + tool-call cards + MetricsStrip
- Server tab with per-session HTTPServerActor wiring (real listener
  per session)
- Image tab with model picker + gen/edit forms + SQLite-backed gallery
- Terminal tab with auto-injected bash tool + Up/Down command history
- API tab with endpoint list + curl/Python/TS/Anthropic/Ollama
  snippets + LAN QR code

---

## Feature coverage

See `APP-SURFACE-AUDIT-2026-04-13.md` for the full per-surface
REAL/STUB/MISSING inventory with file:line anchors. Quick summary:

| Surface | REAL | STUB | MISSING |
|---|---|---|---|
| Chat (`vMLXApp/Chat/`) | 12 | 1 (edit+regenerate UI partial) | 0 |
| Server (`vMLXApp/Server/`) | 10 | 0 | 0 |
| Image (`vMLXApp/Image/`) | 5 | 1 (Flux `.generate()` bodies) | 0 |
| Terminal (`vMLXApp/Terminal/`) | 8 | 0 | 0 |
| API screen (`vMLXApp/API/`) | 7 | 0 | 0 |
| Settings (`vMLXEngine/Settings/`) | 4 | 0 | 0 |
| Engine (`vMLXEngine/`) | 10 | 1 (benchmark driver) | 0 |
| Routes (`vMLXServer/Routes/`) | 23 | 1 (TTS neural backend = placeholder tone) | 0 |
| CLI (`vMLXCLI/`) | 4 | 0 | 0 |
| MCP | 7 | 0 | 0 |
| Flash MoE | Phase 1 + 2a done; Phase 2b (model-side protocol conformance) + Phase 3 (engine wire-up) pending |

---

## Audit docs in this tree

- **`PROGRESS.md`** — full session-by-session changelog, newest at top
- **`APP-SURFACE-AUDIT-2026-04-13.md`** — per-surface REAL/STUB/MISSING
- **`SWIFT-ENGINE-ISSUES-AUDIT.md`** — GH issue cross-reference
  (`jjang-ai/vmlx` + `jjang-ai/mlxstudio`) against the Swift engine
- **`AUDIT-2026-04-13-POST-VENDOR.md`** — hybrid SSM + parser
  auto-dispatch + cross-cutting settings audit
- **`UX-AUDIT.md`** — UI polish findings
- **`SWIFT-NO-REGRESSION-CHECKLIST.md`** — per-release regression matrix
- **`Sources/vMLXLMCommon/FlashMoE/README.md`** — Flash MoE Phase 1 /
  2a architecture + Phase 2b / 3 roadmap

---

## Still remaining

Prioritized list in `PROGRESS.md`. Headline items:

- **Image gen `.generate()` bodies** — Flux/Qwen/Z/SeedVR2/FIBO DiT
  forward passes. Biggest user-visible gap. (FluxBackend.editImage
  wire-up landed 2026-04-14 — still needs model-side `.generate()`.)
- **vision_embedding_cache.py port** — per-image cache for VLM
  continuous batching.
- **MCP Phase 2** — wire MCP tools into `Stream.swift` tool dispatch.
- **Flash MoE Phase 2b** — per-model protocol conformance landed for
  OlmoE, LFM2MoE, GLM4MoE, BailingMoe, PhiMoE, NemotronH SwitchMLP,
  Gemma4 sibling layout, MiniMax, Mistral3 SwitchGLU (2026-04-14).
  Remaining families TBD — tracked in Flash MoE README.

### Recently fixed (2026-04-15 → 2026-05-02)

- **JangPress integration + axis-orthogonality regression guard.** 26-
  pin contract test family covers per-load tier (not per-chat),
  jang_config / generation_config orthogonality, SSM × JangPress (axis
  F vs E), VL multi-turn + audio (Parakeet/Whisper) compose, API
  surface (`GET /v1/cache/jangpress` + cacheStats), RAM accounting
  (mach_task_info quirk + phys_footprint canonical), M-chip support
  (no Metal/version gate), pct sweep × tile-pattern, CLI writeback
  symmetry. JANGPRESS-PRODUCTION.md ships the verification matrix.
- **mlx#3461 unretained-resource race patch** — vendored
  `Sources/Cmlx/mlx/mlx/backend/metal/device.cpp:405` switched from
  `commandBufferWithUnretainedReferences()` to retained
  `commandBuffer()`. Eliminates "Invalid Resource" races under
  TurboQuant KV cache load + Swift structured-concurrency MLXArray
  drop. Cost: ~2.4% throughput.
- **kvCacheQuantization ↔ enableTurboQuant symmetry** (mlxstudio#138
  parity). CLI flags `--disable-turbo-quant`, `--enable-turbo-quant`,
  `--kv-cache-quantization` now writeback BOTH the Bool and the
  canonical String so the resolver doesn't silently re-derive over
  the user's choice.
- **Engine.LoadOptions ↔ GlobalSettings alignment.** Bare-init
  defaults match GlobalSettings tier-1 (`prefillStepSize=2048`,
  `maxCacheBlocks=1000`, `kvCacheQuantization="turboquant"`). 3
  production call sites no longer silently ship stricter caches.
- **NSOpenPanel macOS 26 XPC fix (vmlx#121, #133).**
  Iter 128-129. `Sources/vMLXApp/Common/NSOpenPanelSafe.swift`
  detects XPC failure (no UI rendered in <50ms) and falls back to a
  text-input alert. Applied to all 6 NSOpenPanel call sites
  (ModelDirectoriesPanel, TerminalScreen, SessionConfigForm,
  MCPPanel, AdvancedServerCard, ChatSettingsPopover). Ad-hoc-signed
  dev builds and downloaded-from-source builds no longer silently
  fail their "Add directory…" buttons.
- **MCP headers field (vmlx#131).** Iter 130. MCPServerConfig now
  carries `headers: [String: String]?`; both SSE startup GET and
  per-message POST iterate it. Required for Exa / GitHub / other
  auth-gated remote MCPs. Round-trip + Claude-Desktop
  `mcpServers.<name>.headers` decoding pinned.
- **JANG dense Hadamard dispatch bug.** Fixed instance `__call__`
  override + `dir()` walker miss that caused garbage on JANG dense
  bundles converted with `--hadamard` and bits>=3 (Qwen3.6-27B-JANG_4M-
  CRACK, Gemma 31B). Synced to all 3 bundled-python locations.
- **JANGTQ-VL thread fix.** `mllm-worker_0` dedicated single-worker
  executor — fixes Stream(gpu, 1) Metal crashes on Qwen3.6-MoE-VL
  JANGTQ bundles.
- **DSV4 Jinja kwargs.** Template only branches on
  `reasoning_effort=='max'`; 'high' was a silent no-op. Fixed in 3
  endpoint paths.
- **JANGTQ MXTQ PRNG parity** — `NumPyPCG64.swift` ships bit-
  identical PCG64 port; `JangMXTQDequant` uses it for sign
  generation. The older `TQHadamard.generateRandomSigns` drand48
  path is deprecated with a "do NOT revert" marker. MiniMax-M2.7-
  JANGTQ-CRACK decodes at 46.59 tok/s (Python ref 44.3).
- **JANGTQ `mxtq_bits` per-role dict** — §346. `JangLoader` accepts
  both a flat `Int` and a per-role `{shared_expert, routed_expert,
  ...}` dict, which unblocks Qwen3.6-JANGTQ4.
- **Nemotron Jinja templates** — §341. Jinja runtime now correctly
  evaluates `not in` operator for Nemotron/Cascade chat templates.
- **Thinking-leak audit** — §343/§344/§345. Qwen3.6, Gemma4, MiniMax
  reasoning parsers drained at EOS via `parser.finishStreaming`; §15
  reasoning-off → `.content` reroute verified across families.
- **MCP one-click import** — §340. Paste a Claude Desktop
  `mcpServers` JSON blob; the app imports allowlisted entries.
- **JSON-Schema tool argument coercion** — §338 (vmlx#47). Tool
  args are coerced to declared schema types before dispatch.
- **Image model folder discovery** — §339 (mlxstudio #82/#85/#96).
  Image tab auto-detects Flux / Qwen-Image / Z-Image layouts.

### Known deferred items (2026-04-14)

- **Kokoro neural TTS backend** — scaffolded in
  `Sources/vMLXTTS/Kokoro/KokoroBackend.swift` with a 9-step port plan.
  `/v1/audio/speech` currently returns deterministic WAV tones from
  `PlaceholderSynth` (see `X-vMLX-TTS-Backend: placeholder-tone` response
  header).
- **TTS audio transcoding** — mp3 / opus / flac formats; only WAV emitted.
- **Whisper temperature fallback** — single-temperature greedy only. No
  beam search, no word-level timestamps, no sliding-window for audio
  >30 s, no live verification against v3 (`n_mels=128`).

**Speculative decoding (JANG-DFlash):** LIVE in the dev branch as of
2026-04-15. Full integration in `Sources/vMLXLMCommon/DFlash/` with
`Engine` lifecycle + CacheCoordinator plumbing + admin routes + UI
toggle. Target families with tap adapters: MiniMax, Mistral 4, DeepSeek
V3. Settings: `dflash`, `dflashDrafterPath`, `dflashBlockSize` (16),
`dflashTopK` (4), `dflashNumPaths` (60), `dflashTapLayers`,
`dflashTargetHiddenDim`. CLI: `vmlxctl serve --dflash --dflash-drafter
PATH`. Admin: `GET /admin/dflash`, `POST /admin/dflash/{load,unload}`.
Fallback is silent + logged when drafter/target/tools preclude the
fast path. See `PROGRESS.md` for the full integration log.

**Smelt mode:** Honest UX — the flag flows through settings but the
Swift engine has no partial-expert-loading consumer yet (Python-only
feature). Setting `smelt=true` logs a one-shot warning per request
so users aren't silently no-op'd. Label in SessionConfigForm reads
"Smelt mode (Python engine only)".

---

## Archival notes

optimization, softplus 7.62 → 3.69 µs, Python parity). That tree is
kept as a git-history reference only — it has no origin remote and
its push URL is set to an invalid string so it cannot accidentally
be published. Future kernel edits land in
`swift/Sources/Cmlx/` + `swift/Sources/MLX*/` in this repo and
commit into this repo's git history.
