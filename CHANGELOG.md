# Changelog

All notable changes to vMLX Engine will be documented in this file.

## [Unreleased]

### Fixed
- Fixed the vMLX MiMo-V2.5 MLLM language-model adapter so `mlx_vlm` text turns can pass `inputs_embeds` and receive a `.logits` output object instead of failing with HTTP 500. MiMo long-output quality remains a separate blocker.

### Notes
- Future release notes for the current runtime/model/UI/API issue wave must credit GitHub `@Hornsan1` for reporting many of these issues.

## [1.5.49] - 2026-05-23

### Fixed
- Shipped the current Responses tool-call parser fix so non-streaming
  `/v1/responses` extracts real tool calls emitted inside reasoning text before
  finalizing output items.
- Bundled the materialized DSV4 CSA/HCA pool codec fix so DSV4 pool-quant reads
  reuse the cached materialized pool instead of dequantizing and concatenating
  historical pool segments on every attention read.
- Made DSV4 Flash native composite prefix cache and the materialized pool codec
  the default DSV4 launch path while keeping generic KV q4/q8 suppressed for
  DSV4 and preserving explicit per-session disable.
- Refreshed panel settings, CLI preview, i18n "What's New", cache architecture,
  API/cache, MTP, and VL/media gates so cache defaults, DSV4-only controls, and
  command preview wiring stay aligned.

### Known Follow-ups
- DSV4 Flash long full-output/code-generation quality remains a documented
  release exception pending separate exact-code/runtime-quality clearance.

## [1.5.48] - 2026-05-22

### Fixed
- Aligned Qwen3.6 affine-JANG native-MTP VL routing across engine registry,
  decode-speed launch rows, panel detection, and API policy. Indexed native-MTP
  VL artifacts with real vision and MTP tensors now use the multimodal route,
  while plain affine-JANG Qwen VL remains text-only until the M-RoPE fallback
  issue is cleared.
- Added panel local-path parity coverage for high-risk DSV4, Qwen, Hy3, and
  Nemotron artifacts so parser, reasoning, cache, modality, and launch policy
  cannot silently diverge between the UI and engine.
- Added post-release guards proving explicit Chat/Responses output caps do not
  mutate server startup defaults, and adjusted the release-surface gate to
  accept complete post-release updater state.

### Known Follow-ups
- DSV4 Flash long full-output/code-generation quality remains a documented
  release exception pending separate exact-code/runtime-quality clearance.

## [1.5.47] - 2026-05-21

### Fixed
- Added focused release-regression coverage for PR intake and runtime
  compatibility rows, including max output/context wiring, generation defaults,
  parser parity, cache architecture, native MTP, MCP, VL media cache, packaging,
  and live-only model family gates.
- Hardened built-in coding-tool path resolution so new nested writes are still
  allowed inside the active working directory while traversal and symlink-parent
  escapes remain blocked. Credit: @tomaioo for PR #170 and the arbitrary
  file-write report.
- Documented and tested Unix-domain socket serving via
  `vmlx-engine serve --uds /tmp/vmlx.sock`. Credit: @efortin for PR #168.
- Added a no-heavy JANG compatibility contract for the MiniMax sanitize and
  MoEGate quantize failure class without installing the submitted global
  monkeypatch. Credit: @pperezrubio for PR #155.
- Confirmed the older-engine `--default-repetition-penalty` launch failure is
  superseded by the current no-hidden-sampler-defaults path: stale generic
  `1.10` defaults are migrated away and session launch no longer emits default
  sampler flags from displayed bundle metadata. Credit: @Rishirandhawa for
  PR #77 and the external-engine compatibility report.
- Confirmed VLM JIT keeps `LanguageModel.layers` available through the current
  compiled-module proxy instead of installing a bare `mx.compile` function.
  Credit: @st-adam for PR #154 and the VLM JIT regression report.
- Added no-heavy prefill-loop contracts and accepted the safe slice of PR #163:
  sorted SSM boundary lookup, precomputed state-layer evals, hoisted token-list
  materialization, and an opt-in `--prefill-keep-alloc` /
  `VMLINUX_PREFILL_KEEP_ALLOC=1` chunked-prefill allocator tuning flag. Credit:
  @st-adam for the prefill-loop cleanup work.
- Added no-heavy contracts and accepted the safe cache/index slice of PR #162:
  `MLLMBatch` UID lookup caching for PLD, removal of the redundant SSM
  `lengths * 1` materialization kernel, and removal of duplicate L2 SSM disk
  post-load deepcopy. The submitted sampler-helper hoist was intentionally not
  accepted because current vMLX uses `vmlx_engine.sampling`. Credit: @st-adam
  for the hot-path cleanup work.

## [1.5.46] - 2026-05-20

### Fixed
- **Hy3 Auto/Off requests no longer inherit stale thinking effort**: saved High
  reasoning effort is only forwarded when thinking is explicitly enabled, so
  Auto and Off requests stay model-owned instead of sending old
  `enable_thinking` / `reasoning_effort` state.
- **Legacy session output caps are cleared on startup**: old generic
  `maxTokens` values such as 4096, 12000, 12068, and 32768 are reset to
  model-owned output length, while prompt/context length remains controlled by
  `maxContextLength` / `--max-prompt-tokens`.
- **MiniMax M2 reasoning parser detection is canonical**: MiniMax M2/M2.5/M2.7
  now uses `minimax_m2` across engine registry, stale sidecar overrides, panel
  parser aliases, and CLI launch arguments.

### Verified
- Focused panel request/settings/gateway/MCP tests passed: 496 passed, 3
  skipped; expanded panel release wiring passed: 859 passed, 3 skipped; panel
  TypeScript typecheck passed.
- Focused Python release/server contracts passed: 68 passed; expanded Python
  release slice passed: 1120 passed, 45 skipped; diff whitespace check passed.
- Live source Hy3 JANGTQ2 gate passed with paged/TurboQuant cache hit evidence
  and exact multi-turn recall (`color=blue, animal=cat`).
- Live source MiniMax-M2.7-Small-JANGTQ smoke passed with `tool_parser=minimax`,
  `reasoning_parser=minimax_m2`, paged/TurboQuant cache hits, and multi-turn
  recall.
- Packaged 1.5.46 Sequoia and Tahoe DMGs were Developer ID signed, notarized,
  stapled, Gatekeeper accepted, and passed the mounted-app release gate against
  the bundled Python/JANG runtime.

### Known Follow-ups
- DSV4 Flash long full-output/code-generation production claims remain blocked
  by the existing exact-code/runtime-quality issues documented in v1.5.45.

## [1.5.45] - 2026-05-20

### Fixed
- **DSV4 DSML and XML tool prompts now include concrete examples for real tool
  names without inventing fake arguments**: zero-argument built-in tools render
  as empty invokes, and schema-only DSML prompts get concrete per-tool examples
  so DSV4 can chain built-in and MCP tools cleanly.
- **Ollama `num_predict` disabled sentinels stay model-owned**: `0`, `-1`, and
  `-2` are no longer forwarded as invalid `max_tokens`, while positive
  `num_predict` values and other sampler overrides still pass through.
- **VLM video capability reporting is no longer guessed from vision support**:
  image-only VLMs such as ZAYA1-VL now report `text, vision` and reject
  `video_url` requests clearly instead of reaching a processor crash. Qwen-style
  VLM bundles with explicit video metadata continue to report video support.
- **Top-k and disabled sampling sentinels remain hidden/omitted consistently**
  across the panel and Ollama gateway paths.
- **Kimi K2.6 tokenizer rendering now has its required runtime dependency**:
  `tiktoken` is a core dependency, not audio-only, so Kimi chat-template
  rendering works in source and bundled Python environments.
- **Bundled app packaging strips third-party `.agents` metadata** from
  site-packages so release artifacts do not carry agent/skill sidecar files
  that are not used at runtime.
- **Omitted `max_tokens` now resolves to a bounded model/default budget**:
  explicit request/session overrides still win, bundle `max_new_tokens` is
  honored when present, and no-request/no-bundle-default paths use the bounded
  4096 fallback instead of the old hidden 32768 value.
- **MCP config auto-discovery now runs at startup without a launch flag**:
  `mcp.json` / `mcp.yaml` discovery covers the working directory and user config
  locations, and MCP only initializes when the discovered config contains
  servers.
- **Tool-marker leak cleanup now follows the shared marker registry**: DSML,
  Hunyuan/XML, ZAYA/Zyphra, and related partial tool-call prefixes are handled
  through the same marker list instead of one-off regex coverage.
- **Text-chat chained tool continuation resets streaming state between rounds**:
  follow-up requests clear accumulated tool buffers and parser state, while a
  stall watchdog clears the visible "Generating tool call" state if a buffered
  stream never finishes a tool envelope.

### Verified
- Focused backend checkpoint suite passed: tool-prompt fallback, Ollama
  adapter/defaults, API surface parity, server sampling/max-token guards, media
  diagnostics, and local-model smoke harness contracts.
- Focused panel gateway suite passed for Ollama request translation, including
  disabled `num_predict`/`top_k` omission and explicit override forwarding.
- Focused release-blocker backend suite passed for MCP policy, tool formatting,
  DSV4 contracts, DSV4 paged cache, cross-matrix audits, and engine audit
  coverage.
- Focused panel release-blocker suite passed for update checking, settings
  flow, DSV4 environment wiring, i18n consistency, tool auto-continue, tool
  status responsiveness, chat UI grouping, and reasoning display.
- Live DSV4 and Hy3 Electron UI proofs exercised built-in plus MCP tools in one
  chat with no raw tool-marker leak.
- Live Hy3 JANGTQ_K A/B did not reproduce Chinese visible output under bounded
  English thinking-off prompts and showed paged/TurboQuant cache hits.
- Live ZAYA1-VL JANGTQ_K checkpoint now reports `text, vision` and skips video
  probes instead of advertising unsupported video; blue-image, no-media
  follow-up, multi-turn recall, paged ZAYA-CCA cache hit, and block-L2 paths
  ran through the real server.
- Final release-gate rerun passed the full backend suite, full panel test
  suite, TypeScript typecheck, and bundled-Python/Electron build.
- The packaged app was rebuilt after the v1.5.45 localized "What's New" notice
  update; the fresh bundle contains MCP/MTP/latest.json release text and passed
  Developer ID signing, notarization, strict codesign, Gatekeeper assessment,
  bundled source hash checks, and packaged GUI launch.

### Known Follow-ups
- DSV4 Flash JANGTQ-K is not production-cleared for long full-output prompts:
  the packaged app still passes short/API/tool/cache/composite-cache gates, but
  fresh long VC/code and game-design probes hit `finish=length`; thinking-on
  game-design stayed entirely in reasoning at the 4096-token model-owned
  default budget. Follow-up exact JavaScript identifier probes also corrupted
  Three.js API names even when tokenizer round-trip was clean and sampling was
  deterministic with `repetition_penalty=1.0`; a token-logprob probe showed
  the duplicate `Web` token is generated before response parsing in the chat
  path. The canonical DSV4 chat prompt was rendered separately and preserved
  the requested Three.js identifiers exactly; sending that exact canonical
  prompt through raw completions reproduced the same duplicate `Web` token.
  Single-shot and two-phase DSV4 prefill produced the same result, ruling out
  the prefix-cache N-1 prefill split as the cause. A minimal canonical prompt
  can still emit `WebGLRenderer` correctly, but a fresh identifier-count
  ablation shows the failure starts as soon as a preceding `THREE.Scene` line is
  added: `THREE.Scene` becomes `THREE.ScScene` and `THREE.WebGLRenderer`
  becomes `THREE.WebWebGLRenderer`. Putting the renderer line first preserves
  `WebGLRenderer` but then degrades `Scene` to `THREE.Sc`. Logits on the full
  snippet strongly prefer duplicate `Web` over `GL` after `.Web`. Header-only artifact
  inspection shows the local source bundle has BF16 `head.weight`, while the
  local JANG and JANGTQ-K artifacts both use a quantized output head. A local
  BF16 output-head/final-norm overlay over the JANGTQ-K body still reproduced
  the duplicate `WebWebGLRenderer` failure, so output-head restoration alone is
  not a fix. The matrix now records the precision boundary as a static
  prerequisite. A separate runtime-config audit found the local converted DSV4
  JANG/JANGTQ-K artifacts had `rope_scaling: null` even though the source
  DeepSeek-V4-Flash config requires YaRN scaling for compressed-context layers;
  vMLX now repairs that metadata before DSV4 model construction, but the live
  identifier gate remains the release blocker. After patching the local JANG
  and JANGTQ-K configs to preserve source YaRN `rope_scaling`, a fresh source
  runtime gate on `DeepSeek-V4-Flash-JANG` produced real `paged+dsv4`
  cached-token evidence but still failed deterministic cache equivalence:
  temperature-0 cached follow-up text differed from the `skip_prefix_cache`
  control. DSV4 long code/full-output production claims require source-vs-quant,
  composite-cache equivalence, or a broader rebuilt-artifact/runtime clearance.
  The same short identifier-count gate also fails on the local affine
  `DeepSeek-V4-Flash-JANG` artifact, so the open exact-code blocker is not
  isolated to JANGTQ/TurboQuant routed-expert matmul, parser output, endpoint
  assembly, or cache reuse.
  Because the live `paged+dsv4` path is not byte-equivalent yet, vMLX now
  starts DSV4 Flash with composite prefix/paged/L2 cache reuse disabled by
  default across CLI and the Electron panel. The native composite cache remains
  available only through the explicit diagnostic opt-in
  `--dsv4-enable-prefix-cache` / `VMLX_DSV4_ENABLE_PREFIX_CACHE=1` /
  the panel's "DSV4 Composite Prefix Cache" setting. The local source DSV4
  bundle is larger than this host's RAM, so source-vs-quant live comparison was
  not run here.
- ZAYA1-VL JANGTQ_K still needs a dedicated color/probe investigation: the
  solid-blue image probe answers blue, but the solid-red image probe answered
  white in the current live smoke.
- ZAYA1-VL exact `ACK` obedience in the broad smoke harness remains too weak
  for a pass, although multi-turn recall and cache-hit behavior were coherent.
- Full VLM/video/audio proof remains open for families that genuinely declare
  video/audio support; this release only prevents unsupported image-only VLMs
  from crashing on video requests.

## [1.5.44] - 2026-05-19

### Fixed
- **Mistral Small 4 VLM loads through the real multimodal path again**:
  Mistral3/Mistral4 wrapper detection now preserves vision routing, keeps the
  VLM loader from stripping the active `kv_b_proj`, preserves image token
  templates when processors lack an inline chat template, and sends local image
  paths in the nested shape Mistral/Pixtral processors expect.
- **ZAYA native XML tool streaming no longer leaks raw partial tool markup**:
  Chat Completions and Responses streams now signal tool-call generation while
  buffering partial `<function` / `<tool_call` / Zyphra XML markers, and the
  panel suppresses raw markup while showing an immediate tool status.
- **Interleaved reasoning display handles live replacement and final show-all
  separately**: live streaming shows the latest active reasoning segment while
  completed messages can still expose all visible reasoning segments.
- **Coding-tool config saves are durable and model-derived**: config writers
  preserve backups and force private config file permissions after writing.
- **macOS MLX wheel platform selection is explicit for release packaging**:
  Sequoia-compatible and Tahoe-native bundle lanes pass the intended
  `VMLX_BUNDLE_MLX_PLATFORM` value, while the legacy misspelling remains
  accepted for compatibility.

### Verified
- Live ZAYA1-8B-JANGTQ_K source-server tool-stream proof returned a parsed
  `read_file` tool call with `tool_call_generating` streaming status and no
  visible raw tool XML.
- Live Mistral Small 4 VLM source-server proof returned correct text, red-image,
  and blue-image responses through the multimodal path.
- Live MiniMax-M2.7-JANGTQ_K and DSV4 tool-call probes passed during the
  hardening sweep, including cache-detail and native-cache checks.
- Focused C2 regression gates covered closed GH regressions for hybrid
  TurboQuant KV policy, explicit KV quantization, VLM JIT wrapper preservation,
  Kimi MLA patching, MLLM thread-local stream handling, DSV4 EOS/paged cache,
  Responses system-message normalization, reasoning policy, Nemotron Omni
  packaging, Mistral Small 4 VLM, and Sequoia/Tahoe wheel selection.
- API/cache/MTP/server and panel settings/tool/reasoning/gateway matrices
  passed, and the bundled Python/Electron build verified source parity and
  critical runtime imports.

## [1.5.43] - 2026-05-19

### Fixed
- **Native Qwen3.6 MTP app/API wiring is release-gated on real runtime state**:
  Qwen3.6 MTP artifacts now expose native-MTP status through health and UI
  surfaces, preserve the text+VL loader path, and use validated model-local
  depth tuning when present instead of a blanket depth override.
- **Chat settings, cache controls, and tool auto-continue are integrated across
  the panel and engine**: cache policy controls remain user-visible and
  user-controlled, interleaved reasoning segments are preserved separately from
  visible text, media follow-up payloads keep image/video content, and tool
  auto-continue no longer leaks raw think or tool tags into the chat.
- **Sampling defaults stay model-owned until users override them**: DSV4 JANG
  chat defaults, Qwen generation defaults, MiniMax generation defaults, and
  request-level OpenAI/Responses/Ollama kwargs resolve at request time without
  hidden per-session sampler writes.
- **Model-family cache policy is explicit and fail-closed**: standard KV
  families can use live TurboQuant KV, DSV4 keeps generic KV quantization off
  for its native SWA/CSA/HCA composite cache, ZAYA uses typed CCA state, and
  Qwen3.6 hybrid SSM now uses live TurboQuant KV only for attention layers while
  SSM/ArraysCache companion state remains native full precision. The prior
  global hybrid make-cache patch stays disabled because it cannot preserve
  path-dependent companion state.
- **Hugging Face GUI downloads recover from stale backup endpoints and stale
  tokens**: the bundled worker can fall back to unauthenticated metadata and
  direct Hub resolution when user credentials or cached endpoint state are bad.
- **macOS packaging lanes are split by platform compatibility**: release scripts
  now keep Sequoia/Sonoma-compatible and Tahoe-native DMG lanes separate, with
  startup checks that fail before MLX import when the wrong wheel tag is bundled.

### Verified
- Current-head live gates covered Qwen3.6-27B MXFP4-MTP, MXFP8-MTP, and
  JANG_4M-MTP native-MTP generation, acceptance telemetry, text+VL detection,
  hybrid SSM cache handling, selective live attention TQ-KV, and stored q4
  attention-KV snapshots.
- Task 8 parity gates covered MXFP4, MXFP8, and JANG_4M hybrid live TQ-KV:
  selective attention-layer TurboQuant exactly matched full-precision live KV
  on deterministic probes while preserving 48 native companion layers.
- Task 8 MTP/VL reruns covered MXFP4 D2, MXFP8 D3, and JANG_4M D3 with native
  MTP active, text+VL scope, visible non-empty output, and selective live
  attention TQ-KV active.
- Real Electron/CDP chat proof covered reasoning streaming, run/list/image/video
  tool calls, tool auto-continue, media follow-up content, cache UI toggles, and
  absence of raw think/tool-tag leakage.
- Focused Python and panel tests cover DSV4 cache policy, DSV4 reasoning
  defaults, Qwen native-MTP controls, generation defaults, request-builder
  sampler passthrough, Ollama options mapping, Hugging Face fallback downloads,
  and macOS wheel-tag startup guards.

## [1.5.41] - 2026-05-18

### Fixed
- **DSV4 DSML streaming tool calls no longer leak wrapper text**: streaming
  requests now buffer the full DSV4 `<｜DSML｜tool_calls>` envelope before
  emitting structured tool-call deltas, including the leading-whitespace case
  that previously surfaced raw DSML marker text as assistant content.
- **DSV4, Qwen native-MTP, and ZAYA native-cache release gates were refreshed
  against packaged bundled-Python builds**: the fixed app reports the correct
  native cache schemas, keeps generic TurboQuant KV disabled for incompatible
  composite/hybrid caches, and preserves storage-boundary cache telemetry.

### Verified
- Live DSV4 Flash DSML parser gate passed for non-streaming tool calls,
  tool-result roundtrip, and streaming `tool_choice=required` with no raw DSML
  leakage.
- Packaged smoke gates passed for DSV4 Flash native cache, Qwen3.6 native-MTP
  hybrid cache, and ZAYA CCA cache lifecycle paths.

## [1.5.40] - 2026-05-17

### Fixed
- **Native MTP now honors validated model-local tuning by default**:
  `vmlx_mtp_tuning.json` sidecars select the measured depth before falling
  back to D3, while explicit CLI/UI overrides still win. This fixes
  Qwen3.6-27B-MXFP4-MTP launching at D3 when the validated local sweep selects
  D2.
- **Unvalidated native-MTP profiles fail closed in the app**: blocked sidecars
  and Qwen3.6 JANG_2K diagnostic artifacts no longer expose native-MTP launch
  controls unless explicitly force-enabled for research.
- **Hybrid SSM cache reporting is explicit**: live generic TurboQuant KV remains
  disabled for path-dependent hybrid models, while q4/q8 attention-KV storage
  quantization is reported separately for prefix/paged/L2 cache boundaries with
  SSM companion state and async clean-prefill rederive.

### Verified
- Packaged app release gate passed on Qwen3.6-27B-MXFP4-MTP from bundled
  Python: GUI launch, OpenAI Chat, Responses, Anthropic, Ollama, multi-turn
  recall, cross-request cache hit, cache stats, and soft wake.
- Focused packaged media/speed gate passed: native MTP D2 text+VL detection,
  260-token count row at 50.78 wall tok/s with 171/174 accepted draft tokens,
  red image then no-image text follow-up, red video then no-video text
  follow-up, q4 attention-KV storage telemetry, SSM companion L2 state, and deep
  sleep/wake reload.

## [1.5.39] - 2026-05-17

### Added
- **Native MTP artifact autodetection and health reporting**: Qwen 3.6 text/VL
  bundles with preserved `mtp.*` tensors now report MTP availability, depth,
  scope, tensor counts, and gating reasons through health/status surfaces.
- **Production MTP runtime gates**: native-MTP launch policy now fail-closes
  unless the bundle metadata, tensor evidence, loader route, and cache safety
  checks agree. Supported artifacts default to D3; model-local tuning sidecars
  are opt-in only via `VMLINUX_NATIVE_MTP_USE_TUNING=1`. Preserved-only
  artifacts stay autoregressive unless explicitly verified for native
  self-spec.

### Fixed
- **MTP/settings wiring across CLI, API, and app launch**: stale speculative
  settings are stripped for incompatible families, model-driven defaults stay
  authoritative, and runtime flags are not persisted into unrelated sessions.
- **Qwen 3.6 text+VL loader/runtime parity**: text and vision-capable Qwen
  artifacts share the same MTP detection path while preserving VL assets and
  hybrid cache requirements.
- **Qwen 3.6 MTP+VL app detection parity**: the panel/session detector now
  keeps artifact-backed affine-JANG Qwen MTP+VL bundles on the multimodal path
  when indexed MTP and vision tensors are present, instead of applying the
  older text-only affine hybrid guard.
- **Cache and status visibility**: native cache and MTP status details are
  surfaced without enabling generic TurboQuant KV paths for incompatible
  model-family cache schemas.
- **Source-runtime safety for Kimi/Gemma VLMs**: PyPI/source installs now apply
  the Kimi K2.6 DeepSeek-V3 MLA patch and Gemma 4 mixed `pixel_values` coercion
  at runtime, matching the protections already applied to bundled DMGs.
- **DSV4/JANG guard hardening on main**: current main also includes defensive
  settings and metadata gates for DSV4 affine/JANG artifacts, but DSV4 quant
  research artifacts are not part of this release claim.

### Verified
- Focused Python tests cover native MTP autodetect, MTP policy, bench harness,
  sampler/research helpers, JANG loader metadata, JIT toggles, DSV4 contract
  hardening, SSM companion cache, and multimodal routing surfaces.
- Focused panel tests cover model-config registry, session/settings flow,
  chat settings compatibility, and MTP/native-cache performance display.

## [1.5.37] - 2026-05-16

### Fixed
- **Model generation defaults are no longer copied into hidden session or
  per-model state**: new chats start with no sampling/thinking overrides, and
  startup no longer emits `--default-temperature`, `--default-top-p`,
  `--default-top-k`, `--default-min-p`, `--default-repetition-penalty`, or
  `--default-enable-thinking` from panel metadata. The engine resolves
  `jang_config` / `generation_config.json` per request.
- **Per-chat max output tokens remain user-controlled**: Chat Settings shows
  the model-declared `max_new_tokens` as the default placeholder, but leaves the
  field unset until the user saves a chat override. Explicit `max_tokens` /
  `max_output_tokens` values are respected exactly; no family-specific hidden
  floors are applied.
- **Stale local SQLite sampling rows are cleared once**: historical generic
  values such as temperature `0.7`, top-p `0.95`, top-k `40`, and max-token
  caps `4096`, `12000`, or `12068` are reset to bundle defaults. Per-model
  sampling/thinking write-back is removed.
- **ZAYA1-VL explicit reasoning-on now opens the qwen3 think rail** when the
  VLM processor template is plain and the bundle declares qwen3 reasoning
  support. Auto/off remain unchanged.
- **DSV4 DSML tool-call argument recovery**: canonical DSV4 parser results with
  missing required arguments or raw DSML/HTML-ish markup now fall through to the
  repair parser, which can recover plain `<param name="...">...</param>` bodies.
- **Top-k sampling uses a compact logits path** for bounded top-k requests so
  large-vocabulary models do not sample over a masked full vocabulary.

### Verified
- Python focused gate: sampling, reasoning modes, DSML parser, cache bypass,
  and worker-dequant tests passed.
- Python audit gate: reasoning modes, DSML parser, engine audit, cache bypass,
  and worker-dequant tests passed.
- Full panel Vitest suite passed.

## [1.5.36] - 2026-05-16

### Fixed
- **Packaged engine assets now ship in the Python wheel and bundled app
  `site-packages`**: `vmlx_engine/chat_templates/*.jinja`,
  `vmlx_engine/config/*.yaml`, and `vmlx_engine/metal/*.metal` are declared as
  setuptools package data. This fixes the 1.5.35 packaging gap where the
  installed package had the Stream(gpu,0) runtime fix but omitted the Gemma 4
  fallback chat template, default YAML config, and codebook Metal kernels.
- **Release gates now cover non-Python engine assets**: both the bundled-Python
  verifier and installed-app release gate hash the package-data files against
  source so future releases cannot silently pass with a Python-only source hash.

### Verified
- Built wheel inspection confirms the package-data files are present before
  upload.
- Bundled-Python verification now fails if the app's installed `vmlx_engine`
  copy is missing or drifting from these release-critical assets.

## [1.5.35] - 2026-05-15

### Fixed
- **Single-active cache-hit Stream(gpu,0) crash**: q4/q8 memory-aware prefix,
  legacy prefix, and disk L2 cache hits now defer stored-cache dequantization to
  the scheduler worker stream instead of materializing MLX arrays on the API
  thread.
- **SingleBatchGenerator cache replay stream ownership**: replay tensors are
  rehomed inside the generator-owned MLX stream before sampling/evaluation.
  This fixes the MiniMax/JANG interleaved-reasoning reproduction where a second
  cached request aborted with `RuntimeError: There is no Stream(gpu, 0) in
  current thread.`

### Verified
- Focused regression suite passed for cache-hit worker dequantization,
  single-active batch generation, and cache-bypass behavior.
- Installed-app MiniMax-M2.7-JANG_2L-CRACK smoke passed Chat, Responses, and
  streaming cache-hit paths with no Stream(gpu,0), engine-loop, traceback, or
  invalid-resource errors.
- DMG was signed, notarized, stapled, and accepted by Gatekeeper.

## [1.5.25] - 2026-05-07

### Fixed
- **ZAYA JANGTQ/MXFP4 conversion routing**: `vmlx convert --jang-profile ...`
  now detects `model_type=zaya` and routes through the ZAYA JANGTQ converter,
  including generic `JANG_*` profile aliases such as `JANG_4L -> JANGTQ4`.
  `vmlx convert --bits 4` routes ZAYA through the MXFP4 converter instead of
  the generic affine path.
- **ZAYA no-state MoE layers**: odd ZAYA layers do not carry CCA state. The
  runtime now uses an explicit no-state cache object for those layers instead of
  `ArraysCache(1)`, preventing post-generation cache extraction from slicing
  `None`.
- **Desktop reasoning Auto default**: new and migrated sessions now preserve
  Auto reasoning as `NULL`/omitted instead of forcing `enable_thinking=false`.
  The local request builder omits `enable_thinking` and
  `chat_template_kwargs.enable_thinking` in Auto mode so model-family registry
  detection can choose the correct template behavior.
- **Cached reasoning-state recovery**: upgraded users with old SQLite
  `chat_overrides`, starred profiles, or per-model `reasoning_mode` rows no
  longer keep poisoning new requests with explicit thinking On/Off. A one-time
  migration resets those legacy choices back to Auto; post-upgrade user changes
  are preserved.
- **Ollama parity through both server and desktop gateway**: direct Python
  `/api/chat`/`/api/generate` and the Electron API gateway now accept native
  `think` plus the vMLX `enable_thinking`, `reasoning_effort`, and
  `chat_template_kwargs` extensions. Gateway `/api/generate` now follows
  Ollama semantics: chat-template route by default, raw completions only when
  `raw:true`.
- **ZAYA reasoning Auto now follows the production capability contract**:
  ZAYA/CCA is tools-capable but not yet reasoning-capable in the Python runtime
  gates. The engine and panel now treat stale `supports_thinking=true` bundle
  stamps as incompatible for ZAYA, avoid auto-installing a Qwen3 reasoning
  parser, and resolve Auto to `enable_thinking=false` so visible content is not
  lost inside an unfinished reasoning block.

### Verified
- ZAYA CLI conversion live-verified for JANGTQ4 and MXFP4 from the local
  `Zyphra/ZAYA1-8B` source bundle. The JANGTQ4 output loaded through
  `vmlx serve` and passed OpenAI Chat, Responses, Anthropic Messages, Ollama
  chat, and multi-turn recall smoke tests. TurboQuant KV stayed disabled for the
  CCA cache topology.
- Rebuilt packaged app live-gated ZAYA JANGTQ4 in both Reasoning Auto and
  explicit thinking-off modes. Both gates passed OpenAI Chat, Responses,
  Anthropic, Ollama, multi-turn recall, cache-stat/memory observation, and JIT
  soft sleep/wake. Evidence was captured in private release-gate artifacts.
- Qwen3.6 reasoning Auto live-verified with the desktop request shape: Auto,
  explicit Off, and explicit On all produced visible final content; reasoning
  was present only for Auto/On, and prefix/paged/block-disk cache stats were
  observed on the same session.
- Targeted parity tests passed for the direct Python Ollama adapter, API
  surface translations, Electron Ollama gateway, panel request builder,
  reasoning display, and cached-state migration.

## [1.5.6] - 2026-05-02

### Fixed (DSV4-Flash 14/14 — final)
- **DSV4-Flash + JANGTQ now passes the FULL probe matrix (14/14)**, including reasoning ON (17×23=391), tools (`get_weather` returns valid tool_calls), `enable_thinking=true` with reasoning + content, and sleep/wake roundtrip with coherent post-wake reply. Three remaining v1.5.5 fails fixed by:
  1. **`vmlx_engine/cli.py`** auto-detects DSV4 family at startup and sets `DSV4_POOL_QUANT=0` so `make_cache()` returns `DeepseekV4Cache` (not the `PoolQuantizedV4Cache` peer class). The Compressor + Indexer activation in `DeepseekV4Attention.__call__` gates on `isinstance(cache, DeepseekV4Cache)` and PoolQuantizedV4Cache is NOT a subclass — so the isinstance check returned False, the tri-mode (HSA+CSA+SWA) attention path stayed dormant, the sliding-window cache overflowed at 128 tokens, and every reasoning chain crashed mid-decode with `broadcast_shapes`. Setting it at CLI startup (before the engine loads the model) guarantees the warmup pass + every subsequent `make_cache()` picks up DeepseekV4Cache. Trade-off: lose ~4 GB pool-quant memory; gain end-to-end reasoning + multi-turn long context. Override: explicitly set `DSV4_POOL_QUANT=1` to opt back into the pool-quant path (only safe for ≤128-token contexts).
  2. **`vmlx_engine/server.py`** force-flips `enable_thinking=True` for the DSV4 family on all 4 endpoints (chat-completions, Ollama, Anthropic, Responses) when the client sent anything OTHER than explicit `True` (was: only when explicit `False`). DSV4 chat-mode (`enable_thinking=False`) produces "training-data contaminated" output: hallucinated AI-assistant boilerplate, mixed-language annotation leakage, spam URLs, etc. Thinking-mode is the verified-clean path for this bundle.
  3. **`vmlx_engine/utils/dsv4_batch_generator.py`** prefill is now single-shot (`model(full_ids, cache=cache)`) instead of chunked. Chunking corrupted the DSV4 compressor + indexer pool state, which manifested as `broadcast_shapes (1,N) (1,64,1,128)` mid-decode. The post-warmup model has all kernels JIT-compiled so single-shot prefill stays under the Metal command-buffer watchdog even on long prompts.

Live-verified 14/14 on `DeepSeek-V4-Flash-JANGTQ2` (74 GB on disk, 38 GB resident) at 16.3 tok/s decode. Auto-config requires no shell flag — engine does the right thing out of the box.

### Verified non-regressions
- `Qwen3.6-27B-JANG_4M-CRACK` — `'PONG'` ✓
- DSV4 auto-config does NOT affect non-DSV4 models (gated on `_mc.family_name == "deepseek_v4"`)

## [1.5.5] - 2026-05-02

### Fixed
- **DSV4-Flash + JANGTQ inference end-to-end working via `/v1/chat/completions` and all 5 wire formats**.
  Root cause: `mlx_lm.generate.BatchGenerator`'s prefill / decode loop calls `mx.async_eval` / `mx.eval` on cache state and intermediate tensors that carry MLX-internal stream IDs from cross-thread kernel scheduling. The llm-worker step-executor's default stream is `Stream(Device(gpu, 0), 1)` while MainThread sees `Stream(0)` — when the worker materialises tensors that any C++ kernel tagged with a stream ID neither thread owns directly, MLX raises `RuntimeError: There is no Stream(gpu, N) in current thread.` Live-traced through 24 mitigation iterations: pinning streams, `mx.synchronize()` patches, CPU-stream round-trips, batch-API stubs, internal-stream pre-warm — none survived because MLX C++ allocates streams independently of Python.
  Structural fix shipped:
  - **New `vmlx_engine/utils/dsv4_batch_generator.py`** — DSV4-native batch generator (~250 lines) that mirrors `jang_tools.dsv4.runtime.generate`'s prefill + decode + sample + EOS loop while exposing the `BatchGenerator` API (`insert` / `next` / `next_generated` / `remove` / `extract_cache`). Bypasses `mlx_lm.BatchGenerator` entirely so the cross-thread stream traversal never happens. Single-batch only (`max_num_seqs=1`); multi-batch surfaces a clear `NotImplementedError` pointing the user at the right CLI flag.
  - **Auto-evict finished requests** — generator drops finished requests on next `insert()` so the scheduler can queue the next prompt; without this the engine stuck in a retry loop after the first request finished.
  - **Chunked prefill + warmup** — DSV4 first forward triggers Metal kernel JIT for every routed-expert / hash-router shape combo; one-shot prefill of even a 20-token prompt blew through the Metal command-buffer watchdog (~10s). Generator now warms up MLX kernels with a 1-token forward on first call, then chunks user prefill in `prefill_step_size`-token blocks with `mx.synchronize()` between chunks.
  - **`vmlx_engine/utils/mamba_cache.py`** also patches `PromptProcessingBatch.prompt` and `GenerationBatch._step` to use `mx.synchronize()` instead of cross-thread cache.state eval, plus adds `filter`/`extract`/`prepare`/`finalize` stubs to `PoolQuantizedV4Cache` and `DeepseekV4Cache` (single-batch passthrough, multi-batch raises clear error). These are belt-and-suspenders — even if the DSV4 detection misses, mlx_lm.BatchGenerator now drains via active-stream synchronize instead of stream-graph traversal.
  - **`vmlx_engine/scheduler.py::_create_batch_generator`** detects DSV4 family by sniffing `type(model).__module__` / `__name__` and returns `DSV4BatchGenerator` instead of `BatchGenerator` when `dsv4` / `deepseek_v4` is in the path.
  Live-verified on `DeepSeek-V4-Flash-JANGTQ2` (74 GB on disk, 38 GB resident): "What is the capital of France?" → coherent reasoning text, "What is 2+2?" → `'4'`, "What is my name?" (multi-turn) → `'Your name is Eric.'`, 16.3 tok/s decode. **11/14 probe-matrix PASS** across `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/api/chat` (Ollama), `/v1/messages` (Anthropic), SSE streaming + reasoning_content delta, sleep/wake roundtrip, family default temps. The 3 fails are DSV4 reasoning-template refinements (reasoning_effort=high MMLU-style chain, tool-call invocation, chat_template_kwargs.enable_thinking=true) — not stream/threading; deferred to next iteration.
- **MLX vlm-import side-effect on uvicorn MainThread** — `vmlx_engine/__init__.py` previously imported `mlx_vlm.prompt_utils` at module-load time (gemma4 + kimi_k25 registry patches). That transitively pulls in `mlx_vlm.generate`, which executes `mx.new_stream(...)` at import — binding the stream to MainThread. Moved into `_install_mlx_vlm_registry_patches()` called from `MLXMultimodalLM.load()` so the import (and any module-level stream creation) lands on the loader-executor worker thread.

### Verified non-regressions
- `Qwen3.6-27B-JANG_4M-CRACK` — `'PONG'` ✓ (unchanged)
- `Gemma-4-26B-A4B-it-JANG_4M-CRACK` — short-response streaming intact (1.5.2 fix preserved)
- `Qwen3.6-35B-A3B-JANGTQ4` — JANGTQ MoE 14/14 PASS preserved

### Known issues
- DSV4 reasoning-effort=high / tools / explicit enable_thinking=true return empty content because the reasoning parser's `<think>...</think>` split path needs DSV4-specific wiring. Deferred — DSV4 chat-mode (the user-reported headline) works coherently end-to-end at 16.3 tok/s.
- DSV4 still requires `--max-num-seqs 1` because compressor + indexer pool state can't be sliced per-request without a full re-prefill.

## [1.5.4] - 2026-05-02

### Fixed
- **`/api/show` returned all-empty metadata** (`vmlx_engine/server.py`):
  - `details.family`, `details.parameter_size`, `details.quantization_level`, and `template` were hardcoded empty strings, making vMLX look like a stripped-down or broken Ollama backend in Open WebUI / Continue / Copilot pickers. Now reads `config.json` (model_type → family) + `jang_config.json` (quantization.actual_bits + profile → quantization_level e.g. `Q4_K_JANG_4M`; source_model.parameters → parameter_size) + `chat_template.jinja` (→ template) at request time. Falls back to the empty stub on any read failure so the endpoint never 500s.
- **`/api/show` reported `vision` capability when image input was unavailable** (`vmlx_engine/server.py` + `vmlx_engine/utils/jang_loader.py`):
  - Qwen3.5/3.6-VL hybrid-SSM bundles fall back to text-only via the v1.5.1 fix because mlx_vlm's `Qwen3_5GatedDeltaNet` is broken. The engine's `is_mllm` flag stayed True (correct, for batching/scheduler routing) so `/api/show` continued to advertise `vision` — clients gating on capabilities (Copilot, Continue) would offer image upload that silently does nothing. Loader now sets a module-level `_LAST_LOAD_VLM_FALLBACK` marker that `/api/show` consults; vision dropped from capabilities when fallback fired.

## [1.5.3] - 2026-05-02

### Fixed
- **DSV4-Flash bundles fail to load with `ModuleNotFoundError: jang_tools.dsv4.pool_quant_cache`** — `panel/scripts/bundle-python.sh`:
  - The PyPI `jang` wheel (2.5.9) lagged the local development of `jang_tools.dsv4` modules. `pool_quant_cache.py`, `fused_pool_attn.py`, `fused_pool_attn_kernel.py`, and `build_role_codebooks.py` landed in jang 2.5.10/2.5.11 but were not on PyPI when the v1.5.2 DMG was built. The moment any DSV4-Flash bundle was loaded, the JANGTQ runtime imported `pool_quant_cache` and the request died inside `scheduler.step()`, returning empty content (`{prompt_tokens: 0, completion_tokens: 0}`) without surfacing the error to the client.
  - `bundle-python.sh` now installs `jang_tools` from the local `~/jang/jang-tools` source path when present, with a PyPI fallback (`jang>=2.5.11`) for CI builds. Every future DMG will ship whatever DSV4 runtime the engine actually needs without waiting for a separate PyPI publish.
- **DSV4 `make_cache()` `NameError: name 'long_ctx' is not defined`** — vendored `jang_tools/dsv4/mlx_model.py` fix synced into bundled-python:
  - The 2026-05-01 "tri-mode always active" cleanup left the `long_ctx` variable referenced inside `make_cache()` but the parameter was removed from the signature, so the function raised on first call. The scheduler caught it as a hybrid-detection warning and silently fell back to a plain KV path that doesn't match DSV4's compressor/indexer architecture.
  - Replaced `if long_ctx and layer.self_attn.compress_ratio:` with `if layer.self_attn.compress_ratio:` (matches the docstring; tri-mode is unconditional).
- **DSV4 `PoolQuantizedV4Cache` rejected by mamba_cache merge whitelist** — `vmlx_engine/utils/mamba_cache.py`:
  - `_patched_merge_caches` enumerated KVCache, RotatingKVCache, MambaCache, ArraysCache, CacheList — but raised `ValueError("does not yet support batching with history")` for DSV4's `DeepseekV4Cache` and `PoolQuantizedV4Cache`. With continuous batching enabled, the very first request crashed the engine loop and aborted itself.
  - Added DSV4 cache passthrough for `len(caches)==1` (single-batch is a no-op merge); multi-batch raises a clearer error pointing the user at `--continuous-batching` off / `max_num_seqs=1`.

### Audit (continued from 1.5.2)
- 4th bundle: `DeepSeek-V4-Flash-JANGTQ2`. The 3 fixes above unblock load + non-batched inference; a deeper Metal stream-thread bug (`There is no Stream(gpu, 3) in current thread.`) still kills the first inference because DSV4's compressor / indexer / pool-quant tensors were created on multiple worker threads. Same class as the MLLM `/admin/deep-sleep` bug. Tracked separately.

### Known issues (carry-over)
- DSV4-Flash inference fails with `Stream(gpu, 3)` on the first request when the engine spans multiple worker threads. Fix path: dedicated single-worker executor for DSV4 (mirrors the 1.3.93 JANGTQ-VL `mllm-worker_0` fix).
- `/admin/deep-sleep` on hybrid-SSM MLLM bundles fails with `Stream(gpu, 4)` — same class. Soft-sleep + wake roundtrip works.

## [1.5.2] - 2026-05-02

### Fixed
- **Gemma 4 streaming drops short responses (any reply < 18 chars)** — `vmlx_engine/reasoning/gemma4_parser.py`:
  - `Gemma4ReasoningParser.extract_reasoning_streaming` was buffering output until `len(current_text) >= 18` to detect a possible incoming `<|channel>thought\n` marker. Responses shorter than 18 chars (e.g. `BRAVO`, `OK`, `42`, single-token tool args) finished generating before reaching the threshold, so the buffered prefix was never flushed → stream emitted `role:assistant` opener and the final usage chunk only, with zero `content` deltas. Affected `/v1/chat/completions stream=true`, `/v1/messages` (Anthropic non-streaming internally streams), and any client that drives Gemma 4 via SSE.
  - Replaced static-threshold buffer with a prefix-could-be-channel-marker check: hold only while the accumulated text remains a viable prefix of `<|channel>` or `thought\n`; flush as content the moment the first character disqualifies the marker (e.g. starts with `B` for "BRAVO").
  - Live-verified on `Gemma-4-26B-A4B-it-JANG_4M-CRACK`: streaming "BRAVO" now emits `BRA` + `VO` deltas; Anthropic returns `text="BRAVO"`; full audit matrix went from 10/14 → 12/14 PASS on this bundle.

### Audit
- Cross-cutting wire-format matrix run on 3 dense/MoE/hybrid bundles (Qwen3.6-27B-JANG_4M, Gemma-4-26B-A4B-it-JANG_4M, Qwen3.6-35B-A3B-JANGTQ4): 13/14, 12/14, 14/14 PASS respectively. Verified across `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/api/chat` (Ollama), `/v1/messages` (Anthropic) — reasoning on/off, tools, multi-turn cache, family fallback temps, async re-derive, sleep/wake. Single shared `_resolve_enable_thinking()` resolver across 6 sites; Anthropic adapter has its own 3-source thinking precedence (per spec, default OFF).

### Known issues (carry-over, not fixed in 1.5.2)
- `/admin/deep-sleep` on hybrid-SSM MLLM bundles fails with `There is no Stream(gpu, 4) in current thread.` — Metal stream is owned by the dedicated `mllm-worker_0` executor (1.3.93 thread fix) but FastAPI calls `_engine.stop()` from the event-loop thread which can't access stream 4. `/admin/wake` returns `already_active` because the failed deep-sleep never cleared `_standby_state`. Soft-sleep + wake cycle works.

## [1.5.1] - 2026-05-02

### Fixed
- **dense JANG bundles incoherent output (Qwen3.6-27B-JANG_4M-CRACK, Gemma-4-26B-A4B-it-JANG_4M-CRACK)** — `vmlx_engine/utils/jang_loader.py`:
  - Qwen3.5 / Qwen3.6 hybrid-SSM VL bundles routed through `mlx_vlm.models.qwen3_5` produced garbage tokens because `Qwen3_5GatedDeltaNet.__call__` is missing `cache.advance(S)` and uses a different conv-state slicing path than `mlx_lm`'s working version. The bug spans multiple mlx_vlm classes (decoder layer, text model, attention) — full upstream port deferred. Workaround: detect hybrid-SSM Qwen3.5/3.6-VL bundles and fall back to the text-only `mlx_lm.models.qwen3_5` path which decodes coherently. Override via `VMLX_FORCE_VLM_LOADER=1`. Trade-off: image input is unavailable on these specific bundles via the fallback; text chat (the user-reported bug) works.
  - Removed TurboQuant auto-enable for unstamped JANG bundles. JANG bundles that were calibrated for TQ continue to auto-activate via `jang_config.json::turboquant`; everything else now ships TQ off by default. Mirrors the Swift v2 fix that delivered 25× speedup on Nemotron Cascade.
  - Per-module `mxtq_bits` overrides now thread through `class_predicate` correctly so mixed-precision JANG bundles (Qwen3-MoE, Kimi K2.6, MiniMax) don't silently default shared-expert layers to 3-bit.
- **mlxstudio#138 — `--kv-cache-quantization` ignored by BatchedEngine** (`vmlx_engine/engine/batched.py`):
  - When the user explicitly passed `--kv-cache-quantization q4|q8|none`, the loader still applied TurboQuant if `jang_config.turboquant` was present. Now the explicit flag wins: `skip_turboquant=True` is forwarded to `load_model_with_fallback` whenever a non-TQ KV mode is requested.
- **panel tok/s counter wrong while reasoning is on** (`panel/src/main/ipc/chat.ts`):
  - Heartbeat events used a hardcoded `"0.0"` for tokens-per-second whenever the model was emitting reasoning content (no visible tokens yet). The status bar showed `0.0 t/s` for the entire thinking span. Heartbeat now derives `tps` from the SSE `usage` block (`completion_tokens || output_tokens`) divided by elapsed wall time, so reasoning-token throughput shows correctly.
- **mlxstudio#95 — `pip install vmlx` missing JANG runtime** (`pyproject.toml`):
  - `jang>=2.5.9`, `torch`, `torchvision`, `soundfile` promoted from `[mxtq]` extra to hard dependencies. Plain `pip install vmlx` now Just Works on every JANG / JANGTQ / Nemotron-Omni bundle without a second install step.
- **mlxstudio#100 — reasoning request not honored** (`vmlx_engine/api/models.py`):
  - `reasoning_effort` / `thinking` / `enable_thinking` aliases now normalize to a single canonical field on the request model, so OpenAI-shape, Anthropic-shape, and Ollama-shape requests all reach the chat template with the right kwargs.
- **mlxstudio#119 — DeepSeek-V4-Flash-2bit-DQ not supported**:
  - Added DSV4 family routing + tri-mode default (HSA / CSA / SWA combo, no `DSV4_LONG_CTX=0` legacy short-context fork). DSV4 Flash 8-bit / 2-bit / 2-bit-DQ all decode coherently end-to-end.
- **mlxstudio#99 — DSV4-Flash-8bit infinite `<begin_of_sentence>` loop** (`vmlx_engine/server.py`):
  - Added `_FAMILY_FALLBACK_DEFAULTS` for DSV4 (temperature 0.6, top_p 0.95, repetition_penalty 1.05) and routed all four endpoints (`chat/completions`, `completions`, `responses`, Ollama bridge) through `_family_fallback_for(_model_path)` so DSV4 picks up sane defaults when the client doesn't pass them.
- **mlxstudio#131 — `MCPServerConfig` rejects `headers` field** (`vmlx_engine/mcp/types.py`, `client.py`):
  - Added `MCPTransport.HTTP` enum value, `headers: Optional[Dict[str, str]]` field, and `_connect_http()` using `streamablehttp_client`. JetBrains IDE MCP servers and any HTTP-transport MCP that needs auth headers (Bearer token, custom API key) now work.
- **mlxstudio#132 — Gemma 4 MoE AWQ `AttributeError` on inner-model resolution** (`jang-tools` upstream, vendored fix):
  - `_resolve_inner_model()` now handles all three module-tree conventions: LLaMA-nested (`.model.model`), Gemma-top-level (`.model`), and VLM (`.language_model`). Stops the AWQ converter from crashing on Gemma 4 MoE where the inner LM is at the top level instead of nested.
- **Nemotron 3 Nano Omni** — `model_config_registry.py` now treats `mod == "omni"` as multimodal so the Omni audio + image pipeline receives the right runtime path.
- **MLLM text-only path defensive reset** (`vmlx_engine/mllm_batch_generator.py`):
  - `_position_ids` and `_rope_deltas` now reset between text-only LM dispatches to avoid stale state when an MLLM session switches mid-flight from image → text.
- **bundled-mlx duplicate diagnostic** (`vmlx_engine/cli.py`):
  - `_check_no_duplicate_mlx` now prints a clear "DO NOT TOUCH bundled" message when a user-installed mlx in `~/.local/lib` shadows the bundled copy. Stops the silent hang on cold first-launch.

## [1.3.83] - 2026-04-23

### Fixed
- **mlxstudio#88 — Gemma 4 VLM image requests crash with `TypeError: concatenate()` on multi-image prompts** (bundled `mlx_vlm/models/gemma4/vision.py`):
  - `VisionModel.__call__` guarded `isinstance(pixel_values, list)` but called `mx.concatenate(pixel_values, axis=0)` without coercing per-item. MLX 0.31+ enforces strict type checking, and mlx_vlm's Gemma 4 processor pipeline can hand us a list mixing `mx.array` and `np.ndarray` (one entry per image tile on multi-image requests). The concat rejected the mixed list and crashed before any image tokens were embedded.
  - Fix: per-item coerce to `mx.array` before concat. Applied idempotently via `panel/scripts/bundle-python.sh` so a future rebuild can't silently lose it. `verify-bundled-python.sh` gates the patch marker pre-DMG.
  - Regression guard: `TestIssueGuards::test_mlxstudio_88_gemma4_vision_pixel_values_list_coercion` pins both the marker and the coercion semantic.
  - Credit: precise analysis + suggested fix from @LRBin on the issue.

## [1.3.82] - 2026-04-23

### Fixed
- **mlxstudio#87 / #84 follow-up — "ModuleNotFoundError: No module named 'vmlx_engine'" / 'jang_tools'** when launching a session (`panel/src/main/sessions.ts::findEnginePath`):
  - v1.3.81 fixed the same 10-second subprocess-timeout pattern in `engine-manager.ts::checkEngineInstallation` but **missed** the identical pattern in `sessions.ts::findEnginePath`. On a cold-disk first launch, the `python3 -s -c "import vmlx_engine"` verification probe took > 10 s (MLX + mlx_vlm shared libs), timed out, fell through to the system-binary search, and spawned any stale user-installed `vmlx-engine` found at `/opt/homebrew/bin/vmlx-engine`, `~/.local/bin/vmlx-engine`, etc. That stale binary's Python frequently lacked `vmlx_engine` or `jang_tools` → user saw `ModuleNotFoundError` and blamed the fresh DMG.
  - Fix: `findEnginePath` now verifies the bundled install via the filesystem dist-info read (shared helper `verifyBundledEngineOnFilesystem` in `engine-manager.ts` — no subprocess, no timeout). In packaged mode, if the bundle is broken it fails fast with a clear error instead of falling through to a system binary — a stale user install can never win over a freshly-shipped DMG. System-binary fallback is preserved for dev mode (non-packaged `app.getAppPath()`).

## [1.3.81] - 2026-04-22

### Fixed
- **mlxstudio#83 — QwenCode/Opencode `/init` Metal OOM** (`vmlx_engine/mllm_batch_generator.py`):
  - `_run_vision_encoding` in all three call sites (fast path, chunked path, fallback) used `getattr(self.model, 'language_model', None)` which returned `None` for text-only models routed through the MLLM path (smelt-loaded text models, MLLM wrappers without a `.language_model` attr). Chunking silently skipped → fell through to single-shot `self.model(input_ids, **kwargs)` → a ~22 GB attention-score buffer exceeded the ~9.5 GB Metal single-buffer cap → hard crash. Fixed: use `self.language_model` (already fallback-handled in `__init__`).
  - Hybrid SSM models (Qwen3.5 hybrid, Qwen3.6-27B with linear+full attention layers, Nemotron-Cascade, MiniMax) gated chunked prefill behind `VMLX_ALLOW_HYBRID_CHUNKED_PREFILL=1` by default. A coding CLI `/init` with ~15K tokens through a 24–32 head hybrid model allocates far more than the Metal cap. Fixed: predict `heads × seq_len² × 2 bytes`; if > 8 GB, auto-force chunked prefill with a warning. New escape hatch `VMLX_DISABLE_HYBRID_AUTO_CHUNK=1` raises an error instead of chunking (for users who prefer the hard fail to potentially-wrong output on unverified hybrid families; Qwen3.5 hybrid is verified safe).
- **mlxstudio#84 — "Inference engine not found" loop on fresh installs** (`panel/src/main/engine-manager.ts`):
  - `vmlx_engine.__version__` is hardcoded `"1.0.3"` inside `__init__.py` independently of the wheel metadata. The startup check compared it against `pyproject.toml version="1.3.80"`, always mismatched, and triggered `pip install --force-reinstall --no-deps` into the bundled Python on **every launch**. Inside the signed/notarized app bundle, `site-packages/` has the `@` extended-attributes flag + code-sign protection — pip uninstall succeeds but the reinstall fails with EPERM, leaving the user with **no** `vmlx_engine` module. Next session launch showed "Inference engine not found".
  - The cold-boot `python3 -c "import vmlx_engine"` subprocess also routinely timed out at 10 s (MLX + mlx_vlm pull ~200 MB of shared libs), falsely reporting the engine as missing.
  - Fixed: new `getBundledEngineVersionFromFilesystem()` reads `Version:` directly from `site-packages/vmlx-*.dist-info/METADATA` — zero subprocess, zero timeout. `checkEngineInstallation()` and `checkEngineVersion()` both use it as the primary source. `needsUpdate` only fires when the installed version is known and actually differs. Hash-based source-content update check skipped in packaged builds (futile write into signed bundle). Fallback subprocess timeout bumped 10 s → 30 s for cold-disk first-launch. `execFileSync` replaces `execSync` (no shell interpolation).
- **mlxstudio#84 part 2 — Cmd+/- zoom resets on restart** (`panel/src/main/index.ts`):
  - Electron's default Cmd+/- menu binding mutates `webContents.zoomFactor` in memory only; without persistence it resets to 1.0 every launch. Added: restore saved `ui_zoom_factor` on `did-finish-load`; persist on `zoom-changed` (trackpad/wheel) and on `before-quit` (catches keyboard-accelerator changes which bypass `zoom-changed`). Stored in the existing `settings` k/v table.

### Added
- **mlxstudio#31 part 1 — MCP command allowlist expanded** (`vmlx_engine/mcp/security.py`):
  - Added `java` for JetBrains IDE MCP servers (IntelliJ/WebStorm ship a `java -classpath ... McpStdioRunnerKt` MCP server). Without this, users could not connect JetBrains' built-in MCP to vMLX at all.
  - Added `bun`, `bunx`, `deno` (alternative JS runtimes used by newer MCP servers).
  - Added `python3.10` through `python3.13` (explicit version-pinned entries for MCP configs that hardcode a Python minor version).
  - HTTP/SSE transport is already supported at the engine level (`vmlx_engine/mcp/client.py::MCPTransport.SSE`, `MCPServerConfig.url`) — remaining work is a UI for url-based server configs (v2).
- **Kimi K2.6 runtime support** — full `research/KIMI-K2.6-VMLX-INTEGRATION.md` §1 compliance:
  - **Registry** (`vmlx_engine/model_configs.py`): `kimi_k25` family with `is_mllm=True`, `tool_parser="kimi"` (alias of `kimi_k2`/`moonshot`), `reasoning_parser="deepseek_r1"`, `think_in_template=True`, `cache_type="kv"`.
  - **mlx_vlm dispatch** (`vmlx_engine/__init__.py`): `MODEL_REMAPPING["kimi_k25"] → "kimi_vl"` + `prompt_utils.MODEL_CONFIG["kimi_k25"]` installed at import time, so `apply_chat_template` + `get_model_and_args` route Kimi K2.6 through the existing `kimi_vl` module (same MoonViT-27-block + PatchMergerMLP architecture as Moonlight).
  - **Loader routing** (`vmlx_engine/utils/jang_loader.py::_load_jang_v2_vlm`): detects `model_type=="kimi_k25"` and delegates to `jang_tools.load_jangtq_kimi_vlm.load_jangtq_kimi_vlm_model`, which applies the Kimi-specific **lower VL wired_limit** (52% vs 70%) and the **vision/language command-buffer split** that prevents Metal's ~60 s watchdog from killing the first VL forward on 191 GB 2-bit MoE bundles.
  - **VL prefill chunking** (`vmlx_engine/mllm_batch_generator.py::__init__`): clamps `prefill_step_size` to **32** when Kimi K2.6 is detected — mirrors `jang_tools.kimi_prune.generate_vl`'s chunked prefill; the default 1024/2048 chunk would blow the Metal command-buffer watchdog.
  - **MLA fp32 L==1 SDPA patch** (bundled `mlx_lm/models/deepseek_v3.py`): L==1 MLA-absorb path casts q/k/v/mask to `float32` before `scaled_dot_product_attention` and back to bf16 afterwards. Mirrors the fix already applied for GLM-5.1 / DSV3.2; without it, Kimi K2.6 decode hits a ~7.0 logit-magnitude drift per token and produces repetition loops after ~14 tokens on quantized bundles. Applied at build time via `panel/scripts/bundle-python.sh` (idempotent — checks for the "JANG fast fix" marker before editing).
  - **Doc-prescribed module layout** (§1.1–1.4): new `vmlx_engine/loaders/{load_jangtq,load_jangtq_vlm,load_jangtq_kimi_vlm}.py`, `vmlx_engine/vlm/generate_vl.py`, and `vmlx_engine/runtime_patches/kimi_k25_mla.py` — all thin re-exports of the `jang_tools` production entry points. This makes the doc's code examples work verbatim without needing `jang_tools` in a user's own Python env. The `kimi_k25_mla` installer refuses to edit files under a `vmlx/` path, mirroring `jang_tools.kimi_prune.runtime_patch`'s refusal so a stray dev-run never corrupts the shipped bundle.
  - **jang_tools bundling now reproducible** (`panel/scripts/bundle-python.sh`): jang-tools installed from `$HOME/jang/jang-tools` (override via `JANG_TOOLS_DIR`), so a fresh `bundle-python.sh` rebuild picks up `load_jangtq_kimi_vlm.py` + `kimi_prune/` automatically instead of relying on manual copies.
  - **Bundled-python release gate** (`panel/scripts/verify-bundled-python.sh`): 7 new import checks pin the Kimi surface + 2 runtime asserts pin (a) the bundled `deepseek_v3.py` actually has the fp32 MLA patch and (b) the `kimi_k25` remap is live in `mlx_vlm` at import time. Any future rebuild that drops one of these fails the build before DMG packaging.
  - Regression guards: `TestIssueGuards::test_kimi_k26_runtime_contract` (6-point integration pin) + `TestIssueGuards::test_kimi_k26_cache_stack_mla_compat` (prefix-cache H=1 MLA detection + L2 disk round-trip + scheduler MLA auto-quant-off).

### Regression guards
- `tests/test_vl_video_regression.py::TestIssueGuards::test_mlxstudio_83_mllm_oom_guard_and_lm_fallback` — pins both the `self.language_model` consistency fix and the `_OOM_GUARD_BYTES` / `VMLX_DISABLE_HYBRID_AUTO_CHUNK` auto-chunk override.
- `tests/test_mcp_security.py::TestMCPCommandValidator::test_mlxstudio_31_jvm_and_alt_js_runtimes_allowed` — pins `java`/`bun`/`bunx`/`deno` on the allowlist.

### Tests
- 450 VL regression tests + 57 MCP security tests + 57 MLLM tests + 116 Ollama/Anthropic wire-format tests + 637 cache/prefix/paged/hybrid/ssm tests: all green.
- TypeScript `tsc --noEmit` clean on `panel/` main process.

## [1.3.62] - 2026-04-18

### Fixed
- **MLLM prefix cache was effectively disabled**: `MLLMPrefixCacheManager.__len__` was defined but `__bool__` was not, so `if self._cache_manager` fell through to `__len__ > 0`. Empty cache evaluated False, so the very first store was skipped, and no subsequent request could ever hit. Fix: use `is not None`. Verified on Qwen3.6-35B-A3B-JANGTQ2 — exact `(image, prompt)` repeat now hits with real prefix-token reuse (22 tokens saved in test).
- **Non-stream reasoning parser missed `think_in_prompt`**: OpenAI Chat Completions and Responses API non-stream paths called `request_parser.reset_state(harmony_active=...)` without `think_in_prompt`, while the stream paths passed it correctly. For always-thinking templates (MiniMax M2.x, Qwen3, DeepSeek R1), the parser fell through to "no tags → all content" and reasoning prose leaked into the assistant message body. Fix: compute `think_in_prompt` at both non-stream sites the same way the stream path does. Verified on MiniMax-M2.7-JANGTQ-CRACK 4-cell matrix (stream × non-stream × thinking on/off).
- **Token IDs stored as dummy zeros**: The legacy `store_cache` path wrote `token_ids=[0] * num_tokens`, making downstream prefix matching always return 0 even on exact repeats. Fix: `MLXMultimodalLM.generate` now routes through `store()` with real token_ids when they're already computed for the fetch branch.

### Added
- **`VMLX_ALLOW_HYBRID_CHUNKED_PREFILL=1` opt-in** (vmlx#89): hybrid SSM models (Qwen3.5 GatedDeltaNet + attention) on text-only prompts >~34K tokens allocate `attention_scores` of `(1, heads, 48K, 48K) × 2 B = 147 GB` in a single Metal command buffer, blowing the ~72 GB single-buffer cap. With this env var set, hybrid models route through the chunked prefill path for text-only requests. Default OFF = zero behavior change. Reporter's safety analysis for Qwen3.5 verified: `cache.make_mask(N)` + `ArraysCache.state` carry across chunks.
- **Torchvision-free video processor fallback**: `jang_tools.load_jangtq_vlm` installs a class-level patch on `Qwen3VLProcessor.__call__` that routes video inputs through the image_processor when `video_processor is None` (no torchvision in bundled Python). Temporal merging preserved via `video_grid_thw` rewriting. Both JANG v1 and v2 VLM load paths wire it in.
- **65 regression guards** in `tests/test_vl_video_regression.py` covering: video fallback idempotence + temporal rollup; VLM loader wiring; `apply_chat_template` num_images gotcha; mlxstudio#69 multimodal auto-promotion; hybrid cache shape (10 TQ-KV + 30 ArraysCache); reasoning parser split; content-part extraction (image_url / video_url / mixed); cv2 import error surface; sustained load-bloat; prior-turn `<think>` strip preserving `tool_calls`; §15 reasoning-off UX contracts; all four API paths `think_in_prompt` wiring; vmlx#89 opt-in env var; `enable_thinking` priority chain; Gemma 4 + tools auto-off; Mistral 4 `reasoning_effort` auto-map; MLLM cache populates + hits.
- **Stale test cleanup (21 pre-existing failures, test-side only)**: Gemma 3 tool_parser (hermes → gemma3), reasoning_parser (deepseek_r1 → None); PrefixCacheManager `_lru` → `_lru_by_type`; VLM-aware JIT targets `language_model.model`; SimpleEngine test mocks `.generate` not `.chat`; default `vision_cache_size` 100 → 16; default `max_entries` 50 → 20; MCP `_extract_content` returns string; JANGTQ weight_format error message.

### Real-model verification (M4 Max 128 GB)
- **Qwen3.6-35B-A3B-JANGTQ2**: 7 scenarios — multi-turn, hybrid TQ KV cache, VL single/multi-image, L2 disk cache bit-exact round-trip, video fallback, sustained RAM (+0.00 GB growth over 20 mixed requests).
- **MiniMax-M2.7-JANGTQ-CRACK**: full 4-cell reasoning matrix ship-clean.

### Test suite
- Python: 2163 passed, 0 failed.
- Panel TS (vitest): 1545 passed, 0 failed.

## [1.3.11] - 2026-03-24

### Added
- **API Gateway dashboard**: Unified gateway-centric API page with configurable port, live model list, and format toggle (OpenAI / Anthropic / Ollama)
- **Ollama API streaming**: `/api/generate` now supports `stream: true` (was forced non-streaming). SSE-to-NDJSON translation for both `/api/chat` and `/api/generate`
- **Ollama endpoint docs**: API page shows Ollama endpoints, CLI snippets, and connection info when Ollama format is selected
- **Gateway cancel broadcast**: Cancel requests without a `model` field are broadcast to all running backends — only the backend holding that request ID cancels
- **Query param model routing**: GET/DELETE gateway endpoints (`/v1/cache/stats`, `/v1/audio/voices`, etc.) now accept `?model=X` query parameter for routing
- **Client disconnect abort**: All gateway proxy handlers now destroy backend connections when clients disconnect mid-stream
- **Tray gateway info**: Menu bar shows API Gateway port and "Copy API URL" option
- **Update banner persistence**: Dismiss persists per-version in localStorage (survives page reload)
- **i18n dot-path support**: Translation keys now support nested paths (`app.mode.chat`) for structured locale files

### Fixed
- **Hybrid SSM ndim crash (Bug 5)**: `_cleanup_finished()` passed state dicts to `_truncate_cache_to_prompt_length()` which expected raw KVCache objects. Python dict `.keys()` method was treated as tensor — `.ndim` crashed. Fixed with inline state-dict-aware slicing
- **QuantizedKVCache stale meta_state**: Truncation wrote original offset instead of `(safe,)` — now consistent with plain KVCache branch
- **mllm_scheduler disk cache gaps**: 3 disk cache store paths were missing `_is_hybrid` guard. SSM state can't be truncated — must not persist to L2 disk cache
- **Dead i18n code**: Removed duplicate `i18n/index.ts` (lazy-loading version) — `index.tsx` (synchronous) is the active one
- **Unused import**: Removed `X` from CodingToolIntegration lucide imports

## [1.3.0] - 2026-03-20

### Added
- **Nemotron-H JANG support**: Gate dequantization (8-bit high-to-low), fc1/fc2 weight rename, MTP key filter — 42GB GPU, 46 tok/s
- **Hybrid SSM cache support**: Full caching pipeline (prefix, paged, disk) now works with hybrid SSM models (Qwen3.5-A3B, Nemotron-H)
- **Session status banners**: Chat tab shows loading, sleeping, and stopped banners reflecting true session state
- **Smooth token streaming**: Renderer-side typewriter animation (rAF) for both main content and reasoning — fixes chunky 3-5 token batching

### Fixed
- **Metal crash on disk cache store (P0)**: Background writer was triggering GPU ops on wrong thread. Pre-materialize all arrays on calling thread before enqueuing — preserves bfloat16
- **Paged cache layer mismatch (P1)**: Block reuse now checks cumulative SSM state for last-block position. Fixes "Reconstructed 10 layers but expected 40" for hybrid models
- **Hybrid cache reconstruction (P1)**: Text scheduler now applies `_fix_hybrid_cache` to expand KV-only caches to full model layer count (was only in VLM path)
- **Fresh-cache fallback detection**: Detects when hybrid fix returns empty cache (all KV offsets=0), treats as miss instead of silent context corruption
- **Image generation interval leak**: `clearInterval(touchInterval)` now in `finally` block — no more leaked timers on abort/error for both gen and edit
- **JANG VLM config detection**: Uses `_find_config_path()` for legacy config names (`jjqf_config.json`, `mxq_config.json`) instead of hardcoded `jang_config.json`
- **MTP key filter consistency**: Text loader now uses substring match (`"mtp." not in k`) matching VLM loader behavior
- **ReasoningBox performance**: Plain text rendering during streaming, markdown parsing only when reasoning completes — eliminates 60fps `marked.parse()` on 30K+ chains

### Removed
- Dead `STREAM_THROTTLE_MS` constant and throttle check in streaming pipeline
- Dead `_is_vlm_config()` function in JANG loader
- Dead `weights` parameter from `_fix_quantized_bits()` (made optional)

## [1.0.8] - 2026-03-18

### Fixed
- **Static/noise output on Fill, Kontext, KleinEdit**: Removed incorrect `image_strength` parameter — these models use full denoising with reference conditioning, not latent blending
- **Stream exception handling**: Engine errors (OOM, tokenizer crash) now logged and propagated instead of silently killing the stream
- **Anthropic thinking→tool call**: Close thinking block before opening tool call block (fixes Claude Code SSE)
- **MCP `mcpServers` config key**: `MCPConfig.from_dict()` now falls back to `mcpServers` key
- **Parallel tool call accumulation**: Non-streaming Anthropic path uses `tc.index` for correct multi-tool routing
- **Mask b64 stripping**: `/v1/images/edits` strips data URL prefix from mask before decode
- **Temp file scope**: `/v1/images/generations` temp file creation inside try/finally
- **`--mflux-class` startup error**: Moved from additionalArgs to dedicated config field, strips stale args from old sessions
- **Stop server resets generating state**: No more stale loading skeleton after stopping mid-generation
- **maskBase64 cleared** on new session, model switch, session switch, and session delete

### Added
- **Mask Painter for Fill inpainting**: Brush, rectangle, and eraser tools. Auto-opens on image upload for Fill model.
- **Per-byte download progress**: Smooth progress bars during large file downloads via custom tqdm class
- **Auto-resume interrupted downloads** on app restart (scans `.vmlx-downloading` markers)
- **Loading elapsed time** on Image tab, Server tab, and Chat tab
- **Chat "Loading model..." banner** replaces "Model is not running" during model load
- **Generation persists across tab switches** with elapsed time skeleton
- **Custom model support**: mflux class selector for loading custom/fine-tuned image models
- **Help tooltips** on all image parameters
- **HF auth token** for search, README, and downloads
- **JANG VL inference**: Vision-Language support for JANG quantized models
- **About page**: Credits, Ko-fi, JANG links

### Tests
- Fixed 13 stale Python tests, rewrote 6 engine tests
- 2020 Python + 1545 panel tests pass

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2026-03-18

### Fixed
- **JANG model loading crash**: `mx.utils.tree_flatten` import was wrong (`mx` = `mlx.core`, `tree_flatten` lives in `mlx.utils`). Every JANG model load was failing. Fixed to `from mlx.utils import tree_flatten`.
- **Scheduler memory leak**: `self.requests` dict in `_cleanup_finished` was never cleaned. Grows unbounded on long-running servers. Now calls `self.requests.pop(request_id, None)`.
- **SQLite thread safety**: DiskCacheManager pool connections shared between threads without `check_same_thread=False`. Would crash under concurrent disk cache writes.
- **Memory cache race condition**: `remove()` method had no lock. Now acquires `self._lock`.
- **MCP `mcpServers` key not recognized**: Standard MCP config key silently loaded 0 servers. Now accepts both `servers` and `mcpServers`.
- **PYTHONPATH/PATH blocked in MCP**: Python MCP servers couldn't set import paths. Unblocked PYTHONPATH, PATH, NODE_PATH.
- **Image reiteration race condition**: Redo button used `setSettings` + `handleSubmit` in same sync block — React batching meant old settings were used. Now passes override settings directly.
- **New image session mode mismatch**: Clicking + to start new session preserved mode from last-viewed history entry instead of matching the running model's category.
- **Download window paused state lost**: Paused downloads showed as "Queued" when reopening download window. `getDownloadStatus` now includes job status.
- **Download status bar queue count drift**: Queue count only incremented on new queued events, never decremented when downloads started. Now decrements on download start.
- **Misleading "Start" button for undownloaded models**: Image model picker showed "Start anyway" button that always failed (server rejects models without stored paths). Removed.
- **Streaming timeout variable out of scope**: `stream_chat_completion` used `timeout` but it was only defined in non-streaming paths. Fixed with `_default_timeout`.
- **Anthropic SDK base_url wrong in code snippets**: Was `"${baseUrl}/v1"` causing double `/v1/v1/messages`. Fixed to `"${baseUrl}"`.
- **Session health events false positive**: Health monitor emitted `status: 'ok'` regardless of model state. Now checks `data.status === 'healthy'` before transitioning loading → running.
- **Image fetch timeout**: Node.js default ~5min timeout killed long image edits. Now uses explicit 30-min timeout.
- **Z-Image Turbo full precision rejected**: Was wrongly rejected as "diffusers format". ZImage handles both formats.

### Removed
- **Klein models from image picker**: mflux's `Flux1()` requires `text_encoder_2` which Klein doesn't have. Local loading fails; `from_name()` silently downloads from HF. Removed until mflux adds single-encoder support.
- **Silent HF downloads**: All `from_name()` fallbacks removed from image loading. Models must be downloaded explicitly via the download manager.

### Improved
- **Image redo buttons always visible**: Moved from hover-only overlay to always-visible action bar below each image. Colored to match mode (violet for edit, blue for gen).
- **New image session cleanup**: + button now clears source image, error state, and resets mode to match running model.
- **Download progress**: Per-file JSON tracking with cumulative bytes, speed, ETA. File-count fallback when byte totals unavailable.
- **Concurrent downloads**: Up to 3 simultaneous downloads with pause/resume support.
- **HuggingFace README viewer**: Inline README display in download search with lazy loading and YAML frontmatter stripping.
- **Session timeout**: Increased from 60s to 300s for JANG model loading.

## [0.2.18] - 2026-03-09

### Fixed
- **Stop token cleanup on abort**: Per-request stop tokens added to `BatchGenerator.stop_tokens` are now properly cleaned up when a request is aborted. Previously, stop tokens accumulated indefinitely, eventually causing false-positive stops for unrelated requests.
- **Ghost request time-based reaping**: Ghost requests (orphaned in the engine loop) now have a 30-second time-based fallback in addition to the existing count threshold. Prevents ghosts from stalling indefinitely at just below the count threshold.
- **reasoning_effort dead code**: Removed impossible `if _ct_kwargs is None:` guard in both Chat Completions and Responses API streaming paths. `_ct_kwargs` is always `{}` from `request.chat_template_kwargs or {}`.
- **Disk cache flag without paged cache**: `--enable-block-disk-cache` without `--use-paged-cache` now correctly disables disk cache instead of just warning.
- **Module-level `re` import in scheduler**: Moved `import re` out of per-token hot loop in string stop token matching to module-level imports.
- **CancelledError SSE hang**: Engine loop `CancelledError` handler now calls `_fail_active_requests()` to unblock waiting SSE consumers. Previously, cancellation left active streams hanging.
- **Paged cache block leak on abort**: `abort_request` now uses `delete_block_table()` (decrements ref_counts) instead of `detach_request()` (preserves for LRU). Aborted requests don't enter prefix cache, so `detach` would orphan blocks with permanently elevated ref_counts. Fixed in both `Scheduler` and `MLLMScheduler` (abort + error-recovery paths).
- **VLM disk cache key mismatch**: Disk cache store used `truncated_tokens` (N-1) but fetch used `token_list` (N), causing 100% cache miss rate. Now both use `token_list`.
- **KV dequantize None crash**: `_dequantize_cache()` can return `None` on failure but 3 callers didn't guard against it — passing `None` to `_fix_hybrid_cache` or assigning it as `req.prompt_cache` with truncated `input_ids`. All callers now check for `None` and fall back to full prefill.
- **Reasoning trailing window false positives**: Tool call marker detection searched the full `accumulated_reasoning` buffer, triggering false positives when earlier reasoning text discussed tool syntax. Now uses a 30-char trailing window.
- **DeepSeek Unicode tool markers**: Added `<\uff5ctool\u2581calls\u2581begin\uff5c>` (Unicode fullwidth/block chars) to `_TOOL_CALL_MARKERS` for DeepSeek model variants.
- **GPT-OSS fallback threshold too high**: `_FALLBACK_THRESHOLD` reduced from 10 to 3 chars. Previously, short non-Harmony responses ("Hi", "Yes", "OK") were swallowed.
- **GPT-OSS strip_partial_marker too aggressive**: Minimum partial match length raised from 3 to 5 chars. Previously, common Python keywords ("class", "pass") were falsely stripped.
- **Tool fallback first-tool-only check**: `check_and_inject_fallback_tools()` now verifies ALL tool names are in the prompt (using `all()`), not just the first one. Templates that only rendered some tools went undetected.
- **Mistral tool parser invalid JSON**: New-format tool calls now validated with `json.loads()` — malformed JSON arguments are rejected instead of passed through.

## [0.2.12] - 2026-03-07

### Fixed
- **Critical: tool_choice="none" content swallowing in streaming**: Chat Completions streaming path did not gate `tool_call_active` by `_suppress_tools`, causing content to be silently buffered/swallowed when tool markers were detected despite `tool_choice="none"`. Now correctly disables tool buffering.
- **suppress_reasoning leaks in Responses API**: `response.reasoning.done` event was emitted even when `suppress_reasoning=True`. Reasoning-only fallback in Chat Completions also leaked reasoning as content when suppressed.
- **Non-streaming tool_choice="none"**: Both non-streaming Chat Completions and Responses API paths now skip tool parsing when `tool_choice="none"`.
- **PagedCacheManager crash on invalid input**: `block_size=0` caused `ZeroDivisionError`, `max_blocks<2` caused silent failures. Now raises `ValueError` with clear messages.
- **Hybrid detection silent failures**: `_is_hybrid_model` now logs warnings when `make_cache()` raises instead of silently swallowing exceptions.
- **Memory cache 0-memory fallback**: `compute_memory_limit` now logs a warning when psutil returns 0 bytes, explaining the 8GB fallback assumption.

### Improved
- **First-launch UX**: Auto-creates initial chat for new users instead of showing empty state. Skips `detectConfig` for remote sessions (no local model path to inspect).
- **About page**: App version now reads dynamically via IPC instead of hardcoded. Correct website and GitHub links.

## [0.2.11] - 2026-03-07

### Fixed
- **Hybrid VLM paged cache OOM crash**: Hybrid models (Qwen3.5-VL, Jamba-VL) with paged cache crashed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` after a few requests. Root cause: `fetch_cache()` incremented block ref_counts, but when hybrid models couldn't use the cached KV blocks (missing companion SSM state), the refs were never decremented — blocks accumulated until Metal GPU memory was exhausted. Fix: check SSM state BEFORE `reconstruct_cache()`, call `release_cache()` to decrement block refs when SSM state is missing, then `continue` to skip reconstruction and do full prefill instead. Added 5 new tests in `test_hybrid_batching.py`.
- **Suppress reasoning drops thinking entirely**: When reasoning is toggled off for always-thinking models (MiniMax M2.5, Prism Pro), thinking text is now fully hidden instead of being redirected as visible content. Users see a brief pause then only the final answer.
- **Deprecated MLX API calls**: Replaced `mx.metal.device_info()` → `mx.device_info()` and `mx.metal.set_cache_limit()` → `mx.set_cache_limit()` with backward-compatible fallbacks for older MLX versions.

## [0.2.10] - 2026-03-06

### Fixed
- **Reasoning parser for always-thinking models**: Fixed `effective_think_in_template` being unconditionally set to `False` when user disables reasoning. For models whose templates always inject `<think>` (MiniMax M2.5, Prism Pro), the parser now stays in implicit reasoning mode so it correctly classifies reasoning vs content. The `suppress_reasoning` flag handles hiding reasoning from the user. Fixed in both Chat Completions and Responses API paths.

### Improved
- **Parser dropdown UI**: Reasoning and tool parser dropdown labels now include model names directly (e.g., "Qwen3 — Qwen / QwQ / MiniMax / StepFun"). Help panel auto-opens when a manual parser is selected. More comprehensive model compatibility lists. Auto-detect labels say "(recommended)".

## [0.2.9] - 2026-03-05

### Added

#### Speculative Decoding (Phase 3 — vllm-metal Feature Integration)
- **New module**: `speculative.py` — Speculative decoding using mlx-lm's native `speculative_generate_step()`
  - `SpeculativeConfig` dataclass with model, num_tokens, disable_by_batch_size
  - `load_draft_model()` / `unload_draft_model()` lifecycle management
  - `get_spec_stats()` for API health endpoint integration
  - `is_speculative_enabled()` global state check
- **CLI flags**:
  - `--speculative-model` — path/name of draft model (same tokenizer required)
  - `--num-draft-tokens` — tokens drafted per step (default: 3, sweet spot 2-5)
- **How it works**: Draft model proposes N tokens → target model verifies all N in a single forward pass → accepted tokens skip individual decode → 20-90% throughput improvement with zero quality loss
- **Server integration**: `/health` endpoint now reports `speculative_decoding` status (enabled, draft_model, num_draft_tokens, draft_model_loaded)
- **Startup banner**: Speculative decoding status displayed in security/feature summary
- **Test suite**: 21 new tests in `tests/test_speculative.py`:
  - Config validation (defaults, clamping, warnings)
  - Global state lifecycle (load/unload/enable check)
  - Stats reporting (not configured, partial, fully loaded)
  - Error handling (invalid model → ValueError, auto-disable)
  - CLI argument parsing
  - mlx-lm integration (speculative_generate_step importable, stream_generate accepts draft_model)
  - Server health endpoint integration

### Phase 1 Status: RotatingKVCache
- **Already implemented** — confirmed RotatingKVCache support across: `mllm_batch_generator.py`, `scheduler.py`, `prefix_cache.py`, `disk_cache.py`, `memory_cache.py`, `utils/mamba_cache.py`, `utils/cache_types.py`

## [0.2.8] - 2026-03-03

### Fixed

#### Multi-Turn VLM 0-Token Output (Critical)
- **Root cause**: `model_dump()` without `exclude_none=True` included `image_url=None` on text ContentParts. Jinja2 templates check key existence (`'image_url' in item`) which returned True even for None values, causing Qwen3VLProcessor to count 2× image_pad tokens for 1 image → IndexError → fallback to PyTorch → crash.
- **Fix**: All three Pydantic-to-dict conversion paths now use `model_dump(exclude_none=True)`:
  - `server.py` Chat Completions MLLM path
  - `server.py` Responses API `_resolve_content()`
  - `mllm.py` content extraction (defense-in-depth)
- **Fix**: `batched.py` MLLM single-turn path now passes `extra_template_kwargs` (enable_thinking, reasoning_effort)

#### Hybrid Cache Mismatch Returns Corrupt Cache (Critical)
- **Root cause**: `_fix_hybrid_cache()` returned the short (attention-only) reconstructed cache when its length didn't match expected KV positions, instead of a fresh full-length cache from `make_cache()`. This gave the model a cache with wrong layer count.
- **Fix**: Both mismatch paths now return `language_model.make_cache()` (fresh full cache) or `template` (from already-called make_cache).

#### SimpleEngine MLLM Drops Reasoning Kwargs
- **Root cause**: SimpleEngine `chat()` and `stream_generate()` MLLM paths only forwarded `enable_thinking`, silently dropping `reasoning_effort` and `chat_template_kwargs`.
- **Fix**: Both paths now forward all three kwargs to the underlying mlx-vlm model.

#### Miscellaneous
- Removed hardcoded model name from `_template_always_thinks()` — now uses only dynamic template testing
- `make_cache()` failure in MLLMBatchGenerator init now logs warning instead of bare `except: pass`

### Added

#### Comprehensive Test Suite Expansion
- **64 MLLM serialization tests** (`test_mllm_message_serialization.py`): model_dump behavior, Jinja2 key-existence simulation, multi-turn image counting, _fix_hybrid_cache correctness, SimpleEngine kwargs forwarding, mllm.py model_dump paths
- **89 model config registry tests** (`test_model_config_registry.py`): Every model family's tool parser, reasoning parser, cache type, MLLM flag, think_in_template, and priority — plus cross-family consistency checks (valid parsers, no duplicate model_types, priority ordering, MLLM completeness)
- **Total engine test suite**: 1295+ tests passing (up from 1237)

## [0.2.7] - 2026-03-02

### Fixed

#### Continuous Batching Stability (2026-03-02)

- **Continuous Batching Thread Safety**: Added `threading.RLock()` to protect queue mutations across asynchronous loops and sync `MLLM` vision tasks running over background threads (`step`, `add_request`, `abort_request`). Resolves latent data race failures under heavy loads.
- **Bounded Queues**: Fixed unbounded growth mapping of the stream generation output by explicitly setting max size values (`asyncio.Queue(maxsize=8192)`). Ensures memory safety during downstream socket unresponsiveness scaling.
- **Ghost Abort Subsystem**: Fast-tracked the `_ghost_check_counter` interval from checking every 500 loops to 50 loops on the core Engine allowing rapid recycling of broken API memory references for stability endpoints.
- **Batched Engine Rescheduling Safety**: Gracefully intercepted GPU-metal-level corruption traps within generation steps by ensuring requests accurately respool via `retryable` queue structures dropping erroring chunk pointers automatically without completely abandoning API sessions. 

#### Mamba & SSM Native Paged Routing (2026-03-02)
- **Automatic Multi-Array Cache Re-Routing**: Intercepts `model.make_cache()` structure arrays matching hybrid combinations (`MambaCache` alongside `KVCache`) natively detecting standard LLM truncations violations. Auto switches memory parameters inside the `Scheduler` to natively fall back to compatible Paged and Legacy parameters gracefully to prevent sequence faults.

### Fixed

#### Reasoning Content Leaking as Visible Text During Tool Calls (2026-03-02)
- **Reasoning leak on tool follow-ups**: When using agentic tool calling with thinking models (Qwen3, Qwen3.5-VL), reasoning text leaked into visible content on follow-up requests after tool execution. Root cause: `effective_think_in_template` was forced to `False` when tool results were present, breaking the reasoning parser's `think_in_prompt` state. Fix: keep `think_in_prompt=True` — the parser's streaming extraction handles `<think>`→`</think>`→content transitions correctly regardless of tool results.
- **Duplicate content when reasoning disabled**: When reasoning was turned off (`enable_thinking=False`) but the model still produced reasoning text, content appeared twice. Root cause: end-of-stream tool call extraction re-emitted `cleaned_text` that was already streamed (either as content or as redirected reasoning). Fix: track `accumulated_content` during streaming and subtract already-emitted content from the final emission.
- **False-positive tool call buffer flush**: When tool call markers were detected but no actual tool calls were parsed, the entire `accumulated_text` was flushed as content — including text that was already streamed. Fix: only flush the un-streamed portion.
- **Responses API `enable_thinking` guard**: Added missing `_effective_thinking is False` guard to Responses API streaming path for parity with Chat Completions.
- **Tool fallback injection for broken templates**: Some model chat templates silently drop tool schemas when `enable_thinking=False` (e.g., Qwen 3.5 family). Added `check_and_inject_fallback_tools()` that detects when tools are missing from the rendered prompt and injects a standard XML `<tool_call>` instruction set into the system message. Works for all models — not just Qwen.

#### Integrated Tool Call System — Deep Audit & Fixes (2026-03-02)

- **Responses API `tool_choice` handling**: The Responses API endpoint now fully mirrors the Chat Completions handler — `tool_choice="none"` suppresses all tools, `tool_choice={"function":{"name":"X"}}` filters to the named tool only. Previously the Responses API ignored `tool_choice` entirely.
- **Responses API `suppress_reasoning` parity**: When the client sets `enable_thinking=False` but the model forces reasoning (e.g., MiniMax), the Responses API streaming path now redirects suppressed reasoning as content (matching Chat Completions behavior). Previously it silently dropped reasoning deltas, causing the stream to appear to hang.
- **Responses API JSON schema validation**: The non-streaming Responses API path now validates output against `json_schema` with `strict=True` and returns HTTP 400 on validation failure, matching the Chat Completions behavior. Previously it only prompt-injected JSON instructions with no post-generation validation.
- **`gitCommand` shell injection prevention**: Added shell metacharacter blocking (`;|&`$(){}`) to prevent command injection via `/bin/sh -c`. Dangerous git operations (`push --force`, `reset --hard`, `clean -f`, `branch -D`) already blocked.
- **`run_command` kill reason accuracy**: Added `!killReason` guards on all three kill paths (stdout overflow, stderr overflow, timeout) so only the first reason is preserved. Previously, a second overflow event could overwrite the original reason.
- **`ask_user` always available**: Moved `ask_user` out of `UTILITY_TOOLS` category so it cannot be accidentally disabled by the `utilityToolsEnabled: false` toggle. It's a core IPC tool that should always be available.
- **`insertText` / `replaceLines` null guards**: Added parameter validation to prevent silent corruption (NaN splice index, TypeError on undefined text).
- **`fetchUrl` truncation reporting**: Fixed truncation footer to show original content length instead of the truncated length.
- **`batchEdit` conditional write**: Only writes the file when at least one edit succeeded (prevents unnecessary mtime updates on all-fail).
- **`get_diagnostics` dead code removal**: Removed dead TSC single-file branch. Fixed schema mismatch (`path` was marked required but is optional).

#### Comprehensive Test Suite Expansion
- Added `tests/test_tool_format.py` with 54 new tests covering:
  - `ResponsesToolDefinition.to_chat_completions_format()` conversion
  - `convert_tools_for_template()` with all input formats
  - `tool_choice` suppression and filtering for both APIs
  - `response_format.strict` enforcement
  - `max_tokens` fallback chain behavior
  - Model config registry flags (parser assignments, is_mllm, native tool format)
  - Audio model defaults and settings
  - `_responses_input_to_messages()` conversion (string, list, multimodal)
  - `ToolDefinition` Pydantic model edge cases

### Added

#### VLM Caching Pipeline (Pioneer MLX Feature)
- **Paged KV Cache for VLMs**: Full integration of `PagedCacheManager` + `BlockAwarePrefixCache` into the MLLM scheduler for Vision-Language Models
- **Prefix Cache for VLMs**: Token-level prefix matching and cache reuse across VLM requests — stores KV blocks after generation, retrieves on subsequent requests with shared prompt prefixes
- **KV Cache Quantization for VLMs (Q4/Q8)**: Quantized KV cache storage in prefix cache, reducing VLM cache memory by 2-4x
  - Init-time head_dim validation, group_size auto-adjustment, and round-trip testing
  - Quantize on store (`_quantize_cache_for_storage`), dequantize on fetch (`_dequantize_cache_for_use`)
- **Config Propagation**: `SchedulerConfig` cache settings (`enable_prefix_cache`, `use_paged_cache`, `kv_cache_quantization`, `kv_cache_group_size`, etc.) now properly forwarded to `MLLMSchedulerConfig` via `batched.py`
- **VLM Cache Architecture**: Per-request prefill uses standard `KVCache` (integer offsets), then converts to `BatchKVCache` (mx.array offsets) for batched autoregressive decode — bridging `mlx_lm` and `mlx_vlm` cache expectations
- **Mamba Hybrid VLM Support**: Auto-detects VLMs with mixed KVCache + MambaCache/ArraysCache layers (Jamba-VL, VLM-Mamba, MaTVLM). Uses `model.make_cache()` for correct per-layer cache types, `BatchMambaCache` for batched decode, auto-switches to paged cache for Mamba models

### Fixed

#### VLM Cache Crash: `'list' object has no attribute 'offset'` (Critical)
- **Root Cause**: `BlockAwarePrefixCache.fetch_cache()` returns `(block_table, remaining_tokens)` tuple, but code was assigning this raw tuple as the model cache — passed to every decoder layer as `c` in `zip(layers, cache)`
- **Fix**: Proper tuple unpacking + `reconstruct_cache()` on cache hits
- **Additional Fix**: Removed broken `BatchKVCache.offset` monkey-patch (was preventing `BatchKVCache.__init__` from setting its own offset attribute)

#### VLM Cache Merge: `Slice indices must be integers or None`
- **Root Cause**: Per-request VLM prefill used `BatchKVCache` (which has `mx.array` offsets), but `BatchKVCache.merge()` internally uses `.offset` as a Python slice index
- **Fix**: Changed prefill to use standard `KVCache` (integer offsets), then convert to `BatchKVCache` after prefill for batched decode

#### GLM-4.7 / GPT-OSS Harmony Protocol Support
- **GLM-4.7 Flash and GLM-4.7** now use `openai_gptoss` reasoning parser (Harmony protocol: `<|channel|>analysis/final`)
- Previously mapped to `deepseek_r1` which caused leaked `<|start|>assistant<|channel|>analysis<|message|>` tokens in chat
- `think_in_template=False` for GLM Flash — uses channel markers instead of `<think>` prefix injection
- Reasoning effort selector (Low/Med/High) only appears when GPT-OSS/Harmony parser is active

#### Expanded Model Registry
- **Devstral** and **Codestral** added to both TS and Python registries (don't match `/mistral/i` by name)
- Unified GPT-OSS dropdown label: "GPT-OSS / Harmony — GLM-4.7, GLM-4.7 Flash, GLM-Z1, GPT-OSS-20B/120B"

#### Client-Side Content Cleanup
- **Harmony protocol tokens** (`<|start|>`, `<|channel|>`, `<|message|>`) added to TEMPLATE_STOP_TOKENS fallback
- **Hallucinated tool calls** from Anthropic-trained models (`<read_file>`, `<write_file>`, `<run_command>`, etc.) stripped from content
- Both streaming buffering (line-start pattern) and final cleanup (regex) catch these patterns
- Abort path applies identical cleanup to prevent partial tool XML in saved messages

#### Bundled Python Distribution
- `panel/scripts/bundle-python.sh` creates relocatable Python 3.12 + all deps for standalone distribution
- App checks bundled Python first, falls back to system vmlx-engine binary
- Bundled spawn uses `python3 -m vmlx_engine.cli serve` (avoids shebang issues)
- Engine auto-update on startup: compares installed vs source `pyproject.toml` version
- Setup screen skipped entirely when bundled Python detected

### Fixed

#### GLM-4.7 Flash Reasoning Leak (Critical)
- GLM Flash was configured with `deepseek_r1` parser and `think_in_template=true`
- Model actually uses Harmony/GPT-OSS protocol (`<|channel|>analysis/final`), NOT `<think>` tags
- All reasoning content and raw protocol tokens leaked into visible chat output
- Fixed by switching to `openai_gptoss` parser with `think_in_template=false`

#### Reasoning Effort Visibility
- Low/Med/High reasoning effort buttons appeared for ALL models when thinking was enabled
- Only GPT-OSS/Harmony models support `reasoning_effort` parameter
- Now conditionally rendered only when `reasoningParser === 'openai_gptoss'`

### Previously Added

#### Universal Thinking/Reasoning Toggle
- **Per-chat toggle** in Chat Settings (💡 Enable Thinking checkbox) to turn reasoning on/off
- **Default: ON** — matches current behavior, models produce `<think>` blocks
- **When OFF**: `enable_thinking=False` passed to chat template; models skip reasoning for faster, direct responses
- **Pipeline**: UI toggle → `ChatOverrides` DB → request body → `ChatCompletionRequest` → server → engine `apply_chat_template`
- **Compatible models**: Qwen3, DeepSeek-R1, MiniMax M2/M2.5, GLM-4.7, StepFun, and any model with `enable_thinking` template support
- **Server override**: Streaming handler respects the toggle — when OFF, `think_in_template` is forced false (no `<think>` prefix injection)

#### MiniMax M2/M2.5 Model Support
- **New Tool Parser**: `minimax` parser for MiniMax's unique XML tool calling format (`<minimax:tool_call><invoke><parameter>`)
- **Model Config**: Registered `minimax-m2.5` (priority 5) and `minimax-m2` (priority 10) families
  - EOS token: `[e~[]` (ID 200020)
  - Reasoning: `qwen3` parser (standard `<think>` tags)
  - Native tool format: Enabled (chat template handles `role="tool"` natively)
- **Auto-Detection**: MiniMax models auto-detected by model name pattern
- **Streaming**: Added `<minimax:tool_call>` to streaming tool call markers for proper buffer-then-parse behavior
- **16 new tests**: Comprehensive parser tests covering single/multi tool calls, streaming, type conversion, think tags
- **UI**: Added `minimax` to tool parser dropdown in Session Config

#### Chat Scroll Behavior Fix
- **Issue**: Auto-scroll always yanked user to bottom during streaming, preventing scroll-up to read earlier content
- **Fix**: Added `isNearBottom` detection in `MessageList.tsx` — only auto-scrolls when user is within 100px of bottom
- **UX**: Users can now scroll up freely during streaming; scroll resumes when they return to bottom

#### Full Pipeline Audit — Verified Working
- **Qwen hybrid Mamba+KV cache**: Auto-detected in scheduler, auto-switches to paged cache; cache hits work correctly
- **Chat template application**: `apply_chat_template` passes `tools` + `enable_thinking` kwargs with `TypeError` fallback for unsupported templates
- **Tool parser auto-detect**: Model name pattern → `ModelConfigRegistry.get_tool_parser()` → correct parser (all 14 parsers verified)
- **Reasoning parser auto-detect**: Model name pattern → `ModelConfigRegistry.get_reasoning_parser()` → correct parser (`qwen3`, `deepseek_r1`)
- **API completions (non-streaming)**: `/v1/chat/completions` returns correct OpenAI-spec response format
- **API completions (streaming SSE)**: `choices[0].delta.content` + `usage` fields parsed correctly by Electron panel
- **API responses wire format**: `/v1/responses` SSE events (`response.output_text.delta`, `response.completed`) parsed correctly
- **Stop button**: `chat:abort` handler aborts SSE stream + sends `POST /v1/chat/completions/{id}/cancel` for server-side GPU release
- **Reasoning box**: `max-h-[300px] overflow-y-auto` independently scrollable; auto-expands on stream, auto-collapses 1s after done
- **Tool call display**: `ToolCallStatus` component — collapsible, grouped by tool, args+result shown on expand, not spammed
- **Loop prevention**: `MAX_TOOL_ITERATIONS` defaults to 10 (configurable via ChatSettings slider); auto-continue capped at 2 rounds
- **Agentic tool flow**: Full cycle verified — model → `tool_calls` → execute (MCP or builtin) → push results → follow-up request → stream
- **Streaming stats**: TTFT, TGS, PPS computed from SSE stream in `chat.ts:emitDelta()` — generation-only time (gaps >2s excluded)
- **EOS token handling**: Model-specific EOS tokens flow from `ModelConfig` → `MLXLanguageModel.load()` → tokenizer
- **Native tool format**: `SUPPORTS_NATIVE_TOOL_FORMAT` flag preserves `role="tool"` messages through pipeline for models that support it
- **Model selection**: Electron panel scans local filesystem for MLX models; `/v1/models` returns the loaded model
- **Session config UI**: All parser dropdowns (tool + reasoning) list all available parsers with auto-detect default

#### Request Cancellation (OpenAI-Compatible)
- **Feature**: Stop ongoing inference requests to save GPU compute
- **Endpoints**:
  - `POST /v1/chat/completions/{request_id}/cancel`
  - `POST /v1/completions/{request_id}/cancel`
- **Auto-Detection**: Automatically abort when client closes stream connection
- **Unified Request ID**: Response ID (chatcmpl-xxx) is the request ID
- **Compatibility**: Works seamlessly with exploit.bot cancel button (no frontend changes needed)
- **Documentation**: Complete API docs at `docs/api/cancellation.md`
- **Benefits**:
  - Immediate GPU compute savings when user clicks stop
  - Partial responses preserved (no data loss)
  - Works with `reader.cancel()` pattern (auto-detect)
  - Optional explicit API call for programmatic control
  - < 10ms cancel latency

### Fixed

#### Streaming Unicode Character Corruption (Critical)
- **Issue**: Emoji, CJK (Chinese/Japanese/Korean), Arabic, and other multi-byte UTF-8 characters displayed as replacement characters (`�`) during streaming responses
- **Root Cause**: Single-token decoding split multi-byte characters across tokens, producing incomplete byte sequences
- **Fix**: Integrated `StreamingDetokenizer` from mlx-lm into both `scheduler.py` and `mllm_scheduler.py`
  - Per-request detokenizer pool buffers partial characters
  - Only emits text when complete UTF-8 codepoints are assembled
  - Automatically uses optimized BPE detokenizer when available
  - Falls back to `NaiveStreamingDetokenizer` for compatibility
- **Impact**: All streaming clients (vMLX panel, OpenAI SDK, curl) now correctly display multi-byte characters
- **Verification**: 827 tests passing, extensive live server testing with emoji/CJK/Arabic confirmed clean output

#### Hybrid Model Cache Reconstruction (Qwen3-Coder-Next, Nemotron)
- **Issue**: Models with mixed cache types (MambaCache + KVCache) produced null/empty content on cache hits
- **Root Cause**: Two issues:
  1. **KV duplication**: Storing N tokens but re-feeding last token created duplicate with wrong positional encoding
  2. **MambaCache state mismatch**: Cumulative state from post-generation included output tokens
- **Fix**:
  - **N-1 truncation**: Cache stores N-1 tokens so last prompt token can be re-fed for generation kickoff
  - **Prefill-only forward pass**: For hybrid models, runs `model(prompt[:-1])` separately to get clean cache state
  - **Auto-detection**: Hybrid models automatically switch to paged cache (MambaCache can't be truncated)
  - **Cache-hit skip optimization**: Skips redundant prefill on repeated prompts
- **Files Modified**:
  - `scheduler.py`: Added `_is_hybrid`, `_prefill_for_prompt_only_cache()`, modified cache extraction
  - `prefix_cache.py`: Block hashing uses FULL prompt (N tokens), cache DATA has N-1 tokens
  - `paged_cache.py`: Partial block matching for short prompts
- **Impact**: Cache reuse works correctly for all model architectures (pure KVCache, RotatingKVCache, hybrid MambaCache+KVCache)

#### Memory-Aware Cache System
- **Issue**: Cache eviction needed better memory management for large contexts (100k+ tokens)
- **Fixes**:
  - `cache_memory_percent` set to 30% of available RAM (was hardcoded limits)
  - Per-entry size limit is 95% of max_memory (prevents single-entry domination)
  - `_evict_lru()` no longer calls `gc.collect()`/`mx.clear_memory_cache()` during eviction loop
  - Store path removed `mx.clear_memory_cache()` to avoid GPU operation interference
  - Scheduler guards `mx.clear_memory_cache()` with `not self.running` check
  - Memory-aware cache stores raw KVCache object references (not extracted dicts)
- **Impact**: Stable memory usage for long-running servers with large context windows

#### Metal GPU Timeout Prevention
- **Issue**: macOS kills processes when GPU operations exceed ~20-30s
- **Fix**:
  - `mx.eval()` after KV concatenation in `reconstruct_cache()` materializes lazy ops
  - `BatchGenerator.prefill_step_size=2048` controls chunking (safe for Metal timeout)
  - Scheduler memory multiplier reduced to 1.5x (was 2.5x which was too conservative)
- **Impact**: Stable inference for 50K+ token contexts without GPU timeout crashes

### Added

#### Production Readiness Features
- **Streaming detokenizer pool**: Per-request UTF-8-aware token decoding
- **Comprehensive emoji support**: All emoji types verified working:
  - ✅ Basic emoji (🌟 🎯 🔥 🚀 🐍)
  - ✅ Skin tone modifiers (👋🏻 👋🏼 👋🏽 👋🏾 👋🏿)
  - ✅ Family/relationship (👨‍👩‍👧‍👦 👨‍👨‍👦)
  - ✅ Flag emoji (🇺🇸 🇬🇧 🇯🇵 🇧🇷 🇮🇳)
  - ✅ ZWJ sequences (🏳️‍🌈 👩‍💻 👨‍🚀 🧑‍⚕️)
  - ✅ High codepoints (🦀 🦐 🦒 🧀 🧑 🧠)
  - ✅ Ultra-high codepoints (🪐 🪑 🪒 🫀 🫁 🫂)
- **Hybrid model auto-detection**: Automatically switches to paged cache for MambaCache+KVCache models
- **Cache type detection**: Robust detection supporting all cache types:
  - Fully supported: KVCache, RotatingKVCache, MambaCache, ArraysCache, CacheList
  - Partially supported: QuantizedKVCache (detected but no BatchQuantizedKVCache)
- **Memory-aware caching**: Intelligent eviction based on RAM availability
- **Extensive test coverage**: 827 tests covering all cache types, streaming, and emoji scenarios

#### Documentation
- **MEMORY.md**: Comprehensive project memory with all cache system details
  - KV Cache tensor dimensionality handling (3D vs 4D)
  - mlx-lm BatchGenerator cache flow
  - Model config registry patterns
  - Cache system design decisions
  - Hybrid model cache architecture
  - Metal GPU timeout prevention strategies

### Changed

#### Cache Storage Strategy
- **Block hashing**: Uses FULL prompt tokens (N) for matching
- **Cache data**: Stores N-1 tokens to prevent duplication
- **Paged cache**: Default for hybrid models (auto-enabled)
- **Memory-aware cache**: Default for pure KVCache models
- **Forward prefix matching**: Works for multi-turn chat (each turn extends previous)

#### Scheduler Improvements
- **Detokenizer lifecycle**: Created on first request, cleaned up on finish, cleared on reset
- **Cache extraction**: Per-layer error handling prevents one bad layer from killing all
- **Prefill optimization**: Skips re-extraction on cache-hit requests
- **Chunked prefill**: 2048 token chunks prevent Metal GPU timeout

### Technical Details

#### Cache Key and Value Truncation
- **Store key**: Prompt tokens only (not prompt+output) for exact matching
- **Cache value**: Must be truncated to N-1 tokens before storing
  - N-1 because on cache hit, last prompt token is re-fed for generation kickoff
  - If stored at N tokens, last token's KV is duplicated with wrong positional encoding
- **Hybrid models**: `_prefill_for_prompt_only_cache(prompt[:-1])` runs separate forward pass
- **Cache-hit skip**: Blocks already exist from cold store, no redundant prefill needed

#### KV Cache Tensor Dimensionality
- **3D tensors**: `(n_kv_heads, seq, dim)` - Qwen3-Coder-Next and others
- **4D tensors**: `(batch, n_kv_heads, seq, dim)` - BatchGenerator always produces 4D
- **Detection**: Uses `ndim` check before slicing: `seq_dim = 1 if ndim == 3 else 2`
- **Concatenation**: Axis adapts: `axis=1` for 3D, `axis=2` for 4D
- **Fallback check**: `_is_positional_cache` uses `len(shape) in (3, 4)` not just `== 4`

#### Streaming Detokenizer Implementation
```python
# Per-request detokenizer pool
self._detokenizer_pool: Dict[str, Any] = {}

def _get_detokenizer(self, request_id: str) -> Any:
    if request_id not in self._detokenizer_pool:
        # Prefer tokenizer's optimized detokenizer
        if hasattr(self._actual_tokenizer, "detokenizer"):
            detok = self._actual_tokenizer.detokenizer
        else:
            detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
        detok.reset()
        self._detokenizer_pool[request_id] = detok
    return self._detokenizer_pool[request_id]

# In _process_batch_responses():
detok = self._get_detokenizer(request_id)
detok.add_token(response.token)
new_text = detok.last_segment  # Only emits complete UTF-8 codepoints

# On finish:
detok.finalize()
output.output_text = detok.text
```

### Compatibility

#### Model Architecture Support
- **Pure KVCache**: Llama, Mistral, Qwen (non-Next) - uses memory-aware cache
- **RotatingKVCache**: Models with sliding window attention
- **Hybrid (MambaCache + KVCache)**: Qwen3-Coder-Next (36 Mamba + 12 KV layers), Nemotron
- **ArraysCache**: Alternative cache implementations
- **CacheList**: Composite cache structures

#### Chat Template Compatibility
- All models: Native format support
- Mistral: Fixed tool calling template error with native format
- Qwen, DeepSeek, Granite, Nemotron: Added tool call parsers
- MedGemma: MLLM detection patterns updated

### Testing

#### Test Coverage
- **827 tests passing** across all modules
  - 14 comprehensive emoji tests (all categories verified)
- **Streaming detokenizer tests**: 13 tests covering emoji, CJK, Arabic, cache hits
- **Cache system tests**: All cache types (KV, RotatingKV, Mamba, Arrays, CacheList)
- **Live server tests**: Extensive emoji/unicode streaming verification
- **Integration tests**: Multi-turn conversations, system prompts, cache reuse

#### Verified Scenarios
- ✅ Emoji streaming (no replacement characters)
- ✅ CJK (Chinese/Japanese/Korean) streaming
- ✅ Arabic and RTL text streaming
- ✅ Cache hits producing correct content
- ✅ Hybrid model cache reconstruction
- ✅ Multi-turn conversations with cache
- ✅ 100k+ token contexts without GPU timeout
- ✅ Memory-aware cache eviction under pressure

## [0.2.5] - Previous Release

(Previous changelog entries would go here)
