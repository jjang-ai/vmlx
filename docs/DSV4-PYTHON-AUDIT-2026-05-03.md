# DSV4 Python/Electron Audit - 2026-05-03

Scope: Python engine + Electron panel only. Swift work was not used for these fixes.

## Changes Landed

- `vmlx_engine/cli.py`
  - Detects `deepseek_v4` from the model registry before scheduler config.
  - Sets `DSV4_LONG_CTX=1` unless explicitly set.
  - Sets `DSV4_POOL_QUANT=0` unless explicitly set, so `model.make_cache()` returns `DeepseekV4Cache` and the Compressor + Indexer HSA/CSA path is active.
  - Clamps DSV4 continuous batching to `--max-num-seqs 1`. `DSV4BatchGenerator` is single-batch by design because each request owns one heterogeneous SWA + CSA/HSA cache object.

- `panel/src/main/sessions.ts`
  - Detects model family before building concurrency flags.
  - Panel startup now passes `--max-num-seqs 1` for DSV4 even if the saved/default session profile says 5.

- `panel/src/renderer/src/components/sessions/SessionSettings.tsx`
  - Command preview now mirrors the real DSV4 max-seq clamp.
  - Command preview now shows prompt L2 disk cache and paged block L2 flags when they are actually passed.

- `panel/src/main/ipc/chat.ts`
  - Responses SSE parser now ignores duplicate `(event, sequence_number)` events.
  - Existing tool-call persistence/replay changes remain intact.

- `panel/src/renderer/src/components/chat/ChatInterface.tsx`
  - After stream completion and DB hydration, the renderer now uses the persisted assistant content instead of preserving stale streamed content.

- `/Users/eric/jang/jang-tools/jang_tools/__init__.py`
  - Version corrected to `2.5.18` to match `pyproject.toml`.
  - Reinstalled local JANG into:
    - `panel/bundled-python`
    - `panel/release/mac-arm64/vMLX.app`
    - `/Applications/vMLX.app`

## Live DSV4 Runtime Verified

Bundle:

- `/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ2`

Launch flags matched panel-style production startup, including:

- `--continuous-batching`
- original requested `--max-num-seqs 5` (verified engine clamps to 1)
- `--use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000`
- `--enable-block-disk-cache --block-disk-cache-max-gb 10`
- `--reasoning-parser deepseek_r1`
- `--tool-call-parser dsml --enable-auto-tool-choice`

Startup log verified:

- `DSV4_LONG_CTX=1`
- `DSV4_POOL_QUANT=0`
- `--max-num-seqs 5 -> 1` clamp
- `load_jangtq_dsv4`
- `129 DSV4 routed TQ modules`
- `SwitchGLU` fused decode patch for 43 TQ instances
- canonical `encoding_dsv4.py` chat-template shim
- EOS stop set `{1, 128803, 128804}`
- `DSV4BatchGenerator`
- DSV4-aware paged prefix cache enabled
- block disk L2 enabled with `deepseek_v4_v6` nested-state serialization

## Live API Results

### Duplicate Greeting Repro

Prompt: `hi how r u`

- `/v1/responses` streaming, thinking off:
  - content length 137
  - reasoning length 0
  - repeated SSE sequence count 0
  - no duplicate adjacent answer

- `/v1/chat/completions` streaming, thinking off:
  - content length 137
  - reasoning length 0
  - no duplicate adjacent answer

### Multi-Turn Coherence

Thinking off:

- T1 stored color: `noted`
- T2 recalled: `Blue`
- T3 stored animal: `Stored`
- T4 recalled: `Cat`

Thinking on:

- T1 stored Tokyo with separate reasoning/content
- T2 recalled: `Tokyo`
- T3 stored March 12 with separate reasoning/content
- T4 recalled: `March 12`

Max mode smoke:

- Prompt: `What is 17 + 28?`
- Content: `45`
- Reasoning was emitted separately.

## DSV4 Cache / L2 Results

Length-capped long responses are intentionally not stored for DSV4 cache reuse. That guard fired as expected.

Exact-stop long-prefix cache test:

- Prompt length: 4688 tokens
- First run:
  - cache miss
  - 28.70s
  - output `CACHEOK`
  - stored 4687 prompt-cache tokens
  - wrote 74 block-disk entries
- Second run:
  - paged cache hit
  - 4687 cached tokens
  - 0.42s
  - output `CACHEOK`
- Third run:
  - paged cache hit
  - 4687 cached tokens
  - 0.40s
  - output `CACHEOK`

Observed stats after cache test:

- scheduler hits: 2
- tokens saved: 9374
- allocated blocks: 75
- block disk files: 100
- block disk size: about 93 MB
- block disk writes: 74
- block disk hits: 74

Process-restart L2 replay was started but intentionally stopped after the user asked to be careful with RAM. Do not claim restart-persistent DSV4 L2 is fully verified until that exact restart/re-hit test completes.

## RAM Notes

- DSV4 load transiently rose during `streaming hydrate: stacking 129 TQ groups`.
- After hydration, the process settled lower before request-time cache growth.
- A stale Swift/wrapper `JANGPressE2E` DSV4 process was found and killed; available memory rose from about 38 GB to about 71 GB.
- Current app launch does not start a model server automatically.
- Current block cache footprint for DSV4 test directory: about 93 MB.

## Build State

Built and installed:

- `/Applications/vMLX.app`

Installed runtime versions from outside the repo cwd:

- `vmlx_engine 1.5.9`
- `jang_tools 2.5.18`
- `torch 2.11.0`

Electron build completed. macOS notarization was skipped by electron-builder because notarization options were not configured for this build.

## Not Release-Ready Until

- DSV4 L2 restart-persistent replay is completed under controlled RAM monitoring.
- Full cross-model matrix is run for Qwen 3.6 dense/MoE, Laguna, Nemotron Omni/Nano, Mistral/Pixtral 3.5, MiniMax, Gemma, Parakeet/RADIO paths.
- DMG/notarization pipeline is run with signing/notarization credentials configured.
- Release package is checked from a clean working directory, not the current dirty dev tree.

## Follow-Up Fix Pass - 2026-05-04

Scope: Python engine + Electron gateway correctness. MTP/runtime acceleration
and JANGTQ/kernel speed work are explicitly deferred to a later performance
pass.

### Fixed In This Pass

- DSV4 bundle repetition penalty resolution:
  - `vmlx_engine/server.py`
  - DSV4 bundles that omit `chat.reasoning.default_mode` now prefer
    `repetition_penalty_thinking` over generic UI/CLI defaults. This restores
    the bundle-declared 1.15 thinking penalty in current DSV4 Flash tests.

- Bundled MLA absorb fp32 patch:
  - `panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/deepseek_v32.py`
  - `panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/mistral4.py`
  - Verified both bundled files contain the fp32 SDPA absorb branch used by the
    active virtualenv package.

- Responses API suppressed-reasoning tool calls:
  - `vmlx_engine/server.py`
  - Tool markers found only inside suppressed reasoning are now parsed before
    final Responses API finalization, so they emit as `function_call` output
    items instead of falling into an empty text fallback.

- Ollama gateway parity:
  - `panel/src/main/api-gateway.ts`
  - Ollama `images` are translated into OpenAI `image_url` content parts.
  - Ollama `format: "json"` and schema-object formats map to OpenAI
    `response_format`.
  - `cache_salt` and `skip_prefix_cache` pass through chat and generate routes.
  - OpenAI `reasoning_content` / `reasoning` maps back to Ollama `thinking`.
  - OpenAI tool-call JSON-string arguments are converted back to Ollama object
    arguments.

- L2 disk-cache shutdown flush:
  - `vmlx_engine/disk_cache.py`
  - Pending standard and TQ-native writes now preserve `cache_type` during
    manual shutdown flush instead of silently demoting system/user cache rows
    to assistant cache.

- DSV4 fast-load SwitchGLU speed patch scoping:
  - `vmlx_engine/loaders/load_jangtq_dsv4.py`
  - The global `SwitchGLU.__call__` wrapper now preserves the original method
    and only enables the fused DSV4 fast path on modules marked during DSV4
    fast-load. Other SwitchGLU models fall back to their original behavior.

- DSV4 sidecar invalidation:
  - `vmlx_engine/loaders/load_jangtq_dsv4.py`
  - Sidecar manifests now include `runtime_patch`; fast-load invalidates when
    the loader/runtime patch fingerprint changes.

- Anthropic Omni streaming:
  - `vmlx_engine/server.py`
  - Anthropic `/v1/messages` streaming multimodal Omni responses now adapt
    OpenAI chat-completion SSE through `AnthropicStreamAdapter` instead of
    leaking OpenAI SSE chunks to Anthropic clients.

### Tests Added

- `tests/test_disk_cache_unit.py`
  - Shutdown flush preserves standard queue `cache_type`.
  - Shutdown flush preserves TQ-native queue `cache_type`.

- `tests/test_engine_audit.py`
  - Responses suppressed-reasoning tool calls are extracted before finalization.
  - DSV4 fast-load SwitchGLU patch is marker-scoped and idempotent.
  - DSV4 sidecar manifests record/check runtime patch version.
  - Anthropic Omni streaming uses the Anthropic stream adapter.

- `panel/tests/api-gateway-ollama.test.ts`
  - Source contracts for Ollama image translation, response format mapping,
    thinking output, cache bypass forwarding, and tool-call argument conversion.

### Verified Commands

- `.venv/bin/python -m pytest -q tests/test_dsv4_paged_cache.py tests/test_cache_bypass.py tests/test_anthropic_adapter.py tests/test_reasoning_modes.py tests/test_tool_format.py`
  - `189 passed`
- `.venv/bin/python -m pytest -q tests/test_disk_cache_unit.py`
  - `7 passed`
- `.venv/bin/python -m pytest -q tests/test_engine_audit.py::TestResponsesSuppressedReasoningToolCalls tests/test_engine_audit.py::TestV5DisplayTextInit`
  - `2 passed`
- `.venv/bin/python -m pytest -q tests/test_engine_audit.py::TestDSV4FastLoadSwitchGLUScope`
  - `1 passed`
- `.venv/bin/python -m pytest -q tests/test_engine_audit.py::TestDSV4SidecarManifestRuntimePatch tests/test_engine_audit.py::TestDSV4FastLoadSwitchGLUScope`
  - `2 passed`
- `.venv/bin/python -m pytest -q tests/test_engine_audit.py::TestAnthropicOmniStreamingAdapter tests/test_anthropic_adapter.py`
  - `62 passed`
- `npm run typecheck`
  - passed
- `npm test -- api-gateway-ollama.test.ts`
  - `5 passed`

### Still Open For Later Passes

- DSV4 MTP layers:
  - Current Python and Swift reference runtimes drop `mtp.*` by design.
  - MTP is not required for baseline one-token autoregressive quality.
  - Runtime loading/exposure and any MTP acceleration must be treated as a
    separate inference-speed pass with cache trim/rollback verification.

- DSV4/JANGTQ kernel speed:
  - User target is closer to live reports around 38 tok/s on M5 Max.
  - Focus later on JANGTQ kernels, routed MoE fused paths, dense matmul paths,
    and cross-family performance, not only DSV4.

- DSV4 restart-persistent L2 replay:
  - Existing in-process paged hits are verified, but controlled process-restart
    L2 replay still needs live verification.

- Full live API matrix:
  - Re-run OpenAI chat/completions/responses, Anthropic, Ollama, streaming
    reasoning/tool calls, multimodal payloads, and cache bypass/hit paths
    against live model servers after packaging/install.

## Pre-Existing Baseline Fix Pass - 2026-05-04

Scope: clear existing Python/Electron unit/source-regression failures before
starting any JANGTQ/kernel speed work. No live model loads were run in this
pass.

### Additional Fixed Issues

- API/content normalization:
  - `vmlx_engine/api/utils.py`
  - Assistant messages with `content=None` are preserved and normalized instead
    of being dropped as empty assistant turns.

- CLI cache flag validation:
  - `vmlx_engine/cli.py`
  - Minimal argparse namespaces no longer crash before validation when
    `kv_cache_quantization` is absent.

- Health diagnostics:
  - `vmlx_engine/server.py`
  - `/health` diagnostic probes now avoid manufacturing `MagicMock` child
    attributes, fixing the unit-test hang and preventing accidental recursive
    mock traversal in diagnostics.

- Image model local-load errors:
  - `vmlx_engine/image_gen.py`
  - Unknown image/edit model names raise the class-resolution error first.
  - Missing local model files raise a local-file RuntimeError before optional
    mflux imports for generation loads.
  - Missing mflux is still reported when a real local path reaches the import
    guard.

- JANG model detection:
  - `vmlx_engine/utils/jang_loader.py`
  - Legacy empty `jang_config.json` / `jjqf_config.json` stamps are recognized
    as JANG codec bundles unless they explicitly declare a stock MLX weight
    format.

- Hybrid SSM companion cache:
  - `vmlx_engine/utils/ssm_companion_cache.py`
  - Plain Python stored states now deep-copy correctly on fetch; MLX cache
    objects still use the existing materialized clone path.
  - Existing SSM rederive queue cap remains RAM-conscious at 8 entries.

- DSV4 defaults:
  - `vmlx_engine/server.py`
  - DSV4 family detection for bundle sampling defaults now reads local
    `config.json` directly before falling back to the mutable registry.

- VLM/Kimi/Gemma local runtime guards:
  - `vmlx_engine/__init__.py`
  - `panel/bundled-python/python/lib/python3.12/site-packages/mlx_vlm/models/gemma4/vision.py`
  - `.venv/lib/python3.13/site-packages/mlx_vlm/models/gemma4/vision.py`
  - `.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v3.py`
  - Kimi K2.6 `mlx_vlm` registry patches are now installed lazily through an
    import hook, preserving the no-eager-mlx_vlm-import rule.
  - Gemma 4 vision list pixel values are coerced per item before concatenate.
  - The active virtualenv `deepseek_v3` MLA L==1 decode path has the fp32 SDPA
    absorb fix used by the bundled runtime.

- Source-regression harness:
  - `tests/conftest.py`
  - Legacy packaged-source pins under `/tmp/vmlx-1.3.66-build` now point at
    the current checkout during tests.
  - Live server/model tests in `tests/test_e2e_live.py` and
    `tests/test_tq_e2e_live.py` are marked `slow`/`integration` so default
    unit runs do not accidentally start live workloads.

### Additional Verified Commands

- `.venv/bin/python -m pytest tests/test_health_endpoint.py -q`
  - `6 passed`
- `.venv/bin/python -m pytest tests/test_e2e_live.py tests/test_tq_e2e_live.py tests/test_api_utils.py tests/test_audit_fixes.py tests/test_engine_audit.py -q`
  - `318 passed, 13 deselected`
- `.venv/bin/python -m pytest tests/test_vl_video_regression.py -q`
  - `466 passed, 48 skipped`
- `.venv/bin/python -m pytest -q --maxfail=1`
  - `2750 passed, 70 skipped, 32 deselected, 2 xpassed`

### Notes For Next Audit Pass

- The active virtualenv and bundled Python tree both contain runtime patches.
  The bundled tree is partly gitignored, so packaging must still verify the
  produced app bundle carries the same Kimi/Gemma/MLA fixes.
- The default pytest run excludes slow/integration tests; live DSV4/Laguna/API
  model verification remains a separate controlled-RAM pass.
- JANGTQ/MiniMax/DSV4 kernel speed work remains deferred until the baseline
  correctness audit stays clean across another pass.

## VLM + MiniMax Speed Fix Pass - 2026-05-04

Scope: Python/Electron app plus JANG converter tooling. Swift work is not part
of this pass.

### Fixed In This Pass

- Auto-detected VLMs no longer get forced off by stale saved session state:
  - `panel/src/main/sessions.ts`
  - `panel/src/main/ipc/chat.ts`
  - `panel/src/main/model-config-registry.ts`
  - `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx`
  - Detected VLM metadata wins over old `isMultimodal=false` saves when
    launching and when chat IPC decides whether to preserve media parts.
  - Smelt remains the explicit text-only exception because its partial-expert
    loader does not wire the vision tower.

- JANG VLM detection now reads model-owned metadata instead of only
  `architecture.has_vision`:
  - top-level `has_vision`
  - `architecture.has_vision`
  - `capabilities.modality`
  - top-level `modality`
  - config fallback `vision_config`
  - This fixes current JANG/JANGTQ bundles where `jang_config.json` exists but
    omits `architecture.has_vision`; those were previously able to turn a real
    VL model into text-only mode.

- MiniMax-M2.7-Small JANGTQ speed root cause isolated to model artifact shape:
  - `/Users/eric/jang/jang-tools/jang_tools/convert_minimax_jangtq.py`
  - `/Users/eric/jang/jang-tools/jang_tools/pad_minimax_jangtq_experts.py`
  - The source model has 154 routed experts. Synthetic decode profiling showed
    the per-token MoE/router path is slower at 154 experts than at aligned
    widths such as 160/192/256.
  - The converter now pads non-32-aligned MiniMax artifacts to the next
    32-expert boundary, writes the padded `num_local_experts`, pads router gate
    rows with zeros, pads `e_score_correction_bias` with `-10000.0`, and emits
    inert zeroed TQ tensors for dummy experts.
  - The migration tool applies the same fix to existing artifacts by rewriting
    shards in place rather than adding duplicate-name patch shards. This matters
    because mlx-lm still has loader paths that glob `model*.safetensors` instead
    of honoring only `model.safetensors.index.json`.
  - This is intentionally a model-artifact fix, not a vMLX runtime monkeypatch.

- Shared vMLX text decode overhead removed:
  - `vmlx_engine/utils/mamba_cache.py`
  - mlx-lm `GenerationBatch._step` was materializing a full vocab-sized
    logprob vector every generated token. vMLX does not expose logprobs on the
    text BatchGenerator path, and PLD verification is already gated away from
    plain text BatchGenerator. The vMLX step patch now keeps logprobs transient
    for samplers that need normalized probabilities, but stores `[None]` and
    does not eval full logprobs every token.

- JANG-side documentation:
  - `/Users/eric/jang/jang-tools/CHANGELOG.md`
  - Added an unreleased note documenting the MiniMax expert-alignment converter
    behavior and why it exists.

### Speed Evidence Collected

MiniMax synthetic JANGTQ full MoE decode microbench, matching the Small
hidden/intermediate shape (`hidden_size=3072`, `intermediate_size=1536`,
`num_experts_per_tok=8`, 62 layers):

- 154 experts: about `0.428 ms/layer`, estimated `37.7 tok/s`
- 160 experts: about `0.410 ms/layer`, estimated `39.3 tok/s`
- 192 experts: about `0.400 ms/layer`, estimated `40.3 tok/s`
- 256 experts: about `0.395 ms/layer`, estimated `40.9 tok/s`

Split rotate+gather versus fused rotate+gather was also checked for the same
shape:

- split rotate + gather: about `0.320 ms/layer`, estimated `50.3 tok/s`
- fused rotate+gather: about `0.391 ms/layer`, estimated `41.3 tok/s`

Conclusion so far: enabling the fused rotate+gather kernel would be a speed
regression for this shape. The observed user speed for
`JANGQ/MiniMax-M2.7-Small-JANGTQ` at about 37-38 tok/s matches the 154-expert
synthetic bottleneck; padding to 160 is the first model-file fix to verify live.

### Live Speed Results In This Pass

All commands below used one prompt, `--max-tokens 128`, `--max-num-seqs 1`,
`--prefill-batch-size 1`, `--completion-batch-size 1`, and
`--disable-prefix-cache`. This is a controlled wall-clock bench path; the app UI
may report decode-only t/s differently.

- `MiniMax-M2.7-Small-JANGTQ`, before model padding and before no-logprob
  engine fix:
  - `33.53 tok/s`
- `MiniMax-M2.7-Small-JANGTQ`, after 154->160 artifact padding but before
  no-logprob engine fix:
  - `34.06 tok/s`
- `MiniMax-M2.7-Small-JANGTQ`, raw model decode after padding:
  - full decode: `44.23 tok/s`
  - hidden state only: `46.48 tok/s`
  - lm_head: about `1.56 ms/token`
- `MiniMax-M2.7-Small-JANGTQ`, after no-logprob engine fix:
  - `40.29 tok/s`
- `MiniMax-M2.7-Small-JANGTQ`, installed app bundled Python after reinstall:
  - first cold-ish controlled bench: `35.59 tok/s`
  - immediate warmed rerun: `40.04 tok/s`
  - raw bundled model decode with temp/top-p: `40.31 tok/s`
- `Qwen3.6-35B-A3B-JANGTQ-CRACK`, after no-logprob engine fix:
  - `71.45 tok/s`
- `Qwen3.6-27B-JANG_4M-CRACK`, after no-logprob engine fix:
  - `23.06 tok/s`
- `Qwen3.6-27B-MXFP4-CRACK`, after no-logprob engine fix:
  - `24.34 tok/s`
- `MiniMax-SLURPY-JANGTQ`, regression smoke after no-logprob engine fix:
  - `34.88 tok/s` in this controlled 128-token bench path. This is not directly
    comparable to the user's short UI sample, which reports decode-side t/s.

### Hybrid SSM Prefix-Disabled Speed Fix - 2026-05-04

Root cause found after the no-logprob pass: LLM `Scheduler` initialized the
hybrid SSM companion cache for Qwen/GatedDelta-style hybrid models even when the
bench/app request explicitly disabled prefix cache. The lookup path was skipped,
but cleanup could still queue deferred clean SSM re-derive work and the logs
misleadingly showed `SSM companion enabled`. That is correct behavior only when
prefix cache is enabled; it is hidden overhead for `--disable-prefix-cache`.

Fixes made:

- `vmlx_engine/scheduler.py`
  - initializes `_ssm_state_cache` only when `config.enable_prefix_cache` is
    true;
  - gates SSM companion store and idle re-derive on
    `config.enable_prefix_cache`;
  - logs that hybrid SSM companion is disabled when prefix cache is disabled.
- `vmlx_engine/mllm_batch_generator.py`
  - added `enable_prefix_cache` to the constructor;
  - disables VLM/hybrid SSM companion lookup/store/re-derive when prefix cache
    is disabled;
  - gates paged, memory-aware, legacy, and disk cache fetches on the same flag
    for direct generator construction safety.
- `vmlx_engine/mllm_scheduler.py`
  - threads `config.enable_prefix_cache` into `MLLMBatchGenerator`.
- `tests/test_engine_audit.py`
  - added source contracts pinning the LLM and MLLM SSM companion gates.

Controlled speed results after this fix, same command shape as above:

- `Qwen3.6-27B-MXFP4-CRACK`
  - before SSM gate: `24.34` completion tok/s
  - after SSM gate: `26.83` completion tok/s, `29.55` prompt+completion tok/s
- `Qwen3.6-35B-A3B-JANGTQ-CRACK`
  - before SSM gate: `71.45` completion tok/s
  - after SSM gate: `73.65` completion tok/s, `77.68` prompt+completion tok/s
  - raw direct decode with the same temp/top-p sampler: `73.29` tok/s, so the
    remaining gap to `80-90 tok/s` is model/JANGTQ kernel/runtime side, not
    scheduler overhead in this bench path.
- `Qwen3.6-27B-JANG_4M-CRACK`
  - before SSM gate: `23.06` completion tok/s
  - after SSM gate: `23.97` completion tok/s, `25.28` prompt+completion tok/s
- `MiniMax-M2.7-Small-JANGTQ`
  - unaffected by SSM, but regression checked after the change:
    `40.11` completion tok/s, `42.30` prompt+completion tok/s
- Installed `/Applications/vMLX.app` bundled Python after sync, warmed rerun on
  `Qwen3.6-27B-MXFP4-CRACK`:
  - `26.87` completion tok/s
  - `28.34` prompt+completion tok/s
  - log confirmed: `Hybrid SSM cache detected but prefix cache is disabled;
    SSM companion lookup/store/re-derive is disabled for this run`
- Installed `/Applications/vMLX.app` bundled Python after full app rebuild/sync
  on `MiniMax-M2.7-Small-JANGTQ`:
  - `42.92` completion tok/s
  - `45.27` prompt+completion tok/s
  - log confirmed the fixed artifact shape is active:
    `Patched SwitchGLU class for fused gate+up (62 TQ instances)` and
    `P18 QKV fusion: 1 class(es), 62 instances`

### DSV4 Encoder Auto-Discovery Fix - 2026-05-04

While rerunning the DSV4/API audit slice, the DSV4 tool-format encoder test
failed if the old `/Volumes/.../DeepSeek-V4-Flash-JANGTQ` persisted path was
missing and `DSV4_ENCODING_DIR` was not manually set. That is brittle for the
current setup where models now live under `~/models`.

Fixes made:

- `vmlx_engine/loaders/dsv4_chat_encoder.py`
  - added shallow auto-discovery for local DSV4 encoder directories:
    - `$VMLINUX_MODELS_DIR` or `$VLLM_MODELS_DIR`
    - `~/models/Sources/DeepSeek-V4-Flash/encoding`
    - `~/models/**/DeepSeek-V4-Flash*/encoding` with bounded one/two-level
      glob patterns
    - `/Volumes/*/.../DeepSeek-V4-Flash*/encoding` with the same bounded
      patterns
  - preserves explicit `encoding_dir`, `DSV4_ENCODING_DIR`, and
    `model_path/encoding` precedence.
- `/Users/eric/jang/jang-tools/jang_tools/dsv4/encoding_adapter.py`
  - mirrored the same fallback so direct JANG tooling is not env-var-only.
- `tests/test_tool_format.py`
  - made DSV4 encoder tests deterministic with a temp `encoding_dsv4.py`;
  - added coverage for `~/models/Sources/DeepSeek-V4-Flash/encoding`
    auto-discovery.

### Verified Commands

- `cd panel && npm test -- model-config-registry.test.ts`
  - `4 passed`
- `cd panel && npm test -- settings-flow.test.ts`
  - `156 passed`
- `PYTHONPATH=/Users/eric/jang/jang-tools .venv/bin/python -m pytest -q /Users/eric/jang/jang-tools/tests/test_minimax_jangtq_padding.py`
  - `2 passed`
- `PYTHONPATH=/Users/eric/jang/jang-tools .venv/bin/python -m jang_tools.pad_minimax_jangtq_experts /Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ`
  - dry run: `154 -> 160`, `39` shards, about `1.323 GB` added
- `PYTHONPATH=/Users/eric/jang/jang-tools .venv/bin/python -m jang_tools.pad_minimax_jangtq_experts /Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ --apply`
  - rewrote all `39` shards
- Artifact validation:
  - `config.json::num_local_experts == 160`
  - layer 0 `gate.weight` shape `[160, 3072]`
  - layer 0 `e_score_correction_bias` shape `[160]`
  - dummy expert 159 TQ tensors exist through layer 61
  - rerunning the migrator reports `already_aligned: true`
- `.venv/bin/python -m pytest -q tests/test_engine_audit.py::TestGenerationBatchFastNoLogprobs tests/test_engine_audit.py::TestDSV4FastLoadSwitchGLUScope tests/test_engine_audit.py::TestDSV4SidecarManifestRuntimePatch`
  - `3 passed`
- `.venv/bin/python -m py_compile /Users/eric/jang/jang-tools/jang_tools/pad_minimax_jangtq_experts.py vmlx_engine/utils/mamba_cache.py`
  - passed
- `.venv/bin/python -m py_compile vmlx_engine/scheduler.py vmlx_engine/mllm_batch_generator.py vmlx_engine/mllm_scheduler.py`
  - passed
- `.venv/bin/python -m pytest -q tests/test_cache_bypass.py::TestSchedulerBypassGating tests/test_engine_audit.py::TestHybridSSMCompanionCacheGating tests/test_engine_audit.py::TestGenerationBatchFastNoLogprobs`
  - `12 passed`
- `.venv/bin/python -m py_compile vmlx_engine/loaders/dsv4_chat_encoder.py /Users/eric/jang/jang-tools/jang_tools/dsv4/encoding_adapter.py`
  - passed
- `.venv/bin/python -m pytest -q tests/test_dsv4_paged_cache.py tests/test_reasoning_modes.py tests/test_tool_format.py -k 'dsv4 or DSV4 or dsml' tests/test_cache_bypass.py -k 'dsv4 or DSV4'`
  - `18 passed`
- `.venv/bin/python -m pytest -q tests/test_anthropic_adapter.py tests/test_api_models.py tests/test_multimodal_routing.py tests/test_streaming_reasoning.py -k 'responses or Anthropic or anthropic or multimodal or reasoning_fallback or empty_fallback'`
  - `74 passed`
- `cd panel && npm test -- model-config-registry.test.ts settings-flow.test.ts`
  - `160 passed`
- `cd panel && npm test -- api-gateway-ollama.test.ts reasoning-display.test.ts request-builder.test.ts model-config-registry.test.ts settings-flow.test.ts`
  - `293 passed`
- `cd panel && npm run typecheck`
  - passed
- `panel/bundled-python/python/bin/python3 -s -m pip install --force-reinstall --no-deps /Users/eric/mlx/vllm-mlx`
  - installed current `vmlx_engine` into the repo bundled Python
- `/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 -s -m pip install --force-reinstall --no-deps /Users/eric/mlx/vllm-mlx`
  - installed current `vmlx_engine` into the installed app bundled Python
- Verified both bundled `vmlx_engine/utils/mamba_cache.py` copies contain
  `self._next_logprobs = [None] * len(self.uids)`
- Synced `/Applications/vMLX.app/Contents/Resources/vmlx-engine-source/` from
  the current checkout so the app's bundled-update path does not reinstall
  stale engine source over the fixed bundled Python on restart.
- `cd panel && npm run build`
  - passed; rebuilt `dist/main/index.mjs` with VLM detection changes
- `cd panel && npm run package`
  - passed; produced `panel/release/mac-arm64/vMLX.app`
- Copied rebuilt `app.asar` into `/Applications/vMLX.app/Contents/Resources/app.asar`
  and ran `codesign --force --deep --sign - /Applications/vMLX.app`
  - `codesign --verify --deep --strict /Applications/vMLX.app` passed
- Extracted installed `/Applications/vMLX.app` `app.asar` and verified
  `dist/main/index.mjs` contains:
  - `resolveJangMultimodal`
  - `has_vision`
  - `detected.isMultimodal === true`
- Reinstalled current `vmlx_engine` and current `/Users/eric/jang/jang-tools`
  into both bundled Pythons after the DSV4 encoder fallback fix.
- Re-synced `/Applications/vMLX.app/Contents/Resources/vmlx-engine-source/`
  and re-signed `/Applications/vMLX.app`.
- Verified installed bundled Python contains:
  - `vmlx_engine.loaders.dsv4_chat_encoder._default_encoding_dirs`
  - `jang_tools.dsv4.encoding_adapter._default_encoding_dirs`

### Not Yet Verified Live

- The installed app on disk now has the no-logprob engine fix, the SSM
  prefix-disabled gate, and rebuilt Electron main-process JS for panel-side VLM
  detection. The currently running app process still needs a normal restart to
  load the new `app.asar`; no running `/Applications/vMLX.app` process was
  killed during this pass.
- The Qwen 27B JANG_4M/MXFP4 cases now hit the user's target range in
  prompt+completion throughput under the controlled bench. Completion-only t/s
  is still lower on short/eos-capped outputs because CLI bench includes request
  add, prefill, and stream collection wall time.
- The app UI decode-only t/s should be rechecked after restarting the Electron
  app, because the controlled bench reports total request wall-clock completion
  t/s rather than the UI's decode-only metric.
