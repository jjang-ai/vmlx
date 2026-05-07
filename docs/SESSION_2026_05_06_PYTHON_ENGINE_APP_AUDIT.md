# Python Engine/App Audit — 2026-05-06

Scope: Python engine + Electron app only:

- `/Users/eric/mlx/vllm-mlx`
- `/Users/eric/jang/jang-tools`

Swift work is out of scope for this session.

## Current State

The source tree and the installed app are not the same.

| Component | Observed state |
|---|---|
| Source `pyproject.toml` | `vmlx 1.5.24` |
| Source `panel/package.json` | `1.5.24` |
| Repo-local `panel/bundled-python` | `vmlx 1.5.24`, `jang 2.5.26`, `mflux 0.17.5`, `mlx 0.31.2`, `mlx-lm 0.31.3`, `mlx-vlm 0.4.4` |
| Installed `/Applications/vMLX.app` | bundled Python reports `vmlx 1.5.24`, `jang 2.5.26`, `mlx 0.31.2`, `mlx-lm 0.31.3`, `mlx-vlm 0.4.4`, but `mflux` is still missing |

The live installed app was not mutated in this pass. It still reproduces the
image-model `ModuleNotFoundError: No module named 'mflux'` class even though
the repo-local source bundle is already corrected.

## GitHub Issue Map

Status terms used below:

- **source fixed** means code and repo-local tests cover the bug class.
- **live-source verified** means a local model/API run from source or
  repo-local bundled Python passed.
- **release verified** means a clean signed/notarized DMG or PyPI package was
  installed and the reporter path was reproduced there.

Nothing in this pass is release verified. Do not close issues from source tests
alone.

### jjang-ai/vmlx

| Issue | Status in source | Remaining blocker |
|---|---|---|
| #147 Ollama version probe | Source has `GET/HEAD /` and `/api/version` gateway support. Panel tests passed. | Needs signed app release/live install verification. |
| #146 mflux missing | Source bundle installs and verifies `mflux 0.17.5`; repo-local bundle has it. | Installed app is stale and missing `mflux`; rebuild/reinstall required. |
| #145 image input `TokenizerWrapper` not callable | Source fixed and live-source verified on Qwen3.6-35B-A3B-JANGTQ-CRACK for `/v1/chat/completions` image_url and `/v1/responses` input_image. | Needs signed app release verification. Affine-JANG Qwen hybrid is now text-only by default and rejects media with a clear 400. |
| #144 RAM growth / hidden SSM companion | Source gates SSM companion by prefix-cache enablement and bounds companion cache. Gemma-4-26B image soak stayed flat over 12 salted requests. | Exact 32GB/g4-31b reporter profile pending. |
| #143 JANGTQ conversion access | Not a runtime bug. | Product/support response; no engine fix required unless docs/tool packaging changes. |
| #142 g4-31b crash | Related to Gemma/JANG loading and cache/memory safety. Source has current validators, but exact model not reproduced in this pass. | Needs live load with reporter model. |
| #141 DSV4-Flash conversion tools fail | Related to converter/JANG tooling, not just runtime. | Needs reproduction in `/Users/eric/jang`; do not close from engine tests. |
| #139 VLM no stream in worker thread | Source fixed and live-source verified on Gemma-4-26B image streaming; no `No Stream(gpu)` error, coherent streamed output. | Needs signed app release verification. |
| #138 `--kv-cache-quantization` overridden by TurboQuant | Source tests confirm explicit q4/q8/none is respected; DSV4 uses native SWA/CSA/HSA cache and hybrid SSM disables generic KV/TQ-KV by default. | Needs packaged CLI check. |
| #137 prompt_lookup.py stale docs | Documentation issue. | Pending doc cleanup; not release-blocking for runtime. |
| #136 PFlash speculative prefill | Feature request. | Backlog; no source fix claimed. |
| #135 draft-model speculative under batching | Feature request/perf. | Backlog; no source fix claimed. |
| #134 PLD on hybrid SSM | Feature/perf proposal. | Backlog; overlaps SSM checkpoint/replay work but not a current bug fix. |
| #133 Swift directory picker | Swift issue, not Python/Electron session scope. |
| #132 Gemma 4 AWQ `.model` AttributeError | Fixed in local `jang` source and installed `jang 2.5.26`. | PyPI/release confirmation if publishing. |
| #131 MCP headers / streamable HTTP | Source accepts headers and streamable HTTP. MCP tests passed. | Live remote MCP token test pending. |
| #127 logprobs/top_logprobs | Feature/API parity request. | Needs separate implementation and tests. |
| #126 M1 Max shutdowns | Hardware/thermal/memory safety report. | Needs reporter repro profile and memory/Metal limits; not cleared. |
| #125 flash-moe | Feature/compat request. | Backlog; current source protects incompatible JANGTQ+Smelt/Flash-MoE paths. |
| #124 wrong font with Cyrillic | Panel UI bug. | Needs UI font/render check; not addressed here. |
| #123 how to run model | Support/docs issue. | Needs docs/response, not engine fix. |
| #122 upgrade question | Support issue. | Needs response. |
| #121 macOS 26 ad-hoc NSOpenPanel | Swift/ad-hoc signing issue, not Python/Electron source. |
| #120 nanobind/metallib Sequoia | Source bundle now forces macOS 14 MLX wheels and package floor is 14.5. | Needs clean installed-app test on macOS 15.x/14.5+. |
| #119 DeepSeek-V4-Flash | Source has DSV4 registration/cache/runtime tests and native-cache policy. | Retest after fresh model download. Full long-context output must be read to the end; DSV4 should use SWA/CSA/HSA native compression, not generic TurboQuant KV. |
| #100 image generation optimization | Feature/perf request. | Backlog; mflux packaging fix is separate. |
| #98 Agoragentic listing | Product/listing item. | No engine work. |
| #88 RotorQuant support | Quantization feature request. | Backlog; no source fix claimed. |
| #86 multi-user isolation/security | Feature/security architecture. | Backlog; current local app auth/rate-limit warnings remain. |
| #79 Z-image support/catalog | Image feature request. | Backlog. |
| #78 recommended code harness | Support/docs issue. | Needs response/docs. |
| #76 DFlash support | Feature request related to DSV4/DFlash. | Current DSV4 runtime policy is separate; do not close. |
| #74 auto-evict/API gateway unload | Feature request. | Backlog; sleep/wake/JIT matrix still pending. |
| #60 asymmetric KV cache | Feature/cache research. | Backlog; no source fix claimed. |
| #57 remove models | UI/support issue. | Needs panel flow review. |
| #44 speculative decoding + continuous batching + MLLM | Feature/perf request. | Backlog; not part of this pass. |
| #37 roadmap/benchmarks | Product/docs request. | Needs response/release notes. |

### jjang-ai/mlxstudio

| Issue | Status in source | Remaining blocker |
|---|---|---|
| #107 mflux dependency conflict | Source bundle uses current mflux line compatible with newer deps. | Installed app stale; release required. |
| #106 Nemotron Nano Omni missing mlx-vlm class | Source routes through JANG/Nemotron native path per prior comments. | Live user model check pending. |
| #105 Qwen3.6 Swift crash | Exact Swift crash is out of Python scope, but Python reproduced the same Qwen3.6 affine-JANG corruption class. Source now routes affine-JANG Qwen hybrid text-only and keeps JANGTQ/MXTQ Qwen VLM native. |
| #104 default metallib language version | Source now blocks packaging/support below macOS 14.5 and docs no longer claim macOS 26+. | Need clean app install on macOS 15.x to verify actionable error/no crash. |
| #103 mflux missing | Same as vmlx #146. Source bundle OK; installed app stale. |
| #102 Swift v2 install question | Support/Swift issue. | Out of Python/Electron runtime scope. |
| #101 duplicate MLX/nanobind | Source has startup duplicate-MLX detector. | Clean install verification pending. |
| #100 reasoning enable | Source has reasoning-mode contracts across OpenAI/Anthropic/Ollama paths. | Live model/protocol matrix pending. |
| #99 DeepSeek-V4-Flash BOS loop | Same DSV4 correctness family as vmlx #119. | Retest after fresh DSV4 download; full output must be read to completion. |
| #95 MiniMax JANGTQ missing `jang_tools` | Source hard-depends on PyPI `jang`; bundled verifier imports `jang_tools`. Installed app has `jang 2.5.26`. |
| #91 old broken `!` / stream thread | Old release packaging issue plus VLM stream class. Source tests passed for relevant paths. |
| #90 libc++ symbol on macOS 14.4 | Source now sets package floor to macOS 14.5 and CLI explains the runtime floor. |
| #89 remote desktop client | Feature request; not a correctness blocker. |
| #75 ModelScope mirror | Feature request. | Backlog. |
| #68 generic bug | Needs issue-body triage. | Not mapped yet without reproduction details. |
| #67 LoRA hotswap | Feature request. | Backlog. |
| #61 image gallery delete/copy | UI feature request. | Backlog. |
| #31 MCP HTTP/SSE/java command | Partially overlaps vmlx #131. | Header/streamable HTTP source tests passed; Java command support needs separate verification. |

## Current Issue Triage

### Fixed In Source And Tested

- vmlx #138: explicit `--kv-cache-quantization q4|q8|none` no longer gets
  silently shadowed by live TurboQuant KV. DSV4 is forced onto native
  SWA+CSA/HSA composite cache; hybrid SSM defaults to native cache unless an
  explicit diagnostic override is set.
- vmlx #146 / mlxstudio #103 / #107: repo-local bundle includes `mflux 0.17.5`
  and verifier imports it. Installed `/Applications/vMLX.app` still lacks it.
- vmlx #147: Ollama root/version probes are covered by panel tests.
- vmlx #131: MCP `headers` and streamable HTTP are source-tested.
- vmlx #132: JANG tools `awq.py` accepts both `model.model.layers` and direct
  `model.layers`; fixed in `jjang-ai/jangq@85dfb3a`.
- mlxstudio #90/#104/#101 and vmlx #120: bundle script now forces
  `macosx_14_0_arm64` MLX wheels and package floor is `14.5.0`, avoiding the
  Tahoe `macosx_26_0` metallib-on-Sequoia class.
- Ling/Bailing JANGTQ2 loop: live-source verified after fixes. Root cause was
  server prompt mutation plus hybrid cache codec mismatch. The final bundle
  uses the canonical 92-token Ling prompt, disables live TQ-KV/generic q4 for
  hybrid SSM auto mode, and hits hybrid paged cache with 28 SSM companion
  layers restored.
- Qwen3.6 JANGTQ VLM image path: live-source verified on
  `/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK`; native
  `jang_tools.load_jangtq_vlm` returns `Red square` for both OpenAI chat
  image_url and Responses input_image.
- Qwen3.6 affine-JANG hybrid VLM path: live-source reproduced corrupt text and
  corrupt image output through `mlx_vlm`; direct text JANG loader was coherent.
  Source now routes these affine-JANG Qwen hybrid bundles to text-only by
  default and returns a clear 400 on media requests instead of token soup.
- Gemma-4-26B VLM path: live-source verified non-streaming image, streaming
  image, and Responses input_image. TurboQuant KV telemetry is now shared
  between `/health` and `/v1/cache/stats` for nested MLLM language models.
- ZAYA/CCA: source now preserves `cache_subtype=zaya_cca`, disables unsafe
  prefix/paged/L2/TQ-KV paths for that subtype, and refuses generic JANG loads
  until a real ZAYA runtime implements CCA state restore.

Latest regression runs:

- Python focused engine/API/cache/media suite:
  `854 passed, 48 skipped`.
- Broader API/VLM/reasoning/tool/image suite:
  `859 passed, 6 deselected`.
- Panel:
  `1595 passed`; typecheck passed.

### Live-Source Artifacts Already Present

Local artifacts under `/tmp/vmlx_family_audit` show these latest statuses:

| Model family | Latest artifact | Status | Notes |
|---|---|---|---|
| Gemma 4 CRACK | `live_gemma4_crack_after_cache_fixes.json` | PASS | Cache repeat, APIs, tool path, mixed-attention cache passed. |
| Laguna | `live_laguna_after_cache_fixes.json` | PASS | Basic APIs/cache passed; performance/quality still needs separate benchmark scrutiny. |
| Ling JANGTQ | `live_ling_flash_tq_after_current_fixes.json` | PASS | Non-CRACK Ling row passed. |
| Ling JANGTQ2 CRACK | final live smoke in this session | PASS for loop class | Previous artifact still failed; final rebuilt bundle smoke and cache-on repeat passed. |
| MiniMax M2.7 | `live_minimax_m27_*_after_cache_fixes.json` | PASS | Uniform, Small, and JANGTQ_K rows passed. |
| Nemotron Omni | `live_nemotron_omni_tq2_after_parser_fix.json`, `live_nemotron_omni_tq4_after_parser_fix.json` | PASS | Uses native `jang_tools.nemotron_omni` path, not missing mlx-vlm module. |
| Qwen3.6 dense JANG/MXFP4 | `live_qwen36_dense_*_after_cache_fixes.json` | PASS | Text/API/cache rows passed. Some reasoning-only responses length-capped into reasoning, so long-form UI quality still needs manual read. |
| Qwen3.6 MoE CRACK | `live_qwen36_moe_crack_after_cache_fixes.json` | PASS | Text/API/cache row passed; this does not cover the user's Qwen3.6-27B-JANG_4L Metal timeout screenshot. |
| DSV4 Flash | `live_dsv4_current_main.json` | FAIL | Most checks passed, including long-context full-output and native composite cache, but one cache-repeat coherence checker failed. Do not close DSV4 from this artifact alone. |

### Still Not Cleared

- vmlx #145 exact image-input repro on
  `dealignai/Qwen3.6-35B-A3B-JANGTQ4-CRACK`. Source tests cover the
  `TokenizerWrapper` processor trap, but the exact model+image request has not
  been rerun from a signed app.
- vmlx #139 exact Gemma-4-26B VLM stream request. Source tests cover the stream
  context class, but the reporter's request has not been repeated from a signed
  app.
- vmlx #144 repeated-prompt RAM growth on 32GB M1 Max. Source bounds the SSM
  companion cache and Gemma live-source artifact passed a short cache check,
  but no 15-20 prompt constrained-memory soak has been run.
- vmlx #142 `g4-31b-jang-4m-crack` crash. No exact model reproduction yet.
- vmlx #119 / DSV4. Needs clean retest after the new DSV4 download, with full
  long-context output read to the end and cache repeat tested with a better
  prompt than `blue blue blue blue`.
- mlxstudio #105 / user Qwen3.6-27B-JANG_4L Metal GPU timeout. This is not
  fully cleared until the exact local model path is loaded and the
  warmup/generation path is reproduced. The closest available affine-JANG
  Qwen3.6-27B-JANG_4M corruption class is covered in source now, including the
  forced-MLLM/panel `--is-mllm` route that could previously wrap a text-only
  fallback in the MLLM scheduler. Keep the exact 4L issue open until live
  model verification.
- Installed app packaging. Source bundle is fixed, but `/Applications/vMLX.app`
  still lacks `mflux`, proving release/install is stale.

## Changes Made In This Pass

1. Set Electron `minimumSystemVersion` to `14.5.0` in `panel/package.json`.
2. Fixed panel docs that incorrectly claimed macOS 26+ was required.
3. Pinned bundled Python dependency builds to macOS 14-compatible MLX wheels:
   `mlx 0.31.2`, `mlx-metal`, `mlx-lm 0.31.3`, `mlx-vlm 0.4.4`, and
   `mflux 0.17.5`. This prevents a Tahoe build host from embedding
   `macosx_26_0` wheels into a public app bundle.
4. Added a regression test that pins the 14.5 floor and rejects stale macOS 26+
   docs.
5. Fixed explicit parser opt-outs: `--tool-call-parser none` and
   `--reasoning-parser none` now remain hard disables instead of being
   re-enabled by registry auto-detection.
6. Fixed the `VMLX_DISABLE_TQ_KV` env typo in the CLI and made
   `vmlx_engine.utils.jang_loader` actually honor it before installing the
   live TurboQuant KV cache patch.
7. Added hybrid-SSM cache gating. For model families with non-standard cache
   state such as Ling/Bailing hybrid, auto mode disables live TurboQuant KV and
   generic scheduler q4/q8 KV quantization by default. Standard KV pages and
   hybrid companion state remain native until a typed hybrid codec exists.
8. Fixed a server-only Ling prompt corruption bug. `BatchedEngine` and
   `SimpleEngine` no longer append synthetic generic `<think>\n</think>\n`
   tags when `enable_thinking=False`; they only close an already-open think
   block. Ling's native template already renders its own thinking-off control,
   and the synthetic suffix changed the prompt from the canonical 92 tokens to
   a bad 97-token prompt that could loop.
9. Normalized the SSM companion memory env knob to `VMLX_SSM_STATE_CACHE_MB`
   while accepting the old `VMLINUX_SSM_STATE_CACHE_MB` typo as a fallback.
10. Removed the stale Qwen3.5/3.6 JANGTQ/MXTQ VLM text-only fallback; native
    JANGTQ Qwen VLM is now the default and was live-verified with images.
11. Added a separate affine-JANG Qwen hybrid policy: text-only for correctness,
    explicit 400 on media payloads, and no misleading `vision` capability in
    `/api/show`.
12. Closed the forced-MLLM hole for affine-JANG Qwen hybrid VLM bundles:
    `is_mllm_model(..., force_mllm=True)` now still returns `False`, and the
    Electron panel no longer marks those bundles multimodal from
    `architecture.has_vision=true`. MXTQ/JANGTQ Qwen VLM remains multimodal.
    Regression tests now execute the affine fallback and prove MXTQ reaches the
    native VLM branch instead of only grepping comments.
13. Added shared TurboQuant KV telemetry detection for nested MLLM language
    models so `/health` and `/v1/cache/stats` report the same live-cache state.
14. Added ZAYA/CCA cache-subtype tracking and compatibility policy. ZAYA is not
    allowed to fall through generic hybrid prefix/paged/L2/TQ-KV cache paths.
15. Added a fail-fast ZAYA loader error until `mlx_lm.models.zaya` or a
    `jang_tools.zaya` runtime exists and supports CCA `conv_state`/`prev_hs`.

This addresses the compatibility confusion behind mlxstudio #90/#104: the app
must not advertise or allow a runtime below the bundled MLX wheel's real floor.
It also addresses the Ling JANGTQ2 loop class seen in 1.5.20/1.5.23 where the
engine path, not the direct model path, generated repeated eyes/state-style
loops.

## Verification Run

Repo-local bundled Python build:

- `cd panel && npm run build`
- Result after ZAYA/Qwen/Gemma source changes: completed successfully; bundle verifier imported `mflux`,
  `mlx-vlm`, `jang_tools`, TQ kernels, Kimi/Gemma patches, and
  `vmlx_engine 1.5.24`.

Repo-local bundled Python:

- `panel/scripts/verify-bundled-python.sh`
- Result: all critical imports passed, including `mflux`, `jang_tools`,
  TurboQuant kernels, Kimi remaps, Gemma4 VLM patch, and patched mlx-lm Kimi MLA.

Panel issue tests:

```sh
cd panel && npm test
```

Result: `1595 passed`.

Panel typecheck:

```sh
cd panel && npm run typecheck
```

Result: passed.

Python issue-focused tests:

```sh
.venv/bin/python -m pytest -q \
  tests/test_engine_audit.py \
  tests/test_vl_video_regression.py \
  tests/test_dsv4_paged_cache.py \
  tests/test_cache_bypass.py \
  tests/test_mcp_security.py \
  tests/test_image_api.py
```

Result: `854 passed, 48 skipped`.

Latest focused Python engine audit after Qwen/Gemma/ZAYA additions:

```sh
.venv/bin/python -m pytest -q \
  tests/test_engine_audit.py \
  tests/test_mllm_processor_stream_contracts.py
```

Result: `204 passed`.

Broader API/VLM/reasoning/tool/image suite:

```sh
.venv/bin/python -m pytest -q \
  tests/test_mllm_processor_stream_contracts.py \
  tests/test_mllm_scheduler_cache.py \
  tests/test_mllm.py \
  tests/test_hybrid_batching.py \
  tests/test_tool_format.py \
  tests/test_reasoning_modes.py \
  tests/test_reasoning_parser.py \
  tests/test_streaming_reasoning.py \
  tests/test_reasoning_tool_interaction.py \
  tests/test_mcp_security.py \
  tests/test_image_api.py \
  tests/test_image_gen.py \
  tests/test_image_gen_engine.py \
  tests/test_api_models.py \
  tests/test_api_utils.py
```

Result: `859 passed, 6 deselected`.

Live Ling hybrid JANGTQ2 check:

- Model:
  `/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK`
- Command used repo-local bundled Python, continuous batching, parser opt-outs,
  default `--kv-cache-quantization auto`, and `--default-enable-thinking false`.
- Startup proof:
  - `Hybrid SSM cache model detected — disabling live TurboQuant KV and generic KV quantization for correctness (was: q4).`
  - `Reasoning: DISABLED (--reasoning-parser none)`
  - `TurboQuant KV skipped: VMLX_DISABLE_TQ_KV=1`
  - no `KV cache quantization enabled: q4`
- Exact Russian Three.js prompt that previously generated repeated-eye output:
  prompt tokens returned to canonical `92`, response was coherent Russian/HTML
  content, `512` generated tokens, tail unique-word ratio `0.883`.
- Post-rebuild smoke of the same prompt from the final repo-local bundle:
  prompt tokens `92`, `160` generated tokens, coherent mixed Russian/code
  planning content, tail unique-word ratio `0.877`.
- Cache-on repeat:
  - prompt tokens: `92`
  - `prompt_tokens_details.cached_tokens=86`
  - log: `hybrid paged HIT — 86 tokens (KV + 28 SSM layers)`
  - response remained coherent.
- Multi-turn recall:
  `teal-mango-42` was recalled exactly.

Live Qwen3.6 JANGTQ VLM check:

- Model:
  `/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK`
- Default startup now uses the native JANGTQ/MXTQ VLM fast path, not the stale
  text-only fallback.
- `/v1/chat/completions` image_url red-square request returned `Red square`.
- `/v1/responses` input_image red-square request returned `Red square`.
- `/api/show` includes `vision`.
- Long Russian Three.js prompt produced coherent planning/code content, not the
  previous multilingual token-salad class.

Live Qwen3.6 affine-JANG hybrid check:

- Model:
  `/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK`
- Forced MLLM/VLM path reproduced the user class: corrupt text-only output and
  corrupt image output.
- Direct text JANG loader produced coherent Russian planning/code content.
- Auto server startup now logs the text-only correctness policy and loads
  `mllm=False`.
- A later audit found one remaining forced path: the panel could still pass
  `--is-mllm` from JANG `architecture.has_vision=true`, and Python
  `force_mllm=True` returned before the affine-Qwen text-only policy. Source now
  overrides that forced path, and panel detection mirrors it so the MLLM
  scheduler is not constructed around a text-only fallback.
- The same long Russian prompt is coherent through the server.
- Media requests return HTTP 400 with a clear text-only-runtime message.
- `/api/show` no longer advertises `vision`.

Live Gemma-4-26B VLM check:

- Model:
  `/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK`
- Non-streaming image chat: `The color is red.`
- Streaming image chat: `The square is red.`
- Responses input_image: `The square is red.`
- 12 salted image requests held Metal active memory flat around 15.5 GB with
  no stream-thread errors.
- `/health` and `/v1/cache/stats` now both report nested live TurboQuant KV
  status.

## ZAYA Import

Copied from `erics-m5-max2.local` over the Thunderbolt bridge:

- `/Users/eric/jang/models/Zyphra/ZAYA1-8B`
- `/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ2`
- `/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ4`
- `/Users/eric/jang/models/Zyphra/ZAYA1-8B-MXFP4`
- `jang-tools/examples/zaya/`
- `convert_zaya_common.py`, `convert_zaya_jangtq.py`, `convert_zaya_mxfp4.py`

Header/config verification:

- JANGTQ2/JANGTQ4: `weight_format=mxtq`, no `local_experts`, 120
  `jang_config.tq_in_features` entries, all width `2048`.
- MXFP4: `weight_format=mxfp4`, no `local_experts`, pre-stacked
  `switch_mlp` expert layout.

Runtime compatibility note: ZAYA is CCA hybrid attention. Prefix cache should
stay disabled for the first Python/vMLX port until standard KV plus CCA
`conv_state` and `prev_hs` restore is implemented and tested. TurboQuant KV can
only target standard K/V pages; CCA state must remain native/float32.

Source policy now enforces that note:

- `ModelConfig.cache_subtype` preserves `zaya_cca` from `jang_config.json`.
- CLI disables prefix cache, paged cache, L2 block disk cache, and TQ-KV for
  `family=zaya` / `cache_subtype=zaya_cca`.
- `load_jang_model()` and `load_jang_vlm_model()` refuse ZAYA through the
  generic JANG path until a ZAYA-aware runtime module exists.
- Local ZAYA bundle checks are in `tests/test_engine_audit.py` and passed.
- Repo-local bundled Python smoke confirmed:
  `family zaya cache_type hybrid cache_subtype zaya_cca tool zaya_xml
  reasoning qwen3`, followed by the expected fail-fast
  `ZAYA model_type=zaya requires a ZAYA-aware runtime`.

## Cross-Check Matrix To Keep Honest

Every future "fixed" claim needs an explicit row for these dimensions:

- **Loader/runtime:** generic JANG v1/v2, JANGTQ/MXTQ native, affine-JANG,
  MXFP4, codebook VQ, CRACK variants, stacked/pre-stacked expert tensors,
  sidecar fast-load, safetensors header validation, and exact dependency
  versions in repo-local bundle and installed app.
- **Quant math:** Hadamard signs keyed by input width, codebook bit width,
  `dp_bits` for down projection, matmul shape repair, SwitchGLU fused path,
  per-expert 2D/3D stack handling, and explicit opt-out behavior for
  incompatible Smelt/Flash-MoE/TurboQuant combinations.
- **Cache contract:** standard KV, TurboQuant KV, DSV4 SWA+CSA/HSA native
  composite cache, hybrid SSM KV+companion state, ZAYA CCA KV+`conv_state`+
  `prev_hs`, prefix/paged/L2 disk store/fetch, block-key schema version,
  restart survival, `cache_salt`, `skip_prefix_cache`, and media-aware cache
  keys.
- **API surfaces:** OpenAI chat/completions/responses, Anthropic messages,
  Ollama chat/generate/version/tags, streaming/non-streaming, tool calls,
  `tool_choice`, reasoning on/off/max, `reasoning_effort`, normalized template
  kwargs, and Responses history/tool-output replay.
- **Panel/session:** model auto-detect, stale saved settings, session auto-start,
  forced MLLM, smelt/VLM mutual exclusion, chat UI streaming merge, stop/cancel,
  sleep/wake/JIT, app signing/notarization, and installed bundle parity.

Passing source tests is not enough for a release: each reporter-facing issue
needs source test + installed app test + exact model/API reproduction when the
model is locally available.

## Next Verification Required Before Release

1. Rebuild the signed app from source 1.5.24 and reinstall from the DMG, not by
   mutating `/Applications/vMLX.app`.
2. Confirm installed bundled Python reports `vmlx 1.5.24` and imports `mflux`.
3. Run the bundled verifier from the installed app path.
4. Live-check:
   - image model load path (`mflux`)
   - Ollama `/`, `/api/version`, `/api/tags`
   - Qwen3.6 text/VL routing on a local model
   - DSV4 cache stats show native composite path, not generic TurboQuant KV
   - DSV4 long-context output must be read to completion, not stopped at the
     first plausible sentence
   - cache bypass with `cache_salt` and `skip_prefix_cache`
   - Ling/Bailing hybrid cache with paged hit plus SSM companion fetch
   - MCP remote headers with a non-secret test endpoint or mocked local server

Known blockers not cleared by this pass:

- The installed app bundle still lacks `mflux`; the source bundle is fixed, but
  users need a clean signed/notarized rebuild and install.
- Qwen3.6 27B JANG_4L exact model is still not locally reproduced. The closest
  available affine-JANG Qwen3.6-27B-JANG_4M corruption was reproduced and fixed
  by defaulting that bundle class to text-only. Treat exact 4L Metal timeout as
  open until the exact model path is available and tested from a clean process.
- DSV4 long-context quality is not revalidated here because the user said the
  model is being re-downloaded. DSV4 cache verification must explicitly confirm
  native SWA/CSA/HSA composite cache handling and no generic TurboQuant KV
  override.
- ZAYA/CCA remains import-verified only; runtime cache reuse must stay disabled
  until CCA `conv_state`/`prev_hs` serialization and restore are implemented.

## §8.3 Empty `<think></think>` Injection — Layer-1 Verified (2026-05-06)

Audit doc: `docs/AUDIT-THINKING-TEMPLATE-RENDER.md`
Plan: `docs/superpowers/plans/2026-05-06-thinking-template-render-audit.md`
Tests: `tests/test_thinking_template_render.py`, `tests/fixtures/thinking_template_models.py`

Verdict (Layer 1, tokenizer-only render): audited local bundles honor
`enable_thinking=False` at the chat-template layer when rendered with the same
normalized kwargs as the engine. The blanket engine-side empty-pair append
removed in 94b16d22 is safe to keep removed, but Simple/Batched still retain a
narrow fallback that closes an already-open trailing `<think>` when thinking is
explicitly disabled and tools are absent. Treat that as compatibility debt, not
as a model-template fix.

Fix landed:

- Added a shared source-level chat-template kwarg normalizer that mirrors the
  resolved `enable_thinking` value to both `enable_thinking` and `thinking` for
  normal tokenizer templates. This handles templates that use the alias without
  requiring local bundle edits. Processor/VLM paths keep the stricter
  `enable_thinking`-only shape.

Layer-2 (live engine) deferred to a follow-up cycle for the three
`think_in_template=False` at-risk archs (Ling-2.6, Nemotron-Omni,
Gemma-4) — Layer 1 confirms template correctness; Layer 2 catches the
narrow case where the model auto-thinks despite a coherent prompt.

## §8.6 Post-Claude Source Truth Pass (2026-05-06)

Purpose: verify the 10 local audit commits on top of `94b16d22` and correct
claims that were not source-level fixes.

Findings:

- The local stack was docs/tests only until this pass. No runtime Kimi source
  fix had actually landed despite the commit summary claiming a Kimi template
  patch.
- Kimi live/runtime work is deferred per user direction. The source-level
  compatibility fix that did land is model-agnostic: normal tokenizer template
  rendering now mirrors resolved `enable_thinking` to both `enable_thinking`
  and `thinking`. This covers templates that use the alias without requiring a
  local bundle mutation.
- The prior wording that the engine-side `<think></think>` injection was fully
  removed was too broad. The blanket empty-pair append is removed; the narrow
  close-unclosed-`<think>` fallback remains in Simple/Batched when thinking is
  explicitly off and tools are absent. Track this as compatibility debt.

JANG docs re-read for current compatibility assumptions:

- `/Users/eric/jang/jang-tools/examples/dsv4_flash/README.md`
- `/Users/eric/jang/jang-tools/examples/dsv4_flash/04_long_context.py`
- `/Users/eric/jang/jang-tools/examples/zaya/ATTENTION_ARCHITECTURE.md`
- `/Users/eric/jang/jang-tools/examples/zaya/README.md`
- `/Users/eric/jang/jang-tools/examples/laguna/README.md`
- `/Users/eric/jang/jang-tools/examples/nemotron_omni/README.md`
- `/Users/eric/jang/docs/plans/2026-03-24-turboquant-integration.md`

Compatibility conclusions from that re-read:

- DSV4 has native SWA+CSA/HSA composite cache state. `DSV4_POOL_QUANT` is a
  DSV4-native compressed-pool codec, not generic TurboQuant KV. Generic
  q4/q8 KV/TQ-KV must stay forced off for DSV4. Tests pin both scheduler and
  CLI behavior.
- ZAYA CCA has standard KV plus `conv_state` and `prev_hs`. Prefix/paged/L2
  replay is not complete unless all three state families are restored at the
  same prefix boundary. Current Python policy remains fail-fast plus cache
  disablement until a real ZAYA runtime lands.
- Hybrid SSM async re-derive remains source-wired for prompt-only companion
  state and must be live-tested separately on Ling/Bailing/Nemotron before any
  release claim says the full matrix is production-verified.

Verification in this pass:

- `.venv/bin/python -m pytest -q tests/test_chat_template_kwargs.py tests/test_thinking_template_render.py tests/test_api_surface_parity.py`
  → `28 passed, 6 deselected`.
- `.venv/bin/python -m pytest -q tests/test_engine_audit.py tests/test_vl_video_regression.py tests/test_dsv4_paged_cache.py tests/test_cache_bypass.py tests/test_mcp_security.py tests/test_image_api.py tests/test_chat_template_kwargs.py tests/test_thinking_template_render.py tests/test_api_surface_parity.py`
  → `882 passed, 48 skipped, 6 deselected`.
- `.venv/bin/python -m pytest -q`
  → `2837 passed, 70 skipped, 92 deselected, 2 xpassed`.
- Qwen forced-MLLM/panel routing follow-up:
  `.venv/bin/python -m pytest -q tests/test_engine_audit.py::TestJangVLMFallbacks tests/test_dsv4_paged_cache.py::test_dsv4_serve_path_forces_generic_kv_quantization_off tests/test_chat_template_kwargs.py`
  → `11 passed`.
- Panel routing follow-up:
  `npm test -- --run tests/model-config-registry.test.ts tests/comprehensive-audit.test.ts`
  → `361 passed`.
- Panel typecheck follow-up:
  `npm run typecheck`
  → passed.
- Broader Python source regression after Qwen routing patch:
  `.venv/bin/python -m pytest -q tests/test_engine_audit.py tests/test_vl_video_regression.py tests/test_dsv4_paged_cache.py tests/test_cache_bypass.py tests/test_mcp_security.py tests/test_image_api.py tests/test_chat_template_kwargs.py tests/test_thinking_template_render.py tests/test_api_surface_parity.py tests/test_reasoning_modes.py tests/test_responses_history.py`
  → `897 passed, 48 skipped, 6 deselected`.
- Full default Python suite after Qwen routing patch:
  `.venv/bin/python -m pytest -q`
  → `2840 passed, 70 skipped, 92 deselected, 2 xpassed`.
- Full panel suite after Qwen routing patch:
  `npm test -- --run`
  → `1597 passed`.

GitHub issue state was re-queried with `gh issue list` for both
`jjang-ai/vmlx` and `jjang-ai/mlxstudio`. The map above remains the release
truth: several issue classes are fixed in source, but none should be closed
solely from source tests until a clean signed app/PyPI release is installed and
the reporter-facing path is reproduced.
