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
- **live-source verified** means the relevant real local model was loaded from
  source or repo-local bundled Python and passed a function-specific live
  matrix. The matrix must include multi-turn continuity when the feature is a
  chat/runtime/cache feature; full decoded output must be read to completion;
  streaming and non-streaming paths must be checked when the bug is streaming
  related; reasoning on/off/effort modes must be checked when the family uses
  reasoning; and cache claims must include prefix/paged/L2 hit-store-restore
  counters plus RAM/Metal/process telemetry.
- **release verified** means a clean signed/notarized DMG or PyPI package was
  installed and the reporter path was reproduced there.

Nothing in this pass is release verified. Do not close issues from source tests
alone.

### Live Verification Gate

Source tests are necessary but no longer sufficient for production claims.
Every future runtime/cache/model fix must produce a live artifact with:

- exact model path, git commit, command line, environment overrides, and server
  log path;
- protocol coverage appropriate to the bug: OpenAI chat, Responses,
  Anthropic, Ollama, streaming/non-streaming, tools, VL/audio payloads;
- reasoning coverage for reasoning families: off/on/effort or equivalent
  family modes, with reasoning and final content checked separately;
- multi-turn prompts that require prior-turn recall, not just a single easy
  prompt;
- cache telemetry before/after: prefix/paged hit counts, tokens saved, block
  disk/L2 writes and hits, cache bypass/salt behavior, and native-vs-TQ cache
  policy from logs/stats;
- resource telemetry: process RSS, system memory, MLX active/peak memory when
  exposed, TTFT, tok/s, prefill speed, prompt/completion tokens, and whether
  memory returns after request/server stop;
- full output capture with loop/garbage checks. A test that only checks HTTP
  200 or a short prefix is not sufficient.

### Strict Matrix Correction — 2026-05-07

The live family audit runner previously used a loose substring check for the
Anthropic and Ollama exact-answer probes. That could mark a row passing when
the expected token appeared only inside reasoning/thinking JSON while visible
content was empty or the request ended by `length`/`max_tokens`.

The runner now extracts only visible Anthropic text blocks and Ollama
`message.content`, normalizes that visible content, and rejects `length` /
`max_tokens` finishes for exact-answer probes. Existing artifacts must be read
under this stricter rule. In particular:

- MiniMax-M2.7-JANGTQ_K's saved Anthropic row is not a semantic pass; it
  length-capped in reasoning-like visible text before emitting the requested
  exact answer.
- Qwen3.6-27B-JANG_4M's saved Ollama row is not a semantic pass; the requested
  token appeared in `message.thinking` while `message.content` was empty.
- Cache-hit rows in those same artifacts can still be useful for cache
  telemetry, but they do not prove full protocol parity.

### GitHub Comment Audit

Live comment counts were refreshed with `gh issue list` on 2026-05-07. Issues
with source fixes or source-level test evidence now have an owner comment, or
are explicitly marked as pending push/package when the fix is local-only.

| Bucket | Issues | Comment state |
|---|---|---|
| Source-fixed/source-tested and commented | vmlx #147, #146, #145, #144, #141, #139, #138, #137, #132, #131, #124, #120, #119; mlxstudio #107, #106, #104, #103, #101, #100, #95, #90, #31 | Commented. Do not close until release/live verification criteria are met. |
| Local-only fix, comment says pending push/package | vmlx #141, #137, #124 | Commented after local commits; leave open until pushed/packaged. |
| Not fixed; no public fix claim should be made yet | vmlx #142, #136, #134, #123, #98, #88, #86, #79, #44; mlxstudio #89 | No code/test evidence sufficient for a fix comment. |
| Support/product/Swift/out-of-Python-scope | vmlx #148, #143, #133, #121, #122, #125, #126, #127, #135, #37, #57, #60, #74, #76, #78, #100; mlxstudio #102, #105, #61, #67, #68, #75, #91, #99 | May need responses, but not Python engine fix claims from this branch. |

### jjang-ai/vmlx

| Issue | Status in source | Remaining blocker |
|---|---|---|
| #148 Swift v2 custom-path freeze | Swift v2 app issue. | Out of Python/Electron session scope; do not claim a Python fix. |
| #147 Ollama version probe | Source has `GET/HEAD /` and `/api/version` gateway support. Panel tests passed. | Needs signed app release/live install verification. |
| #146 mflux missing | Source bundle installs and verifies `mflux 0.17.5`; repo-local bundle has it. | Installed app is stale and missing `mflux`; rebuild/reinstall required. |
| #145 image input `TokenizerWrapper` not callable | Source fixed and live-source verified on Qwen3.6-35B-A3B-JANGTQ-CRACK for `/v1/chat/completions` image_url and `/v1/responses` input_image. | Needs signed app release verification. Affine-JANG Qwen hybrid is now text-only by default and rejects media with a clear 400. |
| #144 RAM growth / hidden SSM companion | Source gates SSM companion by prefix-cache enablement and bounds companion cache. Gemma-4-26B image soak stayed flat over 12 salted requests. | Exact 32GB/g4-31b reporter profile pending. |
| #143 JANGTQ conversion access | Not a runtime bug. | Product/support response; no engine fix required unless docs/tool packaging changes. |
| #142 g4-31b crash | Related to Gemma/JANG loading and cache/memory safety. Source has current validators, but exact model not reproduced in this pass. | Needs live load with reporter model. |
| #141 DSV4-Flash conversion tools fail | Fixed locally from reporter stack traces: MLX conversion no longer crashes on `quantization.bits=None`; JANG converter now shrinks tiny DSV4 matrices to MLX-compatible group sizes instead of falling into the bad RTN fallback reshape. | Needs push/package and ideally a real DSV4 conversion smoke before close. |
| #139 VLM no stream in worker thread | Source fixed and live-source verified on Gemma-4-26B image streaming; no `No Stream(gpu)` error, coherent streamed output. | Needs signed app release verification. |
| #138 `--kv-cache-quantization` overridden by TurboQuant | Source tests confirm explicit q4/q8/none is respected; DSV4 uses native SWA/CSA/HSA cache and hybrid SSM disables generic KV/TQ-KV by default. | Needs packaged CLI check. |
| #137 prompt_lookup.py stale docs | Fixed locally: `prompt_lookup.py` now describes scheduler-owned PLD verification/reinsert/cache restore instead of "Phase 2 future"; regression test added; issue commented with verification. | Push/package pending. Not release-blocking for runtime. |
| #136 PFlash speculative prefill | Feature request. | Backlog; no source fix claimed. |
| #135 draft-model speculative under batching | Feature request/perf. | Backlog; no source fix claimed. |
| #134 PLD on hybrid SSM | Feature/perf proposal. | Backlog; overlaps SSM checkpoint/replay work but not a current bug fix. |
| #133 Swift directory picker | Swift issue, not Python/Electron session scope. |
| #132 Gemma 4 AWQ `.model` AttributeError | Fixed in local `jang` source and installed `jang 2.5.26`. | PyPI/release confirmation if publishing. |
| #131 MCP headers / streamable HTTP | Source accepts headers and streamable HTTP. MCP tests passed. | Live remote MCP token test pending. |
| #127 logprobs/top_logprobs | Feature/API parity request. | Needs separate implementation and tests. |
| #126 M1 Max shutdowns | Hardware/thermal/memory safety report. | Needs reporter repro profile and memory/Metal limits; not cleared. |
| #125 flash-moe | Feature/compat request. | Backlog; current source protects incompatible JANGTQ+Smelt/Flash-MoE paths. |
| #124 wrong font with Cyrillic | Triage: screenshot shows U+FFFD replacement characters in assistant output, not a font fallback; Python distributed streaming path now uses a streaming detokenizer instead of `tokenizer.decode([token])`. | The reporter screenshot is from vMLX 2.x/Swift, so leave open until Swift decode path is fixed/verified. |
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
- Ling/Bailing JANGTQ2 loop/corruption class: cache/runtime causes from the
  previous eye-spam failure are fixed in source, but the latest stricter live
  artifact does **not** clear JANGTQ2 CRACK as production quality. It stops
  cleanly and does not loop, but the Russian long prompt returns a weak
  125-character plan. Treat hybrid cache reuse as fixed; keep JANGTQ2 CRACK
  output quality open.
- Qwen3.6 JANGTQ VLM image path: live-source verified on
  `/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK`; native
  `jang_tools.load_jangtq_vlm` returns `Red square` for both OpenAI chat
  image_url and Responses input_image.
- Qwen3.6 affine-JANG hybrid VLM path: live-source reproduced corrupt text and
  corrupt image output through `mlx_vlm`; direct text JANG loader was coherent.
  Source now routes these affine-JANG Qwen hybrid bundles to text-only by
  default and returns a clear 400 on media requests instead of token soup.
- vmlx #124 Unicode replacement glyph class: the screenshot is not a font bug;
  it shows U+FFFD replacement chars in assistant output while the user Cyrillic
  prompt renders correctly. Python single-node scheduler already uses
  `NaiveStreamingDetokenizer`; this pass fixed the distributed generate loop so
  it no longer calls `tokenizer.decode([next_tok_id])` per token.
- vmlx #141 DSV4 conversion crash: source preflight now treats null/unknown
  quantization bits as full precision instead of multiplying by `None`; JANG
  converter now chooses the largest supported MLX group size that divides each
  tensor's input dimension, so tiny matrices like `(4, 32)` use group_size=32
  instead of failing at 128 and then reshaping incompatible fallback scales.
- Gemma-4-26B VLM path: live-source verified non-streaming image, streaming
  image, and Responses input_image. TurboQuant KV telemetry is now shared
  between `/health` and `/v1/cache/stats` for nested MLLM language models.
- ZAYA/CCA: source now preserves `cache_subtype=zaya_cca`, disables unsafe
  prefix/paged/L2/TQ-KV paths for that subtype, and loads through the local CCA
  runtime. Current strict live status is split by bundle: MXFP4 and JANGTQ4
  pass the text/API/cache-disabled contract; JANGTQ2 still fails basic
  instruction quality.

Latest regression runs:

- Python focused engine/API/cache/media suite:
  `854 passed, 48 skipped`.
- Broader API/VLM/reasoning/tool/image suite:
  `859 passed, 6 deselected`.
- Panel:
  `1595 passed`; typecheck passed.
- Focused follow-up after telemetry harness + hybrid cache fixes:
  `13 passed` across `TestMambaCacheCompat` and startup/cache guard tests.
- Broader scheduler/cache/API/reasoning/media regression slice after these
  changes:
  `903 passed, 48 skipped, 6 deselected`.

### Live-Source Artifacts Already Present

Local artifacts under `/tmp/vmlx_family_audit` show these latest statuses:

| Model family | Latest artifact | Status | Notes |
|---|---|---|---|
| Gemma 4 CRACK | `live_gemma4_crack_after_cache_fixes.json` | PASS in prior pass | Cache repeat, APIs, tool path, mixed-attention cache passed. Not rerun in the telemetry follow-up below. |
| Laguna | `live_laguna_tq_strict_tool_after_cache_fixes.json` | PASS | OpenAI chat, Responses tool call, Anthropic, Ollama, paged/L2 cache hit-store-restore passed with telemetry. Speed remains a separate performance blocker. |
| Ling MXFP4 CRACK | `live_ling_flash_mxfp4_crack_after_tq_zombie_fix.json` | PASS | Higher-bit control row passed; cache/runtime is healthy, output has minor mixed-language fragments but no loop. |
| Ling JANGTQ | `live_ling_flash_tq_after_tq_zombie_fix.json` | PASS | Non-CRACK Ling row passed strict cache/API checks; long Russian output is short but not looping. |
| Ling JANGTQ2 CRACK | `live_ling_flash_tq2_crack_after_tq_zombie_fix.json` | FAIL quality | Hybrid cache reuse passed, but the long Russian prompt returned only 125 chars/13 words. Do not mark production quality. |
| MiniMax M2.7 JANGTQ_K | `live_minimax_m27_tq_k_strict_after_cache_fixes.json` | PASS | Mixed gate/up/down bits, reasoning separation, exact multi-turn recall, Responses tools, and L2 hit path passed. Memory pressure is high. |
| MiniMax M2.7 Small | `live_minimax_m27_small_tq_strict_after_cache_fixes.json` | PASS | Exact multi-turn recall and L2 hit path passed with lower RAM pressure. |
| Nemotron Omni | `live_nemotron_omni_tq2_after_parser_fix.json`, `live_nemotron_omni_tq4_after_parser_fix.json` | PASS in prior pass | Uses native `jang_tools.nemotron_omni` path, not missing mlx-vlm module. Not rerun in telemetry follow-up. |
| Qwen3.6 27B JANG_4M CRACK | `live_qwen36_dense_jang_after_cache_fixes.json` | FAIL strict reasoning budget | Cache/runtime passed; thinking-on with 220-token cap stayed inside reasoning and produced no visible content. Separate bisect with 800 tokens answers `Blue`. |
| Qwen3.6 Russian long prompt | `/tmp/vmlx_qwen_russian_game_prompt.json` | PASS on local 4M bundle | The Russian Three.js prompt produced coherent Russian/HTML instead of token salad. This does not clear the user's exact JANG_4L Metal-timeout report. |
| DSV4 Flash | `live_dsv4_current_main.json` | stale FAIL | Not rerun because the model is being re-downloaded. Next DSV4 pass must use native SWA/CSA/HSA cache and read full long-context output to completion. |

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
- Ling JANGTQ2 CRACK long-prompt quality. The latest strict live artifact
  proves the cache stack does not loop, but the generated answer is still too
  weak to call production-ready.
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
15. Added the local ZAYA CCA runtime and JANG registration path. The old
    fail-fast wording is superseded: ZAYA now loads, but only MXFP4 passes the
    current strict text/API/cache matrix.

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
- `load_jang_model()` registers the local `vmlx_engine.models.zaya` runtime so
  JANGTQ/MXTQ ZAYA bundles hydrate through the ZAYA-aware CCA model class
  instead of a generic JANG path.
- `load_zaya_model()` handles BF16/MXFP4/affine ZAYA bundles through the same
  local CCA runtime.
- Local ZAYA bundle checks are in `tests/test_engine_audit.py` and passed.

Current strict live ZAYA status:

- Artifact:
  `/tmp/vmlx_family_audit/live_zaya_three_rows_after_tool_exemplar.json`
- `ZAYA1-8B-MXFP4`: PASS for current text/API/cache contract.
- `ZAYA1-8B-JANGTQ2`: FAIL. It passes contract/cache disablement, but basic
  chat returns the prompt paraphrase instead of `noted`, thinking-on recall
  length-caps inside reasoning with empty visible content, and Responses auto
  tool choice repeats user text instead of a structured `function_call`.
  Manual repetition-penalty probes at 1.0, 1.10, 1.15, and 1.25 did not clear
  the failure.
- `ZAYA1-8B-JANGTQ4`: PASS after adding parser-native Zyphra concrete tool
  exemplar injection. Basic chat/recall/API factual probes and Responses auto
  tool choice now pass.
- For all ZAYA rows, generic prefix/paged/L2/TQ-KV cache is correctly N/A/off
  until typed CCA prompt-state restore serializes standard KV plus
  `conv_state` plus `prev_hs`.

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
- ZAYA/CCA no longer remains import-only. MXFP4 and JANGTQ4 are strict-live
  passing for the current text/API/cache-disabled contract. JANGTQ2 is not
  cleared.
  Runtime cache reuse must stay disabled until CCA `conv_state`/`prev_hs`
  serialization and restore are implemented.

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

## §8.7 Live Telemetry Follow-Up (2026-05-07)

Purpose: make the cache/runtime claims measurable. The cross-family live audit
now captures request artifacts plus process RSS, system RAM, `/health`, MLX
active/peak memory, `/v1/cache/stats`, block-disk directory size, request
latency, token usage, and cache hit details around every major probe.

Source fixes verified by focused tests:

- Hybrid SSM batch-cache empty slots: patched prompt-cache finalize and
  `BatchKVCache`/`BatchRotatingKVCache.extract()` so unused KV slots do not
  crash prompt-cache extraction after recovery or padded scheduling.
- Generic TurboQuant zombie path: `_apply_turboquant_to_model()` now honors
  `VMLINUX_DISABLE_TQ_KV`, uses `model.make_cache()` instead of
  `len(model.layers)`, and skips hybrid SSM live TQ-KV unless
  `VMLINUX_ALLOW_HYBRID_KV_QUANT=1`.
- Hybrid CLI policy: when registry auto-detects `cache_type="hybrid"`, it
  forces `--kv-cache-quantization none` and marks the setting explicit so
  BatchedEngine does not later turn generic TQ-KV back on.
- Sampler-change paged-cache path: preserving block-aware paged cache across
  BatchGenerator recreation fixes the stale "paged hit then Block not found"
  failure. Direct memory/prefix caches are still invalidated when cleared.

Focused regression command:

```sh
.venv/bin/python -m py_compile tests/cross_matrix/run_production_family_audit.py \
  vmlx_engine/scheduler.py vmlx_engine/utils/mamba_cache.py \
  vmlx_engine/utils/tokenizer.py vmlx_engine/cli.py vmlx_engine/model_configs.py \
  tests/test_audio.py tests/test_engine_audit.py

.venv/bin/python -m pytest -q \
  tests/test_audio.py::TestMambaCacheCompat \
  tests/test_engine_audit.py::TestStartupCompatibilityGuards::test_turboquant_disable_env_is_honored_by_jang_loader \
  tests/test_engine_audit.py::TestStartupCompatibilityGuards::test_hybrid_ssm_auto_mode_skips_kv_quant_codecs \
  tests/test_engine_audit.py::TestStartupCompatibilityGuards::test_generic_turboquant_patcher_honors_disable_env \
  tests/test_engine_audit.py::TestStartupCompatibilityGuards::test_generic_turboquant_patcher_skips_hybrid_ssm \
  tests/test_engine_audit.py::TestStartupCompatibilityGuards::test_sampler_recreation_invalidates_pending_prefix_hits
```

Result: `13 passed`.

Broader follow-up:

```sh
.venv/bin/python -m pytest -q tests/test_engine_audit.py \
  tests/test_dsv4_paged_cache.py tests/test_cache_bypass.py \
  tests/test_reasoning_modes.py tests/test_responses_history.py \
  tests/test_api_surface_parity.py tests/test_chat_template_kwargs.py \
  tests/test_thinking_template_render.py tests/test_mcp_security.py \
  tests/test_image_api.py tests/test_vl_video_regression.py
```

Result: `903 passed, 48 skipped, 6 deselected`.

Live telemetry artifacts:

| Row | Artifact | Verdict | Cache/resource evidence |
|---|---|---|---|
| Ling MXFP4 CRACK | `/tmp/vmlx_family_audit/live_ling_flash_mxfp4_crack_after_tq_zombie_fix.json` | PASS | Hybrid SSM native cache; repeat hit `cached_tokens=26`; scheduler `cache_hits=2`, `tokens_saved=26`; L2 `disk_hits=2`, `disk_writes=9`; SSM companion `392MB`; RSS `47.5GB`, MLX peak `65.1GB`, system available `33.5GB`. |
| Ling JANGTQ2 CRACK | `/tmp/vmlx_family_audit/live_ling_flash_tq2_crack_after_tq_zombie_fix.json` | FAIL quality | Cache path passed with the same hybrid SSM/L2 shape (`cache_hits=2`, `tokens_saved=26`, SSM `392MB`), but Russian long prompt returned only `125` chars / `13` words. |
| Ling JANGTQ | `/tmp/vmlx_family_audit/live_ling_flash_tq_after_tq_zombie_fix.json` | PASS | Hybrid SSM native cache; repeat hit `cached_tokens=26`; L2 `disk_hits=2`, `disk_writes=9`; SSM `392MB`; RSS `29.5GB`, MLX peak `30.2GB`. |
| MiniMax M2.7 JANGTQ_K | `/tmp/vmlx_family_audit/live_minimax_m27_tq_k_strict_after_cache_fixes.json` | PASS | Mixed-bit gate/up/down path loaded; exact `noted`/`blue`; reasoning separated; tool call structured; cache `tokens_saved=46`; L2 `disk_hits=1`, `disk_writes=13`; RSS `35.6GB`, MLX peak `78.1GB`, system available `23.9GB`. |
| MiniMax M2.7 Small | `/tmp/vmlx_family_audit/live_minimax_m27_small_tq_strict_after_cache_fixes.json` | PASS | Exact `noted`/`blue`; cache `tokens_saved=46`; L2 `disk_hits=1`, `disk_writes=13`; RSS `5.2GB`, MLX peak `40.4GB`, system available `60.1GB`. |
| Laguna JANGTQ | `/tmp/vmlx_family_audit/live_laguna_tq_strict_tool_after_cache_fixes.json` | PASS | Responses auto tool call emitted structured `list_directory`; cache `tokens_saved=61`; L2 `disk_hits=1`, `disk_writes=13`; RSS `10.0GB`, MLX peak `11.1GB`. Health reports q4 KV cache quantization, but live `turboquant_kv_cache.enabled=false`; this is storage-boundary `QuantizedKVCache`, not live TurboQuantKVCache. |
| Qwen3.6 27B JANG_4M CRACK | `/tmp/vmlx_family_audit/live_qwen36_dense_jang_after_cache_fixes.json` | FAIL strict reasoning budget | Hybrid SSM cache path worked: `cache_hits=2`, `tokens_saved=18`, L2 `disk_hits=2`, `disk_writes=10`, SSM companion about `440MB` in memory and `1.03GB` on disk. The 220-token thinking-on recall probe length-capped inside reasoning with empty visible content. |
| ZAYA MXFP4 | `/tmp/vmlx_family_audit/live_zaya_three_rows_after_tool_exemplar.json` | PASS | CCA unsafe cache tiers disabled (`kv_cache_quantization.bits=0`, `turboquant_kv_cache.enabled=false`); generic exact-hit cache row marked N/A; OpenAI chat, Responses history/tool, Anthropic, Ollama `think:false`, and multi-turn recall passed. |
| ZAYA JANGTQ2 | `/tmp/vmlx_family_audit/live_zaya_three_rows_after_tool_exemplar.json` | FAIL quality | Contract/cache disablement passed; basic chat echoed/paraphrased the prompt instead of `noted`, thinking-on recall length-capped inside reasoning, and Responses auto tool choice repeated user text. Manual repetition-penalty probes at 1.0/1.10/1.15/1.25 did not clear the failure. |
| ZAYA JANGTQ4 | `/tmp/vmlx_family_audit/live_zaya_three_rows_after_tool_exemplar.json` | PASS | Contract/cache disablement, factual API probes, multi-turn recall, and Responses auto tool choice passed after the parser-native Zyphra concrete tool exemplar injection. |

Qwen follow-up:

- `/tmp/vmlx_qwen_bisect_results.json`: thinking-off recall answers `blue`
  immediately; thinking-on recall with `max_tokens=800` answers `Blue` and
  stops. The strict audit failure is a reasoning-budget/template behavior, not
  cache corruption.
- `/tmp/vmlx_qwen_russian_game_prompt.json`: on the available
  `Qwen3.6-27B-JANG_4M-CRACK` bundle, the user's Russian Three.js game prompt
  produced coherent Russian/HTML output (`2490` chars under a `900` token
  cap), not the token-salad/Metal-timeout output from the user's exact
  `JANG_4L` screenshot. The exact 4L model remains open.
- `/tmp/vmlx_qwen_sampler_fix2.log`: after the sampler-change cache fix,
  temperature changes preserve paged hybrid hits twice:
  `hybrid paged HIT — 43 tokens (KV + 48 SSM layers)`. No `Block not found`
  or reconstruction failure appears.

Production interpretation:

- Hybrid SSM cache reuse is now behaving as an owned feature for the tested
  Ling/Qwen rows: KV blocks are paged/L2-stored, SSM companion state is
  restored at the same prefix boundary, and generic live TQ-KV is disabled by
  default for correctness.
- MiniMax and Laguna pass the strict API/cache behavior checks, but MiniMax
  large mixed-bit runs close to memory pressure and Laguna still needs a
  dedicated speed benchmark before a performance claim.
- Qwen3.6 JANG_4M does not reproduce the exact JANG_4L Metal-timeout issue.
  Keep the GitHub issue open until the exact model path is available and tested.
- DSV4 is not cleared in this pass. When the model is available, verify native
  SWA/CSA/HSA composite cache, DSV4 pool compression, prefix/paged/L2 hits, no
  generic TurboQuant KV, and read full long-context output to completion.

## §8.8 MiniMax Thinking-Off Prompt Contract (2026-05-07)

Live MiniMax API parity exposed a prompt-contract bug, not a mixed-bit kernel
bug. `MiniMax-M2.7-JANGTQ_K` passed OpenAI chat thinking-off/reasoning/tool/cache
rows, but failed Anthropic and Ollama exact-answer probes by emitting visible
analysis text before `Paris`.

Raw prompt bisection on `/v1/completions` showed the cause:

- Native JANGTQ_K thinking-off template ends at `]~b]ai\n` with no think
  sentinel. The model opens `<think>...` itself and emits visible reasoning.
- Adding `]~b]ai\n<think>\n</think>\n\n` makes the same prompt return exactly
  `Paris` in two output tokens.
- `MiniMax-M2.7-Small-JANGTQ` already emits an open `<think>\n` in the
  thinking-off prompt, but the engine was closing it as `<think></think>\n`.
  That no-newline sentinel is also unstable. The stable close shape is
  `<think>\n</think>\n\n`.

Source change:

- `vmlx_engine/utils/chat_template_kwargs.py` now owns
  `ensure_thinking_off_sentinel()`.
- `BatchedEngine` and `SimpleEngine` both call it after chat-template render
  and fallback tool injection when `enable_thinking=False` and no tools are
  present.
- Tool requests are deliberately excluded because tool selection may need the
  planning rail; server-side reasoning suppression handles UX for tool flows.

Focused tests:

```sh
.venv/bin/python -m pytest -q \
  tests/test_thinking_template_render.py::test_minimax_thinking_off_adds_empty_thought_sentinel \
  tests/test_thinking_template_render.py::test_thinking_off_sentinel_closes_open_thought_with_stable_shape \
  tests/test_thinking_template_render.py::test_thinking_off_sentinel_does_not_touch_other_families_or_tools \
  tests/test_api_surface_parity.py tests/test_ollama_adapter.py \
  tests/test_tool_format.py::TestFallbackToolPromptFormat
```

Result: `32 passed`.

Live verification:

| Row | Artifact | Verdict | Evidence |
|---|---|---|---|
| MiniMax M2.7 JANGTQ_K | `/tmp/vmlx_family_audit/live_minimax_both_after_stable_think_sentinel.json` | PASS | OpenAI chat, Responses history/tool, Anthropic, Ollama, mixed-bit gate/up/down load, paged/L2 cache coherence all passed. Anthropic exact answer returned `Paris`; Ollama exact answer returned `Paris`; cache `tokens_saved=98`, L2 `disk_hits=2`, `disk_writes=13`, TurboQuant KV active for plain KV layers. |
| MiniMax M2.7 Small JANGTQ | `/tmp/vmlx_family_audit/live_minimax_both_after_stable_think_sentinel.json` | PASS | Regression check for templates that already emit open `<think>` in thinking-off. Anthropic and Ollama exact answers both returned `Paris`; cache `tokens_saved=98`, L2 `disk_hits=2`, `disk_writes=13`, TurboQuant KV active. |
