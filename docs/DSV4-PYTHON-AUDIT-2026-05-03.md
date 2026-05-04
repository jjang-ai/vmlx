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
