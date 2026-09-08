# Changelog

All notable changes to vMLX Engine will be documented in this file.

---

## [1.6.56] - 2026-09-07

- **Misaligned safetensors are repaired on disk during local-model preflight.** A bounded sibling-file copy preserves tensor names, shapes, dtypes and payload bytes, validates the replacement, then atomically replaces each affected shard. Nested media and draft shards are included. Repairs are resumable across interruptions and unchanged bundles are not rewritten. Unknown signed/file-hash contracts and unsafe paths are refused rather than silently invalidated; the supported local MTP proposal-head digest is updated transactionally. This is container realignment, not re-quantization or conversion of codebooks to another dtype.
- **Same-name model folders remain distinct.** Selecting an external or alternate local bundle no longer reuses an unrelated session merely because its directory has the same basename. Real filesystem aliases still resolve to the same model.
- **SSD pool capacity notice and guarded clearing.** Local server sessions show a dismissible notice only when the effective shared-pool ceiling is reached or capacity pressure automatically evicts entries. Manual clears and routine garbage cleanup do not trigger it. Clear requires confirmation, checks the live engine and cache root, refuses active requests or another live pool owner, and removes eligible disk entries while retaining protected data and resident state. Missing disk-only lookup hints are retired so they cannot mask a shorter newly stored prefix. Successful clearing dismisses the notice; the API reports bytes freed and remaining, without promising an empty directory.
- **Intentional no-MTP Flash-Next bundles avoid the redownload warning.** An explicit no-MTP declaration is respected only when the weight index also contains no MTP tensors. Old MTP-bearing layouts still use the update notice.
- **MiniCPM5 native tool and reasoning integration.** Bundles declaring the MiniCPM5 XML function dialect select the dedicated parser and native thinking controls in both the engine and desktop launcher. Complete CDATA arguments preserve whitespace and escaped data; schema types remain authoritative. Generic Llama bundles and model templates are unchanged.
- **Qualified MiniCPM5 affine projections share one packed QKV operation.** Native MiniCPM5 bundles with matching 8-bit, group-size-64 BF16 attention projections use the existing guarded projection grouping, addressing the reproduced partial-prefix continuation failure caused by shape-sensitive projection rounding. Other layouts retain their native path. The loaded projection layout has a separate cache identity, so earlier projection states cannot be restored into the changed runtime; model files and generation limits are unchanged.
- **Streamed answers retain their final punctuation.** Terminal reconciliation no longer mistakes a closing Markdown backtick, bracket or comparison sign for an unfinished tool marker. Chat and Responses preserve delayed detokenizer suffixes while withholding identifiable tool-control markup.
- **Native tool schemas stay authoritative.** Native tool declarations are compared structurally without confusing annotation names inside schema maps or literal values, or treating JSON booleans as numbers. Ordinary application JSON with a `name` field is not interpreted as a tool call without an explicit argument object.
- **MTP cache-hit accounting uses the full prompt.** Adaptive profile selection and governor context telemetry include restored tokens rather than only the freshly processed suffix. This does not change the configured depth or promise a universal speed floor.
- **Live API capabilities and sampling controls.** The API page derives capability badges from the active engine and rejects stale results after a session switch. MTP settings distinguish greedy startup defaults from an enforced Deterministic policy; explicit request sampling overrides remain visible in Auto mode.
- **Video controls report what they did.** A per-request video control the processor cannot honour as sent (a token budget below the smallest frame grid, a minimum pixel count above the budget, an explicit size the loader rounds, or frames dropped to meet a budget on a frame-fallback family) is applied best-effort and the effective frame count, resolution and media tokens are reported in the response `warnings` on every API and in the chat bubble. Nothing is reported when the request was honoured exactly.
- **Per-request image controls.** `image_max_pixels`, `image_min_pixels` and `image_resized_height`/`image_resized_width` on Chat, Responses, Anthropic and Ollama bound each image in a request-local copy before the processor; the bound is aligned to the processor's patch grid, every control is part of the cache identity, and a bound the processor's own floor or grid still changes is reported. `image_token_budget` is reported as unsupported on non-Gemma processors instead of being ignored.
- **Strict media controls.** `media_controls_strict: true` rejects an unmeetable or unsupported control with HTTP 400 `media_controls_unmeetable` on every lane, including the streaming lanes' error events and the Anthropic and Ollama envelopes. Documented in docs/api/media-controls.md.
- **Frame-fallback video budgets are enforced.** On families that present sampled frames as images, a `video_token_budget` now selects the frame count the image processor's pixel floor allows, and sampled frames no longer count against the request's image limit as if they were user images.
- **Prefix cache: a partial hit is no longer discarded after the companion delta is accepted.** A hybrid hit whose SSM checkpoint sat less than a block below the KV hit was re-prefilled in full since v1.6.36.
- **Media prefix cache preserves position state.** The hit path keeps the media positions for the text tail, text-path seeding no longer clears the rope state, and media prefill splits at the clean boundary. This addresses the reproduced cached-versus-cold divergence; bit-identical output is not a guarantee across all models, batching shapes or sampling settings.
- **SSD cache janitor runs while idle.** An idle engine now rescans its cache root after unscanned writes and on a 300-second cadence, so dead-writer temp files and unreferenced payloads age out without a request or a restart; `/health` reports the janitor constants and scan age.
- **Media prefix cache: a short question after a video is reused.** The store aligns into the media span and the fetch asks for the exact boundary, so a follow-up that ends inside the media still restores.

## [1.6.55] - 2026-09-06

- **Native MTP: adopting a running engine keeps its sampling policy and depth.** Adoption mapped the Auto session's own `deterministic-defaults` launch policy to the Deterministic override, which the launcher re-emitted as `greedy-only` on the next restart, pinning explicit request temperatures to greedy. Adoption now recovers the live engine's request policy, depth policy and effective depth (from `/health` first, the process arguments second) and maps them back so a restart relaunches the same policy: Auto stays Auto, the Deterministic override stays Deterministic, a disabled engine adopts as Off, and explicit depth 1/2/3 or Adaptive survive. Only an engine that exposes none of this falls back to the family default (fixed depth 3 for Qwen3.8 MTP families, as before).
- **Native MTP: one evidence rule for every tuning sidecar format.** A nested `native_mtp` or `best_native_mtp_depth` block that merely omitted `validated` could recommend depth 2 or 3, while the flat format demanded matched wall-speed evidence. Both the engine and the desktop registry now accept depth 1 from such a block as the conservative seed and require `validated: true` for anything deeper; blocked and invalidated blocks still contribute nothing.
- **Prefix cache: the terminal durability line reports what was actually stored.** The barrier that holds a request's terminal event until cleanup is durable used to log `cache_persisted=true` as soon as the cleanup fence was released, which happens in a `finally` whether the store succeeded, was skipped, was refused for budget, or raised. Every store site in both scheduler lanes now records its outcome for the request, and the barrier logs `cache_outcome=stored|already_durable|skipped|refused|failed|unknown` with the retained token count, whether the record is durable (an L1-only legacy store is not), the reason, and whether the client actually waited. A request with no recorded outcome is reported as `unknown`, never as persisted.
- **Tool calls: presence and schema are separate checks.** A required argument counts as missing only when the key is absent (or `null` for a non-nullable property); a blank string is a present value whose acceptability belongs to the schema (`minLength`). Parsed arguments are now validated against the tool's JSON Schema (types without coercion, enum, numeric bounds, string length, nested `required`, `additionalProperties`, nullable types). The default mode `warn` delivers the call and reports each violation as a warning so the agent can see why a tool rejects its arguments; `VMLX_TOOL_ARGS_SCHEMA_VALIDATION=enforce` drops such a call with the same explicit warning and error contract as a missing argument, and `off` restores presence-only checking. A malformed input schema is reported and treated as unconstrained; it never drops a call.
- **Tool calls: drop and schema warnings reach every API lane.** Non-streaming Chat Completions and Responses replies never carried the dropped-tool-call / schema-validation diagnostics (the capture was only started by the stream generators, so the warning reached the server log alone), and the chat stream withheld them whenever any tool call had been delivered. Both JSON lanes now merge the diagnostics into `warnings`, and the chat stream delivers them alongside a delivered call, so a warn-mode schema problem or a sibling call dropped for an unavailable name is visible to the caller. Ollama and Anthropic dialects inherit the change through the chat lane.
- **Prefix cache: the periodic global-budget rescan no longer holds a request's terminal barrier.** The first store after the block-disk budget's 30-second reconcile interval rescanned every cache namespace inside the publication critical section the durability fence waits on; on a large cache root that was a 4-5 s pause before the turn's final event. The rescan is now owed to the block-disk writer, which runs it only while quiet (no request waiting or running, nothing queued or in flight, no unfinished fence, a full second without writes) and logs its duration and outcome. Strict fences, missing accounting and ceiling crossings still reconcile inline, so the disk ceiling and the write, fence and publication order are unchanged. Measured on Qwen3.8-Flash-Next-JANG_4S with the shipped configuration: the first store after the interval fell from 4.9-5.0 s to 0.3 s; later stores stayed at 0.27 s.
- **Dependencies: mlx-vlm is bounded below 0.6 on macOS.** A fresh pip/uv install resolved mlx-vlm 0.6.17, whose Qwen3.5 rotary helpers no longer expose `apply_interleaved_mrope`, and Qwen3.8 prefill then failed on the exact-mrope path (`AttributeError` on a 4,320-token prompt). The bundled Python already pinned 0.5.0; the package metadata and lock now agree.
- **Chat history: a tool turn is replayed with its own reasoning on the next turn.** The persisted-history replay indexed reasoning segments by the live loop's step number, which starts at 1, so the next user turn re-rendered the first tool step with the final answer's reasoning and dropped the final reasoning. The rendered prompt no longer matched what the model had seen, and prefix reuse for agentic turns fell back to the boundary before the first assistant turn (340-490 tokens re-prefilled per step). Segments now map by step rank; a follow-up turn extends the previous prompt in full (traced token by token with the engine's prompt dump).

## [1.6.54] - 2026-09-05

### Fixed
- **Native tool prompts are no longer overridden for MiniMax-M2.7, Qwen3.8-27B and Nanbeige.** These families' chat templates render the tool schemas natively (a `<tools>` block plus an example call), but the fallback probe expected a different dialect and injected a second schema on every request, logging "Chat template needs fallback tool schema injection"; the conflicting instructions then produced tool calls that were dropped for empty required arguments. The probe now accepts a rendered schema block that names every tool. Verified across every bundle template on the drive: all 50 renderable templates take the native path.
- **Coding-tool Install works from the app.** The Install button spawned `npm` without the shell's PATH and failed with `spawnSync npm ENOENT`; it now resolves the installer against the usual install locations.
- **Hermes Agent has a manual-setup snippet** (the provider block for `~/.hermes/config.yaml`) like the other coding tools.

### Changed
- **Qwen3.8 Flash-Next: sparse-attention block index is no longer rebuilt from the whole history every token.** Above the 2,048-token sparse budget the QSA indexer re-pooled, re-normalized and re-rotated every completed 4-token block of the conversation on every decode step and on every speculative verify. Completed blocks never change, so their pooled keys are now retained on the cache (appended per newly complete block, truncated on speculative rollback, dropped on any restore) with the raw index history still the only authority. Measured on 4S plain decoding: unchanged at 1.7k context (path inactive), +3% at 6.6k, +7% at 26k; the same retained keys serve depth-3 verification. Kill switch `VMLX_QWEN4_QSA_POOL_RETAIN=0`.
- **Qwen3.8 Flash-Next: exact position angles in the sparse index.** The stock rotary formed position × inverse-frequency with a K=1 matmul whose result differs between one-row and multi-row chunks (up to 3e-4 rad), so the same position got slightly different angles at prefill and at decode. The index now uses an exact elementwise product for both queries and pooled keys.
- **Native MTP safety valve: a one-step seed is judged with a 10% band.** The autoregressive baseline measured at the start of a request is a single synchronized step; on 4S it read 12% faster than the true rate and demoted a depth-3 window that was beating real plain decoding. While the baseline is that seed the valve requires a 10% loss (`VMLX_NATIVE_MTP_SEED_COST_MARGIN`); once the request has a measured plain-decoding or depth-1 cost the exact 1.0 margin applies. The first request after a model load (cold graphs) and requests shorter than two judgment windows no longer decide the next request's starting depth, and a starting depth inherited from the previous request is re-examined after one window instead of 32 cycles. Measured: identical code produced 39 vs 50 tok/s at 1.7k context depending only on what the previous request happened to end in.
- **Qwen3.8 Flash-Next: grouped GDN projections and compiled hyper-connections now cover speculative verification** (2-4 rows), not only single-token decode. The grouped projection is bitwise identical to the separate ones; measured +0.6-0.7% at fixed depth 3 across 1.7k-26k context with byte-identical outputs, and verified in the app over three connected turns plus the raw API tool probe on the same engine. `VMLX_QWEN4_GDN_GROUP_MAX_ROWS=1` / `VMLX_QWEN4_HC_COMPILE_MAX_ROWS=1` restore decode-only.
- **Sparse-index cache (MiniMax-M3 / Qwen3.8 QSA): the raw index history grows in 256-row steps** instead of being re-concatenated at its exact length every token (an O(context) copy per token per sparse layer, 13.6 MB at 26k tokens). Persisted state, restore, clone, trim and the batch merge still see exactly the logical length.
- **Qwen3.8 Flash-Next: the fused expert pair kernel stays off while native MTP is active.** With the single-row pair kernel registered, fixed depth-3 decoding on 4M ran 39 / 43 / 34 tok/s at 1.7k / 6.6k / 26k context against 51 / 46 / 37 with it off, at identical or lower acceptance, and the greedy output became unstable between loads; on 4S the kernel on the draft head cost 15%. Processes that run the bundle's native MTP now keep the stock expert path (`VMLX_QWEN4_FUSED_MOE_PAIR=1` forces the kernel); plain-decoding processes keep the kernel's +3%.
- **Qwen3.8 Flash-Next: the fused expert pair kernel now covers mixed-width bundles.** 4S carries 2-bit gate with 3-bit up projections and 6S 4-bit with 6-bit, layouts the kernel could not unpack, so it silently kept the stock path on those bundles. The generic path handles per-projection widths including 3- and 6-bit; measured +2.2-2.7% on 4S plain decoding with byte-identical outputs. `VMLX_QWEN4_FUSED_MOE_PAIR_MIXED=0` restores the previous behaviour.
- **Sparse-index cache validation and accounting.** The live validator now requires the initialized index extent to equal the logical offset (spare backing capacity can no longer hide an incomplete lane), the single-sequence append fails closed on a desynchronised lane, and retained pooled keys are counted in cache memory accounting.
- **Qwen3.8 Flash-Next: exact position angles in the sparse index are separately switchable** (`VMLX_QWEN4_QSA_EXACT_ROPE=0` restores the stock rotary; pool retention then turns off). Measured: for any chunk of two or more rows the stock rotary rounds position × inverse-frequency with about 1e-3 relative error, so at 26k tokens the highest-frequency dimension differs by about 14 radians between prefill and single-token decode. An opt-in exact angle for the attention rotary (`VMLX_QWEN4_EXACT_ROPE_ATTN=1`) is included for a separate long-context quality evaluation.
- **Native MTP: truthful depth controls.** Both depth policies start every request at the configured depth and never exceed it. `fixed` keeps that depth and steps down only through the explicit AR-safety valve (depth 1, then plain decoding) when a measured window is slower than plain decoding, each transition logged with its reason; `adaptive` may also try depth 1 once against the configured depth's measured cost and keep the measured winner (on 4S at 26k tokens depth 1 ran 18-19 ms/token where depth 3 ran 26.5). `VMLX_NATIVE_MTP_AR_SAFETY=0` disables the valve; `VMLX_NATIVE_MTP_DEPTH_PROBE` overrides the depth-1 comparison for either policy. The per-request finish line now reports policy, configured depth, cycles spent at each depth and confirmed-output throughput over the request's MTP span.
- **Qwen3.8 Flash-Next PLE: duplicate rows in a table gather are read once** on the memory-mapped path (the pread path already did).
- **Tool calls: an explicit `null` for a required argument is a value when the tool's schema declares the property nullable.** The parsed-call validator treated any `null` as a missing argument and dropped the call, so a tool whose schema listed a property as `required` with `type: ["string", "null"]` (or `nullable: true`, or an `anyOf` branch with `type: "null"`, the shape OpenAI strict-mode SDKs emit) lost every call in which the model passed `null` exactly as asked; the client then received a warning chunk and a `reasoning_only_no_content` error instead of the call. Measured on the packaged candidate with Flash-Next 4S: four of four such requests were dropped. Absent keys and blank strings are still missing, and a `null` for a non-nullable property still counts as missing.
- **Native MTP: the plain-decoding baseline is re-measured during the request, and a long answer can no longer get stuck in plain decoding.** The safety valve compared each window against a baseline seeded by one synchronized step at the start of the request, so a baseline that drifted over a long answer was only re-measured after a trip. The governor now runs a bounded calibration: when the cycle wall at the current depth has drifted 25% or more from its anchor, or 768 emitted tokens have passed since plain decoding was last measured, it decodes 8 tokens autoregressively at the live context, records that cost as the baseline, and resumes at the depth it left with a fresh judgment window. Its cost is bounded by spacing rather than by a per-request count (a drift-triggered calibration needs 512 emitted tokens since the last measurement, so at most 8 plain-decoding tokens per 512), and a measurement that reads more than 1.5x the previous one is re-checked after 256 tokens instead of 768, so a reading taken during a transient slowdown cannot stay in force for the rest of a long answer. `VMLX_NATIVE_MTP_AR_CALIBRATION=0` disables it. A calibration is not a loss: it spends no re-entry backoff, teaches nothing to the session profile, and a request that finishes during its plain-decoding steps does not start the next request at depth 1. Re-entry after a real loss is no longer capped per request: the wait between probes doubles per failed or quickly re-tripped probe up to 4,096 plain-decoding tokens, relaxes again after a re-entry that survived 256 output tokens, and probes alternate between depth 1 and the configured depth. The final rung (depth 1 to plain decoding) now needs a second losing window within three windows of the first, because on long answers single 8-cycle depth-1 windows lose by a few percent while the surrounding output wins. Measured through the packaged app before this change: a 9,843-token answer spent 8,406 tokens in plain decoding at 34 ms/token after four re-entries were kept and later re-tripped, while depth 1 measured 20 ms/token at the same 14k context. Measured on Flash-Next 4S at 6.6k context with adjacent plain-decoding controls (1,024-token answers, hot chip): plain decoding 37.9-38.5 tok/s, fixed depth 3 with the valve 44.1, adaptive 45.4, depth 3 with the valve off 47.3, and no 64-token window of either governed run more than 5% slower than the matched control. The per-request AR-tier summary reports calibrations and the wall time spent in plain decoding, and `VMLX_NATIVE_MTP_CYCLE_TRACE=1` logs every verify cycle and every plain-decoding step with the depth, acceptance, baseline in force, epoch, cumulative output tokens and a clock.
- **Native MTP depth ladder: never below plain decoding, and back up when it wins.** The speculative-decoding safety valve now trips as soon as the windowed cost per token exceeds the request's own autoregressive step (no 25% tolerance). A losing configured depth first drops to depth 1, which is often faster than depth 3 on ordinary prose, and only a losing depth 1 drops to plain decoding. Re-entry probes start at depth 1, are judged within a few cycles and abort early when clearly losing, and a promotion probe climbs back to the configured depth only when it measurably beats the depth-1 cost. A request that ended in plain decoding or depth 1 starts the next request at depth 1 with a scheduled promotion probe, so a losing conversation stops paying for a depth-3 opening every turn. Each request gets a bounded number of retries and then stays on its current rung. Measured on Flash-Next 4M over a 14-turn chat to 16k context: mean decode ~33 → ~38 tok/s with far fewer sub-AR turns; 4S unchanged.

## [1.6.53] - 2026-09-04

### Fixed
- **Qwen3.8-Flash-Next MTP works again on the fixed bundles.** 1.6.52 could not detect, load, or use MTP on the re-uploaded Flash-Next bundles: the loader discovered weight files by a filename glob and swept the calibrated proposal-head sidecar (which carries `lm_head.*` copies) into the weight load, aborting every launch with a duplicate-tensor error. Weight files now come from the model index (`model.safetensors.index.json`), so the MTP tensors load from the regular shards that name them and unrelated sidecar files are ignored; a file the index references but is missing still fails loudly.
- **New Qwen3.8 MTP sessions start at fixed depth 3, not adaptive.** The Server panel's displayed depth default was never sent, so fresh Flash-Next and Qwen3.8-27B sessions silently launched with an adaptive depth policy. New sessions for these families now start at fixed depth 3; an explicit depth choice (including switching off the fixed override) is preserved and saved.

### Added
- **Calibrated multi-token-prediction proposal head.** When a Flash-Next bundle ships an imatrix-calibrated q4 proposal head (`mtp_draft/vmlx_mtp_proposal_head.safetensors`, pinned by the bundle stamp), the runtime verifies it by SHA-256 and uses it for speculative drafting; any missing, mismatched, or malformed artifact falls back silently to the runtime-rebuilt head. Speculative verification always uses the checkpoint head, so the calibrated head only affects draft acceptance, never output.
- **On-the-fly AR safety valve for native MTP, at every depth policy, with automatic re-entry.** Fixed-depth MTP sessions (the new Qwen3.8 default) previously ran with no fallback at all: a request whose speculative decoding fell below plain autoregressive speed stayed slow for the whole response. Every verify cycle now compares the recent windowed cost per token against the request's own measured autoregressive step (a floor that grows with context), and drops the request to plain decoding within about 16 cycles when speculation is genuinely losing. The drop is not permanent: while decoding plainly the request measures its true autoregressive speed at the live context and re-probes speculation with exponential backoff, resuming the configured depth as soon as it beats that measurement. It ignores a single stalled cycle, never fires during warmup, applies to chat and API alike, and can be disabled with `VMLX_NATIVE_MTP_AR_SAFETY=0` (re-entry: `VMLX_NATIVE_MTP_AR_REENTRY=0`).
- **Hybrid prefix restores verify attention/recurrent alignment.** Restoring a cached prefix for a hybrid model (attention layers plus Mamba/GatedDelta/KDA state) now checks that every restored attention layer sits at exactly the token boundary the recurrent state was saved at. A mismatch, which would make the two halves of the model decode from different positions, is rejected with a logged warning and the turn prefills in full instead.
- **MTP prompt priming survives block-aligned prompts.** In a multi-turn chat whose prompt length landed on an exact cache-block multiple, the speculative head's prompt history was saved one token past the boundary the prefix cache restores, so every later turn ran without priming (lower acceptance, 25-35 instead of 45-60 tok/s on Flash-Next). The history is now keyed at the boundary the cache actually restores.
- **One-time "model updated" notice for Flash-Next.** Loading a Qwen3.8-Flash-Next model (any tier or variant) whose local copy predates the MTP fix shows a one-time, dismissible in-app notice — localized in all supported languages — offering to redownload the model; a copy that already carries the fix never prompts.

### Fixed (UI)
- **Language picker menu is reachable.** The title-bar language dropdown was clipped by the title bar and painted over by the main content; it now renders above all content so every language is selectable.

---

## [1.6.52] - 2026-09-02

### Fixed
- **Concurrent batch requests to Qwen3.8-Flash-Next no longer crash.** The QSA sparse-attention layers keep a native index lane that the generic continuous-batching cache promotion discarded, so the first co-batched decode step aborted with `'BatchKVCache' object has no attribute 'update_index'`. A specialized batch cache preserves the sparse index through merge, extract, filter, extend, trim, and finalize. Single (non-batched) requests were unaffected, which is why one-at-a-time calls worked while concurrent load failed.
- **Native MTP restores greedy startup defaults for every enabled mode.** Auto had regressed to keeping the bundle's sampled temperature as the default, so app-managed MTP sessions (Qwen3.8-27B, Flash-Next) silently ran stochastic decoding. Enabling MTP now pins temperature 0 / top_p 1 as the startup default for chat and API alike; an explicit per-request temperature still wins. Chat Settings displays the pinned default.
- **GLM-5.3-Flash stops replaying prior-turn reasoning into the prompt.** The template's `clear_thinking` defaulted false, re-embedding each earlier turn's full reasoning and growing the prompt until it tripped the prefill guard. It now defaults to the flat-history contract; explicit callers still override.
- **External-drive model bundles load past macOS AppleDouble sidecars.** `._`-prefixed metadata files on exFAT/FAT volumes are no longer mistaken for safetensors shards.
- **Honest launch-memory estimate for the SSD-backed PLE.** Qwen3.8-Flash-Next keeps its quantized PLE table on SSD; the resident estimate now accounts for that, and the Metal wired-limit recommendation is advisory-only (never blocks the launch).

### Added
- **codex 0.150 compatibility.** The gateway model list carries an additive `models` field for codex's model-manager alongside the standard `data`. codex must be configured with `wire_api = "responses"` (it no longer supports the chat wire).
- **Opt-in BF16 DSA indexer state for GLM-5.3-Flash** (`VMLX_GLM5_DSA_BF16`, default off).

### Changed
- **GLM-5.3-Flash speculative-verification hyper-connection transforms are fused by default**, and the multimodal load path now runs the post-hydration acceleration hook.

---

## [1.6.51] - 2026-08-31

### Fixed
- **Native MTP detection includes supplemental weight shards omitted from the base Hugging Face index.** The Server panel and Python runtime now discover preserved MTP heads in split layouts such as Qwen3.8-27B, restoring the Native MTP controls and effective runtime status without loading tensor payloads. Unreadable MTP-labeled supplements fail closed, while unrelated custom sidecars do not invalidate an otherwise sound index.

---

## [1.6.50] - 2026-08-31

### Performance
- **Qwen3.8 Flash Next now selects routed-MoE fusion atomically for mixed-bit bundles.** Fully compatible JANG 2L and 4M stacks retain the fused affine path, while mixed-layout 4S and 6S stacks keep the faster stock MLX decode path instead of interleaving a small number of custom layers. Exact fixed-D3 measurements restored 4S from 74.8 to 87.5 tok/s and 6S from 75.4 to 82.9 tok/s while preserving 100% draft acceptance.
- **GLM-5.3 Flash decode and tight-memory prefill use their qualified native paths by default.** KDA decode fusions remain shape-gated, and memory-pressure prefill chunks now stop at recurrent-state boundaries rather than splitting through an unfinished SSM transition.

### Fixed
- **Tool terminals are published only after the parsed assistant turn is durable.** Chat Completions and Responses tool loops persist the exact KV and architecture-native companion boundary before the terminal event can trigger the next harness step.
- **Tool contracts remain stable across growing multi-turn histories.** The engine reconstructs the effective tool schema at the owning request boundary and drains terminalizing requests without converting a normal tool completion into an abort.

---

## [1.6.49] - 2026-08-31

### Performance
- **Qwen3.8 Flash Next fixed-depth MTP keeps the selected D3 path active through real tool loops and long ordinary chat.** Request-local sampling, prompt-history priming, affine MoE projection pairing, and exact depth telemetry now stay aligned instead of silently falling back to a different policy. The release proof preserves stochastic sampling, tool-result continuation, hybrid cache restoration, and fixed depth 3 on the same live session.

### Added
- **Every production inference loader now preflights the resolved local model bundle before constructing weights.** The shared gate recursively covers base, MTP, vision, audio, nested Diffusers, JANG affine, and JANGTQ safetensors without requiring optional model configuration files. Valid safetensors payload offsets are preserved; only objectively reconstructible standard weight-index defects are repaired through a locked, fsynced, atomic replacement.

### Fixed
- **Incomplete or corrupt local model artifacts fail closed before allocation.** Active downloads, missing indexed shards, malformed headers, invalid present signature JSON, and failed index repairs can no longer reach a partial model load. Hub identifiers are checked after their one authoritative snapshot resolution, while unchanged local bundles use an external fingerprinted integrity stamp.
- **Agentic API behavior remains source-matched across direct and Electron-gateway routes.** Chat Completions, Responses, Anthropic, and Ollama stream/non-stream tool calls, tool-result continuation, cancellation recovery, reasoning separation, and raw capture pass against the same Qwen3.8 Flash Next fixed-D3 engine used by the real Electron proof.

---

## [1.6.48] - 2026-08-30

### Performance
- **GLM-5.3 native MTP verification now batches affine KDA projections across the speculative slab.** The optimized verifier preserves sequential convolution and recurrent state, exact accepted-prefix rollback, and output bytes while using the stock MLX affine QMM/TensorOps/NAX path. Matched live measurements selected fixed D2 for the current GLM-5.3-Flash JANG MTP bundle; D3 remains slower for that bundle and is not forced globally.

### Added
- **Native-MTP and cache benchmarks can use stable prompts, exact depth arms, and typed prompt-SSD restart namespaces.** The harness records durable request identity and family-native cache restoration so speed claims remain attributable to AR, D1, D2, or D3 instead of mixed cache state.

### Fixed
- **Hugging Face downloads no longer fail after transfer begins when the Hub client refreshes its progress bar.** The app's structured byte-progress adapter now implements the `refresh()` method required by current `huggingface_hub` download paths.

---

## [1.6.47] - 2026-08-30

### Performance
- **Qwen3.8 Flash Next starts adaptive native MTP from measured D3 when bundle evidence is absent or incomplete.** The request-local controller still compares wall value and may demote or restore depth; complete bundle tuning remains authoritative. Matched 4S and 4M sweeps selected D3 with byte-identical AR/MTP outputs and measured adaptive speedups of 2.19x and 1.81x respectively.
- **Flash-Next primes its native MTP head from prompt history.** Cold prompts and exact restored-prefix tails reuse the already-computed pre-norm trunk state, including the engine's N-1 partial-terminal-block boundary, without introducing an unversioned SSD schema.

### Fixed
- **Incomplete native-MTP tuning records no longer pin a bundle to an unmeasured optimum.** A flat best-depth record above D1 must cover every supported depth through its declared ceiling. Complete D1/D2/D3 tuning, including Qwen3.8-27B's measured profile, remains authoritative.
- **Dense Qwen3.5 prompt priming is fail-safe.** The implementation remains available for explicit measurement, but stays off by default because its matched 27B trials improved acceptance without a repeatable wall-time win.
- **Signed-candidate Actions restore only hash-sealed retained evidence for the exact source SHA.** Clean GitHub checkouts no longer discard the current live/packaged receipts required by the production prepackage gate; the gate still revalidates those artifacts before building.
- **Dedicated release runners can use a prevalidated notarytool Keychain profile.** Developer ID signing still uses an ephemeral per-run keychain, while the existing App Store Connect credential remains non-exportable in the runner's login Keychain.
- **Candidate provenance validation runs on the runner's system Python.** The version check no longer assumes Python 3.11's `tomllib` before the workflow creates its isolated release environment.
- **Dedicated release runners may keep the Developer ID identity non-exportable.** Candidate CI validates the exact ShieldStack team identity in the login Keychain instead of requiring a PKCS#12 round trip.
- **Candidate dependency setup rejects macOS's legacy system Python.** Release CI now creates its isolated environment from the documented Homebrew Python 3.11–3.14 range.
- **Candidate Python provenance stays checkout-local.** Release CI now creates its CPython 3.13 environment with `uv`, preventing Homebrew framework resolution from escaping the candidate checkout's `.venv` provenance gate.
- **Packaged engine sources no longer share hard links with the release checkout.** The Electron `afterPack` hook atomically detaches `extraResources` source files before signing, preserving the strict single-link provenance gate while keeping packaged bytes identical.

---

## [1.6.46] - 2026-08-30

### Added
- **Native GLM-5.3-Flash runtime.** vMLX now serves GLM-5.3-Flash text bundles through its owned `glm5_next` implementation, including KDA linear attention, DSA sparse indexing, mHC, affine MoE, native reasoning/tool parser aliases, native MTP attachment, and typed SSD prompt-cache state.
- **Visible wired-memory guidance.** When a model is close to the effective Metal wired-memory limit, the app shows the measured model/limit comparison and the exact temporary `sysctl` recommendation while still allowing the user to continue.
- **Peak benchmark profiles and acceleration telemetry.** The Server panel exposes repeatable best-case profiles plus request-exact MTP and fused-decode status instead of inferring acceleration from configuration alone.

### Performance
- **Adaptive Native MTP now protects the AR baseline.** Fixed-depth probes use their requested depth, reuse matched AR controls, warm kernels before measurement, expire losing observations, and activate a depth only when a validated profile or current measurement beats autoregressive decoding. Qwen3.8 Flash Next selects its proven depth-2 profile; unmeasured or slower GLM profiles stay on AR.
- **Qwen and GLM decode paths use fewer Metal dispatches.** This release groups affine projections and adds source-gated fusions for Qwen QSA/GatedDeltaNet/PLE and GLM KDA/DSA/mHC/MoE components while retaining family-specific math and runtime telemetry.

### Fixed
- **Agentic coding API loops are release-gated across every supported protocol.** Chat Completions, Responses, Anthropic, and Ollama now have exact raw-capture coverage for streaming/non-streaming reasoning, required and explicit tool calls, tool-result continuation, direct final answers, malformed/cancelled requests, and replay through both direct and Electron gateway routes.
- **Qwen3.8 Flash Next parser routing remains correct for previously mis-stamped bundles.** Engine detection, app launch arguments, and saved-session migration resolve the affected Flash-Next bundles to the native Qwen tool parser without changing unrelated Qwen families.
- **MLLM streaming finalization no longer drops or duplicates terminal text.** Final detokenizer bytes and normalized suffixes are reconciled once across streaming rails.
- **Adaptive MTP settings are applied at the request boundary.** App/UI selection, fixed depth controls, sampling compatibility, health telemetry, and the effective runtime policy now describe the same request.
- **Busy request lifecycle health remains observable.** Gateway cancellation and exact backend ownership stay visible while long prefills or tool continuations are active.

---

## [1.6.45] - 2026-08-28

### Added
- **Server-exact prefix-cache fetch provenance.** `/health` and `/v1/cache/stats` now report, per request, exactly which mechanism produced the restored token count — chain-hash block hit, N-1 partial-index fallback, DSV4 delta checkpoint, rotating-SWA anchor normalization, or a genuine miss with its reason — plus the restored token count, RAM-vs-SSD source, and an internal-continuation label so a retry pass can never masquerade as the base request's own result. Pure observability: recording provably never changes what the cache returns.
- **`/v1/completions` accepts `reasoning_effort` and `enable_thinking`.** Both fields were previously dropped silently. An explicit value is honored; when neither is sent, the thinking rail resolves through the bundle's own stamped chat contract (`jang_config.chat.reasoning.default_mode`) exactly like the DSV4 completions rail — unstamped bundles keep the historical no-thinking behavior.
- **Protected release automation.** Dev builds and signed release candidates now flow through separate GitHub Actions pipelines; unsigned dev artifacts can no longer be confused with notarized candidates.
- **Engine-owned lifecycle load progress.** The engine now reports its real load/wake lifecycle — phase, measured shard units, model_loaded, authoritative ready, and a per-attempt generation — mirrored as structured stdout lines during cold start and embedded in /health (which is how an externally triggered API JIT wake becomes visible in the app). The panel maps this contract to the bar on every surface: Server session card, opened session/chat header, the visible Chat tab, and Create Session. Determinate percentages appear only for measured work (weight shards, capped at 99); phases without a real denominator render as an indeterminate animated bar; 100% is reserved for the engine's readiness barrier. RSS/Metal residency remains a separate diagnostic readout and is never the percentage. Stale events from an older attempt (Stop, restart, PID replacement) are discarded by generation, and pages opened mid-load hydrate the current state instead of starting blank.

### Fixed
- **Messages sent while a model is loading no longer time out.** The chat path queued no message and failed after a blind 30-second health poll while a legitimate multi-minute load was in progress; /health answering HTTP 200 was also mistaken for inference readiness (it answers 200 in every state). A message sent mid-load now queues exactly once until the engine's authoritative readiness — watching live load progress — and the inference inactivity watchdog arms only when inference actually starts. An explicit Stop cancels the queued message immediately (the request's endpoint is registered before the wait so Stop can find it), and a mid-wait engine process replacement aborts the queued message instead of silently riding the new process.
- **A loading session can no longer masquerade as running.** The session view promoted any health event to "running", unmounting the load bar mid-load and enabling premature sends that then timed out.
- **Save & Restart is now one atomic main-process lifecycle operation.** The renderer previously orchestrated update → stop → start over separate IPC calls; an explicit user Stop landing between the pair could not cancel the restart's start (the start captured the post-Stop lifecycle epoch at entry and passed), leaving an untracked engine running while the UI said Stopped. `restartSession` carries the generation captured when the restart begins — after its own stop, it starts only if no later explicit Stop advanced the generation. The cancellation check also now runs before single-model detection/adoption side effects, closing an early-return adoption path that never consulted the epoch.
- **`vmlx-serve doctor` no longer false-negatives on native-MTP checkpoints (#262).** Doctor's inference test built the MTP MoE per-expert while native-MTP bundles store it stacked, so weight binding failed on checkpoints that `serve` loads and runs fine. Doctor now applies the same native-MTP sanitize patches as the serve path before building the model tree (still a pure smoke test: the MTP decode runtime stays off, like its existing TurboQuant skip).
- **DSV4 `/health` native-cache attestation no longer omits the CSA indexer pool.** The health endpoint's `components` list disagreed with the memory-estimate function's own list (4 vs 5 entries); both now derive from one constant, so `csa_indexer_pool` is visible again.

---

## [1.6.44] - 2026-08-28

### Performance

- Qwen3.8 Flash Next PLE row lookup now performs contiguous, page-aligned reads
  from its SSD-backed table instead of issuing one random read per candidate.

### Security

- Patched production npm dependencies: DOMPurify, js-yaml, PostCSS, uuid,
  nanoid, and picomatch. The Electron/extract-zip advisories require an
  Electron major upgrade and are deferred to the next release.

### Fixed

- An explicit session Stop now cancels any queued or in-flight start, so a
  Stop landing inside a Save & Restart can no longer leave an untracked
  engine running while the session reads Stopped.
- Ollama `/api/chat` and `/api/generate` report a deep-sleep JIT wake as
  `load_duration` and include it in `total_duration`; prefill/decode splits
  stay wake-independent.
- The vendored Qwen 3.5 / Qwen 3.5 MoE runtime overlay purges stale upstream
  submodules before installing, so importing Qwen4 code first can no longer
  produce a half-vendored runtime missing the router-gate quantization fix.
- A sidecar JANG capabilities stamp that names no model family no longer
  selects cache or parser behavior; detection falls back to structural
  config.json architecture detection and logs the fallback.
- Session Settings cache labels are translated in Spanish, Japanese, Korean,
  and Chinese instead of falling back to English.

- Electron tolerates orphaned terminal `read EIO` and `write EIO` events while
  continuing to surface unrelated stream and filesystem failures.
- Native MTP preserves sampled-acceptance and tool-boundary semantics, releases
  aborted PLD history, and prevents stale tool reasoning from leaking into a
  later no-tool turn.
- Message-triggered wake keeps model progress visible and serializes competing
  deep-sleep wake paths so one model instance owns the transition.
- Long local DeepSeek V4 prefills use backend-progress-aware inactivity
  timeouts, including a keepalive-safe client floor, instead of aborting at a
  short wall-clock deadline.

---

## [1.6.43] - 2026-08-27

### Fixed

- DeepSeek V4 Flash 0731 bundles that omit the auxiliary tokenizer encoder now
  use the packaged canonical 0731 encoder after strict bundle fingerprinting;
  unknown revisions still fail closed.
- Deep Sleep detaches generic compiled model graphs before engine teardown so
  repeated dense-model wake cycles do not retain another compiled model
  closure in process memory.
- Auto-sleep honors the earliest configured light or deep deadline, and active
  sessions keep the backend-reported light/deep state instead of displaying a
  stale sleep depth.
- Exactly-once tool replays use one bounded answer-only recovery, and tool cards
  are emitted only when a complete call reaches validation or execution.

---

## [1.6.42] - 2026-08-27

### Added

- Qwen3.8 Flash Next now has an owned MLX runtime for its hybrid QSA/Gated
  DeltaNet architecture, SSD-backed PLE bigram/trigram tables, image and video
  inputs, native reasoning tiers, and its preserved in-model MTP head.
- The Electron server settings expose detected Qwen3.8 native MTP mode and
  adaptive or fixed D1-D3 depth policy with persisted launch-argument parity.

### Performance

- Qwen3.8 decode fuses its Gated DeltaNet projections and hyper-connection
  path while retaining the exact mixed-precision JANG contract.
- Native MTP uses request-local rollback/commit state and rolling confirmed
  tokens per wall second to probe adjacent depths without target replay.
- PLE rows remain file-backed and row-addressed on SSD instead of materializing
  the checkpoint's full n-gram table in unified memory.

### Fixed

- Qwen3.8 full-precision SSD prefix blocks preserve all QSA K/V/indexer state
  together with typed Gated DeltaNet and PLE companion state across divergent
  suffixes and process restarts, with retained paged-cache RAM disabled.
- Mixed image/video histories preserve modality order, processor scaling,
  embeddings, media cache keys, and request-local MTP state.
- App-managed native MTP enforces a complete greedy sampler contract, including
  neutral repetition penalty, while the explicitly enabled sampled path uses
  stochastic speculative acceptance instead of silently disabling MTP.
- Tool-bearing requests remain capped at D1 so adaptive probing cannot cross an
  uncommitted tool-call boundary.

---

## [1.6.41] - 2026-08-26

### Performance

- Supported M5 runtimes now use the native Qwen MTP verification kernel while
  retaining stock MLX as the default verifier elsewhere and preserving shared
  request-policy mutual exclusion.
- DSV4 process-restart cache publication now waits for the exact SSD durability
  fence before shutdown, avoiding a needless cold prefill when the prior turn's
  native composite blocks were still being serialized.

### Fixed

- Mixed-SWA changed-tail requests now reconstruct the bounded rotating and
  full-attention cache state from SSD across process restarts instead of
  rejecting a valid off-boundary partial prefix.
- Multimodal tool-intent checks now read normalized Pydantic content parts from
  the current user turn, preventing an older text-only tool instruction from
  being reused when the latest image or video turn says not to call a tool.
- Explicit stream-interval settings survive session creation and engine launch
  instead of being overwritten by the panel default.
- New chats ignore dead tool directories rather than advertising unavailable
  coding tools to the model.

---

## [1.6.40] - 2026-08-25

### Performance

- Qwen3.5-family head-dimension-256 prefill tails now stay on MLX's fused
  scaled-dot-product attention path for the one-shot prefill lane while
  preserving the existing chunked-prefill fallback.

### Fixed

- Hybrid Qwen text, image, and video requests now resolve the same scoped
  media and generation-prompt discriminators for native GatedDelta companion
  checkpoints as for paged KV blocks. Longest-prefix SSD hits therefore
  restore the exact pre-media native state across chats and process restarts
  without retaining a second in-RAM cache.
- MiniMax-M3 adaptive streams now keep ambiguous initial reasoning private
  until a real reasoning boundary or marker-free completion establishes the
  output lane, preventing late end-only reasoning markers from leaking
  planning into visible content.
- Electron chat settings now persist and forward OpenAI frequency and presence
  penalties through the same validated request domain as the Python CLI and
  gateway APIs.
- Session creation reports live engine startup progress and allows the active
  model's bounded startup window to complete instead of timing out early.
- Release packaging now rejects a bundled Python runtime whose installed MLX
  or MLX-Metal distribution version differs from the selected release build.

---

## [1.6.39] - 2026-08-25

### Performance

- Native SSD prefix-cache validation now reads safetensors metadata instead of
  loading each full DSV4, ZAYA, or mixed-SWA payload before reconstruction.
  The reconstruction worker remains the only full-payload reader, eliminating
  one redundant full SSD read per retained block while preserving exact native
  typed state and the full-reader compatibility fallback.
- SSD LRU access-only touch batches no longer trigger a whole-root byte-accounting
  reconciliation while serving a cache hit.
- Qwen3.5-family native-MTP depth-one decode avoids an unnecessary cycle fence;
  deeper drafts and other families retain the conservative synchronization.

### Fixed

- OpenAI frequency and presence penalties plus `logit_bias` are now validated
  and forwarded through text, multimodal, Chat Completions, legacy
  Completions, Responses, Anthropic, and Ollama paths without logging biased
  token identifiers.

---

## [1.6.38] - 2026-08-25

### Performance

- Updated the macOS runtime and Electron bundle to MLX 0.32.2. On an M5 Max,
  the directly affected small-row q4 matmul shape improved by about 24% with
  identical output bytes, and Qwen3.8's 2K head-dim-256 causal-attention
  primitive improved by about 2.2x while remaining within the upstream fused
  kernel's numerical tolerance.

### Fixed

- Hybrid multimodal SSD cache repair now teaches the media lane the exact
  block-aligned KV boundary that previously missed its native SSM companion.
  For Qwen hybrid prompts whose match ends before or inside the final expanded
  image/video run,
  the auxiliary pass first derives the complete media-conditioned embeddings
  and mRoPE positions, then snapshots only the exact matched prefix. A later
  hit before media begins re-encodes those full conditioned embeddings but
  forwards only the uncached suffix over restored native KV+SSM state. Other
  wrappers fail closed rather than pairing partial placeholders with a full
  media tensor.
- Backported mlx-vlm's Qwen3-VL scalar-repeat correction for MLX 0.32.2, so
  image and video prefills convert temporal grid counts to Python integers
  instead of failing before generation with an incompatible `mx.repeat` call.
- Inherited MLX's sliced-array quantization fix and MXFP8 scale round-up, so
  future conversions cannot bind stale sliced storage or saturate block maxima
  because an E8M0 scale rounded down. Existing JANG model bundles are not
  rewritten by this dependency update.

> **⚡ DeepSeek V4 Flash — status as of vMLX Engine 1.6.25**
>
> As of **1.6.25** the engine ships the full DSV4-Flash speedup + coherency rollup accumulated across 1.6.19 → 1.6.25 (this file). CRACK builds targeting DSV4-Flash-0731 (`dealignai/DeepSeek-V4-Flash-0731-JANG-CRACK`) require vMLX 1.6.25+ to see these numbers.
>
> **Speedups now landed:**
> - MLX allocator cache scales with installed RAM (128 GB → 23 GB, cap 24 GB) — **~11% faster decode at every context length** (1.6.22).
> - Prefill projects only the final position through `lm_head` on chunked prefill (1.6.25).
> - Extended prefill+decode delta-chain KV store for cross-turn reuse; extended-store armed for short prompts too; idle-time shadow re-key of the predicted visible transcript (1.6.25).
> - Batched TQ restore decode — single-run fold + one deferred eval across layers (1.6.25).
>
> **Coherency + long-context fixes now landed:**
> - DSV4 fails closed on misaligned/incomplete composite-cache; SWA ring + q8 CSA/HCA pool segments preserved through prompt snapshots and SSD/L2 reconstruct; long prefills use compression-ratio-aligned chunks to avoid Metal command-buffer timeouts (1.6.19).
> - DSV4-0731 session caps derived from bundle (Low reasoning, DSML, top-p, q8 native pool quant, activation-QAT, cache tiers); native pooled-cache snapshots preserve short append checkpoints, lossless L2 writes, valid eviction ancestry, partial-tail replay, SSD-only op, RAM-to-SSD refault without repeated validation reads (1.6.20).
> - Delta-cache restore reads its block size and anchor interval from the cache records themselves (1.6.21 + 1.6.22).
> - Answer-reserve split (`clamp(25% of budget, 256, 2048)`) — DSV4 always emits a visible answer, even on long-reasoning prompts (1.6.23). Reserve floor capped at half the budget so budgets under 512 tokens still split (1.6.24). Override via `DSV4_ANSWER_RESERVE`.
> - Metal **live-buffer count** cap for DSV4 long generations — no more `Resource limit (499000) exceeded` crashes at ~12k tokens (1.6.24).
> - Honest hardware context ceiling from pool geometry + floor-width transient margin + active-growth allowance; context-dependent prefill transient in the hw ceiling — DSV4 refuses/clamps oversized requests up front with a clear message instead of crashing mid-generation (1.6.25).
>
> **Companion jang_tools fix** (not vMLX code but required for the DSV4-Flash 2-bit line): CSA/HCA pool BF16 → q8 promotion ceiling raised from 2 MiB → 16 MiB, eliminating the mid-decode promotion event that caused DSV4-Flash-2L to loop past ~15k tokens.

---

## [1.6.37] - 2026-08-25

### Release hardening

- Refreshed the Electron live-proof contract to bind release evidence to the current locked-off in-memory cache label and current SSD/native-cache settings.
- Kept the mixed-media parser contract explicit: image, video, and audio markers remain in the user's request order so video-first prompts cannot pair features with the wrong placeholder.
- Staged the verified 1.6.36 runtime and cache work under a new synchronized source, engine, and panel version for release packaging.

## [1.6.36] - 2026-08-24

### Fixed

- **DeepSeek V4 Flash foreground replies no longer wait behind idle predicted-transcript cache work.** An incoming request is now marked before terminal cleanup begins, so the SSD shadow re-key lane cannot mistake it for an idle engine and start a full prompt prefill first. On `dealignai/DeepSeek-V4-Flash-0731-JANG-CRACK`, the directly affected follow-up fell from 9.789 s to 4.204 s TTFT while preserving 97.88% prefix reuse and 32.0 tok/s decode.
- **DeepSeek V4 native SSD cache writes accept zero-sized anchor tensors.** Empty topology-owned tensors are serialized as valid cache state instead of failing the durability fence and leaving most of a conversation chain unavailable after the turn.
- **Exactly-once tool follow-ups retain their tool schema.** A model can now continue from the tool result without losing the schema that owns the call or invalidating the reusable prompt branch.
- **Multimodal and mixed-attention cache branches keep the correct history and native state.** Step media/tool continuations, Muse image/video transitions, Gemma media tails, and mixed-SWA restores now select usable causal prefixes instead of replaying stale or incompatible snapshots.

### Performance

- **SSD-only prefix caching no longer builds full-cache NumPy copies or retains writer buffers after durability.** Blocks stream directly to safetensors files, duplicate physical blocks are deduplicated, and restore admission accounts for measured SSD lookup cost before choosing a hit.
- **Architecture-native cache state is the default.** The app exposes the effective native policy in Server Settings and does not force generic TurboQuant encoding onto models whose instantiated cache topology uses another representation. In-RAM paged, vision, and SSM payload caches remain disabled in the SSD-only product path.

### Compatibility

- Raised the bundled/runtime JANG floor to **2.5.46**, matching the public JANG release used by the verified DeepSeek V4 path.
- The reported DeepSeek V4 Flash bundle now produces visible completions through both the Electron Chat UI and raw Responses streaming on the release source. The live hardware-safe prompt ceiling is model- and machine-derived; this release does not promise the bundle's declared 1M context on every Mac.

## [1.6.35] - 2026-08-21

### Fixed

- **dots3-note returned an EMPTY answer on most cached turns (regression in 1.6.34).** Restored plain-KV layers are zero-padded to the 256-token cache step while `offset` stays logical, and dots3 adopted the padded buffers — so its sparse-attention indexer counted pad rows as real keys, put the causal frontier behind by the pad size, and generation ran to `max_tokens` with nothing visible. The trigger was any restore whose cached length was not a multiple of 256, which is the overwhelming majority: every restore observed in testing (2176, 2232, 2245, 3438, 7043, 10634) was misaligned. Fixed by adopting the logical extent, refusing to let a padded physical length become an offset, and declining the adoption loudly rather than attending over zeros.
- **`--prefix-cache-max-bytes` was silently ignored on every VL/multimodal session** — the MLLM scheduler read a field its config never declared, so the flag did nothing and the budget fell back to the RAM-percent default.
- **`reasoning_effort: "minimal"` turned reasoning OFF on hy3 bundles.** The lowest effort tier matched no branch and fell through to the template's `no_think` fallback, while every other family clamps it to `low`. Unrecognized efforts now warn instead of silently disabling reasoning.
- **`/v1/messages` silently dropped `min_p`, `repetition_penalty`, `cache_salt` and `skip_prefix_cache`.** The server already resolved all four; the Anthropic request model simply never accepted them.
- **API gateway dropped the top-level `images` array on `/api/generate`,** so a vision request became text-only and the model answered about nothing, with no error. The gateway also dropped `options.seed`, making identical requests reproducible against the engine port and silently non-deterministic through the gateway.
- **Ollama responses carried no usable timings.** `total_duration` was hardcoded to 0 and the streaming path sent no duration fields at all, so every client that renders throughput as `eval_count / eval_duration` showed nothing. Durations are now measured, with the prefill/decode split taken from real time-to-first-token and omitted where it cannot be measured.
- **LFM2-VL bundles whose MLP tensors were renamed to the llama `gate_proj/up_proj/down_proj` spelling failed to load** with 270 unhandled parameters. The llama names are now aliased back to LFM2's own `w1/w3/w2`.

### Performance

- **The second message of a long conversation no longer stalls.** To store a reusable prefix the engine needs the cache as it stood at the prompt boundary, and because recurrent state cannot be recovered after generation it re-prefilled the entire prompt — a second full forward pass that ran after the response was dispatched and blocked the next request. Profiled at 40.8% of engine time (~28s at 15.4k tokens). That boundary cache is now assembled from state already in hand: append-only attention KV sliced back to the boundary, plus the recurrent snapshot captured before generation advanced it. Second-message time-to-first-token drops from ~37s to ~2.4s on Qwen3.8-27B at 15k context. Layouts where slicing would be wrong (ZAYA CCA, mixed-SWA rotating windows, quantized attention) are excluded and keep re-prefilling.
- Embeddings requests batch internally, so a long-chunk batch can no longer ask Metal for an impossible single allocation.

### Added

- **Low-memory Macs are advised to use SSD-only caching at session startup.** On 36GB and below, the in-RAM paged KV mirror competes with model weights for the same unified memory while measuring within ~2% of SSD-only on a 15k-token workload. The notice is advice only — nothing is disabled, no argument is changed, and the configuration runs exactly as requested.

## [1.6.34] - 2026-08-20

### Performance

- **Native MTP overhaul (Qwen3.8, dots3-note):** aligned draft-head context cache plus generalized skip-replay — rejected drafts no longer replay through the main model at any depth, and restore-aware acceptance gates keep heads warm across cache restores. Warm in-app decode on Qwen3.8-27B-JANG_4D rises from ~22 to 38–44 t/s; dots3-note reaches ~41 t/s.
- **DFlash2 session prefix reuse:** end-of-turn cache checkpoints, draft hidden-state gap splice, and prompt-boundary snapshots (safe under thinking-strip) — warm multiturn TTFT drops ~20x; 60+ t/s sustained on Qwen3.8 with DFlash2. The DFlash runtime (`dflash`, `dflash-mlx`) now ships in the bundle.
- **Stream interval defaults to 8:** the renderer can no longer backpressure-stall the engine emit loop on long conversations; legacy interval-1 sessions are lifted automatically.
- **dots3-note deep context:** the prefill admission valve projects per-chunk (removing a phantom ~17k refusal), SSM companion snapshots exempt positional full-latent slots (quadratic memory growth fixed), and the L2 block store gains a 4GB pending-write budget with drain (no more lineage drops under store bursts). 56k-token conversations verified in-app on a 128GB Mac with the wired limit raised.
- mlx 0.32.1, mflux 0.19.0.

### Fixed

- **Store-path memory explosion that could take down the whole machine.** The prefix-cache block store imported numpy slice views into MLX in a way that dragged a full-layer-sized buffer per block (~140GB of live Metal on an 11k-token hybrid store). At the default wired limit this silently killed the engine minutes after a turn; with a raised limit it could wedge the machine into a watchdog kernel panic. Slice imports are now slice-sized, background clean prefills return their chunk transients per chunk, and the store-time pass stores the SSM companion so the idle re-derive skips its duplicate full-prompt pass.
- **Transient DFlash2 resume crash** (`[full] Negative dimensions not allowed`) on short follow-up turns in resumed conversations — upstream RotatingKVCache growth math now keys on physical fill.
- **Qwen3.5-family video input works.** Sending a video failed with "Image features and image tokens do not match" — the frame-fallback placeholder rewrite (the designed qwen video path) was never wired into the app's engine lane. Videos now route as sampled frames and temporal questions answer correctly.
- **Removed the RAM preflight that refused large model loads.** Estimates now only advise; models the hardware can genuinely hold (e.g. a 101GB dots3-note bundle on a 128GB Mac) load again.
- **Native MTP actually engages by default.** Bundle-temperature auto-detection had kept MTP permanently off; detection is now compatible-only with a deterministic-defaults sampling policy (non-zero-temperature sessions log a loud AR fallback instead of silently degrading).
- Prefill admission rejections now include a wired-limit advisory naming the exact `sysctl`, the macOS ~84%-of-RAM default, and the reset-on-reboot behavior.
- Hybrid prefix cache: bf16 cumulative-state dtype round-trip; plain-KV restores are step-padded.

## [1.6.33] - 2026-08-17

### Fixed

- Stale Error badge and title no longer survive restarts on sessions whose models load fine (boot-time reconcile).

## [1.6.32] - 2026-08-16

### Added

- Cross-turn peak-walk admission valve for hybrid models: deep incremental
  conversations that previously aborted the engine with an uncatchable Metal
  command-buffer OOM now receive a clean HTTP 413 refusal before the fatal
  forward. The valve fits the measured between-turn Metal peak walk per
  process, logs every engagement, and refuses at the device working-set
  limit; hardened against interleaved conversations, mid-size chunked
  prefills, retries, and failure paths. Conversations continue past a
  refusal by restarting the model (the conversation restores from the disk
  cache — proven live at 101k tokens) or with a fresh cold prefill.
  `VMLX_TURN_PEAK_ADMISSION` / `VMLX_TURN_PEAK_ALLOWANCE_MB`.
- 100k+ multiturn proven end-to-end on the Qwen3.8 hybrid line: restart +
  disk-cache restore continued a refused conversation to 101,673 tokens with
  96k restored, byte-equal replay at maximal reuse, and byte-equal answers
  after eviction-forced cold refault.
- Deep-span allocator cache clear ahead of large forwards on both the fresh
  prefill and cache-hit lanes (`VMLX_DEEP_SPAN_CACHE_CLEAR_TOKENS`); SSM
  companion RAM budget default raised to 1536MB (`VMLX_SSM_STATE_CACHE_MB`).
- Chat UI: context-exhaustion and effort-substitution notices; deep-refusal
  bubbles carry the actual remedy (restart restores from disk).

### Fixed

- Admission declines and prompt-too-long are 413 on every API door (chat,
  completions, responses, messages, ollama) via app-level handlers; three
  routes carried a latent error in their inline handlers and three doors had
  no handling at all.
- An oversized single cache payload (deep mixed-SWA layer states) could
  never be written under the pending-write byte budget, permanently
  truncating disk-cache coverage for its descendants; oversized payloads now
  admit exclusively after the write queue drains. Applies to both the block
  disk store and the SSM companion store.
- `reasoning_effort` now forwards identically across the chat, responses,
  and messages routes; companion disk entries are touched on RAM hits so hot
  entries survive disk eviction; media prefix boundaries capture and reuse
  byte-equal on matching replays for allow-listed VL families.
- Panel: raw database error replaced with a clear message when a port is
  held by another session; MTP mode/depth and all restart-required settings
  verified to persist across engine restart, sleep/wake, and app relaunch.
- Native-MTP text models (stock batch generator) wedged every request in a
  TypeError retry loop after the cold-prefill-split change; the scheduler
  now probes the active generator's insert() capability and omits the
  kwarg where it does not apply.

### Release-note catch-up (1.6.29 - 1.6.31, shipped 2026-08-15)

- .29/.30/.31 shipped the Qwen3.6/3.8 enablement line with native MTP
  autodetection, MiniMax M2.7 MTP engagement, disk-L2 writer contention
  fixes, context-exhaustion hygiene across families, mlx.fast hot-path
  work, and the notice-chip chat UI. See the release pages for details.

## [1.6.28] - 2026-08-13

### Added

- Support for the new Qwen3.6-27B bundle line (and the upcoming Qwen 3.8
  line), whose metadata-only stamps previously killed the server at startup
  or load. The `qwen3_coder` tool-parser name now resolves to the XML-function
  parser its templates actually emit, capability-only stamps route through the
  stock loader instead of the JANG codec, and the bundled multi-token
  prediction head is constructed, correctly quantized per the stamp's
  per-module overrides, and engaged automatically at the stamp's trained
  speculative depth. Measured on the 4-bit bundle: 23.3 to 31 tokens/sec
  (+33%) with no flags, greedy output byte-identical with the head on or off,
  and image and video probes answering correctly through both API and app.

### Fixed

- A prefix-cache hit could change the answer on seven measured model families,
  and the opt-in switch that trades cache reuse for answer stability only
  reached three of them. It was reachable only from inside the family policy
  chain, and `minimax`, `muse_glimmer` and `step3p7` hold no branch there. The
  switch now runs ahead of that chain, and `nanbeige` — the family with the
  highest measured divergence rate — has been added to the list it was missing
  from. Proven end-to-end: with the switch armed, both turns reproduce the cold
  answer byte for byte. Absence from the list is not exemption; no family has
  been shown exempt, and the accompanying notes now say so.

- A long-running request could be killed as unresponsive while it was
  generating normally. The engine's liveness counter double-counted output
  tokens, and a recovery restart reset half of the sum, so the value went
  backwards mid-request; the streaming timeout only credits a counter that
  increases. Both the text and multimodal schedulers had the defect. Operator
  logs that report "still progressing (N tokens)" now show the true token
  count rather than twice it.

- Explicitly disabling a parser was silently ignored. `--reasoning-parser none`
  had no effect on any streaming surface and `--tool-call-parser none` was
  inert in the module entry point, while the startup banner reported both as
  disabled.

- A derivation step beginning with `=` rendered as literal dollar-sign text in
  chat instead of as maths.

- The Hugging Face token was normalised on only two of the five paths that read
  it, so a token with surrounding whitespace worked in some places and failed
  in others.

- Muse Glimmer failed to load on a fresh `pip install vmlx`. mlx-vlm 0.6 began
  shipping its own `muse_glimmer` package, and vMLX handed the namespace to it
  on sight — but upstream names its preprocessing module differently, so the
  server exited with a missing-module error before it could start, and had it
  started it would have used a forward pass without this port's corrections.
  vMLX now keeps its own validated runtime unless an upstream package genuinely
  provides the same interface. The macOS app was never affected: it pins the
  older mlx-vlm that has no upstream package, which is why the app tested clean
  while the published wheel did not.

- Muse Glimmer failed to load on a fresh `pip install vmlx`. mlx-vlm 0.6 began
  shipping its own `muse_glimmer` package, and vMLX handed the namespace to it
  on sight — but upstream names its preprocessing module differently, so the
  server exited with a missing-module error before it could start, and had it
  started it would have used a forward pass without this port's corrections.
  vMLX now keeps its own validated runtime unless an upstream package genuinely
  provides the same interface. The macOS app was never affected: it pins the
  older mlx-vlm that has no upstream package, which is why the app tested clean
  while the published wheel did not.

## [1.6.27] - 2026-08-11

### Added

- Muse Glimmer 30B runs, with vision and video. Its text tower, windowed
  vision tower, recipient-routed reasoning rail (`to=self` / `to=user`) and
  ATEM tool dialect are live and verified on all three bundles
  (JANG_2D / 4M / 6M). Single- and multi-image prompts and video clips are
  described correctly, including temporal order. Reasoning depth is controlled
  by the `reasoning_strength` template kwarg (low/medium/high/xhigh); this
  family ignores `enable_thinking` and `reasoning_effort` entirely.

### Fixed

- Muse video requests failed outright — the engine handed the image processor
  a 4-D float frame array PIL cannot construct from, so every clip died with
  "Cannot handle this data type". The processor now normalizes array frames
  (squeeze, CHW→HWC, float→uint8) and no longer treats images and videos as
  mutually exclusive.
- Muse recipient markers could reach the visible answer. A bracketed control
  token (`<|message|>`, `<|eom|>`, `<|eot|>`, `<|start|>`) that slipped into
  the answer rail on a malformed or max-tokens-cut stream is now scrubbed from
  visible content — a no-op on well-formed output, and never applied to
  reasoning or to tool bodies (which the ATEM parser needs verbatim).
- The Anthropic `thinking.budget_tokens` control armed nothing. It now maps to
  `max_thinking_tokens` (the runtime reasoning clamp), matching the OpenAI
  Responses `reasoning.budget_tokens` behavior, instead of only setting a
  template kwarg no runtime reads.
- Ollama `think` string levels (`"low"` / `"medium"` / `"high"`) were silently
  dropped — they normalized to neither true nor false, so thinking never even
  engaged. A level now enables thinking and selects the reasoning effort.
- Muse Glimmer showed no reasoning-strength controls. Chat Settings offered only
  the Auto/On thinking toggle, so the four depths the model actually has were
  unreachable from the app. The panel derived its effort buttons solely from a
  bundle's `reasoning_effort_levels`, but Muse declares `reasoning_strength` with
  a `modes` list instead; effort levels are now also read from `modes`, and
  `xhigh` became a first-class level (labelled "Extra High" in every locale).
- The prefix cache was capped far below the context window. New sessions were
  created with a 1000-block index, which at the 64-token block addresses only
  63,936 tokens — a long prompt could report zero reuse on an exact repeat and
  run slower than a cold prefill. New sessions now index 262,144 tokens, and
  existing sessions are migrated.
- Vision models could run with paged RAM off in the app while the same model ran
  paged from the command line, so the app silently used a slower cache path.
  The two now agree, and the migration that was meant to correct existing
  sessions — which never actually ran, and marked sessions as already-migrated —
  is fixed.

- Muse Glimmer produced fluent nonsense. Four divergences from Gemma were
  missing from the port: the checkpoint's zero-centered RMSNorm gains were
  used unshifted (every norm ran at gain ~0 instead of ~1), Q and K skipped
  their weightless per-head norm and the query-side `qk_scale_factor`, the
  embedding lookup skipped its RMSNorm, and the sliding and full attention
  layers shared a single mask so sliding layers could attend past their
  window.
- ATEM tool calls never reached clients. The parser returned a pre-nested
  OpenAI envelope while the server builds that envelope itself, so every
  call raised inside the dispatcher and the raw `<atem:...>` markup was
  passed through as visible chat text with `tool_calls: null`.
- DeepSeek V4 Flash warm long-context responses drop ~37%. The answer pass
  differs from the reasoning pass by only the 3-token generation rail, but
  the terminal-anchor guard admitted a 1-token tail at most, forcing a
  fallback to the block-aligned checkpoint and a 215-token re-prefill
  (5.8 s at 127k) to change one token. Measured 21.97 s → 13.74 s with
  byte-identical output.

Rolling summary above. Per-commit detail: `git log v1.6.24..HEAD`.

## [1.6.26] - superseded by 1.6.27 (in-development rollup)

### Fixed

- Decode no longer stalls while the app polls engine health. During generation
  `/health` now serves the last idle snapshot with live queue/status overlays
  instead of rebuilding its full diagnostics on every poll, removing a
  330–410 ms inter-token hitch per 5 s poll on every model.
- DeepSeek V4 Flash long-output decode is smooth. The extended-store delta
  chain now batches its per-256-token snapshot into one async materialization
  instead of a few hundred blocking evals on the decode thread, so sustained
  generations hold their rate (long-context long-output measured stall-free).

Rolling summary above. Per-commit detail: `git log v1.6.24..HEAD`.

## [1.6.25] - superseded by 1.6.26 (in-development rollup)

Rolling summary above. Per-commit detail: `git log v1.6.24..HEAD`.

## [1.6.24] - 2026-08-04

### Fixed

- Long DeepSeek V4 Flash generations no longer fail with
  `[metal::malloc] Resource limit (499000) exceeded`. Metal caps the *number*
  of live buffers, not just their bytes, and the output guard only modelled
  bytes — so it advertised a safe cap of 17,575 tokens for a model that dies
  around 12,000. Measured by sampling the live-buffer count while decoding,
  DeepSeek V4 Flash retains about one buffer per layer per generated token
  (~42 on a 43-layer model) because its compressor/indexer pools are
  cumulative; `mx.clear_cache()` frees none of it, so the ceiling is hard.
  The guard now projects that ceiling as well and takes whichever limit binds
  first, so an oversized request is refused up front with a clear message and
  an unspecified one is clamped, instead of both crashing twelve minutes in.
  Conventional KV caches are measured at 0.000 buffers per token (Qwen 3.6 held
  its count flat across 600 decode steps) and are deliberately left uncapped.
- DeepSeek V4 Flash answers again at small output budgets. The answer reserve
  had a flat 256-token floor, which made every budget under 512 unsplittable —
  so reasoning consumed all of it and the answer came back empty, the same
  failure the reserve exists to prevent. Measured before: 160, 256 and 400 all
  returned no content while 512 and above answered. The floor is now capped at
  half the budget, so 128 and up split; every budget at or above 512 is
  unchanged.
- The chat Max Tokens field no longer sits blank while a budget is silently in
  force. Bundles are not required to declare `max_new_tokens`; when one does
  not, the engine resolves its own default and clamps it to projected headroom,
  and the panel now shows that resolved number as the placeholder instead of
  nothing.

## [1.6.23] - 2026-08-04

### Fixed

- DeepSeek V4 Flash now produces a visible answer on prompts that need a long
  response. The model does not reliably emit `</think>`, so reasoning consumed
  the entire output budget and the answer was left with nothing: a 2000-token
  budget produced 7,980 characters of reasoning and zero content, 4000 produced
  15,980, 8000 produced 31,980. The engine already had a never-empty answer
  pass for exactly this, but it could only arm when the caller supplied
  `max_thinking_tokens`, which no caller does, so it never ran.
  When no split is requested, DeepSeek V4 Flash now reserves part of the output
  budget for the answer, which makes that answer pass reachable. The reservation
  is a bounded reserve rather than a percentage, so deep reasoning is not taxed:
  `clamp(25% of the budget, 256, 2048)` tokens, leaving 75% of a 2000-token
  budget and 99.5% of a 384K budget for thinking. Budgets under 512 tokens are
  left unsplit. Set `DSV4_ANSWER_RESERVE` to override, or `0` to disable.
  No tokens are forced or injected and sampling is unchanged. Applies to Chat
  Completions, Responses, Anthropic and Ollama, streaming and non-streaming.
  Other model families are unaffected.

## [1.6.22] - 2026-08-02

### Performance

- DeepSeek V4 Flash decodes about 11% faster at every context length. The MLX
  cache ceiling was a fixed 8 GB, which made MLX free and re-allocate decode
  graph buffers it could have reused; it now scales with installed memory
  (16 GB machine → 4 GB, 128 GB → 23 GB, capped at 24 GB). Peak process memory
  is unchanged, because this bounds reclaimable cache rather than reserving it.

### Fixed

- Tool markup carrying a non-string name, such as `{"name": 12345}`, no longer
  produces a tool call whose function name is empty. Affected the Hermes, Qwen,
  Nous and Gemma 4 parsers; such markup is now treated as ordinary content.
- Tool-choice enforcement errors report the caller's own `tool_choice` instead
  of always naming `'required'`, so a request that pinned a specific function
  can tell which setting failed.
- DeepSeek V4 Flash delta-cache restore reads its block size and anchor interval
  from the cache records rather than assuming fixed values.

## [1.6.21] - 2026-08-02

### Fixed

- Streams that stop at `max_tokens` report their token usage again. A
  reasoning-only turn cut short by the token cap returned without emitting the
  terminal usage chunk or `[DONE]`, so OpenAI-compatible clients requesting
  `stream_options.include_usage` received no usage at all, and the Anthropic
  `/v1/messages` route — which rebuilds usage from that stream — reported zero
  input and output tokens with no `cache_read_input_tokens` even when the
  prompt was served almost entirely from cache.
- DeepSeek V4 Flash delta-cache restore reads its block size and anchor
  interval from the cache records themselves instead of assuming fixed values,
  so the two stay consistent with whatever geometry wrote them.

### Changed

- Built-in tools name an unusable working directory and the setting that fixes
  it, instead of surfacing a raw filesystem error per tool call.
- Reasoning-rail DSML tool calls are recovered from their markup span rather
  than discarded, and prompt tool catalogs stay turn-independent so an agentic
  loop keeps reusing its cached prefix.
- Cache reporting exposes live block occupancy alongside allocator pin state,
  and the Anthropic surface reports prefix-cache reads.

## [1.6.20] - 2026-08-01

### Changed

- DeepSeek V4 Flash 0731 sessions derive native Low reasoning, DSML tooling,
  top-p guidance, q8 native pool quantization, activation-QAT availability,
  and cache-tier capabilities from the loaded bundle and effective runtime.
- Server and Chat settings distinguish DSV4 native compiled decode and pooled
  cache state from unsupported generic TurboQuant or whole-model cache modes.
- Cache activity separates current-process reads, writes, misses, and evictions
  from persistent namespace and managed-root occupancy.
- A guarded opt-in affine MoE decode path is available for DSV4 JANG bundles;
  native MLX remains the production default because the current M5 A/B did not
  improve throughput. Credit: Andrew Hornsby (@hornsan1) for PR #248.

### Fixed

- Native DSV4 context admission, stop-token handling, tool-prompt ownership,
  DSML failure behavior, and tool-result continuation remain consistent across
  Electron and OpenAI-compatible API paths.
- DSV4 native cache snapshots preserve short append checkpoints, lossless L2
  writes, valid eviction ancestry, partial-tail replay, SSD-only operation, and
  RAM-to-SSD refault without repeated validation reads.
- The Electron renderer warns when DSV4 top-p differs from bundle guidance,
  keeps cache status requests session-scoped, prevents loaded-session header
  overlap, and renders math without treating currency as a delimiter.
- Explicit one-tool prompts whose final contract says the visible answer must
  be exact now enter the no-more-tools continuation pass instead of leaving the
  model in an open-ended tool-capable continuation.

### Distribution hardening

- Production packaging attests the native Python DSV4 encoder and the exact
  clean JANG runtime source, including nested affine defaults and mixed module
  quantization metadata.
- Sequoia and Tahoe remain separately signed and notarized artifacts from one
  exact source revision.

### Checkpoint boundary

- Broader multi-family, modality, and maximum-context coverage remains tracked
  for the next checkpoint and is not implied by this DSV4-focused release.

## [1.6.19] - 2026-07-28

### Changed

- Model sampling defaults remain owned by explicit request values and the
  selected bundle's `jang_config.json` / `generation_config.json`; vMLX no
  longer synthesizes a Laguna-specific top-k value.
- Fresh and restored Electron settings hydrate model-derived sampling and
  output defaults without painting stale placeholder values as saved
  overrides.
- Native MTP health and profiling expose cache lifecycle, acceptance, and
  phase timing without changing generation policy.
- Looped-transformer cache identities include the effective repeated layer
  layout, preventing persistent cache reuse across incompatible loop counts.
- Release and UI proof tooling binds captures to the exact Electron, gateway,
  Python engine, bundle, request, and endpoint provenance.

### Fixed

- DeepSeek-V4 Flash now fails closed on misaligned or incomplete native
  composite-cache state. Product sessions keep reusable prefix, paged, and L2
  cache paths disabled until cached-output equivalence is independently proven,
  while pool quantization remains bundle-derived and generic TurboQuant KV stays
  disabled for this architecture.
- DeepSeek-V4 Flash native long-context state now preserves ratio-zero SWA
  rings and lossless, dtype-aware q8 CSA/HCA pool segments through prompt
  snapshots and SSD/L2 reconstruction. Long prefills use smaller
  compression-ratio-aligned chunks beyond the proven short-context band to
  avoid Metal command-buffer timeouts.
- Electron session adoption cannot silently inherit an unsafe or malformed
  DeepSeek-V4 cache configuration; unsupported cache controls and contradictory
  cache help are hidden for these sessions.
- Qwen native-MTP loading preserves already converted backbone normalization
  tensors while applying conversion only to MTP-owned tensors.
- Hybrid TurboQuant prompt-batch splits preserve MLX dtype metadata without
  retry loops or stalled streams.
- SSD cache eviction preserves causal parent chains, and disk payload encoding
  runs off the inference completion path after safe MLX detachment.
- Memory-pressure cache trims preserve the maximum safe block-aligned SSD
  prefix and reconcile saved-token telemetry to the prefix actually consumed.
- Ollama preserves explicit top-k request semantics, including zero.
- Prompt-token accounting follows mlx-lm's BOS handling for text-only engines.
- Literal currency no longer consumes the delimiter of a following valid
  single-dollar math expression in the Electron renderer.

### Distribution hardening

- Public source and bundled Python attest the exact clean JANG 2.5.37 source
  tree used by the release, including affine bundles whose module overrides
  intentionally mix group sizes.
- Sequoia and Tahoe artifacts remain separate, signed, notarized outputs bound
  to one exact source revision.

### Checkpoint boundary

- Laguna XS 2.1 artifact-level long-reasoning repetition is not hidden by an
  app-side sampler clamp, forced close tag, prompt coercion, or output cap.
  Bundle publishers should validate corrected artifacts independently.
- Broader exhaustive family, modality, long-context, cache-pressure, and
  performance coverage continues after this checkpoint and is not implied by
  the release version alone.

## [1.6.18] - 2026-07-25

### Changed

- Reasoning Auto mode preserves each bundle's native policy unless a request
  supplies an explicit supported thinking budget. Reasoning-marker aliases
  remain on the reasoning rail instead of appearing in visible content.
- Tool capabilities are scoped to the current turn. Tool availability changes,
  exact authorization, multi-tool result continuations, cancellation identity,
  and final-pass usage remain tied to the correct request.
- Fresh-session controls derive sampling, output, template, parser, and native
  MTP settings from the selected bundle without turning inherited values into
  hidden saved overrides.
- The block-disk cache uses one process-safe physical size budget across
  model/config namespaces and typed companion state. Disk publication is
  bounded and moved off the request-completion path, with coordinated LRU
  eviction, clear, shutdown, and restart accounting.
- Cache telemetry distinguishes the attempted prefix, applied reuse, RAM or
  SSD blocks, uncached suffix, reconstruction work, fallback prefill, and
  aggregate SSD usage.
- Experimental codebook-VQ weights use a bounded RAM LRU and reload their
  original compressed model shards after eviction instead of creating a
  second unbounded SSD spill cache.
- The Electron renderer uses KaTeX for supported math while preserving
  currency, literal Markdown/HTML, code fences, assistant XML, and readable
  in-progress reasoning text.
- The update banner and bundle-specific missing-template warning are localized
  across the shipped locales. Obsolete release and Swift-migration notices were
  removed.
- The Python package and Electron app advance to 1.6.18 and require JANG
  2.5.34.

### Fixed

- Laguna tool-result history, incomplete terminals, mixed-SWA cache safety,
  and affine-JANG JIT exclusions no longer reuse incompatible state or discard
  the model's first-pass terminal.
- Responses retains attached media across tool-result continuations, and
  Electron permits playback of local attachment media.
- Ollama gateway streams preserve backend failures instead of converting them
  into successful terminal events.
- Deep sleep and wake release stale executors and clear MLX residency before
  the engine resumes.
- Include-based chat templates are accepted only when their referenced
  template exists inside the selected bundle; missing or escaping references
  produce a current bundle-specific warning.
- Cache construction failures and model unloads drain asynchronous SSD writers
  and release shared budget ownership instead of leaving stale workers or
  capacity leases.
- Public-repository hygiene rejects private evidence, generated captures,
  malformed metadata, and machine-specific release material.

### Distribution hardening

- The Sequoia and Tahoe packaging path now requires both DMGs, bundled Python,
  JANG dependency, signing identity, and release metadata to bind to one exact
  source revision.
- Direct or cross-checkout packaging paths fail closed when source, tools,
  dependencies, artifacts, or version metadata drift.

### Checkpoint boundary

- This checkpoint does not promote deferred exhaustive family-by-family
  long-context/performance coverage, every modality/restart combination,
  architecture-wide cache eviction/refault soaks, extended gateway/LAN failure
  recovery, or general JANG conversion certification.
- Thanks to GitHub `@Hornsan1` for reporting many of the runtime, model, UI,
  and API issues addressed in this checkpoint.

## [1.6.14] - 2026-07-20

### Changed
- Gemma 4 image composition now preserves the bundle-native media-before-text
  order and exposes the supported 70/140/280/560/1120 image-token budgets
  through the Electron and Responses paths. Media cache identity includes the
  selected budget so incompatible vision prefixes cannot alias.
- Laguna and LFM use their native model/template reasoning policy in Auto mode
  instead of inheriting a generic family override. Reasoning and visible
  content remain separate on the API and Electron streaming rails.
- Narrow settings drawers and icon-only accessibility states remain usable at
  minimum window widths, and Ollama preserves backend error details.

### Fixed
- Gemma reasoning-only recovery stays on a fresh direct context and no longer
  replays an invalid reasoning-only assistant turn through the real Gemma
  template. The degraded control prefix is buffered rather than leaking into
  visible content.
- Nemotron Omni conversation identity now includes media bytes. Replaying the
  same text with different audio resets stale persistent KV/SSM state, and a
  post-media Electron follow-up preserves and rehydrates the prior media only
  for a bundle-grounded `nemotron-h` Omni route.

### Verified checkpoint boundary
- Current-source validation covers real Electron Start materialization, Gemma
  image/cache/reasoning rows, Nemotron Omni audio attachment and no-attachment
  continuation, and progressive Chat and Responses
  reasoning/content/terminal events.
- This checkpoint does not promote retained PARTIAL/OPEN rows. Gemma exact
  small-text OCR, stochastic strict-format reliability, Omni process-restart
  media-state restoration, broader signed-app family repetition, long-context
  and network-failure soaks, and the remaining model/parser/media matrix stay
  explicitly deferred.

## [1.6.13] - 2026-07-20

### Fixed
- Electron now preserves authoritative terminal usage and failure state after
  a Chat Completions or Responses stream emits progressive partial output and
  then fails. The safe partial prefix remains visible, interruption UI is not
  replayed into model history, and immediate same-chat recovery completes.
- Anthropic Messages and every Ollama streaming rail now terminate injected
  mid-stream engine failures in their native wire format. Ollama chat,
  templated generate, and raw generate emit `{"error":"..."}` and never
  synthesize a false `done:true` after the failure.

### Verified checkpoint boundary
- Literal unbuffered HTTP failure/recovery pairs cover Chat Completions,
  Responses, Anthropic Messages, Ollama chat, templated generate, and raw
  generate. Real Electron dev proof covers the Chat and Responses UI failure
  surfaces; adapter routes have current production-handler live proof.
- This checkpoint preserves the v1.6.12 model/cache/media evidence and its
  explicit deferred rows. Packaging does not promote untested family,
  long-context, stochastic, media, parser, or signed-app repetition rows.

## [1.6.12] - 2026-07-19

### Changed
- Block-disk L2 can now restore reusable full and partial prefix blocks even
  when paged RAM is explicitly disabled. When both tiers are enabled, memory
  remains first choice and disk refaults only missing reusable blocks.
- Eligible hybrid and full-KV attention components retain q4 TurboQuant cache
  storage while native/typed architectures keep their own codecs and rederive
  rules. TurboQuant remains a KV-cache codec, distinct from JANG affine,
  JANGTQ/MXTQ Hadamard-codebook, and base MLX MXFP model weights.
- Model-derived Electron settings more accurately surface quant format,
  parser/reasoning/cache policy, eager Start materialization, single-model
  switching, sleep/wake lifecycle, and port/LAN rollback.

### Fixed
- Abandoned non-stream Chat, Anthropic, and Ollama gateway requests now cancel
  upstream inference before response headers; immediate recovery requests no
  longer inherit a stuck scheduler slot.
- Progressive reasoning/content/tool output and terminal usage ordering were
  hardened across the current Qwen, HY3, Step, MiniMax, Anthropic, Ollama, and
  Responses routes, including explicit no-tool turns and malformed native tool
  prefixes.
- q4 memory-prefix serialization no longer falls back to unquantized storage
  when a minimal or early scheduler boundary has not initialized paged-cache
  state.
- Immediate session stop preserves completed first-turn L2 state, and media
  decoding retains its persistent scheduler owner for cancellation/recovery.

### Verified checkpoint boundary
- Current-source live Electron and API validation covers the named cache,
  streaming, tool, media, lifecycle, and gateway rows; those scoped results
  are not generalized to untested artifacts or parser families.
- Historical generated `build/` proof output is no longer tracked as source
  truth. Clean checkouts classify absent generated/staged evidence as OPEN,
  while partial, stale, or mismatched present artifacts remain FAIL.
- Broader long/stochastic model-family, media, parser, and signed-app soak rows
  remain explicitly deferred after this release checkpoint.

## [1.6.11] - 2026-07-15

### Changed
- Hybrid Qwen 3.5/3.6 models, including Bonsai 1-bit and ternary JANG bundles, now keep stored attention KV lossless by default. Live TurboQuant KV encode/decode remains active after the configured threshold, while unsafe persisted SSM companion restore is explicitly quarantined and restart requests safely full-prefill.
- Cache telemetry now reports byte ceilings, resident usage, block-disk L2 activity, and hybrid SSM evictions in the Electron cache and performance panels.

### Fixed
- Preserved coherent hybrid multi-turn generation across memory hits, restart boundaries, SSM async rederive, and paged attention-prefix reuse without promoting restored hybrid prefixes into unsafe longer entries.
- Enforced the paged-cache RAM byte ceiling before disk promotion and restored valid partial paged prefixes across process restarts.
- Stabilized Bonsai/Qwen and DeepSeek-V4 native tool prompts and Electron tool-result continuation; terminal DeepSeek-V4 cache hits can extend and store correctly.
- Bound Qwen's canonical `run_command` fallback example for the natural `to run: ...` request form so required command arguments are not emitted empty, and moved create-session port selection to the main process so it rejects ports already owned by live OS listeners.
- Kept Liquid LFM2.5 on its native tool transcript: the Electron client no longer prepends the competing generic coding-agent prompt, explicit requests retain only the requested native schema, and shell results are returned to the model as structured JSON (`exit_code` plus `stdout`) before final-answer synthesis.
- Added native MiniMax-M3 video preprocessing through the Responses route while preserving the model's typed sparse-attention cache (`attention_kv`, `msa_idx_keys`, and absolute block indices) instead of substituting generic KV.
- Cache clear now removes hybrid SSM companion disk state as well as attention-prefix state.

### Verified
- Real current-source Electron UI proof on the M5 Max covered Bonsai 27B 1-bit, Bonsai 27B ternary, DeepSeek-V4-Flash, Laguna-M.1, and MiniMax-M3: visual model loading, source-derived Chat Settings, Responses routing, reasoning Auto/Off/On, coherent multi-turn continuation, native tools, and visible cache telemetry.
- Bonsai proof covered affine-1 and ternary loading, hybrid GDN/SSM execution, live TurboQuant KV, paged attention reuse, bounded RAM, SSM eviction, block-disk L2 writes, image, and video. MiniMax-M3 proof covered native video plus typed sparse-cache memory and disk hits; Laguna covered `paged+tq` and `paged+disk+tq`; DeepSeek-V4 covered native composite paged hits with no competing model resident.
- Qwen3.6-27B hybrid proof covered Electron-selected live-port probing, Responses routing, Auto/Off/On reasoning, exact three-turn Off-mode continuation, native shell-tool execution, image understanding, paged+SSM hits, block-disk L2 hits and writes, SSM eviction under its byte ceiling, and model-RAM release after UI stop.
- LFM2.5-8B hybrid proof covered current-source Electron loading, Responses routing, Auto/Off/On reasoning, coherent multi-turn recall, one native `run_command` execution with a visible final answer, paged+SSM reuse, a 100% disk-L2 restart restore (70 block hits, zero misses), bounded 71/1000 block residency, and about 5.2 GB returned after UI stop.
- Current source passed the 876-test changed-engine suite, the 161-test focused cache suite, six block-disk LRU/restart tests, 2,209 Electron panel tests with three intentional skips, and TypeScript typecheck.

### Release boundary
- Hybrid SSM disk restore remains quarantined pending a fidelity-safe persisted state format; same-process SSM L1, async rederive, paged attention reuse, and block-disk writes remain enabled. MiniMax-M3 intentionally uses its typed native sparse cache rather than generic paged/TurboQuant KV.
- This source candidate does not by itself clear the repository-wide package/notarization gate; broader installed-app/model-family rows remain independently governed by the release manifest.
- Thanks to GitHub `@Hornsan1` for reporting runtime, model, UI, and API issues addressed in this release candidate.

## [1.6.10] - 2026-07-14

### Added
- Added lossless runtime support for JANG discrete-affine 1-bit storage. One-bit codes stay compact on disk and widen only to MLX's native two-bit slots during load; schema-2 tensor manifests remain authoritative for per-module bits and group sizes across text and multimodal towers.
- Runtime-verified affine-JANG Qwen hybrid bundles retain image and video routing instead of being forced text-only.

### Changed
- SSD block L2 now defaults on whenever compatible paged prefix caching is active in the engine and Electron UI. The visible toggle remains authoritative and emits an explicit `--disable-block-disk-cache` opt-out when disabled.

### Fixed
- Completed multimodal requests now release their paged-block references before detaching request state, restoring LRU eviction, new-prefix admission, and continued L2 writes after the in-memory block pool reaches capacity.

### Verified
- Real Electron UI proof on the M5 Max covered Bonsai 27B 1-bit and ternary text, reasoning modes, native shell tools, exact image and video prompts, multi-turn long-context recall, hybrid GDN/SSM companion state, q4 attention-cache storage, L2 restart hits, and forced block eviction/admission.
- Release candidate gates passed 276 focused engine/cache tests, 2,206 panel tests with 3 intentional skips, and TypeScript typecheck.

## [1.6.8] - 2026-07-12

### Fixed
- Reasoning models no longer truncate their visible answer mid-sentence. The engine's fallback output budget (used when a request sends no `max_tokens`, no explicit CLI/session default is set, and the bundle omits `max_new_tokens`) was a flat 4096 tokens shared by the hidden `<think>` reasoning phase and the visible answer; a verbose reasoner spent it all on reasoning and the answer starved (observed live on MiniMax-M2.7: reasoning 17400 chars, answer 183 chars, `finish=length`). Reasoning-capable models now fall back to a larger budget (detected via registry family capability — `reasoning_parser`/`think_in_template` — with the active parser as a fallback), while non-reasoning models keep the modest 4096 loop guard. The projected Metal headroom guard clamps the value per-model, so it never exceeds safe headroom (MiniMax-M2.7 80GB → 14522; MiniMax-M3 79GB → 16384). No request/response schema, endpoint, field, or resolution precedence changed — only the fallback value for reasoning models. Verified live across reasoning ON/OFF/AUTO and all API surfaces (OpenAI chat/completions/responses, Anthropic messages, Ollama, MLLM/VL).

### Notes
- Verified MiniMax-M3 (lightning/MSA sparse cache, paged-incompatible → routed to its own `MiniMaxM3SparseCache`, not forced-paged): warm-vs-warm greedy determinism byte-identical, prefix-cache hit + clean prompt-boundary re-prefill, memory-aware prefix cache bounded by its 11 GB ceiling.

## [Unreleased]

### Fixed
- Fixed the vMLX MiMo-V2.5 MLLM language-model adapter so `mlx_vlm` text turns can pass `inputs_embeds` and receive a `.logits` output object instead of failing with HTTP 500. MiMo long-output quality remains a separate blocker.
- Kept the Qwen3.5 dense native-MTP GatedDeltaNet patch compatible with `gdn_sink` propagation; current source and installed-app Qwen35/Qwen27 MXFP8 MTP smokes completed without the reported `gdn_sink` crash. Qwen35 speed is healthy in the smoke, while Qwen27 speed/equivalence remains open.
- Added packaged Qwen27 MTP speed evidence: MXFP8-MTP is stable but slow under model-owned stochastic defaults, clears decode only under the explicit deterministic native-MTP policy, and still has prompt-processing rows below target; JANG_4M-MTP decodes faster but also has a PP nuance.
- Recorded current Gemma4 12B JANG_4M source and installed-app speed gates above the 45 tok/s default target. Installed-app topk64 remains a nuance because its median was below target even though the top-k primary-regression threshold did not fire.
- Recorded installed-app Gemma4 12B JANG_4M VLM recovery evidence: a small image request succeeds with cache stack enabled, and a forced image-prefill rejection returns HTTP 413 without poisoning the next text request.
- Refreshed no-heavy VL/media, API/cache, cache-architecture, Step3.7 crash-class, and admin sleep contracts after the Gemma4 VLM recovery proof; all five current artifacts pass and remain scoped as route/policy evidence, not full live model-family clearance.
- Fixed MiMo V2.5 JANG_2L VLM paged-prefix cache hits for asymmetric full/SWA KV heads. MiMo full-attention layers use `num_key_value_heads=4`, while SWA `RotatingKVCache` layers use `swa_num_key_value_heads=8`; the VLM cache store path was slicing every layer down to the primary count, causing second-turn cache hits to fail with a 4-head vs 8-head concatenate error. The runtime now preserves all config-declared valid KV head counts, paged cache schema is bumped to `paged_n1_keys_v6`, and the live MiMo cache-stack repro now passes with a 25-token paged cache hit.
- Refreshed API surface, tool-call, reasoning/template, panel tool security, MCP/gateway policy, release-surface, and cancellation contracts after the MiMo tool blocker. Streaming detokenizer tests remain skipped in the current environment and are not counted as streaming proof.
- Added a live source-level streaming API proof with Gemma4 12B JANG_4M under continuous batching, paged prefix cache, block-disk L2, and q8 KV: the streamed Chat Completions response emitted visible chunks and final `[DONE]`.
- Added installed-app streaming API parity evidence for the current `/Applications/vMLX.app` bundled runtime with Gemma4 12B JANG_4M: streaming Chat Completions emitted visible chunks and final `[DONE]`.
- Refreshed bundled Python and staged-app package evidence after the MiMo cache fix. The staged app now has current engine/JANG hash parity and packaged integrity passes, while `/Applications/vMLX.app` remains stale until a new app is installed/notarized.
- Tightened release-regression ledger expectations so non-deferred live-model blockers remain explicit and do not get mistaken for packaged-integrity failures.

### Notes
- Future release notes for the current runtime/model/UI/API issue wave must credit GitHub `@Hornsan1` for reporting many of these issues.
- JANGQ/JANG tools MiMo V2 support is on `jjang-ai/jangq` main at `d1316c3`.
- MiMo V2.5 JANG_2L text/cache validation does not provide full MiMo release
  clearance; tool behavior, VL/audio/video, and performance remain separate
  rows.
- MiMo V2.5 JANG_2L tool behavior remains a release blocker: a forced XML-function tool call stayed HTTP 200 but produced raw malformed `<tool_call>` text and punctuation garbage with zero parsed tool calls.

## [1.6.7] - 2026-07-12

### Changed
- Paged KV cache now defaults ON for autodetected text families when continuous
  batching and the prefix cache are active. Detection is per-family: dense text
  and hybrid-SSM/linear-attention families (Qwen3.5/3.6, Zaya, Nemotron-H,
  LFM2.5, Laguna, Hy3, Step-3.7) launch paged; multimodal/VL bundles, MiniMax-M3,
  openPangu-v2, and Gemma 4 stay on their own cache paths. DeepSeek-V4-Flash keeps
  its composite opt-in. Explicit `--use-paged-cache` / `--no-paged-cache` always
  win. The Electron panel is fully reconciled with the engine: the visible cache
  toggle, the launch flag, and the engine's effective policy now match for every
  family (no silent OFF), with a migration that flips only untouched prior-default
  text sessions.
- Reasoning defaults ON for every reasoning-capable family (Eric directive),
  applied uniformly across Chat/Responses/Anthropic/Ollama, with family guards
  (Mistral none/high, MiniMax custom-off, Laguna template-default-OFF) still
  authoritative. `max_thinking_tokens` is honored and gated to engine-answer-pass
  families on both streaming and non-streaming paths.

### Fixed
- Never-empty answer after runaway reasoning: a reasoner that consumes the whole
  token budget inside its hidden rail still streams a visible answer instead of
  empty content, across chat and `/v1/responses`, for qwen3.5/3.6, DeepSeek-V4,
  Step-3.7 and the MiniMax families. The answer pass runs a fresh context for
  deepseek_v4/step3p7 (fixes a malformed appended reasoning turn) and no longer
  leaks truncated reasoning or a `<think>`/`<thinking>` re-entry into the visible
  answer.
- MLLM/VL paged cache now honors the same RAM byte ceiling as the text path
  (`--cache-memory-mb`/`--cache-memory-percent`). Previously the MLLM scheduler's
  block pool was bounded only by `--max-cache-blocks`, so under paged-default-ON a
  forced-paged VL/MLLM model (e.g. Step-3.7 video) had no byte ceiling on its
  in-RAM block KV mirror. Verified live: Step-3.7 (q4 KV, 3 GB ceiling) held flat
  at +0.27 GB RSS across an 18-turn growing multiturn.
- Memory-cache lock inversion (community #233 deadlock), a `stop_token_ids`
  attribute crash, and a lockless cache-clear race.
- Paged cache RAM is now bounded by a byte ceiling: completed-request block
  references are released after store so the ceiling can evict (Wave-18 RAM fix),
  and the disk-L2 longest-prefix off-by-one that re-fed the wrong matched block
  was corrected.
- Loaded MLLM `/v1/completions` (plus stream and Ollama raw) now route through the
  chat rail instead of feeding an unframed prompt to the MLLM tokenizer, which had
  caused constant-token degeneration on Gemma-4 and other VL bundles.
- `gpt_oss` streaming no longer leaks harmony analysis into visible content.
- Laguna uses the glm47 tool parser (GLM arg_key/arg_value format, fixes silent
  streaming tool-call loss); ZAYA/ZAYA1-VL pin `zaya_xml` as a defense against a
  stale bundle stamp.
- `/v1/capabilities` reports truthful cache/TQ policy descriptors for hybrid and
  mixed-SWA families; the panel launch RAM estimate no longer over-counts
  lazy-mmap JANG bundles.

### Notes
- Step-3.7-Flash video in the Electron UI was investigated (Codex NO-GO) and found
  sound: the engine, qwen3 reasoning parser, step3p5 tool parser and mRoPE all
  produce a clean reasoning/answer split under an adequate output budget. Step-3.7
  is a very long reasoner whose JANG_K bundle omits `max_new_tokens`; at the 4096
  fallback it can be capped mid-reasoning, in which case the never-empty floor
  emits truncated reasoning as the answer (documented behavior, not a parser leak).

## [1.6.6] - 2026-07-10

### Added
- Tencent Hy3 (hy_v3) family runtime: model support with clean handling of
  MTP-dropped bundles (`num_nextn_predict_layers=0` loads without a headless
  speculative head; capabilities endpoint reports the drop), hunyuan tool
  parser and qwen3 reasoning rails wired from the registry, and the native
  MTP runtime for bundles that preserve the head. Native MTP autodetects the
  profitable draft depth (env > measured `vmlx_mtp_tuning.json` sidecar >
  family fallback (hy_v3 -> depth 1) > generic D3); on the 2K-MTP bundle
  depth 1 is a measured ~+10-14% decode win while depths 2/3 collapse
  acceptance. The dead legacy JANG_2K profile block that silently disabled MTP
  was removed.
- Variant-suffixed special-token dialects (e.g. Hy3's `:opensource` spellings)
  are canonicalized at the token-to-text boundary, so reasoning split,
  tool-call parse, and think-tag stripping work on bundles that ship only
  suffixed specials.
- Reasoning parity across surfaces: a concrete `reasoning_effort` implies
  `enable_thinking` for reasoning-capable families when thinking is otherwise
  unspecified, applied uniformly across OpenAI Chat/Responses, Anthropic, and
  Ollama — so families with their own reasoning trigger (e.g. gemma4) engage
  reasoning via `reasoning_effort`. Family guards (Mistral none/high,
  MiniMax custom-off, unsupported families) remain authoritative.
- Text chat/completions honor a request-local `seed`; `/v1/capabilities`
  responds for the active model.

### Fixed
- The defensive multi-eos stop set now installs for tokenizers that only ship
  variant-suffixed eos/role tokens. The resolver falls back through the same
  dialect map, so the registry keeps one canonical spelling per family and
  role-flip stop protection is no longer silently inert on such bundles.
- Reasoning-runaway answer pass: a native always-reasoning or degraded runaway
  reasoner that consumes the whole token budget inside its hidden rail still
  emits a visible final answer instead of empty content. The answer pass draws
  the remaining budget with a bounded floor (never the unbounded fresh-budget
  the earlier path used). Applies to qwen3.5/3.6, gemma4, openpangu, MiniMax-M3
  and MiniMax-M2. Fixed alongside answer-pass double-emission, a display path
  that dropped visible text before an orphan `</think>`, and (Ollama) a
  streamed final answer that was misrouted into `message.thinking`.
- MLLM stochastic decode normalizes logits once before the shared sampler
  contract (prefill and decode now agree); invalid `native-mtp-depth` overrides
  are ignored instead of silently falling back to D3.
- Paged-cache block hashing uses a canonical type-tagged, length-delimited
  encoding for request-conditioned keys, so distinct multimodal/adapter
  conditions cannot collide onto the same KV blocks.
- Disk L2 (block) cache stores bfloat16 natively instead of a lossy bf16->fp16
  cast that overflowed large values to infinity and drifted low bits (the
  source of warm-vs-warm divergence on the paged+L2 path).
- L2 cache identity now includes a weight-artifact fingerprint plus sub-second
  config fingerprints, so replacing weights in place under the same path no
  longer reuses stale KV.
- Architecture/cache detection fails closed to a conservative native-cache
  path instead of failing open into an incompatible policy.
- TurboQuant KV `compress_after` is a real engine field wired end-to-end; live
  encode is available for diagnostics but defaults OFF (measured not to
  beneficially trade coherence or memory on any family), and capability/log
  strings now describe the actual encoded state rather than unearned 3-bit /
  memory-savings claims.

### Notes
- Bundle-side (jang converter, shipped separately): Hy3 bundles now stamp
  audited chat sampling defaults (`temperature 0.9, top_p 0.9, min_p 0.05`).
  The 2-bit routed tail was measured to loop at temp 0.9 with untruncated
  sampling (3/6 at top_p 1.0); either floor eliminates it (0/20 live).
  Explicit request parameters always override bundle defaults.

## [1.6.5] - 2026-07-09

### Fixed
- The prefix cache never hands decode the stored entry. `_clone_cache_for_fetch` used to fall back to returning the cached object itself whenever a layer failed the isolate-clone gate; decode writes through every layer it is given, so the stored prefix was mutated in place and later hits replayed a polluted context. This is the same defect already fixed for `RotatingKVCache`, the MiniMax-M3 sparse cache and `TurboQuantKVCache`. A cache that cannot be isolated is now a clean miss: the exact- and forward-match paths recompute, and the hit counters only advance once a clone succeeds.
- Cumulative SSM/conv state (`MambaCache` / `ArraysCache`) is now cloned rather than shared. It cannot be *reduced* to a shorter token count, but `_clone_cache_for_fetch` is only ever called at the entry's full cached length, where nothing needs reducing — so the state arrays are copied. Reverse-match truncation, which is a true reduction, still refuses and takes a clean miss. This was latent rather than live: both schedulers auto-switch hybrid models onto the paged cache, so a hybrid layer list only reaches the memory-aware cache when hybrid detection fails open (`make_cache()` raising). Correctness no longer depends on that detector.
- A worker-side clone failure now takes a clean miss instead of cloning inline on the API thread, which would have resurrected the 1.6.4 stream bug (empty 200 on every cache hit).

### Notes
- `tests/test_memory_cache.py` and `tests/test_cache_isolation.py` asserted the old aliasing contract (“Same reference, no copy”) using cache stubs that the clone gate rejects, so they had only ever exercised the stored-reference fallback. They now use real `KVCache` layers and assert isolation.
- Verified live on Qwen3.6-27B (16 attention + 48 SSM layers): decode determinism, the prefix-cache pollution guard and multiturn recall all pass, KV+SSM cache hits are still served, and no cache is ever rejected as non-isolatable.

## [1.6.4] - 2026-07-09

### Fixed
- Prefix-cache hits no longer die with `RuntimeError: There is no Stream(gpu, 0) in current thread`. MLX binds every array to a completion event on the stream of the thread that created it; the prefix-cache isolate-clone was built on the API/event-loop thread, so the llm-worker could not evaluate logits derived from it. The throw was swallowed and the request returned an empty 200, which made every cache hit on a single-sequence model (Laguna) generate zero tokens. The clone is now built on the worker that owns the stream. Batched generators re-home cache tensors themselves, which is why gemma-4 never showed it.
- `TurboQuantKVCache` layers are no longer handed to decode by reference on a prefix-cache hit. TQ is not a `KVCache` subclass, so the clone gate rejected it and returned the stored entry; TQ is monotonic-growth, so decode appended into the cached prefix and later hits replayed a polluted context. `_truncate_cache` now rebuilds TQ layers as fresh independent caches. Only mixed layer lists stored a live TQ object, so the pure-TQ families (MiniMax-M2.7, Qwen3.6) never surfaced it.

### Added
- `VMLX_SWA_TQ=1` (opt-in, default off): per-layer TurboQuant KV on the full-attention slots of mixed sliding/full attention models, leaving the sliding slots on their native `RotatingKVCache` so the `mixed_swa_kv_v1` window metadata survives. gemma-4-12B maps to 8 TQ + 40 rotating; gemma-4-26B-A4B to 5 + 25; Step-3.7-Flash to 12 + 33. Verified byte-equal to the default path on all three. See `docs/MIXED-SWA-TURBOQUANT-KV.md`.
- Regression test `tests/test_turboquant_prefix_cache_clone.py` pinning TQ prefix-cache isolation.

### Changed
- ZAYA text models now reason by default. `architecture_hints.default_enable_thinking` was hard-stamped `False`, so a request that did not pass `enable_thinking` rendered a closed empty `<think></think>` block and the model never opened its reasoning rail. `think_in_template` stays `False` — ZAYA opens its own rail and the qwen3 reasoning parser routes it to `reasoning_content`. An explicit `enable_thinking` still wins in both directions.

### Notes
- TurboQuant KV does not compress during decode on any model: `compress_after` defaults to `0` and nothing sets it, so a `TurboQuantKVCache` holds plain float KV. Only `_recompress_to_tq` (paged / disk-L2 reconstruction, and the paged cache is off by default) ever calls `.compress()`. `VMLX_SWA_TQ` is therefore output-identical to the default path today and is kept opt-in until `compress_after` is wired end-to-end, which will change decode numerics for every TurboQuant family.
- Pre-existing, unchanged by this release, both reproducible with `VMLX_SWA_TQ` off: prefix-cache hits are not byte-faithful to a cold run on gemma (warm is stable — the q4 stored-prefix round-trip), and gemma-4-26B-A4B's greedy answer differs solo vs co-batched at `--max-num-seqs 2` intermittently.

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
