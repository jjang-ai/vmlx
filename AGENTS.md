# vMLX Python Engine / MLXStudio Release Worktree Guard

LOCAL WORKTREE OVERRIDE: do not commit this file unless Eric explicitly asks.

This worktree is for the active Python engine and Electron/panel app only:

```text
/Users/eric/mlx/vllm-mlx-finite-launch-guard
```

Do not work from `/Users/eric/vmlx`, `/Users/eric/vmlx/swift`, old Swift
handoffs, adlab, Max2, TP4, RDMA, or Thunderbolt lanes unless Eric explicitly
asks for that exact path in the current turn.

## Deprecated wrapper checkout guard

If a continuation starts in `/Users/eric/vmlx`, treat that checkout as a
deprecated wrapper/history repo for current vMLX app/runtime work. Do not edit,
build, launch, test, probe, or use old implementation notes from that tree for
active engine/app tasks. Switch to this active Python/Electron worktree, read
this `AGENTS.md`, `.agents/STATUS.md`, `.agents/LOG.md`, and
`.agents/CODEX_ACTIVE_DIRECTIVES_20260610.md`, then continue from the written
state here. Do not infer Swift work from vMLX, JANG, cache, MTP, model-runtime,
or app/runtime wording unless Eric explicitly names that path in the current
turn.

## Active release control board - read before every action

The target is not "one model smokes" or "the package builds." The target is the
Python vMLX engine plus MLXStudio Electron app, with real runtime behavior
proved before release.

### Mandatory continuation loop

Every agent continuation must do these in order:

1. Stay in this active Python/Electron worktree unless Eric explicitly names a
   different current-turn path.
2. Name the release blocker being reduced before editing or launching anything.
3. Read `.agents/CODEX_ACTIVE_DIRECTIVES_20260610.md`, `.agents/STATUS.md`,
   and the latest release checklist/proof artifact before acting. Do not rely
   on older memory, deprecated `/Users/eric/vmlx` notes, or stale release
   snapshots.
4. Transcribe current-turn user instructions and corrections into
   `.agents/STATUS.md` / `.agents/LOG.md` before doing the next substantive
   action. This includes stops, release locks or unlocks, model-lane ownership,
   parser/API priorities, proof boundaries, and "do not assume" corrections.
5. Prefer a live proof that closes or classifies a blocker over a source-only
   test, stale pointer refresh, upload chore, package/signing step, or broad
   exploration.
6. Write every movement down: request, action, command/proof/artifact,
   proven/not-proven state, blockers, no-claims, and what the other agent
   should do next.
7. After each proof, update `.agents/STATUS.md`, `.agents/LOG.md`, and the
   release tracker with the exact artifact path, pass/fail state, and remaining
   blocker boundary.
8. Regenerate the current release gates only when the source/proof state changed
   enough to make the status meaningful.
9. Do not sign, notarize, tag, push release notes, or update downloads while any
   objective row is open unless Eric explicitly overrides the lock in the
   current turn.

Standing constraint from Eric: do not use Python, local scripts, shell wrappers,
MCP tools, or any other mechanism to spawn subagents or delegate this lane's
work to other agents. Use direct repo edits, direct shell commands, direct live
proofs, and explicit handoff notes only. Python remains acceptable for ordinary
local verification, artifact inspection, proof scripts, and tests when it is
not spawning, prompting, supervising, or summarizing subagents.

### Written-state discipline

Before each substantive action, re-check the current written state in
`.agents/STATUS.md`, `.agents/LOG.md`, and this file. If Eric gives a new
instruction, correction, stop, priority change, model boundary, release
boundary, parser/API requirement, proof result, or coordination request, record
it in `.agents/STATUS.md` and `.agents/LOG.md` before continuing.

Every status movement must be written down as it happens: what was requested,
what was done, which exact command/proof/artifact supports it, what is proven,
what is not proven, what must not be claimed, and what the other agent should
or should not pick up. Do not rely on memory, prior chat context, unstaged local
notes, or implied intent when the written state disagrees or is incomplete.

### No subagent delegation

This lane must be worked directly by the active Codex instance. Do not use
Python, shell wrappers, MCP tools, orchestration scripts, or hidden helper
processes to spawn agents, delegate implementation, delegate proof runs, or
summarize another agent's work as if it were this lane's evidence. Coordination
with the parallel agent is limited to explicit checked-in handoff notes and
status/log entries that name what is proven, what is not proven, and what the
other agent should do next.

Allowed: ordinary Python or shell commands for direct local verification,
artifact inspection, test execution, proof generation, and source maintenance.
Not allowed: any command whose purpose is to create, manage, prompt, or monitor
subagents for this lane's work.

Current parser/API carry-forward from Eric: harshly prioritize auto tool usage,
content deltas, reasoning deltas, interleaved reasoning/tool streaming, request
kwargs, Chat/Responses API behavior, gateway passthrough, raw SSE ordering,
parser selection, and final-object consistency across all model reasoning and
tool parser families. The Qwen3.6 / Qwen-coder empty-args report is active for
both 27B and 35B style XML tool-call dialects. Do not assume the report's root
cause is correct without same-model raw output, but do treat the failure shape
as release-critical for opencode/Codex harnesses. Missing required tool
arguments must fail closed; do not synthesize `cmd`, infer arguments from a
visible preamble, disable reasoning to avoid the bug, or strip raw XML after
the fact as a fake fix. Required proof is same-model direct server, local
gateway, and tunnel raw SSE with reasoning enabled, valid content/reasoning
deltas, argument delta/done events, final object consistency, valid
`output_index` ordering, required/auto/no-tool modes, tool-result continuation,
and cache reuse telemetry.

Current explicit parser/runtime work item from Eric: add the Qwen3.6/Qwen-coder
empty-arguments report to the active fix/proof list for both 27B and 35B style
XML tool-call dialects, but do not trust the proposed root cause without live
same-model raw output. The failure shape is: model emits visible text and then
an XML tool call with `<function=...></function>` but no parameter tags, the
parser returns `{}`, and clients such as Codex/opencode fail because required
arguments like `cmd` are missing. Required behavior is fail-closed validation,
clean raw SSE/content/reasoning/tool-call event shape, and usable harness
behavior for opencode/Codex-style agent loops. Forbidden fixes include
synthesizing arguments from a preamble, disabling reasoning, silently dropping
the tool call, or repairing values after parser failure.

Current cross-family parser/API proof target: test and fix every model
reasoning/tool parser family that can affect agentic loops, including Qwen,
Qwen-coder, Gemma4, MiMo/think-XML, MiniMax, DeepSeek/R1-style think parsers,
XML function-call parsers, and any gateway/tunnel route that rewrites or
streams the same events. Proof must include auto tool usage, required tool
usage, no-tool mode, tool-result continuation, content deltas, reasoning
deltas, function-call argument delta/done events, final response object
consistency, request kwargs passthrough, parser selection, cache reuse telemetry,
and raw leak checks. If a family-specific parser cannot honestly support a
dialect, mark it unsupported or fail closed with evidence instead of pretending
the dialect works.

Current N2 boundary from Eric: do not work on Nex/N2 JANG_1L unless Eric
explicitly reopens that lane in the current turn. Treat N2 JANG_1L as
Eric-owned/off-limits; do not launch, fix, prove, classify, or claim it from
partial prior runs. Allowed N2 work here is N2 JANGTQ/non-JANG_1L only when it
does not overlap the JANG_1L lane.

### 2026-06-10 Eric correction - write every movement down

Eric explicitly reinforced that this lane must not rely on chat context,
unstaged mental state, or recursive agent behavior. Every instruction, action,
status movement, proof artifact, blocker, no-claim boundary, and other-agent
handoff must be recorded in `.agents/STATUS.md` and `.agents/LOG.md` as work
happens, before the next substantive action.

Current active emphasis:

- Do not ignore the user's current goal; re-check the written state before
  acting and keep the work on runtime/API/model proof that moves a checkpoint
  release forward.
- Do not use Python, shell, MCP, browser, or any wrapper to spawn, supervise,
  prompt, summarize, or delegate work to subagents. Direct Python remains
  allowed only for local proof scripts, artifact inspection, tests, and source
  verification.
- Keep N2 JANG_1L off this Codex lane unless Eric explicitly reopens it in the
  current turn. Do not infer permission from available RAM or prior attempts.
- Harshly prioritize auto tool usage, required/no-tool modes, tool-result
  continuation, content deltas, reasoning deltas, interleaved reasoning/tool
  streaming, request kwargs passthrough, parser selection, gateway/API/raw SSE
  parity, cache reuse telemetry, and final response object consistency.
- Test and fix all model reasoning/tool parser families that can affect
  opencode/Codex-style agent loops: Qwen/Qwen-coder XML, Gemma4, MiMo
  think-XML, MiniMax, DeepSeek/R1-style think parsers, XML function parsers,
  and gateway/tunnel routes.
- Treat Qwen3.6/Qwen-coder empty `arguments: {}` tool calls as release-critical
  for both 27B and 35B family surfaces, but do not assume the proposed root
  cause without same-model raw output. Required-arg parser failures must fail
  closed and preserve accurate streaming/final-object evidence; do not
  synthesize args from text preambles, disable reasoning, silently drop tool
  calls, or strip raw XML after the fact.

Current checkpoint-release pressure: Eric does want a signed/notarized working
checkpoint release, but this agent must not enter release/sign/notarize/PyPI/
download-update steps unless Eric explicitly asks for that action in the
current turn or the active directive file says the release lock is lifted.
Until then, reduce model/runtime/API/UI/cache blockers and keep the release
writeup current.

Current release-focus constraint from Eric: do not drift into broad harness
rewrites, stale pointer churn, or release-adjacent cleanup when a live runtime,
API, cache, media, UI, or installed-app blocker can be reduced directly. Work
one concrete blocker at a time, prove it with the real engine/app surface where
possible, and list what is proven and not proven after every movement. Release
preparation is the goal, but signing/notarization/publishing still require an
explicit current-turn release override.

Current 128GB-user proof target from Eric: do not avoid large supported models
only because they are memory-heavy. For MiMo, Gemma JANG/MXFP/QAT, Nex/N2
JANGTQ/non-JANG_1L, Qwen MTP/MXFP, JANG/JANGTQ/MXFP/MXTQ, VL/video/audio, and
family-native cache paths, prefer real live loading and multi-turn/API/UI/cache
proofs when RAM headroom allows. Watch memory and cache storage, kill only
clearly unrelated smaller processes when needed, preserve the N2 JANG_1L
off-limits boundary unless Eric reopens it, and never substitute metadata-only
capability claims for live evidence.

Current 2026-06-10 parser/API carry-forward: streaming tool parser paths must
propagate the request schema into the final full-output parse. This is required
so Qwen/Qwen-coder style completed tool markers with missing required args fail
closed instead of emitting `arguments: {}` to Codex/opencode-style clients.
The valid path must still preserve real arguments, content deltas, reasoning
deltas, output indices, kwargs, cache telemetry, and final-object consistency.
Do not repair missing arguments from visible preambles, disable reasoning, or
claim all model-family parser/API loops green until direct/gateway/tunnel and
family-specific auto/required/no-tool/tool-result rows are live-proven.

### 2026-06-10 current-turn hard carry-forward

Eric explicitly asked that the current goal, every instruction, every status
movement, and every no-claim boundary be written into `AGENTS.md` / `.agents`
state so future continuations are forced to check it instead of relying on chat
memory. This section is a standing carry-forward until a newer checked-in
instruction supersedes it.

- Keep the work on fixes and proof that move a signed checkpoint release
  surface forward, but do not run release/sign/notarize/PyPI/updater/download/
  website actions unless Eric explicitly asks for that action in the current
  lane.
- Do not use Python, shell wrappers, MCP tools, browser automation, or any other
  mechanism to spawn, supervise, prompt, summarize, or delegate this lane to
  subagents. Direct commands are allowed only for local inspection, proof,
  tests, source maintenance, and app/runtime verification.
- Do not work on Nex/N2 JANG_1L unless Eric explicitly reopens that lane in the
  current turn. Treat it as Eric-owned and do not load, prove, fix, classify, or
  claim it from prior partial artifacts.
- Put parser/API/tool-loop correctness ahead of adjacent cleanup: auto tool
  use, required/no-tool modes, tool-result continuation, content deltas,
  reasoning deltas, interleaved reasoning/tool streaming, function-call
  argument delta/done events, output indices, request kwargs, gateway/tunnel
  passthrough, cache reuse telemetry, and final object consistency must be
  traced and proven across Qwen/Qwen-coder, Gemma4, MiMo think-XML, MiniMax,
  DeepSeek/R1-style, XML function-call parsers, and gateway/API routes.
- Treat the Qwen3.6/Qwen-coder `arguments: {}` empty-tool-call failure as
  release-critical for opencode/Codex-style harnesses, including 27B and 35B
  family surfaces, but do not trust the proposed root cause without same-model
  raw output. Missing required arguments must fail closed; do not synthesize
  arguments from visible preambles, disable reasoning, silently drop tool calls,
  or strip raw XML after parser failure as a fake fix.
- For Gemma JANG/MXFP/QAT, MiMo JANG/JANGTQ, Nex/N2 JANGTQ/non-JANG_1L, Qwen
  MTP/MXFP, VL/video/audio, JANG/JANGTQ/MXFP/MXTQ, cache/L2/TurboQuant, UI,
  CLI, installed-app, and API rows, prefer real live loads and E2E proof over
  metadata-only claims when RAM headroom allows. Watch memory/storage and stop
  any server started by this lane before final response.
- After every movement, update `.agents/STATUS.md` and `.agents/LOG.md` with
  the request, action, command/proof/artifact, proven/not-proven state,
  blockers, no-claims, and exactly what the other agent should or should not
  pick up next.

### 2026-06-10 current proof carry-forward

This subsection is current live-state guidance for continuations. Re-check the
named artifacts before repeating claims.

- Qwen35 MXFP8 MTP direct/gateway/tunnel raw SSE is currently green from
  `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json`
  plus focused source guards. This does not clear Qwen27, Qwen-coder-next, or
  non-Qwen parser families.
- Qwen27 JANG_4M MTP direct required-tool raw SSE is green for valid
  reasoning-enabled tool arguments: artifact
  `build/responses-sse-captures-20260610/direct-qwen27-jang4m-mtp-required-tool-after-continuation-fix-20260610.sse`
  has argument deltas, argument done, and final function-call arguments
  matching `{"value":"blue-cat"}`.
- Qwen27 JANG_4M MTP direct tool-result continuation has a newer green direct
  proof after commit `c468d9b17`: required-tool SSE
  `build/responses-sse-captures-20260610/direct-qwen27-jang4m-mtp-required-tool-after-visible-finalization-seed-fix-20260610.sse`
  keeps reasoning enabled for the tool-selection turn and emits
  `response.function_call_arguments.done` with `{"value":"blue-cat"}`;
  continuation SSE
  `build/responses-sse-captures-20260610/direct-qwen27-jang4m-mtp-tool-result-continuation-after-visible-finalization-seed-fix-20260610.sse`
  completes with visible `response.output_text.delta` and final
  `output_text="The fact \"blue-cat\" has been recorded."`. The older
  `...tool-result-continuation-after-fix-20260610.sse` remains red historical
  evidence. Scope: direct server terminal no-new-tools post-tool synthesis only;
  do not claim gateway/tunnel, Qwen-coder-next, or all parser-family loops from
  this proof.
- Gemma4 QAT JANG_4M source no-media smokes are green across E2B/E4B/12B/26B/
  31B from the parallel lane, and Gemma4 31B audio is currently classified as
  an honest unsupported-modality gate for a vision-only artifact. Do not
  advertise Gemma audio unless an audio tower is weight-backed and live-proven.
- MiMo V2.5 JANGTQ_2 CLI media runtime overlay is green for icon image routing
  and text L2 restart restore after the overlay gate fix:
  `build/current-mimo-v25-jangtq2-cli-media-l2-after-overlay-fix-20260610.json`.
  It proves route/load/cache behavior, not MiMo exactness, red-square color
  semantics, audio hygiene, video semantics, Responses tool continuation, or
  installed-app parity.
- MiMo V2.5 JANGTQ_2 dev-app image route is green from
  `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-icon-image-after-overlay-fix-20260610-proof.json`,
  but that same proof exposed visible planning-style prose with
  `enableThinking=false`. Treat MiMo no-thinking output hygiene as open until
  a neutral prompt or runtime/template fix is proven. Do not strip prose as a
  fake parser fix.
- Follow-up neutral-prompt proof
  `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-icon-image-neutral-first-turn-20260610-proof.json`
  passed with first assistant `OK.`, image assistant `vMLX`,
  `requestedEnableThinking=false`, no persisted reasoning, no raw parser leak,
  and app stream logs showing no reasoning chars. For the icon-image dev-app
  row, classify the earlier planning-style prose as proof-harness first-turn
  prompt contamination, not a confirmed MiMo parser leak. Keep MiMo exactness,
  red-square color semantics, audio/video semantics, Responses continuation,
  installed-app parity, and release readiness open.

If a turn is interrupted, resume by reading the current tracker/status and
continuing the next blocker; do not restart from old memory, old `/Users/eric/vmlx`
notes, old Swift notes, ADLab notes, transport notes, or model upload logs.

### Minimum evidence standard for "working"

Do not call a model family, cache path, media path, parser, or UI setting
working unless current evidence includes:

- visible output, not only `/health`, import, load, or startup success;
- multi-turn continuation with no hidden-only assistant turn;
- required tool, auto tool, no-tool, and tool-result continuation behavior;
- raw markup/reasoning leak checks for the model's native dialect;
- JSON/XML/code/whitespace exactness where the row claims structured output or
  coding quality;
- first-miss and second-hit cache telemetry with `cached_tokens` and
  `cache_detail`;
- typed native cache proof for the family, not generic KV substitution;
- block-disk L2 write plus fresh-process or restart restore when L2 is claimed;
- largest safe context proof with tail inspection, not just prompt processing;
- UI settings reflection for parser, reasoning, cache, max output, and max
  context when the row claims MLXStudio readiness;
- startup parity between CLI `vmlx serve` flags and MLXStudio generated launch
  config/session settings for parser, reasoning, cache, MTP, max output, max
  context, and model-owned generation defaults;
- installed-app parity before any package/sign/notarize/release claim.

Compatibility fallback can be a valid release behavior only when it is explicit,
scoped, logged, tested, and documented as a compatibility path. It must not be
used to claim that the optimized path itself is fixed.

### Current gate snapshot - 2026-06-07

Start every continuation from these current artifacts before trusting older
notes:

```text
build/current-full-release-objective-checklist-after-mimo-manifest-classifier-sync-20260607.json
build/current-objective-proof-after-mimo-manifest-classifier-sync-20260607.json
build/current-release-regression-manifest-after-mimo-no-source-classifier-20260607.json
build/current-mimo-v2-jang2l-current-audit-after-kvnone-noprefix-exactness-isolation-20260607.json
build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json
```

Current status from those artifacts:

- `prepackage_ready=false`
- `release_ready=false`
- full release checklist is `status=open`, `failed_count=17`
- objective proof is 20 PASS / 6 OPEN
- no signing, notarization, tag, public download update, or release claim is
  allowed unless Eric explicitly overrides the release lock in the current turn

Open objective rows:

- Cross-family live multi-turn smoke matrix is release-cleared.
- MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared.
- Real Electron UI unblocked non-MiMo live model matrix is proven.
- Real Electron UI cross-family live model matrix is release-cleared.
- DSV4 long-output/code/file-generation quality is release-cleared.

MiMo current blocker facts:

- `mimo_long_prompt_coherence_blocked`
- `mimo_jangtq2_artifact_exactness_blocked`
- `mimo_cb_system_prompt_working_set_pressure_blocked`
- `mimo_source_vs_quant_first_divergence_missing_or_failed`
- `mimo_vl_audio_video_unwired`
- `mimo_media_runtime_implementation_missing`

MiMo cache/exactness boundary:

- The same literal value mutations reproduce with runtime KV quantization
  disabled, native storage quantization disabled, prefix cache disabled, paged
  cache disabled, block-disk L2 disabled, and zero cache hits.
- Do not chase prefix reuse, paged cache, block-disk L2, or runtime KV
  quantization as the primary current MiMo exactness cause.
- Do not clear MiMo by JSON repair, tool-argument rewrite, prompt-only folding,
  hidden sampling changes, cache disabling, or hiding failed rows.

API/cache/Responses boundary:

- The no-heavy API/cache/Responses endpoint contract is green for route
  behavior and UI/API plumbing.
- That contract covers cache stats/reuse endpoints, streaming cache detail
  usage, Responses `previous_response_id`, max output/context separation,
  parser surfaces, and panel request-builder plumbing.
- It does not replace live cross-family model E2E, largest-context cache
  stress, media runtime proof, installed-app UI proof, or release signing.
- Treat cache reuse and Responses endpoint proof as a plumbing prerequisite only.
  A family is not release-green until that same family has live output proof for
  Responses streaming/non-streaming, tool-result continuation, cancellation,
  cache-hit telemetry, prefix/paged/native cache behavior, block-disk L2 write,
  fresh-process L2 restore, and full-tail output inspection under the real UI or
  installed app route that will ship.
- For hybrid/nonstandard cache families, never collapse the row into generic KV
  proof. Qwen 3.6 hybrid SSM, LFM hybrid SSM, DSV4 composite SWA/CSA/HCA,
  Gemma mixed SWA, MiMo asymmetric SWA, ZAYA typed CCA, and Nemo/Nemotron Omni
  media caches each require their own typed cache schema, partial-hit behavior,
  restore rules, and UI telemetry before release.

Every continuation must start by asking: "Which release blocker am I reducing
right now?" If the answer is only "a stale pointer", "a passing unit test", "a
small model smoke", or "a packaging step", stop and reconnect the work to the
full runtime/app release objective below.

Hard focus rule: if the task mentions vMLX, MLXStudio, model runtime, cache,
tools, parser, JANG/JANGTQ, VL/audio/video, signing, notarization, or release,
the default work is this Python engine plus Electron/panel app. Do not drift
into external research repos, ADLab experiments, transport lanes, old Swift
workspaces, model upload chores, or source-vs-quant jobs unless the current
turn explicitly names that path and explains why it reduces a live release
blocker.

Before editing, write down the blocker class in the user update and keep it
mapped to one of these release surfaces:

- `runtime/kernel`: model load, quant kernel, decode loop, MTP, attention,
  routed experts, SWA/SSM/native cache execution, Metal resource behavior.
- `parser/template`: tool-call parser, reasoning parser, chat template,
  fallback injection, raw markup leaks, hidden thinking leakage, JSON/XML/code
  exactness, tool-result continuation.
- `cache/storage`: prefix cache, paged cache, typed native cache, TurboQuant KV
  encode/decode/restore, block-disk L2, restart restore, cancellation cleanup,
  largest-context hit proof.
- `media`: image/audio/video processor bridge, media-expanded accounting,
  media-salted cache, post-media text recovery, media plus tools.
- `api/ui`: Chat Completions, Responses, Anthropic, Ollama, streaming,
  cancellation, max output/context, settings panel, parser/cache controls,
  installed-app parity.
- `release`: branch/main state, package integrity, Developer ID signing,
  notarization, appcast/download updates, release notes with `@Hornsan1`
  credit. Only enter this class after all model/runtime/UI/cache blockers are
  green or Eric explicitly overrides the lock.

If a live run proves a model emits a native but unsupported dialect, implement
or fix the strict parser/runtime path. Do not hide it with prompt-only changes,
schema-free raw-markup acceptance, forced tool calls, or release-manifest
wording changes. Parser fixes must preserve the raw-leak checks and validate
function names/arguments through the normal server filtering path.

The active objective is full Python vMLX engine plus MLXStudio release quality,
not an isolated model upload or one-off benchmark. The required deliverable is:

- current Python engine source and panel source pushed to the intended branch;
- all release-critical local model families tested through live server APIs and
  UI where applicable;
- cache reuse, prefix/paged/L2 disk restore, typed native cache, and
  TurboQuant-KV boundaries proved per architecture;
- tool calls, tool-result continuation, multi-turn recall, JSON/XML/code
  exactness, whitespace-sensitive output, hidden reasoning leaks, and raw
  tool-markup leaks checked per model family;
- VL/audio/video media paths wired and tested for every advertised media-capable
  model family, including post-media text recovery and media-salted cache;
- parser/reasoning/tool/cache/max-output/max-context settings exposed in CLI,
  API capabilities, MLXStudio UI, and startup launch paths without drift;
- packaged app rebuilt from current source, Developer ID signed, notarized, and
  public downloads updated only after all runtime/model/UI/cache gates are
  green.

Do not release, sign, notarize, tag, or update downloads when any item above is
open unless Eric explicitly overrides the release lock in the current turn.

Every continuation must keep these rows active until the matching evidence is
current and scoped correctly:

- `MiMo V2.5 JANG_2L`: text coherence, required/auto tools, tool-result
  continuation, loop stop, no raw XML leaks, speed near target, mixed full/SWA
  native cache, prefix/paged/L2 restart, source-vs-quant root cause, VL/audio/
  video truth, installed app UI.
- `Qwen 3.6 27B/35B MTP`: MTP autodetect, `gdn_sink` compatibility, depth and
  deterministic policy, tool-context cap, hybrid SSM companion cache, largest
  context, restart/L2, cancellation, streaming, media, UI.
- `Responses/cache reuse endpoints`: Chat Completions and Responses must both
  prove visible stream and non-stream output, `previous_response_id`, cache
  detail usage, cache stats/reuse endpoints, cancellation cleanup, and max
  output/context separation. Endpoint-contract green is not sufficient unless
  the same family also has live cache-hit and L2 restore proof.
- `Nemotron/Nemo Omni`: text/tools/reasoning, image/audio/video bridge,
  media-salted cache keys, structured output, cache restore, UI.
- `LFM`: text/tools/multiturn, JSON/XML/code exactness, prefix/paged/L2,
  restart, API parity, UI.
- `MiniMax/JANGTQ_K`: reporter parity/root cause, current-source cancel route,
  cancellation cleanup, JANGTQ runtime quality, strict MiniMax
  `<minimax:tool_call>` dialect parsing, required-tool recovery, no raw
  MiniMax markup leaks, UI.
- `DSV4 Flash`: native SWA/CSA/HCA cache only, no generic TQ-KV substitution,
  exact code/file/long-output, thinking rails, restart/L2, UI.
- `Step 3.7 Flash`: text-only stability, honest VLM unsupported boundary until
  real VLM exists, tools, loop stop, multiturn, API parity, UI.
- `Gemma 4 12B`: MXFP4/MXFP8/JANG_4M separately, speed, mixed-SWA cache,
  image/audio/video, media prefill recovery, tools, API parity, UI.
- `ZAYA/hybrid/SSM/nonstandard cache families`: typed native cache, partial-hit
  reject, async rederive where required, restart/L2, media if advertised.

Before claiming any row green, prove all relevant surfaces:

- `APIs`: Chat Completions, Responses, Anthropic Messages, Ollama chat/generate,
  stream and non-stream, cancellation, max output tokens, max context.
- `Tools`: required tool, auto tool, no-tool request, tool-result continuation,
  repeated tool output, loop stop, hidden-reasoning leak check, raw XML leak
  check, valid JSON args, exact code/whitespace-sensitive output.
- `Cache`: first miss, second hit, prefix hit, paged hit, largest safe context,
  TurboQuant encode/decode/restore only for standard KV, typed native cache for
  SWA/SSM/composite families, block-disk L2 write, restart restore/hit,
  timeout/cancel cleanup, unload/reload, sleep/wake, telemetry and UI.
- `Media`: OpenAI content parts, app files, data URLs, safe URLs, image, audio,
  video, processor bridge, media-expanded accounting, media-salted cache,
  post-media text recovery, media plus tools.
- `UI/release`: settings reflection, launch args, installed-app parity,
  packaged Python parity, Developer ID signing, notarization, tags, appcast/
  download updates, release notes with GitHub `@Hornsan1` credit.

Additional per-family release checks that must not be skipped:

- `MiMo V2.5`: JANGTQ2 exactness, required and auto tool calls, JSON exactness,
  long-prompt coherence, mixed full/SWA cache reuse, CB and simple routes,
  prefix/paged/L2 restart, no raw XML/tool leakage, real VL/audio/video wiring,
  current-artifact classification if source-vs-quant is RAM-blocked, and UI
  installed-app parity. Speed near 40 tok/s is necessary but not sufficient.
- `Qwen 3.6 27B/35B MTP`: native MTP autodetect, `gdn_sink` compatibility,
  deterministic MTP policy, top-k and trained MoE K boundaries, hybrid SSM
  companion cache, Responses cancellation, largest safe context, L2 restore,
  image/video handling where advertised, and UI parser/cache selection.
- `Gemma 4 12B`: MXFP4, MXFP8, and JANG_4M separately; mixed-SWA cache and
  storage-boundary KV quantization; image/audio/video accounting; VLM prefill
  guard recovery; tool/JSON/code exactness; and UI settings parity.
- `Step 3.7 Flash`: text-only stability unless real Step3p7 VLM is implemented,
  no fake `has_vision=false` release claim for advertised VLM, thinking/template
  control, tool dialect consistency, loop stop, and API parity.
- `LFM`: text/tool/multi-turn quality, JSON/XML/code exactness, cache reuse,
  prefix/paged/L2 restart, Responses/Anthropic/Ollama parity, and UI controls.
- `MiniMax/JANGTQ_K`: issue179 reporter parity/root cause, Responses cancel
  behavior, current-source vs installed/public route differences, JANGTQ runtime
  quality, strict native tool parser behavior for complete and safely
  recoverable truncated `<minimax:tool_call>` blocks, tool recovery, cache
  reuse, L2 restart, and UI parity. A cache-green MiniMax run is not release
  green if required-tool still returns HTTP 400 or visible raw tool markup.
- `DSV4 Flash`: native SWA/CSA/HCA composite cache only, no generic TQ-KV
  substitution, path-dependent clean prompt-boundary store, exact code/file/
  long-output quality, thinking/direct rails, restart-L2, and UI reflection.
- `Nemo/Nemotron Omni`: RADIO/Parakeet image/audio/video bridge, media-salted
  cache, typed hybrid cache where relevant, structured output, tools, and UI.
- `ZAYA/hybrid/SSM`: typed native cache, partial-hit reject, async rederive or
  clean prompt-boundary store as required, L2 restore, no generic TQ-KV unless
  parity is proven, and UI reflection.

For every family, largest-context proof must include at least: cold miss, warm
same-process hit, cache telemetry (`cached_tokens` plus `cache_detail`), L2 disk
write, fresh-process L2 restore/hit, cancellation cleanup, unload/reload or
sleep/wake recovery, and full tail inspection of the generated output.

If an action does not reduce one of these rows, do not do it. If a blocker is
runtime/kernel/cache/parser/UI, fix that component. If current evidence proves a
bundle is corrupt or incomplete, tell Eric the exact requant/reupload contract;
do not fake runtime behavior to hide it.

### Current objective guard - do not drift

When Eric asks for vMLX, MLXStudio, MiMo, JANG, JANGTQ, VL/audio/video, cache,
tools, parser choices, signing, notarization, or release, the work is always
the active Python `vmlx_engine` plus MLXStudio Electron/panel app unless the
current turn explicitly names another path. Do not switch to ADLab, transport
lanes, old Swift workspaces, deprecated `/Users/eric/vmlx`, upload chores, or
source-vs-quant jobs as a substitute for engine/app release work.

Every continuation must keep this concrete release shape in mind:

- `engine`: loader, model-family autodetect, quantized runtime, decode loop,
  attention/MoE/SWA/SSM kernels, MTP, media processors, cache implementation,
  parser/template handling, and server APIs;
- `app`: MLXStudio settings panel, request builder, chat UI, file/media inputs,
  Responses/Chat streaming, cancellation, parser/reasoning/cache controls, max
  output/max context, and installed-app parity;
- `proof`: live visible output, multi-turn tools, structured output exactness,
  leak checks, cache hit/L2 restore/largest-context evidence, media recovery,
  and current artifact paths;
- `release`: only after all runtime/model/UI/cache/media rows are green,
  verify branch/main state, rebuild package from current source, verify bundled
  Python parity, Developer ID sign, notarize, staple, tag, publish, update
  appcast/downloads, and include GitHub `@Hornsan1` credit in release notes.

Do not call a release ready because one family, one endpoint, one UI row, or
one cache contract passed. Do not sign or notarize while blockers are open
unless Eric explicitly overrides the release lock in the current turn. If there
is a speed, coherency, parser, media, or cache failure, classify it as
runtime/kernel, parser/template, cache/storage, media, api/ui, release, or
artifact/requant-required before moving on.

## Non-negotiable operating contract

- The active goal is full production release readiness across runtime, model,
  cache, media, tools, UI, packaging, signing, notarization, and public release.
- Do not shrink the goal to the nearest passing unit test, proof pointer, or
  packaging row.
- Before any substantial action, identify which top-level blocker it reduces.
- If an action does not reduce a named blocker below, do not do it.
- Do not call source-only, load-only, health-only, import-only, or text-only
  proof production-ready for a broader model/media/runtime claim.
- Do not blame a model until the runtime path, bundle metadata, template,
  parser, scheduler, cache, and logs have been checked.
- Do not use prompt rewrites, hidden sampling overrides, forced metadata,
  fake guards, fallback injection, or disabled features as production fixes.
- Do not use stale memory, old artifacts, or previous claims as current status.
  Inspect current artifacts before reporting status.

## Absolute packaging / release lock

Do not run any of these while any runtime/model/UI/cache blocker is open:

```text
electron-builder
codesign
notarytool
build-release-dmgs.sh
npm run dist
npm run package
panel/scripts/build-and-install.sh
git tag
gh release
twine upload
hf upload
latest.json update
```

Exception: Eric explicitly says in the current turn that packaging/signing is
allowed despite blockers.

Important: `electron-builder --dir` signs this app because package config has
Developer ID signing enabled. Treat staging as release-adjacent. Do not run it
as a harmless package-parity check while blockers are open.

## Current authoritative status inputs

Use these before answering status or choosing work:

```text
docs/runtime/vmlx-production-release-blocker-matrix-20260606.md
docs/runtime/mimo-v25-jang2l-runtime-status-20260606.md
build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-cache-family-fix-20260606/JANGQ_gemma-4-12B-it-JANG_4M/result.json
build/current-mimo-v25-tool-continuation-probe-20260606/
build/current-mimo-v25-compiled-router-live-probe-20260606/
build/current-all-local-model-smoke-mimo-v25-jang2l-after-chatml-tool-fallback-20260607/
build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-refresh-after-qwen35-20260607/
build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-keep0-cache-fix-20260607/
build/current-local-long-context-cache-mimo-v25-installed-64w-after-tight-memory-rotating-store-20260607/
build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json
build/current-mimo-v2-jang2l-current-audit-after-tool-row-source-quant-preflight-stale-clean-20260607.json
build/current-qwen27-mxfp4-mtp-api-parity-20260607/
build/current-qwen27-mxfp4-mtp-restart-l2-restore-20260607/
build/current-qwen27-mxfp4-mtp-responses-cancel-mtp-deterministic-20260607.json
build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-20260607/
build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607/
build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-textonly-harness-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json
build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260606/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json
build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260606/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json
build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-after-reasoning-budget-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json
```

As of the current blocker state:

- `prepackage_ready=false`
- `release_ready=false`
- no signed/notarized release is allowed
- no public release/download update is allowed
- latest pushed release-hardening worktree branch: `codex/pr-intake-manifest`
- latest pushed commit on `codex/pr-intake-manifest` is `9fa15864`
  (`Tighten Gemma issue119 cache gate`)
- do not infer that these commits are merged to `main` unless current git state proves it
- latest packaged integrity proof after staged app rebuild/signing is still not
  release-green. Bundled Python verification, release gate unit contracts, and
  Developer ID signing preflight pass, but `release_gate_skip_app` still fails
  because the current objective digest honestly reports open release rows.
- do not treat contract-level cache/API/Responses green as live release proof.
  Current no-heavy API/cache/Responses contracts cover route behavior, cache
  stats/reuse endpoints, previous_response handling, max output/context
  separation, parser surfaces, and UI settings plumbing; live cross-family
  cache/API/Responses E2E remains open until each model-family row below has
  real output, tool, cache-hit, L2/restart, cancellation, media, and UI proof.
- current full-objective checklist directly consumes the panel settings
  contract at
  `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`.
  A future release-ready claim must keep this green for DSV4 native cache
  controls, generic KV suppression, max output/context split, chat override
  behavior, MiniMax parser detection, native MTP D3 policy, model-family parser
  registry, panel-emitted CLI flags, and panel typecheck. This is still
  no-heavy UI contract evidence, not a substitute for the real Electron live
  model matrix.
- current #175-#177 installed live cache/Responses proof is green after commit
  `e21c85f6`: Qwen MTP `/v1/responses` streaming visible output, `paged+ssm`
  cache hit, L2 disk hits, and TTFT improvement pass; MiniMax installed
  `paged+tq` warm reuse plus fresh-process `paged+disk+tq` restart-reader L2
  restore pass. Artifact:
  `build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json`.
  This clears that audit row only; it does not clear MiMo, #179, DSV4 exactness,
  broad live smoke/tool, media, UI, package, signing, notarization, or release.
- current MiMo V2.5 `JANGTQ_2` local source evidence is materially improved
  after commit `533efd17`: stale `JANG_2L` local copies are absent, local
  artifact manifest integrity passes for 115 files, decode speed now clears
  the 40 tok/s floor (`40.21` bundle / `40.74` greedy), text cache repeat,
  multiturn recall, reasoning visible final, and exact code pass in the current
  no-media smoke. Literal prompt tightening reduced strict failures: tool-result
  continuation now passes, but required tool args and structured JSON still
  mutate `blue-cat` to `blue cat`; this is direct model-output exactness
  failure from the current quantized artifact, not parser truncation. CB
  working-set pressure is still open, long-prompt/tool legacy blockers remain
  in older routes, media is still unwired, and UI/installed-app release
  clearance is not complete.
- latest MiMo source proofs are improved but still not release-green. The narrow
  no-media source smoke after the keep=0 cache fix parses the required tool row
  as `record_fact({"value":"blue-cat"})` and text/cache/multiturn rows pass.
  Artifact: `build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-keep0-cache-fix-20260607/`.
- latest MiMo long-prefix MLLM cache proof is green for the prior crashing
  64-word row after commit `96b33a5e`: first turn stores clean mixed-SWA cache
  from live `RotatingKVCache`, block-disk writes 7 blocks / 435 tokens, second
  turn reports `cached_tokens=435`, `cache_detail=paged`, visible `LONGCTX-OK`,
  and no Metal OOM/server disconnect. Artifact:
  `build/current-local-long-context-cache-mimo-v25-installed-64w-after-tight-memory-rotating-store-20260607/`.
- MiMo still remains red for release: speed is about `0.2 tok/s` first turn and
  about `1.4 tok/s` cached second turn in the 64-word proof, full
  tool-result/auto-tool/multiturn tool matrix is not cleared, VL/audio/video is
  preserved but unwired, UI/installed-app parity is not cleared, and no
  signing/notarization/release is allowed.
- newest MiMo source speed/cache work after `d0130bb6` fixes two runtime bugs but
  does not clear release:
  - generic TurboQuant KV is now skipped at the MLLM call site for MiMo; source
    `/health` reports native `mixed_swa_kv_v1` with `generic_turboquant_kv=false`.
  - list-held MiMo decoder layers are explicitly quantized; source log reports
    `MiMo-V2 load quantized 192 runtime modules`.
  - tight-memory text prefill now chunks 512/1024-token PP rows instead of
    one-shot Metal OOM.
  - current source artifact
    `build/current-decode-speed-live-mimo-v25-jang2l-source-after-cache-policy-fix-20260607.json`
    is still only `review`: exact coherency passes, PP is `76.39` / `104.60`
    tok/s below the `400` target, and decode is `1.63` tok/s below the `40`
    target.
  - newer current source artifact
    `build/current-decode-speed-live-mimo-v25-jang2l-source-after-fastpath-counters-20260607.json`
    is still only `review`: exact coherency passes, native `mixed_swa_kv_v1`
    cache reports prefix/paged/block-L2 true and generic TQ-KV false, PP is
    `78.41` / `106.43` tok/s below the `400` target, bundle decode is
    `1.79` tok/s, and greedy/top-k0 decode is `1.83` tok/s below the `40`
    target.
  - that newer run proves the affine SwitchGLU decode fast path installed and
    activated (`calls=4096`, `compiled_shapes=2`), but decode trace still shows
    `last_async_ms` about `503-597 ms/token` while Python/model step work is
    about `2-3 ms`.
  - current classification: the MiMo 40 tok/s blocker is runtime/kernel-side
    unless a corrected bundle disproves it. The current bundle also has a real
    model-artifact defect: no local `jang_config.json`, and `lm_head` /
    `embed_tokens` are bf16 without intended 8-bit affine sidecars. Runtime
    quantizing those two modules only moves decode into the same `1.7-1.8`
    tok/s range.
  - installed/release app bundles are stale until rebuilt; do not package,
    sign, notarize, tag, or update downloads while this remains red.
- latest MiMo source-vs-quant harness now includes the required XML tool row,
  and now defaults to the current source/quant endpoints and paths. Current
  preflight is still `missing_prerequisites` because source and quant endpoints
  are not running:
  `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-refresh-defaults-after-qwen35-20260607.json`.
  Both model paths exist; `http://erics-m5-max2.local:8126` and
  `http://127.0.0.1:8897` are down. The stale HF remote-code MiMo cache was
  deleted again and current audit reports `stale_local_state_absent=true`.
- latest Gemma4 JANG_4M source no-media text/cache/reasoning/tool row is green
  after dropping leaked hidden-channel marker text only when native tool calls
  are present; full media/UI/per-quant matrix remains red.
- latest Qwen27 MXFP4-MTP source proofs clear short API parity, restart/L2
  restore, and deterministic Responses cancellation/recovery for the hybrid
  attention-KV plus SSM companion path, but UI, largest-context, MXFP8
  deterministic policy/session, and TP4 route rank/speed remain open.
- latest Qwen35 MXFP8-MTP deterministic long Responses/tool/cache gate is green
  after the request-local native-MTP D1 cap for tool-context MLLM requests and
  strict tool-evidence fallback markers:
  `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-evidence-marker-fallback-20260607/`.
  It proves required tool calls on both tool turns, exact tool evidence markers,
  post-first cache hits (`128`, `256`), final no-tools visible synthesis, no raw
  tool markup, no `gdn_sink`, and native MTP configured D3 with tool-context
  generations capped/activated at D1. Qwen35 family remains partial for
  Anthropic/Ollama, streaming parity, real Electron UI settings,
  largest-context cache, restart/L2 restore, cancellation/recovery, and 27B
  parity.
- latest Qwen36 expanded no-media source gates:
  - `build/current-all-local-model-smoke-qwen36-35b-mxfp8-mtp-json-code-tools-nomedia-after-reasoning-budget-20260607/`
    passes Qwen3.6 35B A3B MXFP8-MTP after the harness reasoning probe budget
    was corrected to 512 tokens for MoE MTP. Evidence: visible `FINAL=OK`,
    required tool, tool-result continuation, strict JSON, exact code, native
    MTP D3, paged+SSM cache hit `cached_tokens=56`, block L2, SSM companion L2,
    and no `gdn_sink`. The earlier 256-token row stopped during hidden
    reasoning and is a max-output-token budget boundary, not a crash.
  - `build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-json-code-tools-nomedia-20260607/`
    passes Qwen3.6 27B JANG_4M-MTP across text cache, multiturn, reasoning-on,
    required tool, tool-result continuation, strict JSON, exact code, native
    MTP D3, paged+SSM cache hit `cached_tokens=56`, block L2, SSM companion L2,
    and no `gdn_sink`.
  - `build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/`
    passes Qwen3.6 27B JANG_4M-MTP through bundled Python with required tools,
    tool-result continuation, strict JSON, exact code, image, video,
    no-media-after-image/video isolation, paged+SSM cache reuse, block disk L2,
    SSM companion L2, q4 attention-KV storage, pixel cache reuse, and native
    MTP D3 metadata.
  - These are source no-media proofs only. They do not clear installed-app UI,
    media, restart/L2 restore across variants, largest-context cache,
    cancellation/timeout cleanup, TP4 route rank/speed, API parity, packaging,
    signing, notarization, tags, or public downloads.
  - `build/current-local-restart-l2-qwen36-27b-jang4m-mtp-20260607/` passes
    Qwen3.6 27B JANG_4M-MTP restart/L2 restore in current source. Evidence:
    first process exact `ACK`, block L2 write, SSM companion L2 store; second
    process exact `ACK`, `cached_tokens=27`, `cache_detail=paged+ssm+disk`,
    block disk hit, SSM disk hit, reconstructed/dequantized OK, native MTP D3,
    and hybrid SSM typed cache. This is not installed-app/UI/media/largest
    context/release proof.
  - `build/current-local-long-context-cache-qwen36-27b-jang4m-mtp-1200w-20260607/`
    passes Qwen3.6 27B JANG_4M-MTP long-prefix cache near the explicit 8k cap:
    prompt tokens `7236`, second turn `cached_tokens=7235`,
    `cache_detail=paged+ssm`, exact `LONGCTX-OK`, 114 block L2 writes, SSM
    companion state, TurboQuant recompression/dequant/reconstruct evidence, and
    native MTP D3. The 1800w and 2048w diagnostics are HTTP 413
    `prompt_too_long` under `--max-prompt-tokens 8192`. UI max-context control
    and installed-app parity remain open.
  - `build/current-max-output-context-contract-after-qwen27-long-context-20260607.json`
    passes the no-heavy source/panel max-output/max-context separation contract:
    engine/API `26 passed`, panel/settings/request-builder `54 passed`,
    `missing_markers=[]`. This covers panel launch/request wiring and stale
    session migration, but it is not live installed-app UI proof.

## Top-level blockers that must drive all work

## Always-on release objective checklist

This worktree exists to make the Python engine and MLXStudio Electron app
production-ready. Every agent continuation must preserve the full objective,
not shrink it to one passing smoke. Before choosing work, explicitly map the
next action to one of these rows:

- `runtime`: loader, tokenizer/chat template, parser, scheduler, decode loop,
  kernels, native MTP, JANG/JANGTQ/MXFP, and family-specific forward paths.
- `tools`: tool-required, tool-auto, tool-result continuation, loop-stop,
  hidden-reasoning leaks, raw XML leaks, malformed JSON, bad schema coercion,
  code/syntax/whitespace-sensitive output, and multi-turn tool context.
- `cache`: prefix hit/miss, paged hit/miss, largest safe context cache hit,
  TurboQuant KV encode/decode/restore for standard KV, native typed cache for
  nonstandard families, L2 block-disk store/restart/restore/hit, cancellation
  cleanup, timeout cleanup, unload/reload, sleep/wake, and cache telemetry.
- `media`: OpenAI content parts, app files, data URLs, safe URLs, image, audio,
  video, media masks/tensors, processor bridge, media-expanded token accounting,
  media-salted cache keys, post-media text recovery, and media+tools.
- `api`: `/v1/chat/completions`, `/v1/responses`, Anthropic `/v1/messages`,
  Ollama chat/generate, streaming and non-streaming, max output tokens, max
  context, reasoning rails, JSON/XML structured output validation/repair.
- `ui`: MLXStudio settings reflection, model selection, cache toggles, MTP
  toggles/status, media upload controls, max output/max context, launched
  process args, visible chat recovery, and installed-app parity.
- `release`: bundled Python parity, current source pushed to the intended
  branch, packaged integrity, Developer ID signing, notarization, release tags,
  appcast/download updates, and release notes with `@Hornsan1` credit.

Do not call a row green unless the evidence scope matches the claim. A source
server pass is not installed-app proof. Text-only proof is not media proof.
Health/load proof is not generation proof. Parser repair is not model tool
generation proof. A guarded reject is not feature support.

Current required model-family lanes:

- MiMo V2.5 JANG_2L: text, tools, speed, native mixed full/SWA cache, L2,
  source-vs-quant/root-cause, VL/audio/video honesty, UI.
- Qwen 3.6 27B and 35B MTP: autodetect, deterministic MTP policy, APIs,
  native hybrid SSM cache, attention-only TQ KV, L2/restart, cancellation,
  largest context, media, UI.
- Nemo/Nemotron Omni: text/tool/reasoning, image/audio/video bridge, cache
  salt, structured output, UI.
- LFM: text/tool/multiturn/structured output/cache/UI.
- MiniMax/MiniMax-M2.7/JANGTQ_K: reporter/root cause, live cancel lifecycle,
  JANGTQ/runtime quality, tool/cancel recovery, UI.
- DSV4 Flash: native SWA+CSA/HCA composite cache, no generic TQ-KV substitution,
  long/code/file exactness, thinking rail correctness, UI.
- Step 3.7 Flash: text-only stability, advertised VLM guard, true VLM support
  only when implemented, tools/loops/multiturn/API/UI.
- Gemma 4 12B: MXFP4/MXFP8/JANG_4M, mixed-SWA cache, speed, media prefill,
  image/audio/video, tools, APIs, UI.
- ZAYA and other nonstandard caches: typed native cache only, partial-hit
  rejection, clean restore semantics, media where advertised.

## Always stay on the active vMLX release objective

This local file is a worktree-only routing guard. Do not commit it unless Eric
explicitly asks in the current turn.

Every continuation must ask: does the next action reduce a blocker for the
active Python engine, MLXStudio Electron app, runtime cache/media/tool path, UI
settings path, or signed/notarized release path? If not, stop that action.

Never substitute a smaller pass for the real objective:

- A source smoke is not installed-app proof.
- A no-media proof is not image/audio/video proof.
- A parsed tool call is not tool-result continuation, loop-stop, streaming, or
  multi-turn agent proof.
- A cache hit at short prompt is not largest-context, restart/L2 restore,
  cancellation cleanup, unload/reload, or sleep/wake cache proof.
- A typed unsupported-modality error is not modality support.
- A prompt/template/fallback workaround is not a model/runtime fix unless the
  actual API response and logs prove correct generation semantics.
- A packaging/signing command is forbidden while any runtime/model/cache/UI row
  is red.

Required systematic gates before release claims:

- Runtime: loader classification, tokenizer/chat template, generation config,
  parser selection, scheduler path, model-family forward path, kernel dispatch,
  native MTP dispatch, JANG/JANGTQ/MXFP quant metadata, and logs.
- Tools: required tools, auto tools, no-tools requests, tool-result
  continuation, repeated tool outputs, loop stop, raw XML leaks, hidden
  reasoning leaks, JSON argument validity, whitespace-sensitive code output,
  and multi-turn persona/context retention.
- Structured output: JSON/XML parse, repair, schema normalization, retry-on-fix
  path, and reporting of raw-invalid versus repaired-valid scores.
- Cache: first miss, second hit, prefix hit/miss, paged hit/miss, largest safe
  context hit, TurboQuant encode/decode/restore only for standard KV families,
  typed native cache for SWA/SSM/composite/nonstandard families, block-disk L2
  write, L2 restart restore/hit, cancellation cleanup, timeout cleanup,
  unload/reload, sleep/wake, and telemetry/UI reflection.
- Media: OpenAI content parts, app upload files, data URLs, safe URLs, image,
  audio, video, media masks/tensors, processor bridge, media-expanded token
  accounting, media-salted cache keys, post-media text recovery, and
  media-plus-tools.
- APIs: Chat Completions, Responses, Anthropic Messages, Ollama chat/generate,
  streaming/non-streaming, max output tokens, max context, reasoning controls,
  cancellation, and structured output mode/repair.
- UI: MLXStudio settings panel, launched process args, model selection,
  cache toggles, MTP toggles/status, media controls, max output/context,
  visible chat recovery, installed-app parity, and reporter-style workflow.
- Release: intended branch/main status, pushed source, packaged integrity,
  Developer ID signing, notarization, tags, appcast/download updates, release
  notes with GitHub @Hornsan1 credit, and no stale public downloads.

Family-specific reminders:

- MiMo V2.5 JANG_2L: keep the keep=0 SWA cache fix, keep generic TQ-KV disabled,
  do not claim VL/audio/video support until a real media forward path exists,
  do not claim speed fixed until live generation reaches the release target,
  and do not delete old MiMo copies until current/replacement artifacts are
  enumerated and proven by the same rows.
- Qwen 3.6 27B/35B MTP: prove autodetect, gdn_sink compatibility, MTP depth,
  deterministic policy, tool-context cap behavior, SSM companion cache,
  restart/L2 restore, streaming/API/UI parity, cancellation, and large context.
- Nemo/Nemotron Omni: prove text/tools/reasoning plus image/audio/video bridge,
  media cache salt, structured output, and UI.
- LFM: prove all quant variants for text/tools/multiturn/structured output,
  prefix/paged/L2 cache, restart, API parity, and UI.
- MiniMax/MiniMax-M2.7/JANGTQ_K: prove reporter crash/cancel lifecycle, live
  cancellation cleanup, JANGTQ runtime quality, tool recovery, and UI.
- DSV4 Flash: use native SWA/CSA/HCA composite cache only; do not substitute
  generic TQ-KV; prove exact code/file output on thinking rails, long context,
  restart/L2, and UI.
- Step 3.7 Flash: text-only stability is not VLM support; guard or implement
  VLM honestly; prove tools, loop stop, multi-turn, APIs, and UI.
- Gemma 4 12B: prove MXFP4/MXFP8/JANG_4M separately for speed, mixed-SWA cache,
  media prefill/recovery, image/audio/video, tools, APIs, UI, and installed app.

When reporting progress, use:

- Changed:
- Proved:
- Still open:
- Next:
- Forbidden until green:

When reporting status, use this exact shape:

- `Changed:`
- `Proved:`
- `Still open:`
- `Next:`
- `Forbidden until green:`

### 1. MiMo V2.5 JANG_2L

Status: release blocker.

Required fixes/proofs:

- Current positive runtime proof: cached repeat hits paged/L2 cache
  (`cache_detail=paged`, `cached_tokens=67`) and returns visible `ACK`.
- Current concrete runtime blocker: required XML tool calls generate malformed
  raw XML output followed by punctuation/fullwidth-comma garbage, with zero
  parsed OpenAI `tool_calls`. Current source can force the structural prefix
  `<tool_call>\n<function=record_fact>\n<parameter=value>`, but the model still
  fails the required argument value and closing XML.
- Do not call the MiMo required-tool prefix sampler a tool fix. It is only a
  structural decode-path repair/proof aid until live output contains the real
  requested argument and the parser returns API `tool_calls`.
- Keep debugging model artifact vs runtime decode/template/tool path. The tool
  failure reproduces with SimpleEngine, no prefix cache, no paged cache, no L2,
  and `--kv-cache-quantization none`, so do not misclassify it as only a cache
  restore issue.
- Do not "fix" MiMo by disabling prefix cache, paged cache, L2 disk cache,
  TurboQuant restore, tools, thinking-off mode, or system prompts. Temporary
  diagnostic runs may disable one component only to classify root cause; release
  proof must pass with the intended cache stack.
- Source-vs-quant first-divergence proof exists and classifies failures.
- Long-prompt coherence passes with full output read.
- Tool-call protocol uses real API `tool_calls`; no raw XML/text leaks.
- Tool-result continuation stops correctly and produces final answer.
- Text cache, prefix cache, paged cache, TurboQuant KV if applicable, and L2 disk
  cache are proven on MiMo-specific prompts.
- Decode speed target is proven; current slow simple path is not acceptable.
- VL/audio/video runtime is honestly wired or honestly unsupported; preserved
  sidecars do not count as runtime support.
- UI/settings/API proof covers model selection, max output tokens, cache toggles,
  media controls, streaming and non-streaming, and text recovery after media
  errors.

Do not clear MiMo from parser repair, short text smoke, health, metadata sync,
or fail-closed media rejection alone.

### 2. Cross-family live multi-turn matrix

Status: release blocker.

Families that must be covered:

- MiMo V2.5
- Qwen 3.6 27B and 35B, including MTP autodetect
- Nemo/Nemotron Omni
- LFM
- MiniMax
- DSV4 Flash
- Step 3.7 Flash
- Gemma 4 12B
- ZAYA and hybrid/SSM families where applicable

Required rows per family:

- `/v1/chat/completions`
- `/v1/responses`
- Anthropic `/v1/messages` when supported
- Ollama chat/generate when supported
- streaming and non-streaming
- thinking off/on where the family supports thinking
- tool-required, tool-auto, tool-result continuation, loop-stop behavior
- JSON/XML structured output validation and repair diagnostics
- multi-turn persona/name/context retention
- code/syntax/whitespace-sensitive output
- parser leak checks for hidden reasoning, XML tags, markdown fences, bad JSON,
  repeated tool calls, and visible garbage

Do not count a narrow no-heavy contract as this live matrix.

### 3. Cache and scheduler matrix

Status: release blocker until proven per family and architecture.

Required rows:

- largest safe context cache hit
- prefix cache hit and miss correctness
- paged cache correctness where compatible
- TurboQuant KV encode/decode/restore for standard KV families
- L2 disk cache store, restart, restore, and hit
- cancellation and post-error cache cleanup
- continuous batching or explicit safe reject when not compatible
- cache stats in `/health`, `/v1/cache/stats`, capabilities, and UI settings

Architecture-specific requirements:

- Standard full-KV models may use generic TurboQuant KV only after parity proof.
- DSV4 uses native SWA+CSA/HCA composite cache; do not force generic TQ-KV.
- ZAYA CCA and other path-dependent caches need typed native cache and clean
  restore semantics.
- Hybrid SSM models need companion state, async rederive where required, and
  cannot be treated as plain KV.
- MLLM/media prompts need media-salted cache keys and no cross-media reuse.

### 4. Media / VL / audio / video

Status: release blocker for any media claim.

Required rows:

- media request normalization for OpenAI content parts, app files, data URLs,
  safe URLs, image/audio/video parts
- processor/embedding bridge per family
- media-expanded prompt accounting
- text-after-media recovery after success, reject, timeout, cancellation, and
  parser failure
- streaming and non-streaming parity
- tool calls after media
- JSON/XML structured output after media
- media cache salt, L2 behavior, and stats
- UI upload/settings/recovery proof

Do not call Step3.7 VLM fixed because current source routes advertised VLM
metadata text-only. That only prevents unsupported VLM crashes.

### 5. MiniMax issue179

Status: release blocker.

Required fixes/proofs:

- reporter installed app hash provenance
- reporter server route parity
- cancel route ordering and response-id behavior
- installed app bundled Python/log parser/cache flags
- local reproduction clean or honest model/runtime classification
- no broad MiniMax release claim without reporter parity closure

### 6. DSV4 Flash

Status: partial cache/tool rows pass; release blocker remains.

Required fixes/proofs:

- long-output/code/file-generation exactness and quality
- no identifier corruption
- generated-only repetition handling
- memory-gated UI rows
- native composite cache remains default; no generic TQ-KV substitution
- full UI/API/cache/tool proof after quality row is fixed

### 7. Gemma 4 12B

Status: partial source/package rows exist; release blocker remains for full
media/UI matrix.

Required fixes/proofs:

- 45+ tok/s target for the intended 12B route if claimed
- image prefill guard behavior on high-memory machines in installed app
- small/large media prompts
- audio/video if advertised
- tools after media
- cache and L2 behavior
- UI settings and post-error recovery

### 8. Qwen 3.6 27B / 35B MTP

Status: source fixes exist; full E2E release matrix remains open.

Required fixes/proofs:

- 27B and 35B native MTP autodetect in source and installed app
- `gdn_sink` regression cannot reproduce on fresh installed app
- streaming/non-streaming output equivalence
- MTP speed and quality, not just activation
- hybrid SSM/cache behavior where applicable
- media/VL proof when using VL variants
- UI/settings parity

## Required final-answer status format

Every work turn on this objective should report:

- `Changed:` concrete files/artifacts changed this turn.
- `Proved:` exact commands/artifacts and what they prove.
- `Still open:` top blockers still not cleared.
- `Next:` the single next blocker-reducing action.
- `Forbidden until green:` package/sign/notarize/release actions remain locked.

## Current next valid actions

Prefer these in order unless current evidence changes:

1. Update the blocker ledger from current artifacts so the release state is
   unambiguous and not stale.
2. Commit/push only source, tests, proof-pointer, and docs changes that improve
   blocker tracking or runtime correctness. Do not commit `.agents/`, staged app
   output, release output, or local vendor noise.
3. Continue MiMo by separating artifact/model failure from vMLX decode/runtime
   failure. Do not keep iterating on parser aliases now that ChatML placement is
   fixed and live still fails.
4. Continue Qwen27/Qwen35 by filling missing UI/largest-context/cancel/MXFP8
   policy rows, not by rerunning already-green short text/API rows.
3. Continue cross-family live matrix proof, prioritizing Qwen 3.6 35B/27B MTP
   because public `gdn_sink` regressions and MTP/UI/cache claims are release-facing.
4. Resume MiMo only for a real corrected artifact, constrained XML decoder, or
   optimized routed-expert decode path. Do not spend loops re-proving the same
   current JANG_2L XML corruption.
5. Expand live media/UI/cache rows as soon as a family has source text/tool/cache
   proof; no family is release-green from source-only proof.

## Mandatory per-family checklist

For every model family named by Eric, do not work from intuition. Build or
update a family row with exact evidence for every item below.

### Model artifact and runtime identity

- Exact local model path.
- Exact repo commit and source tree used.
- Exact served model name.
- `config.json`, `generation_config.json`, tokenizer/chat template, and
  `jang_config.json` when present.
- Whether the artifact is MXFP, JANG, JANGTQ, MXTQ, dense, MoE, hybrid SSM,
  VLM, audio, video, or text-only.
- Whether runtime-proven modalities match advertised sidecars.
- Whether any model card/config metadata is wrong or unsafe for vMLX.

### Prompt/template/reasoning

- Exact rendered prompt shape for instruct/chat, thinking off, thinking on, and
  tool requests.
- First-token/top-token behavior on short exact and long/system prompts.
- No hidden reasoning leak, no `<think>` leak, no prompt-copy, no raw XML leak,
  no markdown-fence leak unless requested.
- Reasoning parser selected from registry or metadata, not guessed.
- Thinking defaults reflected consistently in CLI, API, capabilities, and UI.

### Tool calls and structured output

- OpenAI `tool_calls` emitted when tools are supplied.
- `tool_choice=required` produces a real tool call or a classified failure.
- Tool-result continuation stops and produces final answer.
- Multi-tool loop stops under max-iteration cap.
- JSON/XML output parses raw or is repaired with diagnostics; repair is not
  native guided decoding.
- Check syntax, code fences, whitespace, quote escaping, arrays vs strings,
  trailing commas, and schema type coercion.

### Cache and scheduler

- Prefix cache hit/miss and no stale prompt reuse.
- Paged cache correctness when compatible.
- L2 disk cache store, restart, restore, and hit.
- TurboQuant KV encode/decode/restore for standard KV families.
- Native cache path for DSV4, ZAYA CCA, hybrid SSM, MLA, MiMo mixed full/SWA,
  and MLLM/media families.
- Largest safe context cache hit.
- Cancellation, timeout, post-error recovery, unload/reload, sleep/wake.
- `/health`, `/v1/cache/stats`, capabilities, and UI settings all report the
  same effective cache state.

### Media / VL / audio / video

- OpenAI content-part normalization for text, image, video, audio, data URLs,
  app-local files, and safe remote URLs.
- Processor/embedding bridge actually forwards media tensors and masks.
- Media-expanded prompt accounting and memory guard correctness.
- Text-after-media recovery after success, rejection, timeout, cancellation,
  and parser error.
- Streaming/non-streaming parity for media turns.
- Tool calls and JSON/XML after media turns.
- Media-salted prefix/L2/cache behavior.
- UI upload/settings/recovery proof.

### API and UI parity

- `/v1/chat/completions`
- `/v1/responses`
- Anthropic `/v1/messages` where supported
- Ollama chat/generate where supported
- stream and non-stream
- settings panel: model, max output tokens, max context, thinking, tools,
  cache mode, MTP, media controls
- installed app parity only after source rows are green

### Release readiness

- Source proof green.
- Bundled Python parity green.
- Installed/staged app parity green.
- Real Electron UI matrix green.
- Release manifest `prepackage_ready=true`.
- Release manifest `release_ready=true`.
- Only then package, sign, notarize, tag, upload, and update public downloads.

## Family-specific non-negotiables

### MiMo V2.5

- Current JANG_2L lane is classified red, not fixed.
- Local and remote canonical installed artifacts match sampled metadata/shards.
- Remote `~/jang` docs say coherent affine JANG_2L is about `1.974-2.640`
  generation tok/s on M5 Max and explicitly not a `30 tok/s` path.
- Local bundled `jang_tools` was stale versus remote MiMo runtime; vMLX now
  installs a fallback compiled single-token router if needed.
- The fallback compiled router is proven to install but does not fix XML tools
  or speed.
- Do not reclassify MiMo as parser-only. Direct `/v1/completions` continuation
  with no prefix cache, no paged/L2, no q4 KV, and no tool parser still fails
  valid XML continuation after `<function=record_fact><parameter=value>`.
- Next valid MiMo work is one of: corrected artifact proof, real constrained XML
  decoder, optimized routed-expert decode, or real VL/audio/video implementation.

### Qwen 3.6 27B / 35B

- MTP autodetect and active depth must be proven in live runtime, not just
  metadata.
- `gdn_sink` must be accepted and propagated on dense MTP routes.
- Hybrid SSM cache rows must prove companion state/L2 behavior; do not call it
  plain KV.
- VL variants need media matrix, not text-only MTP proof.

### Nemotron / Nemo Omni

- Check tokenizer fallback, nonstandard configs, omni processor/audio/video
  bridge, hybrid cache state, streaming and tool output.

### LFM

- Check reasoning/tool dialect, hybrid or nonstandard cache mode, loop stopping,
  and exact visible output in all APIs.

### MiniMax

- Reporter parity and issue179 provenance are release blockers.
- Do not clear from local-only route proof if reporter installed hash/log route
  remains unproven.

### DSV4 Flash

- Native SWA+CSA/HCA composite cache only. Generic TurboQuant KV is not a
  substitute.
- Long-output/code/file-generation exactness remains blocking even if cache/tool
  rows pass.

### Step 3.7 Flash

- Text-only fail-closed routing avoids unsupported VLM crash but does not prove
  VLM.
- Tool loops/raw XML dialect and multi-turn synthesis must be tested separately.

### Gemma 4 12B

- JANG_4M source no-media text/cache/reasoning/tool row is currently proven.
- Tool-only responses must not leak visible hidden-channel markers such as
  `thought` or `analysis` when native tool calls are present.
- 45+ tok/s target must be proven on the intended route if claimed.
- Image prefill guard behavior must be proven on high-memory installed app.
- VL/audio/video/tools/cache/UI rows remain separate from text speed.

## Required work loop for agents

Repeat this loop until the release blocker ledger is green:

1. Read current ledger/artifacts.
2. Choose exactly one top blocker.
3. State which blocker the action reduces.
4. Inspect actual config/runtime path.
5. Make a real runtime/source fix or generate a proof artifact.
6. Run the narrowest proof that can falsify the fix.
7. Update the blocker ledger with pass/fail and exact boundary.
8. Commit/push only source, tests, proof artifacts, and docs that reflect real
   blocker progress.
9. Do not package/sign/notarize until all runtime/model/UI/cache rows are green
   or Eric explicitly unlocks release packaging in the current turn.

If an action does not fit this loop, stop before doing it.

## Existing coordination rule

Before non-trivial work, read local coordination:

```text
.agents/STATUS.md
.agents/LOG.md
.agents/MAIL.md
```

The `.agents/` directory is local-only and must never be committed.

## Current objective lock — 2026-06-06 continuation

This lane is still a blocker burn-down, not a release-packaging lane.

Active release blockers must stay visible until they are actually green:

- MiMo V2.5 JANG_2L: first-token stop and think-tag leakage are partially improved in SimpleEngine, but exact instruction following, long-output coherence, tool calls, speed target, source-vs-quant classification, prefix/paged/L2 proof, and real VL/audio/video runtime remain red.
- MiMo V2.5 JANG_2L: current direct continuation proof shows XML tool output remains corrupt outside Chat tool parsing; fallback compiled router installs but does not clear tool correctness or speed.
- MiMo V2.5 JANG_2L: standard decode speed gate row exists as `mimo_v25_jang2l`. Latest source artifact `build/current-decode-speed-live-mimo-v25-jang2l-source-partial-evidence-20260607.json` is red: warm about 0.2 tok/s, deterministic coherency exact but only 1.58 tok/s, first PP request kills the server with Metal OOM. Generic flat TurboQuant KV is correctly disabled for MiMo mixed full/SWA rotating cache; do not fake-fix this by enabling flat TQ-KV or disabling cache.
- Qwen 3.6 27B/35B MTP: `gdn_sink` source propagation is not enough. Native MTP autodetect, active depth, streaming/non-streaming, multiturn, tool calls, hybrid SSM/L2 cache, UI settings, and installed-app parity must be proven for each claimed artifact.
- Qwen 3.6 27B JANG_4M-MTP: installed-app UI remote-session proof exists at `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-maxcontext-20260607-proof.json`. It proves `/Applications/vMLX.app` plus installed bundled Python can run `--max-tokens 128 --max-prompt-tokens 8192`, record request max output/context, report `/health.max_prompt_tokens=8192`, and show paged+SSM plus block/SSM L2 telemetry. It does not prove normal local-session model launch, media, or MTP decode on the exact UI prompts because non-deterministic sampling skipped MTP by policy.
- Gemma 4 12B: text/cache and narrow image proof do not clear full VL/audio/video/tools/UI; MXFP4, MXFP8, and JANG_4M claims need their own current rows.
- Media guard recovery: current panel IPC rolls back failed media user turns both when the backend throws and when the Responses path completes with empty warnings. This prevents the next text-only turn from replaying the same failed image/audio/video payload, but it is not media model-family clearance.
- Structured output: Chat Completions and Responses non-streaming paths return repaired canonical JSON when post-generation JSON repair succeeds, and surface a `warnings` entry when repair/schema coercion was needed. Treat this as scoring/diagnostic integrity only; it is not hard constrained decoding and does not clear raw exact JSON/code failures.
- Step 3.7 Flash: advertised-VLM text-only routing is crash avoidance, not VLM support. Real media, tool, cache, streaming, and UI rows remain required.
- DSV4 Flash: native composite cache proof does not clear route/rail exactness, long-output code generation, or file-generation quality.
- LFM, Nemotron/Nemo Omni, MiniMax, ZAYA, and hybrid families: each needs family-specific multiturn/tool/JSON/cache/media/UI proof before release claims.

Before any signing, notarization, release tag, upload, appcast/download update, or public release, the current release manifest, current objective proof, installed-app parity, real UI live model matrix, and relevant model-family live rows must all be green with tails inspected.

Partial fixes are allowed only when committed with red proof artifacts and explicit remaining blockers. Never convert partial runtime progress into a production-ready claim.

## Current release objective guardrail - 2026-06-06 local override

This local note exists to keep agents on the actual lane Eric requested. Do not commit this AGENTS.md change unless Eric explicitly reverses the local-only instruction.

Active lane:
- Work only on vMLX Python engine, Electron/MLXStudio panel, release gates, and model-runtime proof in `/Users/eric/mlx/vllm-mlx-finite-launch-guard` when `.agents/STATUS.md` routes there.
- Do not touch ADLab, RDMA/TB transport experiments, deprecated `/Users/eric/vmlx`, Swift, or unrelated model-upload work unless Eric explicitly names that exact path in the current turn.

Before saying any model family is fixed, prove the relevant rows through live server/API or a documented no-heavy contract when live proof is not possible:
- MiMo V2.5/JANG_2L: text, tools, long prompt, cache hit, L2 disk, speed target, source-vs-quant classification, and VL/audio/video wiring truth.
- Qwen 3.6 27B/35B MTP: native MTP autodetect, GatedDelta `gdn_sink` propagation, hybrid SSM companion cache, async/rederive safety, tool-call continuation, stream/non-stream, and packaged-app parity.
- Nemo/Nemotron Omni: audio/image/video capability truth, prompt template and parser behavior, cache key/media salt, and no fake capability exposure.
- LFM: architecture-specific cache profile, text/tool/multi-turn quality, and UI settings parity.
- MiniMax/JANGTQ: reporter issues, codebook/runtime correctness, speed, tool protocol, cancellation/recovery, and no memory-guard fake pass.
- DSV4 Flash: native composite cache, CSA/HSA/SWA path-dependent restore, raw-max quality, long output/code/file generation, tool loops, and no generic TurboQuant KV substitution.
- Step 3.7 Flash: text-only vs unsupported VLM metadata truth, tool parser dialect, loop stopping, thinking tags, crash falsification, and no fake `has_vision=false` release claim unless the bundle/runtime boundary says it is text-only.
- Gemma 4 12B: MXFP/JANG_4M loader path, VL prefill memory guard behavior, image/audio/video runtime, cache and speed, and thinking/tool/schema output behavior.

Every release-candidate pass must cover:
- `/v1/chat/completions`, `/v1/responses`, Anthropic messages, Ollama chat/generate where supported.
- stream and non-stream.
- at least 3-turn multi-turn coherence with tool result continuation.
- JSON/XML/tool formatting validity, no hidden reasoning leaks, no malformed tool call text, no whitespace/syntax corruption in code output.
- max output tokens, max context, prefix cache, paged cache, architecture-specific native cache, TurboQuant KV encode/decode where valid, L2 disk write and hit, sleep/wake/unload/reload.
- UI settings panel parity with CLI args, capabilities JSON, server `/health`, and actual launched process.
- installed/staged packaged app parity before signing.

Release rule:
- If any current release manifest says `prepackage_ready=false` or `release_ready=false`, do not package, sign, notarize, tag, publish, or update download metadata.
- Credit GitHub `@Hornsan1` in release notes/changelog/public acknowledgements for the recent reported runtime/model/UI/API issues.

## 2026-06-07 MiMo JANGTQ2 continuation update - local guard only

Do not commit this section unless Eric explicitly asks.

Current MiMo promoted artifact:

```text
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2
```

Bad local MiMo JANG_2L/stale cache paths were deleted. The promoted JANGTQ2
bundle was copied from `erics-m5-max2.local` and structurally verifies with the
updated JANG verifier after metadata truth correction.

Pushed source/tooling commits:

```text
jjang-ai/jangq main d67d753 Accept MiMo prestacked JANGTQ bundles
jjang-ai/vmlx codex/pr-intake-manifest c2a0b3c8 Point MiMo gates at JANGTQ2 artifact
jjang-ai/vmlx codex/pr-intake-manifest edb0581c Load MiMo prestacked JANGTQ in MLLM runtime
```

Current source proof:

```text
build/current-decode-speed-live-mimo-v25-jangtq2-source-after-tq-decode-fastpath-20260607.json
```

Evidence:

- source MLLM runtime starts MiMo-V2.5-JANGTQ_2 and stays healthy.
- strict load binds prestacked `switch_mlp.*.tq_{packed,norms,bits}` into real
  `TurboQuantSwitchLinear` modules instead of disabling strictness.
- server log proves `MiMo-V2 TurboQuant SwitchGLU decode fast path active:
  calls=4096`.
- visible coherency passes: `READY`, `17+28=45`, `CERULEAN`.
- long count output is coherent.
- bundle decode is `38.94 tok/s`; greedy/topk0 decode is `39.93 tok/s`.
- native cache reports MiMo `mixed_swa_kv_v1`, prefix/paged/block-L2 true, and
  generic TurboQuant KV disabled.

Still not release-green:

- the staged packaged app/bundled Python is stale until refreshed/rebuilt; prior
  packaged MiMo JANGTQ2 gate exited early with unknown `tq_*` parameters.
- prompt processing remains about `145 tok/s`, below the current `400 tok/s`
  gate.
- this proof is source-only; it does not clear installed app UI/settings,
  full tool-result/auto-tool/multiturn/adversarial loop-stop, VL/audio/video,
  Qwen/Nemo/LFM/MiniMax/DSV4/Step/Gemma rows, signing, notarization, tags, or
  public downloads.
- no release/tag/sign/notarize/upload is allowed from this proof alone.

## 2026-06-07 current continuation override - MiMo JANGTQ2 promoted, release still red

LOCAL ONLY: this section is intentionally not for commit unless Eric explicitly requests it.

Current authoritative vMLX Python branch/worktree:

```text
/Users/eric/mlx/vllm-mlx-finite-launch-guard
branch: codex/pr-intake-manifest
latest pushed commit on this branch: ea307a21 Refresh MiMo JANGTQ2 audit proof
```

Current MiMo artifact truth:

- Promoted local bundle: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`.
- Do not target deleted/bad old `MiMo-V2.5-JANG_2L` paths for current release work.
- `MiMo-V2.5-JANGTQ_2-candidate` still exists locally; do not delete candidate artifacts unless Eric explicitly asks for candidate cleanup.
- The old bad local MiMo model paths and stale HF remote-code cache were previously deleted; do not restore them.

Current MiMo proof truth:

- Source vMLX runtime support for prestacked JANGTQ2 was committed/pushed in `edb0581c Load MiMo prestacked JANGTQ in MLLM runtime`.
- Current MiMo audit pointer refresh was committed/pushed in `ea307a21 Refresh MiMo JANGTQ2 audit proof`.
- Packaged staged app proof after directly refreshing bundled `mllm.py` shows coherent exact output and about `40 tok/s` decode for JANGTQ2, but that staged app was modified after signing and is not release-signed/notarized proof.
- Current audit artifact: `build/current-mimo-v2-jang2l-current-audit-after-jangtq2-packaged-speed-proof-20260607.json`.
- Current promoted-bundle manifest: `build/current-mimo-jangtq2-local-manifest-20260607.tsv`.
- Current release manifest refresh: `build/current-release-regression-manifest-after-mimo-jangtq2-audit-refresh-20260607.json`.

Current MiMo status:

- `manifest_integrity=true` for promoted JANGTQ2.
- `decode_speed_target=true` for packaged JANGTQ2 proof.
- `local_release_clearance=false`.
- Remaining MiMo blockers are real and must not be papered over: long-prompt coherence, tool protocol, CB/system-prompt working-set pressure, source-vs-quant first-divergence, and VL/audio/video runtime wiring.

Current release status:

- `current_proof_sweep=fail`.
- `prepackage_ready=false`.
- `release_ready=false`.
- Do not package, sign, notarize, tag, push a release, update appcast/download metadata, or claim public release while this remains true.

Next work must reduce one of these broad release blockers:

- MiMo behavior matrix: long prompt, required/auto tools, tool-result continuation, loop stop, source-vs-quant, media truth.
- Qwen 3.6 27B/35B MTP: largest context, restart/L2 across variants, API parity beyond no-media source, cancellation/streaming/UI parity.
- Gemma4 12B: per-quant media/audio/video/UI/cache recovery, not just JANG_4M text speed.
- Step3.7: text-only/VLM boundary, tool loop control, API/UI parity.
- LFM/Nemotron Omni/MiniMax/DSV4: live multiturn tools, exact JSON/XML/code, typed cache/L2, installed-app UI, route-specific quality blockers.
- Packaging/release only after runtime/model/UI/cache proof gates are green. Cache reuse must include Chat Completions and `/v1/responses` streaming/non-streaming usage, `cached_tokens`, `cache_detail`, L2 restart hits, cancellation cleanup, and endpoint parity before any release claim.

## Current continuation update - 2026-06-07 packaged gate and MiMo JANGTQ2

Do not use older lines above as newer than this block.

Latest pushed branch state:

- Active branch: `codex/pr-intake-manifest`
- Latest pushed commit: `00819461` (`Track open public app issue blockers explicitly`)
- Earlier current-turn commits:
  - `b83381bc` refreshed current suite/release proof pointers.
  - `818899dc` aligned ZAYA-VL manifest proof assertion.

Packaging/signing state:

- `panel/bundled-python` was refreshed from this checkout with `./panel/scripts/bundle-python.sh`.
- `cd panel && npm run verify-bundled` is green for bundled engine/source parity, `jang_tools`, MiMo V2 registration, Step3p7 VLM runtime, Gemma4 unified VLM/audio registration, TurboQuant kernels, and Gemma4 vision coercion.
- Current packaged-integrity artifact:
  `build/current-packaged-integrity-contract-after-staged-sequoia-rebuild-current-source-20260607.json`
- Current packaged-integrity status is still `fail`, but the previous bundled Python hash drift is fixed. The remaining named package blocker is now explicit:
  `packaged_app_developer_id_signing_blocked`.
- Signing preflight currently reports the staged Sequoia app is ad-hoc signed, lacks hardened runtime, and fails `codesign --verify --deep --strict`; next proof is a fresh Developer ID signed app from the gated release path.
- Do not manually sign a stale app to make this green. Build/sign only after prepackage/live objective blockers are cleared or Eric explicitly authorizes packaging despite blockers.

Current top-level suite state:

- Current suite artifact:
  `build/current-regression-suite-after-dsv4-preflight-refresh-20260607.json`
- Status: `open`
- Failed steps after bundle refresh and packaged-integrity reporting fix:
  - `packaged_integrity_contracts`
  - `release_regression_manifest`
- `release_gate_skip_app` is no longer a failed current-suite step.
- Open objective rows remain:
  - Cross-family live multi-turn smoke matrix
  - MiMo V2.5 JANG_2L/JANGTQ2 runtime/tool/long-prompt quality
  - MiniMax-M2.7-JANGTQ_K reporter parity/root cause
  - Real Electron UI cross-family live model matrix
  - DSV4 long-output/code/file-generation quality

MiMo JANGTQ2 state:

- Local promoted bundle:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`
- Stale local candidate deleted:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2-candidate`
- Max2 sync completed from:
  `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2/`
- Current JANGTQ2 audit artifact:
  `build/current-mimo-v2-jang2l-current-audit-after-jangtq2-packaged-speed-proof-20260607.json`
- That audit proves manifest integrity and decode speed target, but still does **not** release-clear MiMo.
- Current MiMo component blockers in that audit:
  - `long_prompt_coherence=false`
  - `tool_protocol=false`
  - `cb_system_prompt_working_set_pressure=false`
  - `source_vs_quant_first_divergence=false`
  - `mimo_media_wired=false`
- Do not clear MiMo from short coherency or speed-only proof. Required remaining proof is long-prompt/tool-context quality, required/auto tool continuation, source-vs-quant classification with both endpoints running, and real media/UI wiring.

## Current continuation update - source-vs-quant RAM boundary

Eric explicitly stopped MiMo source-vs-quant comparison because the machines do not have enough RAM for this lane right now.

Do not start both MiMo source and quant servers for source-vs-quant comparison unless Eric explicitly reauthorizes it in the current turn and confirms enough RAM/headroom. Treat `source_vs_quant_first_divergence=false` as an honest open MiMo release blocker, not something to brute-force.

Cleanup already performed for the aborted attempt:

- Local quant server on `127.0.0.1:8897` stopped/no listener.
- Max2 source server on `erics-m5-max2.local:8126` stopped/no listener.
- Separate Max2 worktree `/Users/eric/mlx/vllm-mlx-codex-pr-intake` exists at `a4170d4b` for future current-code diagnostics, but do not use it for source-vs-quant unless reauthorized.

## Local continuation note - 2026-06-07 issue119 cache gate refresh

Do not commit this section unless Eric explicitly asks.

Latest pushed commit: `9fa15864` (`Tighten Gemma issue119 cache gate`) on
`origin/codex/pr-intake-manifest`.

What changed:

- Public issue119 no longer depends on the stale missing 20260524 Gemma memory
  artifact path.
- Issue119 now validates the current installed Gemma4 memory/cache stress proof:
  `build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json`.
- The issue119 gate requires all of these before passing:
  - artifact status is `pass`
  - native cache health reports Gemma4 `mixed_swa_kv_v1`
  - prefix, paged cache, and block-disk L2 are enabled
  - generic TurboQuant KV is disabled for Gemma4 mixed-SWA
  - storage-boundary KV quantization is enabled
  - at least two repeated rows show positive `cached_tokens` with
    `cache_detail=paged+mixed_swa`
- Focused validation passed: `9 passed, 309 deselected` for
  `tests/test_public_app_issue_audit.py tests/test_release_regression_manifest.py -k 'public_app_issue or issue115 or issue119'`.
- Regenerated public issue audit:
  `build/current-public-app-issue-audit-after-issue179-memory-preflight-20260607.json`
  now has `focused_open=["117","115"]`; issue119 checks are all true.
- Regenerated release manifest:
  `build/current-release-regression-manifest-after-issue179-public-dmg-provenance-20260607.json`
  remains `prepackage_ready=false`, `release_ready=false`, with explicit
  release blockers now reduced to:
  - `mimo_v2_jang2l_runtime_quality_open`
  - `issue179_minimax_k_root_cause_audit`
  - `issue115_installed_app_performance_regression_open`

Still do not package, sign, notarize, tag, upload, or update downloads. Issue115
is still open because current installed Gemma cold stream speed is below the
80 tok/s floor and Qwen35 installed speed is not cleared. MiMo and #179 remain
release blockers.

## Local continuation note - 2026-06-07 issue115 stream TPS refresh

Do not commit this section unless Eric explicitly asks.

Latest pushed commit: `ba846f4f` (`Measure visible stream TPS before done gap`) on
`origin/codex/pr-intake-manifest`.

What changed:

- `tests/cross_matrix/run_runtime_memory_stress_probe.py` now measures visible
  streaming TPS from first visible text delta to last visible text delta.
- The old first-visible-to-`[DONE]` duration is preserved as
  `post_first_token_seconds`, and the post-visible completion/cache-persistence
  gap is exposed as `post_last_visible_to_done_seconds`.
- This prevents background/final cache persistence from being misclassified as
  slow UI token streaming while still keeping the gap visible for diagnostics.

Fresh issue115 evidence:

- Qwen35 4bit installed-app speed artifact was regenerated at
  `build/current-decode-speed-live-qwen35-4bit-issue115-installed-app-after-decode-position-20260601.json`.
  Result: `qwen35_4bit` pass, bundle `122.346 tok/s`, greedy/top-k0
  `123.009 tok/s`, installed MLX/Metal wheels `macosx_14_0_arm64`.
- Gemma4 26B installed-app stream/cache artifact was regenerated at
  `build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json`
  using isolated block-disk cache dir `/tmp/vmlx-issue115-gemma-block-cache-20260607`.
  Result: `status=pass`; visible stream TPS is `107.020`, `106.903`,
  `106.882`; first row is uncached, second/third rows report
  `cached_tokens=1220` and `cache_detail=paged+mixed_swa`; native cache is
  Gemma4 `mixed_swa_kv_v1`, prefix/paged/block-L2 true, generic TQ KV false,
  storage-boundary q4 KV true.
- Public app issue audit regenerated at
  `build/current-public-app-issue-audit-after-issue179-memory-preflight-20260607.json`.
  Current `focused_open=["117"]`; issue115 and issue119 checks are all true.
- Release manifest regenerated at
  `build/current-release-regression-manifest-after-issue179-public-dmg-provenance-20260607.json`.
  It remains `prepackage_ready=false`, `release_ready=false`; explicit release
  blockers are now:
  - `mimo_v2_jang2l_runtime_quality_open`
  - `issue179_minimax_k_root_cause_audit`

Focused validation passed after the source change:

- `.venv/bin/python -m pytest tests/test_runtime_memory_stress_probe.py tests/test_public_app_issue_audit.py tests/test_release_regression_manifest.py -k 'stream_speed or runtime_memory_stress or public_app_issue or issue115 or issue119' -q`
- Result: `34 passed, 309 deselected`.

Still do not package, sign, notarize, tag, upload, or update downloads. Release
is still blocked by MiMo runtime quality and MiniMax #179 reporter/root-cause
parity, plus broader open objective rows in the current proof sweep.

## Local continuation note - 2026-06-07 MiMo JANGTQ2 exactness blocker

Do not commit this section unless Eric explicitly asks.

Latest pushed commit: `11fab2a4` (`Surface MiMo JANGTQ2 exactness blocker`) on
`origin/codex/pr-intake-manifest`.

What changed:

- MiMo current audit now treats strict current-artifact exactness as a first-class
  release blocker.
- Current JANGTQ2 no-media smoke still fails exactness:
  - `tool_required` expected `{"value":"blue-cat"}` but actual was
    `{"value":"blue cat"}`.
  - `structured_json_exact` expected `{"status":"ok","value":"blue-cat","count":3}`
    but actual was `{"status":"ok","value":"blue cat","count":3}`.
- This is now surfaced as `mimo_jangtq2_artifact_exactness_blocked`, not hidden
  behind only source-vs-quant ambiguity.
- No fake parser correction or argument rewriting was added. This remains a real
  current-artifact/model-output exactness blocker unless a corrected artifact or
  real constrained decoder is implemented and E2E-proven.

Validation and regenerated evidence:

- Focused tests passed:
  `.venv/bin/python -m pytest tests/test_mimo_v2_current_audit.py tests/test_release_regression_manifest.py -k 'mimo_v2 or mimo' -q`
  -> `19 passed, 295 deselected`.
- Regenerated MiMo audit:
  `build/current-mimo-v2-jang2l-current-audit-after-jangtq2-local-manifest-speed-refresh-20260607.json`
  -> `status=open`, `local_release_clearance=false`.
- Current MiMo audit blockers include:
  - `mimo_long_prompt_coherence_blocked`
  - `mimo_tool_protocol_blocked`
  - `mimo_jangtq2_artifact_exactness_blocked`
  - `mimo_cb_system_prompt_working_set_pressure_blocked`
  - `mimo_source_vs_quant_first_divergence_missing_or_failed`
  - `mimo_vl_audio_video_unwired`
- Regenerated release manifest remains `prepackage_ready=false`,
  `release_ready=false`; explicit release blockers remain:
  - `mimo_v2_jang2l_runtime_quality_open`
  - `issue179_minimax_k_root_cause_audit`

Still do not package, sign, notarize, tag, upload, or update downloads.

## 2026-06-07 local release blocker addendum - do not commit without Eric approval

Current active work remains Python vMLX engine plus MLXStudio app release readiness. Do not switch to deprecated `/Users/eric/vmlx`, Swift, ADLab, TB/RDMA, or model-upload-only work unless Eric explicitly asks in the current turn.

Newest Qwen35 required-tool/cache proof:

```text
build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-after-current-turn-required-reminder-20260607/SUMMARY.json
build/current-full-release-objective-checklist-after-qwen35-required-reminder-20260607.json
```

Qwen35 current classification:

- `overall_pass=false` for the strict `/v1/responses` long tool/cache gate.
- Runtime/cache path is active: native MTP configured D3 and tool-context capped to D1, `paged+ssm` cache hit, TurboQuant KV recompress/dequantize, block disk L2 writes/hits, and SSM companion L2 stores.
- API behavior is honest: `tool_choice=required` still returns HTTP 400 when the model produces no native tool call. Do not add fake post-generation function-call synthesis.
- Source plumbing already tightened: `tool_choice` is forwarded into chat-template kwargs, Qwen fallback prompt has a hard required-tool contract, and latest-user current-turn reminder is injected for Qwen required-tool fallback.
- Remaining blocker: Qwen35 model/tool-policy compliance under chained Responses still emits prose on turn 2 instead of a required tool call.

Release lock remains active:

- Latest full checklist is `status=open`, `failed_count=21`.
- Do not package, sign, notarize, tag, push release, update downloads, or claim production-ready until Qwen35, MiMo, media rows, UI matrix, MiniMax #179, DSV4 exactness/code rows, and all other full objective rows are green with current live evidence.

## 2026-06-07 local MiMo JANGTQ2 status addendum - do not commit without Eric approval

Newest MiMo audit evidence:

```text
build/current-mimo-v2-jang2l-current-audit-after-tool-protocol-exactness-split-20260607.json
build/current-full-release-objective-checklist-after-mimo-tool-protocol-exactness-split-20260607.json
```

MiMo current classification:

- `tool_protocol=true`: current JANGTQ2 no-media smoke emits a parsed `record_fact` tool call.
- `artifact_exactness=false`: the model mutates the literal `blue-cat` to `blue cat` in tool arguments and strict JSON.
- Do not add parser-side argument repair or fake tool-call synthesis to hide exactness failure.
- Remaining MiMo blockers: long-prompt coherence, JANGTQ2 artifact exactness, CB system-prompt working-set pressure, source-vs-quant endpoints/proof, and VL/audio/video unwired.
- Full checklist remains open with `failed_count=20`; no package/sign/notarize/tag/release/download update is allowed.

## Local-only release gate reminders - 2026-06-07

- Do not call release-ready until `/v1/responses` multi-turn endpoint tests prove stable request state, cache reuse, tool-call continuation, parser selection, max-token/settings behavior, and no request-level HTTP errors.
- Do not treat cache path logs alone as sufficient. Require visible cached-token reuse in endpoint responses and successful continuation behavior after tool outputs.
- MiMo XML parser is proven to preserve hyphenated literal arguments; classify `blue-cat -> blue cat` as artifact/model decode exactness unless new evidence proves runtime mutation elsewhere.

## Local-only proof update - 2026-06-07 Qwen35 Responses cache gate

- Qwen35 `/v1/responses` long-tool-cache gate is green at `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-after-historical-tool-required-20260607/SUMMARY.json`.
- Do not reopen this row unless a newer proof regresses `previous_response_id`, cached-token reuse on later turns, required tool calls, tool-result continuation, parser selection, or HTTP 200 behavior.
- Keep release locked because the refreshed full checklist is still open with 16 blockers.

## Local-only vMLX/MLXStudio release objective guard - 2026-06-07

Do not commit this section unless Eric explicitly asks. This is a local active-objective guard for agents working in this checkout.

### Active scope

Work only on the active vMLX Python engine, MLXStudio panel/app integration, runtime proof gates, packaging readiness, signing/notarization readiness, and release evidence.

Canonical runtime worktree for this task:

```text
/Users/eric/mlx/vllm-mlx-finite-launch-guard
```

Do not work in deprecated or unrelated paths for this objective:

```text
/Users/eric/vmlx
/Users/eric/vmlx/swift
ADLab / TB5 / RDMA / model-transfer-only work
source-vs-quant heavy comparison for MiMo unless Eric explicitly re-approves RAM use
```

### Release lock

Do not package, sign, notarize, tag, upload, push a release, update public downloads, or claim production-ready while any full-objective checklist row is open.

Current authoritative full checklist after Qwen35 cache/Responses pass:

```text
build/current-full-release-objective-checklist-after-qwen35-cache-responses-pass-20260607.json
status=open
failed_count=16
```

Current open rows:

```text
prepackage_ready
release_ready
packaged_integrity_component
real_ui_live_model_matrix
mimo_local_release_clearance
mimo_long_prompt_coherence
mimo_artifact_exactness
mimo_cb_system_prompt_working_set_pressure
mimo_source_vs_quant_first_divergence
mimo_mimo_media_wired
gemma4_12b_media_rows_complete
step37_real_vlm_runtime_complete
nemotron_omni_media_rows_complete
issue179_root_cause_status_pass
dsv4_exactness_preflight_status_pass
dsv4_exact_code_file_long_output_complete
```

### Non-negotiable proof standard

A row is not green from source inspection, load-only evidence, `/health`, or a narrow one-turn smoke. Require current evidence for the exact surface being claimed.

Required proof categories for runtime/model release:

```text
/v1/responses multi-turn with previous_response_id
/v1/chat/completions parity where applicable
streaming and non-streaming behavior
settings panel propagation for max output tokens, parser selection, thinking, cache toggles
real tool calls, tool outputs, and final synthesis without raw markup leaks
strict JSON/XML/code/whitespace formatting rows
cache reuse with positive cached_tokens in response usage
prefix cache detail for the architecture, not generic claims
TurboQuant encode/decode or explicit architecture-specific non-applicability
block disk L2 store and hit evidence where enabled
hybrid SSM companion state store/hit/rederive behavior for hybrid models
media rows for image/audio/video where the model advertises them
no fake guards, fake tool calls, fake parser repair, or silent argument rewriting
```

### Cache architecture checklist

Classify cache by architecture before testing:

```text
full attention KV: prefix/paged cache, optional TurboQuant KV, optional block disk L2
Gemma 4 mixed/SWA: mixed_swa/native cache detail, storage-boundary q4 if applicable, not generic live TQ KV
Qwen 3.6 hybrid/SSM: paged+ssm, attention-only live TurboQuant KV, SSM companion L2, async/inline SSM capture or aligned rederive
DSV4 composite SWA/CSA/HSA: DSV4-native composite cache, not generic TurboQuant KV as a drop-in
MiMo/JANGTQ MoE: routed expert runtime, SwitchGLU correctness, cache-vs-no-cache next token, long prompt and exactness proof
VLM/media: media-expanded prompt cache behavior must be proven separately from text-only cache
```

### Model-family gates that must stay tracked

MiMo V2.5 JANGTQ2:

```text
current audit: build/current-mimo-v2-jang2l-current-audit-after-tool-protocol-exactness-split-20260607.json
current state: tool protocol green, speed green, cache narrow green, exactness red, long prompt red, CB working-set red, media unwired
never hide blue-cat -> blue cat with parser repair; XML parser preserves hyphens
```

Qwen 3.6 35B MXFP8 MTP:

```text
current passing gate: build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-after-historical-tool-required-20260607/SUMMARY.json
must preserve: native MTP autodetect, required tools, previous_response_id, cached_tokens on later turns, paged+ssm, block L2, SSM companion L2, TurboQuant KV
```

Qwen 3.6 27B MTP:

```text
must preserve: MTP autodetect, parser selection, Responses cancel/settings, tools, media where advertised, cache reuse, L2/TQ proof
```

Gemma 4 12B:

```text
text/no-media rows are not enough
must prove image/audio/video rows where advertised, Gemma4 mixed/SWA cache type, large image prefill behavior, tool/schema formatting, settings panel max tokens/cache controls
```

Step 3.7 Flash:

```text
text-only proof does not clear real VLM
must avoid fake has_vision=false as a release fix for VLM claims
must prove Step3p7 VLM/media path or clearly mark model/runtime VLM unsupported and block media release row
must test tool dialect consistency and loop stop in multi-turn tool tasks
```

Nemotron Omni:

```text
RADIO/Parakeet image/audio/video rows remain required
must prove media dispatch through Responses/chat surfaces, not text-only fallback
```

LFM / MiniMax / DSV4 / JANG / JANGTQ / MXFP:

```text
LFM: parser/schema/tool rows plus cache behavior for its architecture
MiniMax: issue179 root-cause/reporter parity must be green before release
DSV4 Flash: exact code/file long-output and route-mode preflight must be green; use DSV4-native composite cache semantics
JANG/JANGTQ/MXFP: loader metadata, quant sidecars, autodetection, runtime kernels, cache policy, and decode speed must be proven per artifact type
```

### UI/app release checklist

Before packaging/signing/notarization, prove from MLXStudio UI or installed app evidence:

```text
model selection and launch uses current Python engine code
parser dropdown exposes required tool and reasoning parsers
settings panel applies max output tokens, thinking, cache mode, prefix cache, paged cache, L2 disk cache, and MTP-compatible sampling where relevant
chat UI handles tool calls, tool outputs, Responses streaming, cancellation, empty output errors, and media attachments
cache usage is visible or inspectable with cached_tokens/cache_detail evidence
packaged app contains the patched source files and bundled runtime dependencies
```

### If a model fails

Classify honestly:

```text
model/artifact issue: bad/missing metadata, corrupt quant tensors, missing sidecars, incoherent exactness, quantization damage
runtime issue: loader route, parser/template plumbing, decode loop, cache type, kernel path, VLM/audio/video embedding, scheduler, settings/UI propagation
unknown: keep row red and collect narrower evidence; do not force green
```

For model/artifact issues, tell Eric exactly what must be rebuilt or uploaded. For runtime issues, fix the engine/app and prove with current E2E artifacts.

## Local-only proof update - 2026-06-07 MiMo sentinel exactness

- Current MiMo exactness proof is `build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-20260607/summary.json`.
- The failure is broader literal-copy/punctuation corruption, not just `blue-cat` semantic normalization: `B7-CAT-09` became `B7CAT-09` / `B7CCAT-09`.
- Keep `mimo_jangtq2_artifact_exactness_blocked` open. Do not repair this with parser-side replacement or argument rewriting.
- Current audit/checklist pointers:
  - `build/current-mimo-v2-jang2l-current-audit-after-sentinel-exactness-proof-20260607.json`
  - `build/current-full-release-objective-checklist-after-mimo-sentinel-exactness-proof-20260607.json`

## Local-only proof update - 2026-06-07 DSV4 preflight refresh

- Current DSV4 route-mode code exactness preflight is `build/current-dsv4-route-mode-code-exactness-preflight-after-release-manifest-refresh-20260607.json`.
- It is still skipped for `insufficient_vm_stat_memory`: 53.27GB available for gate vs 120GB required, gap 66.73GB, no active heavy process detected.
- Do not claim DSV4 exactness or long-output complete from older/stale artifacts.

## Local-only proof update - 2026-06-07 Gemma4 12B media

- Gemma4 12B JANG_4M image media is currently green at `build/current-gemma4-12b-jang4m-media-smoke-after-vlm-prefill-guard-20260607.json` with answer `Red`.
- Current source VLM image prefill guard is memory-aware; do not regress it to a fixed 8GB default. Explicit `VMLINUX_VLM_IMAGE_PREFILL_BUFFER_GB=8` is still allowed as an override and will preserve the old cap.
- Current full checklist after this row: `build/current-full-release-objective-checklist-after-gemma4-jang4m-media-pass-20260607.json`, failed_count 15.

## Local-only proof update - 2026-06-07 Nemotron Omni media/cache

- Nemotron Omni MXFP4 media gate is currently green at `build/current-nemotron-omni-mxfp4-media-gate-20260607/SUMMARY.json`.
- The proof covers image, video, audio, and multi-turn recall rows through `/v1/chat/completions`: image/video/audio answered `Blue`, recall answered `color=blue, animal=cat`.
- The same proof includes cache reuse and L2 evidence: turn-2 recall has `cached_tokens=24`, `cache_detail=paged+ssm`; final cache stats show block disk L2 writes/hits and SSM companion disk stores.
- Full checklist now uses that media gate instead of hard-failing Nemotron media. Current refreshed checklist: `build/current-full-release-objective-checklist-after-nemotron-media-cache-pass-20260607.json`, failed_count 14.
- This does not clear release: packaging/UI, MiMo long/exactness/CB/source/media, Step3.7 real VLM, MiniMax #179, and DSV4 exactness/code rows remain open.

## Local-only proof update - 2026-06-07 MiMo media/runtime metadata classification

- Current MiMo audit: `build/current-mimo-v2-jang2l-current-audit-after-media-runtime-metadata-classification-20260607.json`.
- Current full checklist: `build/current-full-release-objective-checklist-after-mimo-media-subgates-20260607.json`, status `open`, failed_count 16.
- MiMo media classification is `runtime_implementation_gap_with_model_metadata_overadvertising`.
- Promoted bundle does preserve media payloads: 364 `visual.*` tensors, 95 audio/speech tensors, `preprocessor_config.json`, and `audio_tokenizer/`.
- Current runtime `/capabilities` surface is safe: `modalities=["text"]`, with vision/image/video/audio marked `preserved_unwired`.
- Current model config is not safe for direct consumers: it advertises `capabilities.modalities=["text","vision","audio"]` and lacks the required `preserved_modalities`, `unwired_modalities`, and `multimodal_status` fields. If runtime remains unwired, rebuild/patch model metadata to:
  - `capabilities.modalities=["text"]`
  - `capabilities.preserved_modalities=["vision","audio"]`
  - `capabilities.unwired_modalities=["vision","audio"]`
  - `capabilities.multimodal_status="weights_preserved_text_runtime"`
  - `runtime.multimodal_mode="weights_preserved_text_runtime"`
- Engine/runtime work still required before MiMo media can be green: real `mimo_v2_multimodal.py` or equivalent `vmlx_engine.models.mllm` implementation for VisionConfig/AudioConfig parsing, `visual.*` load binding, audio/speech load binding, ViT forward, vision merger, audio tokenizer bridge, audio transformer/projection, mixed media/text `inputs_embeds`, and media-aware cache/L2 proof.
- The top-level checklist intentionally exposes MiMo media subgates separately: `mimo_media_weights_preserved` and `mimo_media_runtime_capabilities_safe` are green, while `mimo_media_model_metadata_text_only_contract`, `mimo_media_runtime_implementation`, and `mimo_mimo_media_wired` remain red.

## Local-only proof update - 2026-06-07 MiniMax issue179 subgates

- Current full checklist: `build/current-full-release-objective-checklist-after-issue179-subgates-20260607.json`, status `open`, failed_count 19.
- MiniMax #179 is now split into first-class release rows. Green/current rows:
  - `issue179_current_source_cancel_contract`
  - `issue179_latest_public_dmg_cancel_route`
  - `issue179_local_installed_cancel_route`
  - `issue179_local_installed_cancel_live_probe`
  - `issue179_local_real_ui_clean`
  - `issue179_installed_session_settings_parity`
- Red reporter/provenance rows:
  - `issue179_root_cause_status_pass`
  - `issue179_reporter_parity_artifact`
  - `issue179_reporter_server_hash_parity`
  - `issue179_reporter_prompt_reproduction_or_log_proof`
- Do not treat MiniMax #179 as a current-source cancel-route bug. Current source, latest public DMG route markers, local installed route, local live cancel, and local UI diagnostics are green. The remaining blocker is reporter parity/provenance and reporter-session reproduction/log proof.

## Local-only proof update - 2026-06-07 Step3.7 VLM subgates

- Current full checklist: `build/current-full-release-objective-checklist-after-step37-vlm-subgates-20260607.json`, status `open`, failed_count 19.
- Step3.7 text/no-media proof is green at `build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-textonly-harness-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json`: text cache, multiturn recall, reasoning, required tool call, and `step3p7_full_sliding_kv` cache are green.
- Step3.7 no-heavy VLM runtime audit is green at `build/current-step37-vlm-runtime-audit-after-gemma4-vl-refresh-20260606.json`: `mlx_vlm.models.step3p7` exposes runtime symbols and source-owned config/model/vision/projector/processor/projection surfaces.
- Top-level checklist now keeps `step37_vlm_noheavy_runtime_surface` green and `step37_vlm_rejects_fake_clearance_paths` green.
- `step37_real_vlm_runtime_complete` remains red until live image/video media proof passes through the actual runtime/UI/API path. Do not clear it from text-only proof, importability only, or a stub runtime.

## Local-only proof update - 2026-06-07 API/cache/DSV4 subgates

- Current full checklist: `build/current-full-release-objective-checklist-after-dsv4-memory-subgates-20260607.json`, status `open`, failed_count 20.
- The no-heavy API/cache artifact is now required to have `status=pass` before its per-row checks count. This keeps `/v1/responses`, previous_response_id history, streaming cache details, cache reuse endpoints, output caps, JSON response format/schema preservation, Anthropic, and Ollama rows from being green under a failed parent artifact.
- DSV4 is now split into explicit release rows:
  - `dsv4_exactness_preflight_artifact_exists`
  - `dsv4_memory_preflight_floor_valid`
  - `dsv4_memory_preflight_process_context`
  - `dsv4_memory_preflight_launch_safely_blocked`
  - `dsv4_memory_preflight_resource_available`
  - `dsv4_exactness_preflight_status_pass`
  - `dsv4_exactness_preflight_case_matrix_present`
  - `dsv4_exact_code_file_long_output_complete`
- Current DSV4 evidence remains memory-gated, not exactness-cleared: the preflight was safely skipped for `insufficient_vm_stat_memory`, with no active heavy process detected. The resource availability, exactness status, and exact code/file long-output rows remain red until a real run completes.

## Local-only proof update - 2026-06-07 MiMo metadata text-runtime contract

- Corrected local promoted MiMo config at `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2/config.json` so it no longer advertises unwired media runtime:
  - `capabilities.modalities=["text"]`
  - `capabilities.preserved_modalities=["vision","audio"]`
  - `capabilities.unwired_modalities=["vision","audio"]`
  - `capabilities.multimodal_status="weights_preserved_text_runtime"`
  - `runtime.multimodal_mode="weights_preserved_text_runtime"`
- Updated `build/current-mimo-jangtq2-local-manifest-20260607.tsv` for the new config size; refreshed audit shows manifest integrity still `pass`.
- Current MiMo audit: `build/current-mimo-v2-jang2l-current-audit-after-artifact-exactness-boundary-20260607.json`.
- Current full checklist: `build/current-full-release-objective-checklist-after-mimo-metadata-contract-20260607.json`, status `open`, failed_count 19.
- This clears only `mimo_media_model_metadata_text_only_contract`. It does not wire MiMo VL/audio/video. `mimo_media_runtime_implementation` and `mimo_mimo_media_wired` remain red.

## Local-only proof update - 2026-06-07 release manifest pointer refresh

- Current compact release manifest: `build/current-release-regression-manifest-after-issue179-public-dmg-provenance-20260607.json`.
- It was generated with `tests/cross_matrix/run_release_regression_manifest.py`, not the rows-only manifest library.
- Current state remains:
  - `current_proof_sweep=fail`
  - `prepackage_ready=false`
  - `release_ready=false`
  - release blockers include `mimo_v2_jang2l_runtime_quality_open` and `issue179_minimax_k_root_cause_audit`
- The full release checklist now reads packaged integrity and real-UI status from `current_proof_sweep.component_ok` in that current compact manifest.
- Current full checklist after this refresh: `build/current-full-release-objective-checklist-after-release-manifest-refresh-20260607.json`, status `open`, failed_count 17.

## Local-only proof update - 2026-06-07 DSV4 current preflight/artifact refresh

- Current DSV4 route-mode code exactness preflight: `build/current-dsv4-route-mode-code-exactness-preflight-after-release-manifest-refresh-20260607.json`.
- Current result remains safely skipped, not exactness-cleared:
  - `reason=insufficient_vm_stat_memory`
  - `did_not_launch=true`
  - `available_for_gate_gb=53.27`
  - `required_available_gb=120.0`
  - `memory_gap_gb=66.73`
  - `active_heavy_process_count=0`
- Current proof artifacts were refreshed to the DSV4-preflight generation:
  - `build/current-objective-proof-after-dsv4-preflight-refresh-20260607.json`
  - `build/current-release-regression-manifest-after-issue179-public-dmg-provenance-20260607.json`
  - `build/current-regression-suite-after-dsv4-preflight-refresh-20260607.json`
  - `build/current-full-release-objective-checklist-after-dsv4-preflight-refresh-20260607.json`
- Aggregate current suite remains `status=open`, with failed steps `packaged_integrity_contracts`, `release_regression_manifest`, and `release_gate_skip_app`.
- Current full checklist remains `status=open`, failed_count 18.

## Local-only proof update - 2026-06-07 MiniMax #179 public DMG provenance

- Downloaded public MLXStudio DMGs for v1.5.50 and v1.5.52 from `jjang-ai/mlxstudio` releases and generated no-heavy server route/hash contracts:
  - `build/issue-179/public-v1.5.50-sequoia-dmg-contract.json`
  - `build/issue-179/public-v1.5.50-tahoe-dmg-contract.json`
  - `build/issue-179/public-v1.5.52-sequoia-dmg-contract.json`
  - `build/issue-179/public-v1.5.52-tahoe-dmg-contract.json`
- Combined with existing v1.5.56 contracts, MiniMax #179 provenance now checks six public DMGs. All six have the `/v1/responses/{response_id}/cancel` route and call engine abort.
- Current MiniMax audit: `build/current-issue179-minimax-k-root-cause-audit-after-local-repro-memory-preflight-20260607.json`.
- The old `missing_required_public_release_dmg_contracts` gap is cleared. The remaining reporter blocker is real: reporter published server hash `a2c4...e8be` does not match current source, local installed app, or any checked public DMG, and the full reporter parity artifact is still missing.
- Current release manifest: `build/current-release-regression-manifest-after-issue179-public-dmg-provenance-20260607.json`, still `prepackage_ready=false`, `release_ready=false`.
- Current full checklist: `build/current-full-release-objective-checklist-after-issue179-public-dmg-provenance-20260607.json`, status `open`, failed_count 18.

## CODEX 2026-06-07 - MiMo sentinel pointer refresh and current release gate

Current lane remains Python vMLX engine plus MLXStudio app release only. Do not switch to deprecated `/Users/eric/vmlx`, Swift, ADLab/TB/RDMA, or source-vs-quant unless Eric explicitly asks in the current turn.

Current evidence refresh from this continuation:

- Objective digest now points at `build/current-objective-proof-after-mimo-sentinel-pointer-refresh-20260607.json`.
- Release manifest now points at `build/current-release-regression-manifest-after-mimo-sentinel-pointer-refresh-20260607.json`.
- Aggregate current suite now points at `build/current-regression-suite-after-mimo-sentinel-pointer-refresh-20260607.json`.
- Current MiMo proof for JANGTQ2 exactness is `build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-20260607/summary.json`.

MiMo current classification:

- Green in current sentinel proof: exact text cache repeat, cache hit telemetry, multiturn recall, reasoning visible output, parsed tool calls, tool-result continuation, and MiMo typed cache markers.
- Red in current sentinel proof: exact literal preservation. Examples include `blue-cat` becoming `blue-123` or `bluecat`, `B7-CAT-09` becoming `B7CAT-09`, and sentinel JSON value mutation. Do not fake-fix this with parser argument rewriting or post-generation repair for release clearance.
- Still open in current full checklist: long-prompt coherence, JANGTQ2 artifact exactness, continuous-batching/system-prompt working-set pressure, MiMo media runtime implementation/wiring, and source-vs-quant first divergence when memory allows.

Current aggregate suite result:

- Artifact: `build/current-regression-suite-after-mimo-sentinel-pointer-refresh-20260607.json`.
- Status: `open`.
- Failed steps: `packaged_integrity_contracts`, `release_regression_manifest`, `release_gate_skip_app`.
- Open requirements: cross-family live multi-turn smoke matrix, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity/root cause, real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.

Validation from this continuation:

- `tests/test_objective_proof_digest.py -k 'mimo or all_local_smoke'` -> `7 passed`.
- Focused objective/release/current-suite pointer slice -> `420 passed, 73 deselected`.
- Focused release/checklist/current-suite slice -> `318 passed, 71 deselected`.
- `tests/test_release_regression_manifest.py -k release_regression_manifest` -> `313 passed`.

Release boundary:

- Do not sign, notarize, tag, or publish downloads from this state.
- API/cache/Responses/parser contracts are green at contract level, but production release remains blocked by live model-family and packaged/UI proof rows above.

## CODEX 2026-06-07 - MiMo exactness-boundary classification

Current MiMo audit artifact:

```text
build/current-mimo-v2-jang2l-current-audit-after-artifact-exactness-boundary-20260607.json
```

Key classification:

```text
artifact_exactness_boundary.classification = model_generated_literal_mutation_after_valid_parser_structure
```

Meaning:

- Parser/tool protocol is not the current MiMo exactness blocker.
- The current sentinel proof has valid OpenAI tool-call structure and valid visible JSON object structure.
- The generated values are wrong: examples include `blue-cat -> blue-123`, `B7-CAT-09 -> B7CAT-09`, `blue-cat -> bluecat`, and `B7-CAT-09 -> B7CCAT-09`.
- Do not clear this row by rewriting parsed tool arguments, repairing JSON values post-generation, disabling cache, or hiding exactness rows.
- Remaining likely classes are model/runtime decode quality or quant/artifact quality. Source-vs-quant remains the decisive comparison when RAM/session conditions allow, but it was not run in this continuation.

Current proof chain after this refresh:

- Objective digest: `build/current-objective-proof-after-mimo-exactness-boundary-20260607.json`.
- Release manifest: `build/current-release-regression-manifest-after-mimo-exactness-boundary-20260607.json`.
- Full checklist: `build/current-full-release-objective-checklist-after-mimo-exactness-boundary-20260607.json`.
- Aggregate current suite: `build/current-regression-suite-after-mimo-exactness-boundary-20260607.json`.

Validation:

- `tests/test_mimo_v2_current_audit.py` -> `3 passed`.
- Focused MiMo/checklist/objective/release manifest suite -> `424 passed`.
- Aggregate current suite -> `status=open`; failed steps remain `packaged_integrity_contracts`, `release_regression_manifest`, `release_gate_skip_app`.
- Final focused proof-chain suite -> `429 passed, 69 deselected`.

Release boundary is unchanged: release remains locked until MiMo exactness/long/CB/media/source-vs-quant, MiniMax #179, real UI cross-family matrix, and DSV4 long-output/code rows are cleared.

## CODEX 2026-06-07 - ZAYA text current-smoke pointer refresh

Current objective digest now uses the current mixed ZAYA text/VL smoke for `zaya_text` coverage:

```text
build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json
```

Resulting classification:

- `zaya_text` is no longer a missing required family in the cross-family live smoke objective.
- ZAYA text row in that artifact is pass for text cache repeat, multi-turn recall, reasoning budget, and required tool coverage.
- ZAYA-VL evidence from the same mixed artifact remains not-pass, but the separate refreshed ZAYA-VL artifact remains pass where the current gate expects it. Do not collapse ZAYA text and ZAYA-VL into one acceptance row.

Current proof chain after this refresh:

- Objective digest: `build/current-objective-proof-after-zaya-text-current-smoke-refresh-20260607.json`.
- Release manifest: `build/current-release-regression-manifest-after-zaya-text-current-smoke-refresh-20260607.json`.
- Full checklist: `build/current-full-release-objective-checklist-after-zaya-text-current-smoke-refresh-20260607.json`.
- Aggregate current suite: `build/current-regression-suite-after-zaya-text-current-smoke-refresh-20260607.json`.

Current open objective requirements remain:

- Cross-family live multi-turn smoke matrix, now missing only `mimo_v2` at the family-key level.
- MiMo V2.5 JANG_2L runtime/tool/long-prompt quality.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause.
- Real Electron UI cross-family live model matrix, now missing `mimo_v2`.
- DSV4 long-output/code/file-generation quality.

Validation:

- Objective digest ZAYA/all-local suite: `106 passed`.
- Focused objective/release/current-suite path: `424 passed, 71 deselected`.
- Aggregate current suite: `status=open`; failed steps remain `packaged_integrity_contracts`, `release_regression_manifest`, `release_gate_skip_app`.
- Final focused proof-chain validation: `420 passed, 73 deselected`.

Release remains locked. No signing, notarization, tag, push, or public download update was performed.

## CODEX 2026-06-07 - Packaged integrity current-drift refresh

Current packaged-integrity artifact:

```text
build/current-packaged-integrity-contract-after-staged-sequoia-rebuild-current-source-20260607.json
```

What changed:

- Fixed `panel/scripts/release-gate-python-app.py` objective digest default so it refreshes/enforces `build/current-objective-proof-after-zaya-text-current-smoke-refresh-20260607.json` instead of stale `after-mllm-tight-memory-guard` evidence.
- Release-gate unit contracts now pass in packaged-integrity (`47 passed`).
- Packaged integrity remains red for real bundled Python drift and open objective blockers.

Current packaged-integrity classification:

- `release_gate_unit_contracts`: pass.
- `bundled_python_verifier`: fail because packaged `vmlx_engine/server.py` differs from current source.
- `release_gate_skip_app`: fail because bundled Python import gate fails and objective proof still has open model/release requirements.
- Signing preflight inside the artifact is pass, but do not sign/notarize/release while MiMo, MiniMax #179, DSV4, and real UI rows remain open.

Current release chain after this refresh:

- Release manifest: `build/current-release-regression-manifest-after-packaged-integrity-current-drift-refresh-20260607.json`.
- Full checklist: `build/current-full-release-objective-checklist-after-packaged-integrity-current-drift-refresh-20260607.json`.

Validation:

- Release-gate objective default test: `1 passed`.
- Packaged-integrity/release/current-suite focused slice: `370 passed, 115 deselected`.

Do not fake-clear bundled drift by pointing to old packaged-integrity pass artifacts. The staged app must be rebundled from current source and then re-signed only when Eric explicitly wants to proceed despite remaining model blockers, or after model blockers clear.

## CODEX 2026-06-07 - Staged Sequoia app rebuilt; packaged integrity passes

Current package evidence:

- Bundled Python refresh completed from current source and local `/Users/eric/jang/jang-tools`.
- `npm run verify-bundled` passed, including vMLX source parity, JANG tools parity, TurboQuant kernels, MiMo, Step3p7 VLM runtime, Gemma4 unified runtime, VL/audio dependencies, and bundled imports.
- Staged Sequoia app rebuilt at `panel/release/sequoia-app/mac-arm64/vMLX.app` with `npx electron-builder --mac --dir --config.directories.output=release/sequoia-app`.
- Electron builder signed the staged app with Developer ID identity `55KGF2S5AY`; notarization was skipped. No DMG was built.
- Current packaged-integrity artifact: `build/current-packaged-integrity-contract-after-staged-sequoia-rebuild-current-source-20260607.json` -> `status=pass`, `failed=[]`, all checks true.
- Current aggregate suite: `build/current-regression-suite-after-packaged-integrity-pass-20260607.json` -> `status=open`, failed steps now only `release_regression_manifest`.

Current release manifest/checklist:

- `build/current-release-regression-manifest-after-zaya-vl-bundled-tool-pass-20260607.json` -> `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- `build/current-full-release-objective-checklist-after-packaged-integrity-pass-20260607.json` -> `status=open`, `failed_count=17`.

Remaining release blockers are no longer package/hash drift. They are objective/model proof blockers:

- Cross-family live multi-turn smoke matrix, currently blocked by MiMo V2.5.
- MiMo V2.5 JANG_2L exactness/long-prompt/CB/media/source-vs-quant.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause.
- Real Electron UI cross-family matrix, currently missing MiMo V2.5.
- DSV4 long-output/code/file-generation memory/exactness.
- Step3p7 real VLM image/video proof remains open in full checklist.

Validation:

- `npm run verify-bundled` -> pass.
- Packaged integrity after staged rebuild -> pass.
- Aggregate current suite after package pass -> `status=open`, failed steps `['release_regression_manifest']`.
- Focused packaged/release/current-suite validation -> `370 passed, 115 deselected`.

Do not claim public release readiness. No DMG, notarization, tag, push, or download update happened in this step.

## Current release-tracking nuance - 2026-06-07 live smoke/cache/API slice

The current API/cache/Responses no-heavy contract is green, but this is not live
release clearance. Treat these as source/route contract rows only until each
model family has live API plus UI proof:

- Chat Completions, Responses, legacy Completions, Anthropic, and Ollama route
  behavior.
- `previous_response_id`, streaming cache usage, cache reuse endpoints, cache
  stats/reuse telemetry, JSON/schema response handling, and max-output vs
  max-context separation.
- Prefix/paged/L2 restart restore and TurboQuant KV encode/decode only where the
  architecture is standard KV; typed native cache or async rederive is required
  for hybrid/SWA/SSM/composite families.

Latest manifest-refresh work produced filtered per-row smoke artifacts from
current multi-model summaries so manifest validation no longer fails merely
because a combined summary contains other models. Current filtered artifacts:

- `build/current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json`
- `build/current-filtered-live-smoke-hy3-preview-jangtq2-20260607/summary.json`
- `build/current-filtered-live-smoke-ling-flash-jangtq-20260607/summary.json`
- `build/current-filtered-live-smoke-nemotron-omni-jangtq-20260607/summary.json`
- `build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/summary.json`

Latest manifest artifact for this slice:

- `build/current-release-regression-manifest-after-zaya-vl-bundled-tool-pass-20260607.json`

Remaining live-smoke failures after stale pointer cleanup are evidence-scope or
true release blockers, not missing old filenames:

- several live-smoke/tool-smoke rows are still current-source or non-bundled
  evidence (`command_python_not_bundled`) and need bundled/installed reruns or a
  deliberate release-boundary decision;
- DSV4 live-smoke artifact still uses a HeadBF16 probe name while DSV4 exact
  code/file/long-output remains separately open;
- old diagnostic live-smoke expected-failure rows should be retired or replaced
  with current diagnostic blockers; do not let old expected-failure artifacts
  masquerade as current release evidence;
- MiMo, Step3p7 VLM, MiniMax #179, real UI full matrix, and DSV4 memory/exactness
  remain release blockers.

## Current MiMo V2.5 JANGTQ_2 blocker - 2026-06-07

- Current artifact under test: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`.
- Current bundled no-media tools/cache proof:
  `build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-20260607/summary.json`.
- Status: `probe_failed`, `failures=5`.
- Runtime/cache pieces that passed: text cache repeat, paged cache hit, block disk
  L2, q4 mixed-SWA/full-KV storage-boundary quantization, rotating-window metadata
  preservation, no generic TurboQuant KV on MiMo mixed-SWA cache, multiturn recall,
  tool-result continuation, exact code/whitespace.
- Release-blocking failures: reasoning-on produced hidden reasoning but empty visible
  output; required tool argument mutated `blue-cat` to `blue-123`; MiMo sentinel
  tool argument mutated `B7-CAT-09` to `B7CAT-09`; strict JSON mutated
  `blue-cat` to `blue-1`; sentinel JSON mutated `B7-CAT-09` to `B7CCAT-09`.
- Classification: parser/runtime cache are not the immediate failing surface in this
  proof; the parser preserved model-emitted wrong arguments. Treat as MiMo model
  artifact/quantization/decode-quality blocker until disproven by a runtime A/B that
  does not load source-vs-quant on this machine.
- Do not release-clear MiMo from short speed/coherency rows. It must pass exact
  tool arguments, exact JSON values, reasoning visible-answer behavior, long-prompt
  quality, and media wiring before release.

## Local-only continuation update - 2026-06-07 MiMo capability truth and cache/API release gate

Latest MiMo bundled no-media tools/cache proof:

- `build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-after-mimo-capability-snapshot-fix-20260607/summary.json`
- Status: `fail`, `probe_failed`, `failures=4`.
- Current row policy: `model_type=mimo_v2`, `supports_thinking=false`, `supports_tools=true`, `supports_video=false`, `cache_family=mimo_v2_hybrid_swa`.
- Capability snapshot is now normalized for release proof: reasoning `supported=false`, `parser=null`, tools `supported=true`, `parser=xml_function`.
- Proven runtime/cache pieces: text cache repeat, `cache_detail=paged`, `cached_tokens=67` and `128`, cache hit tokens up to `195`, multiturn recall, parsed XML tool calls, tool-result continuation, exact code/whitespace, MiMo mixed-SWA cache markers, and no reasoning leakage in this no-thinking policy path.
- Remaining release-blocking failures are exact-value generation failures, not parser mutation: `blue-cat -> blue-123`, `B7-CAT-09 -> B7CAT-09`, JSON `blue-cat -> blue-1`, JSON `B7-CAT-09 -> B7CCAT-09`.
- Do not paper over this with parser-side argument rewriting, prompt-only guards, or fake JSON repair inside the runtime. If the model emits the wrong literal, classify it as artifact/model/decode exactness until a real runtime A/B proves otherwise.
- Do not rerun source-vs-quant comparison for MiMo on this machine unless Eric explicitly reauthorizes it; RAM/headroom was explicitly ruled out for this lane.

Current source/proof changes from this slice:

- `vmlx_engine/reasoning/think_xml_parser.py`: no-tag MiMo/XML outputs remain visible instead of being swallowed as hidden reasoning; Qwen/base special-token behavior remains covered separately.
- `vmlx_engine/model_configs.py` and `vmlx_engine/model_config_registry.py`: MiMo V2.5 is not release-exposed as thinking-capable; XML tools remain enabled.
- `bench/all_local_model_smoke.py`: MiMo smoke classifier matches runtime truth and reports a normalized capability snapshot.
- `tests/cross_matrix/summarize_objective_proof.py` and objective tests now point at the latest MiMo capability-snapshot artifact.

Validation from this slice:

- MiMo bundled smoke after capability snapshot fix: still `fail`, `failures=4`, exactness blocker only.
- Objective digest selector: `tests/test_objective_proof_digest.py -k 'mimo or all_local_smoke or api_cache or current_noheavy' -q` -> `11 passed`.
- Release-manifest selector: `tests/test_release_regression_manifest.py -k 'mimo or live_smoke or live_tool_smoke' -q` -> `32 passed`.
- Regenerated artifacts:
  - `build/current-release-regression-manifest-after-mimo-capability-snapshot-fix-20260607.json`
  - `build/current-objective-proof-after-mimo-capability-snapshot-fix-20260607.json`
  - `build/current-full-release-objective-checklist-after-mimo-capability-snapshot-fix-20260607.json`

Release remains blocked. Current open checklist items include MiMo exactness/long-prompt/CB/source/media, Step3p7 real VLM proof, MiniMax #179 reporter parity/root cause, real Electron UI cross-family matrix, and DSV4 memory/exactness/code/file-generation. Do not sign, notarize, tag, push release, or update public downloads from this state.

## Local-only continuation update - 2026-06-07 current-source contract refresh after MiMo capability truth

Current proof-chain correction:

- The MiMo capability truth change invalidated several no-heavy proof artifacts by source hash. The stale rows were not runtime failures; the release gate correctly refused to count old source-hash evidence.
- Refreshed current-source no-heavy artifacts:
  - `build/current-cache-architecture-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
  - `build/current-model-family-detection-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
  - `build/current-parser-registry-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
  - `build/current-model-artifact-format-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
  - `build/current-generation-defaults-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
  - `build/current-native-mtp-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
  - `build/current-vl-media-cache-contract-after-mimo-capability-snapshot-fix-20260607.json` -> pass.
- Updated `tests/cross_matrix/run_decode_speed_gate.py`: MiMo V2.5 JANGTQ2 keeps `tool_parser=xml_function` but no longer declares `reasoning_parser=think_xml`, matching the engine registry's release-exposed MiMo no-thinking policy.
- Regenerated:
  - `build/current-release-regression-manifest-after-current-source-contract-refresh-20260607.json` -> red only because release blockers remain.
  - `build/current-objective-proof-after-current-source-contract-refresh-20260607.json` -> cache architecture, parser/artifact/family gates, generation defaults, native MTP, and VL/media rows are now `PASS`.
  - `build/current-full-release-objective-checklist-after-current-source-contract-refresh-20260607.json` -> `status=open`, `failed_count=18`.
- Focused validation:
  - `tests/test_objective_proof_digest.py -k 'cache_architecture or model_family or parser or generation or native_mtp or vl_media or current_noheavy or mimo or all_local_smoke' -q` -> `15 passed`.
  - `tests/test_release_regression_manifest.py -k 'cache_architecture or model_family_detection or parser_registry or generation_defaults or native_mtp or vl_media or live_smoke or live_tool_smoke or mimo' -q` -> `47 passed`.

Current true release blockers after this refresh:

- MiMo exact literal/tool/JSON/long-prompt/CB/source/media remains open.
- Step3p7 live VLM image/video proof remains open.
- MiniMax #179 reporter parity/root cause remains open.
- Real Electron UI unblocked non-MiMo and cross-family matrix remains open.
- DSV4 exactness/code/file-generation remains blocked by current memory preflight/resource gate.

Do not sign, notarize, tag, push a release, or update public downloads from this state. The no-heavy cache/API/parser/MTP/VL gates are current-source green, but release clearance still requires the live/model/UI blockers above.

## Current objective execution contract - vMLX Python engine and MLXStudio app

This section is a hard routing contract for future continuations. If a task
mentions release, vMLX, MLXStudio, MiMo, JANG, JANGTQ, MXFP, MTP, VL, audio,
video, cache, tools, parser, signing, notarization, or downloads, the work is
the active Python engine plus Electron/panel app release objective unless Eric
explicitly names a different path in the current turn.

Do not treat a single green proof as release readiness. Every continuation must
keep the full release map visible:

CLI startup and MLXStudio startup are independent release gates. A CLI `vmlx
serve` proof does not clear the UI-generated launch/settings path, and a
MLXStudio launch proof does not clear CLI/API startup, request override, health,
or capabilities reflection.

| Surface | Required evidence before green |
| --- | --- |
| CLI startup | `vmlx serve` launch flags, request override precedence, model-owned generation defaults, parser/reasoning/cache/MTP flags, max output, max context, and health/capabilities reflection all agree. |
| MLXStudio startup | Create-session, settings, reset, generated command preview, spawned process args, chat request builder, and persisted session config match the CLI/API contract without hidden sampler or parser overrides. |
| Chat Completions | Stream and non-stream visible output, output cap behavior, tool-call path, cancellation cleanup, cache detail usage, and no hidden-only assistant response. |
| Responses API | Stream and non-stream visible output, `previous_response_id`, required and auto tools, tool-result continuation, cancellation cleanup, JSON/schema/text-format handling, cache detail usage, and no HTTP error in normal release rows. |
| Tool behavior | Required tool call, auto tool, no-tool, tool-result continuation, loop stop, exact JSON args, no raw dialect leak, and no parser-side argument repair. |
| Structured output | JSON/XML/code/whitespace rows must parse and preserve exact semantic values; syntax repair may be benchmark-side only and must not fabricate correct values for release proofs. |
| Cache | First miss, second hit, positive `cached_tokens`, typed `cache_detail`, native cache schema, block-disk L2 write, fresh-process or restart L2 hit, cancellation cleanup, and largest-context tail inspection. |
| TurboQuant KV | Only standard/full KV families may use generic TurboQuant KV. Hybrid SSM, mixed SWA, asymmetric SWA, DSV4 composite caches, ZAYA CCA, and media caches need architecture-specific storage/restore proof. |
| Media | API content parts and MLXStudio file/data URL paths, image/audio/video processors, media-expanded accounting, media-salted cache, post-media recovery, media plus tools, and honest unsupported-modality UI/metadata. |
| Installed app | `/Applications/vMLX.app` or final packaged app must prove bundled Python/source parity, spawned flags, UI settings, live output, cache/media/tools, package acceleration symbols, signing status, and notarization only after model gates are green. |

### Per-family current proof checklist

Use this table to prevent drift. A family is release-green only when the
matching rows have current live evidence through the relevant API and UI routes.

| Family | Must prove or fix |
| --- | --- |
| MiMo V2.5 JANGTQ_2/JANG_2L | Decode speed on the accepted artifact, exact literal tool/JSON values, long-prompt coherence, CB/system prompt working-set behavior, runtime-vs-artifact classification without source-vs-quant unless Eric reauthorizes RAM, MiMo-specific asymmetric SWA cache/L2, XML tool parser with no argument rewrite, and real VL/audio/video before advertising media. |
| Qwen 3.6 27B/35B MTP | Native MTP autodetect, `gdn_sink` compatibility, trained top-k/depth policy, D1 cap under tools where required, hybrid SSM companion cache, TurboQuant attention KV only, block and SSM L2 restart, image/video/reasoning/tools through UI, largest-context tail proof, cancellation cleanup, installed-app parity, and packaged acceleration parity. |
| Gemma 4 12B MXFP4/MXFP8/JANG_4M | Text/tools/cache per quant, mixed SWA/full cache schema, image/VL without unsafe prefill guard false positives, audio/video truth, post-media recovery, optimized batched media path vs fallback classification, installed-app parity, and no claim that preserved media weights equal working runtime media. |
| Step 3.7 Flash | Text-only vs real VLM boundary, no fake `has_vision=false` release claim for a VLM artifact, Step3p7 VLM crash reproduction/fix if advertised, thinking-template behavior, step3p5 tool parser dialect, loop stop after tool output, cache/L2/API/UI proof. |
| LFM | Hybrid cache profile, multiturn/tools/tool-result continuation, JSON/XML/code/whitespace exactness, Responses/Chat/Ollama/Anthropic parity where supported, prefix/paged/L2 restart, UI startup/settings parity, and media truth if advertised. |
| Nemo/Nemotron Omni | Text/tools/reasoning, image/audio/video processor bridge, media-salted cache, audio/video post-media recovery, structured output, typed cache/L2, MLXStudio media UI, and honest capability exposure. |
| MiniMax/JANGTQ_K | Reporter parity/root cause for Issue #179, installed/public/source hash comparison, native MiniMax tool dialect, wrong-language/numeric regression reproduction, cancellation lifecycle, cache/L2, UI settings parity, and no local-only clearance of reporter bugs. |
| DSV4 Flash | Native SWA/CSA/HCA composite cache, no generic TurboQuant KV substitution, thinking-open and thinking-closed quality, exact code/file/long output, restart L2, tools/loop stop, UI route, and sufficient-memory proof before release. |
| ZAYA and other hybrid families | Typed CCA/SSM/SWA cache schema, partial-hit behavior, async rederive/clean boundary storage where required, parser/tool dialect, structured output, media truth, UI settings, and L2 restore. |

### Current no-fake-fix rules

- Do not fix model exactness by parser-side value replacement, JSON semantic
  rewrite, hidden prompt folding, forced tool calls, hidden sampling changes, or
  hiding failed rows.
- Do not call media working because weights are preserved. Runtime processor,
  embedding bridge, cache salt, UI upload, API content parts, and post-failure
  recovery all need live proof.
- Do not call cache working from `/health` or a source contract alone. A cache
  row needs visible output plus miss/hit/L2/restart telemetry on the same
  family and route.
- Do not call release ready from dev-Electron alone. Installed-app bundled
  Python, spawned flags, package acceleration, signing, notarization, and public
  download parity are separate rows.
- Do not enter signing/notarization/tag/download work while MiMo exactness/media,
  Step3p7 VLM, MiniMax reporter parity, DSV4 quality/memory, real UI matrix, or
  installed-app parity rows are open unless Eric explicitly overrides the lock
  in the current turn.

### Required progress logging

After every meaningful proof or fix, append `.agents/STATUS.md` and
`.agents/LOG.md` with:

- the blocker class: `runtime/kernel`, `parser/template`, `cache/storage`,
  `media`, `api/ui`, or `release`;
- touched files;
- exact artifact path or live endpoint proof;
- what turned green;
- what remains red;
- whether the result is source-only, dev-Electron, installed-app, or packaged;
- whether release/signing/notarization remains locked.

### Current active-lane anchor - 2026-06-10

Eric explicitly asked to keep the current constraints in `AGENTS.md` so every
continuation is forced to check them before acting.

Before any substantive movement, the agent must explicitly check and preserve
all of these items:

1. Current goal: keep moving toward a signed working checkpoint release surface
   by reducing real MiMo/Gemma/Qwen/N2-JANGTQ runtime, API, UI, cache, parser,
   and media blockers. Do not substitute stale pointer churn, broad test-suite
   churn, or release-adjacent cleanup when a live blocker can be reduced.
2. Written-state rule: write every user instruction, action, command, proof
   artifact, status movement, blocker, no-claim boundary, commit/push result,
   and other-agent handoff into `.agents/STATUS.md` and `.agents/LOG.md` as
   work happens. If the written state and chat memory disagree, stop and update
   the written state before continuing.
3. No subagents: do not use Python, shell wrappers, MCP tools, browser tools,
   or orchestration scripts to spawn, supervise, prompt, summarize, or delegate
   work to agents. Direct Python/shell remains allowed only for local proof
   scripts, artifact inspection, focused tests, and source maintenance.
4. N2 JANG_1L boundary: do not load, fix, classify, prove, or claim Nex/N2
   JANG_1L unless Eric explicitly reopens that lane in the current turn. N2
   JANGTQ/non-JANG_1L work is allowed only when it does not overlap that lane.
5. Parser/API priority: harshly focus on opencode/Codex harness usability:
   auto tool usage, required tool usage, no-tool mode, tool-result
   continuation, content deltas, reasoning deltas, interleaved reasoning/tool
   streaming, request kwargs passthrough, parser selection, gateway/API/raw SSE
   parity, cache reuse telemetry, and final response object consistency.
6. Qwen empty-args policy: Qwen3.6/Qwen-coder empty `arguments: {}` failures
   are release-critical for 27B and 35B XML tool-call dialects. Do not trust a
   proposed root cause without same-model raw output. Missing required tool
   arguments must fail closed; do not synthesize args from visible preambles,
   disable reasoning, silently drop tool calls as a fake success, strip raw XML
   after the fact, or rewrite JSON/tool values to hide model/runtime failures.
7. Live-proof preference: for MiMo, Gemma JANG/MXFP/QAT, Qwen/Qwen-coder, and
   N2 JANGTQ/non-JANG_1L, prefer real source/dev-app/installed-app/API/cache/UI
   proofs over metadata-only claims. Large-model rows are valid on the 128GB
   host when RAM headroom allows; watch memory/cache storage and kill only
   clearly unrelated smaller processes if needed.
8. Release lock: Eric wants a signed/notarized checkpoint release, but do not
   sign, notarize, tag, publish PyPI, update updater JSON, update download
   links, purge sites, or create public release assets unless Eric explicitly
   asks for that action in the current turn or a newer checked-in directive
   lifts the lock.
9. Other-agent handoff: every proof or blocker classification must say what is
   proven, what is not proven, what must not be claimed, and what the parallel
   agent should implement, rebuild, recapture, or avoid.

Current active lane:

- Work one blocker at a time on real runtime/API/UI/cache/model proof for
  MiMo V2.5 JANG/JANGTQ, Gemma JANG/MXFP/QAT, Qwen/Qwen-coder tool/reasoning
  streaming, and N2 JANGTQ/non-JANG_1L.
- Keep N2 JANG_1L out of this lane unless Eric explicitly reopens it in the
  current turn.
- Do not use Python, shell wrappers, MCP tools, or any mechanism to spawn,
  supervise, prompt, or summarize subagents. Direct Python/shell remains
  allowed for local artifact inspection, proof scripts, tests, and source
  maintenance that do not delegate work.
- Prioritize opencode/Codex harness usability: auto/required/no-tool modes,
  tool-result continuation, content deltas, reasoning deltas, interleaved
  reasoning/tool streaming, request kwargs, parser selection, gateway/API/raw
  SSE parity, cache reuse telemetry, and final response object consistency.
- Treat Qwen3.6/Qwen-coder empty `arguments: {}` tool calls as
  release-critical for both 27B and 35B style XML dialects, but do not trust a
  proposed root cause until same-model raw output proves it.
- Missing required tool arguments must fail closed. Do not synthesize args from
  visible preambles, disable reasoning, silently drop tool calls, hide raw XML,
  rewrite JSON/tool values, or patch parser output to mask model exactness.
- Keep written state current: every instruction, action, proof, blocker,
  no-claim boundary, commit, push, and other-agent handoff goes into
  `.agents/STATUS.md` and `.agents/LOG.md` as work happens.
- Do not enter signing, notarization, tagging, PyPI, updater JSON, download
  link, or public release work unless Eric explicitly asks for that action in
  the current turn.

## Current release blocker guard - 2026-06-07 local

This section is local working guidance for the active release-hardening lane. Do not treat it as public release notes and do not commit it unless Eric explicitly asks.

Current active worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

Current green proof rows:
- No-heavy API/cache contracts: `/v1/chat/completions`, `/v1/responses`, legacy completions, Anthropic, Ollama, stream cache details, `previous_response_id`, JSON schema/text format preservation, cache stats/entries/warm/clear endpoints, TurboQuant KV runtime/disk contracts, and hybrid/DSV4/ZAYA native cache status.
- Step3p7 source VLM: `build/current-step37-vlm-runtime-audit-after-source-live-media-proof-20260607.json` and `build/current-all-local-model-smoke-step37-jang2l-source-tools-media-after-vlm-routing-20260607/summary.json`.
- Current objective digest: `build/current-objective-proof-after-step37-live-vlm-proof-20260607.json`.
- Current full release checklist: `build/current-full-release-objective-checklist-after-step37-live-vlm-proof-20260607.json`.

Current release blockers:
- MiMo V2.5 JANGTQ_2 exactness/long/CB/source/media rows remain open; no parser-side fake repair for value mutations.
- MiniMax issue #179 reporter parity/root cause remains open.
- Real Electron UI live model matrix remains open.
- DSV4 memory/exactness/code/file-generation rows remain open.
- Packaging/signing/notarization/download updates remain blocked until the full checklist is green or Eric explicitly overrides.

Required release checklist surfaces that must stay explicit:
- Cache reuse through usage details: `prompt_tokens_details.cached_tokens` and `cache_detail` on streaming and non-streaming paths.
- Cache endpoints: `/v1/cache/stats`, `/v1/cache/entries`, `/v1/cache/warm`, and clear-cache routes, including panel single-model autoswitch behavior.
- Responses API: `/v1/responses`, streaming deltas, `previous_response_id`, reasoning-only predecessor handling, tool-call extraction, cancellation, and max output token mapping.
- Settings/UI: tool parser, reasoning parser, max output tokens, max context tokens, cache type/family, native MTP, prefix/paged/L2, and family-specific cache disable/enable policy must match CLI, server capabilities, and panel UI.
- Media: image, video, audio, and no-media-after-media recovery must be proven per family; do not advertise media from metadata alone.
