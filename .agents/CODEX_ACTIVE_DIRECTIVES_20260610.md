# Codex Active Directives - 2026-06-10

This file is the current hard lane guard for Codex work in
`/Users/eric/mlx/vllm-mlx-finite-launch-guard`.

## Check Before Action

Before starting any model load, proof run, release step, packaging step, PyPI
step, source edit, or commit, Codex must read this file and confirm the current
allowed lane in the next status update.

Every movement must be logged in `.agents/STATUS.md` and `.agents/LOG.md`:

- what was requested
- what was actually done
- what command/proof/artifact resulted
- what is proven
- what is not proven
- what is blocked
- what must not be claimed
- what the other agent should do next

## Current User Direction

- Do not ignore Eric's goal.
- Write down every instruction, every status change, and every movement.
- Before doing substantive work, write the current-turn instruction/correction
  into `.agents/STATUS.md` and `.agents/LOG.md` so the agent is forced to check
  it instead of drifting.
- Do not use Python, shell wrappers, MCP tools, or any other mechanism to spawn
  or manage subagents for this lane. Direct Python is allowed only for local
  verification, artifact inspection, proof scripts, tests, and source
  maintenance that does not prompt, supervise, or summarize subagents.
- Do not work on N2 JANG_1L unless Eric explicitly reopens that lane.
- Eric said: "im reamming nex n2 jang1l forget about it".
- Eric then corrected the agent after an accidental N2 JANG_1L launch.
- Treat N2 JANG_1L as Eric-owned until explicitly reassigned.

## Current Goal

Keep moving toward a signed working checkpoint release surface and runtime
quality for user-relevant model families, while keeping proof honest.

Primary active lanes for this Codex instance:

- MiMo V2.5 JANG/JANGTQ exactness, media gates, cache/L2/API/UI proof.
- Gemma JANG/MXFP/QAT honest modality detection, VL/video proof, cache/API/UI
  proof, and no fake audio advertisement without weight-backed audio tower.
- Qwen/Qwen3.6/Qwen3-coder Responses/tool/reasoning streaming parity, including
  direct/gateway/tunnel raw SSE, output indices, args deltas, final object
  consistency, and tool-result continuation.
- Harsh priority: auto tool usage, content deltas, reasoning deltas, interleaved
  reasoning/tool streaming, request kwargs, Chat/Responses API behavior,
  gateway passthrough, raw SSE ordering, parser selection, and final-object
  consistency must be tested and fixed across all model reasoning/tool parser
  families. Do not hide failures by synthesizing tool args, disabling
  reasoning, dropping kwargs, or stripping raw parser leaks after the fact.
- N2 JANGTQ/non-JANG_1L rows only if they do not overlap Eric's N2 JANG_1L
  work.
- Installed app/package/release-surface proof only when it follows the proper
  documented signing/notarization/updater workflow.

## Explicit No-Claims

- Do not claim N2 JANG_1L is fixed, proven, release-clear, cache-clear, or
  tool/API-clear from any partial or aborted run.
- Do not treat load-only proof as generation/cache/tool proof.
- Do not patch parser/JSON repair to mask MiMo exactness failures.
- Do not synthesize tool arguments or disable reasoning to hide Qwen tool-call
  failures.
- Do not advertise audio from config/projection tokens only; audio must be
  weight-backed and live-proven.
- Do not publish, tag, sign, notarize, upload, or PyPI-release unless Eric
  explicitly asks for that action in the current lane.

## Latest Correction

At 2026-06-10 02:56 PDT, Codex accidentally launched an N2 JANG_1L proof after
Eric had said to forget that lane. The runner and server were stopped. Port
8876 was verified clear. The aborted run produced only:

- `build/current-n2-jang1l-live-chat-cache-baseline-refresh-20260610.server.log`

There is no JSON proof artifact for that run. It is not a proof and must not be
used as release evidence.

Codex also drafted an unproven N2 JANG_1L server baseline patch and removed it
immediately. No N2 JANG_1L source fix remains from that mistake.

## Next Allowed Work

Proceed one lane at a time, with status updates before commands:

1. MiMo V2.5 JANGTQ_2 exactness/logit/artifact diagnosis.
2. Qwen/Qwen3.6 Responses raw SSE tunnel recapture or source/gateway parity
   follow-up.
3. Cross-family auto-tool/reasoning/content-delta/gateway/API/kwargs parser
   contract: test and fix all model reasoning and tool parsers.
4. Gemma JANG/MXFP/QAT VL/video/cache/API/UI proof and honest modality gating.
5. Installed-app/release-surface verification only after current source proof is
   worth packaging, or if Eric explicitly asks for a checkpoint release action.
